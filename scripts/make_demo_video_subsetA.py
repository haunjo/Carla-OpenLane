"""
CARLA-OpenLane teaser video — 8 diverse Subset-A scenes (Argoverse2, 7 cameras)

Camera grid layout (from OpenLane-V2 gt_generator._render_surround_img):

  Row 1:  ring_front_left | ring_front_center (cropped) | ring_front_right
  Row 2:  ring_side_left  | ring_rear_left | ring_rear_right | ring_side_right
          (Row 2 is resized to match Row 1 width)

BEV panel on the far right (full height).

Output: demo_teaser_subsetA.mp4
"""
import cv2, json, os
import numpy as np
from pathlib import Path

# ── Scene list ────────────────────────────────────────────────────────────────
BASE = Path('/home/user/Carla-OpenLane/LaneSegNet/data/Carla-OpenLane/subset_A')
OUT  = Path(__file__).parent / 'demo_teaser_subsetA.mp4'

SCENES = [
    ('train', '0001', 'Clear  |  Afternoon  |  Urban'),
    ('train', '0069', 'Clear  |  Afternoon  |  Downtown'),
    ('train', '0120', 'Clear  |  Afternoon  |  Highway'),
    ('train', '0195', 'Clear  |  Afternoon  |  Suburban'),
    ('train', '0268', 'Cloudy |  Dusk       |  Downtown (Complex)'),
    ('train', '0073', 'Rain   |  Afternoon  |  Downtown'),
    ('train', '0194', 'Heavy Rain | Afternoon | Suburban'),
    ('train', '0015', 'Clear  |  Night      |  Urban'),
]
FRAMES_PER_SCENE = 6

# ── Video config ──────────────────────────────────────────────────────────────
FRAME_W, FRAME_H = 1920, 1080
HEADER_H = 52
FPS      = 4
BEV_W    = 560
GRID_W   = FRAME_W - BEV_W - 6   # ~1354
CONTENT_H = FRAME_H - HEADER_H   # 1028

# ring_front_center crop rows (portrait 1550×2048 → square-ish 1550×1550)
FC_CROP_TOP    = 356
FC_CROP_BOTTOM = 1906  # height after crop = 1550

# Camera order for each row (indices into CAMS list)
CAMS = ['ring_front_center', 'ring_front_left', 'ring_front_right',
        'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right']
# idx: 0=FC, 1=FL, 2=FR, 3=RL, 4=RR, 5=SL, 6=SR
ROW1_ORDER = [1, 0, 2]          # FL, FC, FR
ROW2_ORDER = [5, 3, 4, 6]       # SL, RL, RR, SR
DIVIDER_W  = 4                   # px between cameras

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
C_BG    = (28, 32, 38)
C_HDR   = (38, 44, 56)
C_GRID  = (65, 65, 65)
C_WHITE = (255, 255, 255)
C_DIM   = (150, 150, 150)
C_BLUE  = (200, 160, 40)

C_REGULAR   = (80,  200,  80)
C_INTERSECT = (30,  165, 255)
C_LL        = (200, 200, 200)
ATTR_COLOR  = {0: (120,120,120), 1: (40,40,230), 2: (40,200,40), 3: (30,220,220)}
INTERP_T    = 150

C_BEV_BG   = (22,  26, 32)
C_BEV_GRID = (42,  46, 52)
C_LANE_REG = (200, 200, 200)
C_LANE_INT = (50,  185, 255)

BEV_X = (-20, 50)
BEV_Y = (-25, 25)


# ── Utilities ─────────────────────────────────────────────────────────────────

def interp_arc(points, t=INTERP_T):
    pts = []
    for p in points:
        p = list(p)
        if not pts or p != pts[-1]:
            pts.append(p)
    if len(pts) <= 1:
        return None
    pts = np.array(pts, dtype=np.float64)
    n   = len(pts)
    eq  = np.linspace(0, 1, t)
    chord = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = chord.sum()
    if total == 0:
        return None
    chord /= total
    cumarc = np.zeros(n)
    cumarc[1:] = np.cumsum(chord)
    bins  = np.clip(np.digitize(eq, cumarc), 1, n - 1)
    denom = np.where(chord[bins - 1] == 0, 1e-9, chord[bins - 1])
    s     = (eq - cumarc[bins - 1]) / denom
    return pts[bins - 1] + (pts[bins] - pts[bins - 1]) * s[:, None]


def project(points_3d, intrinsic, extrinsic):
    pts = np.array(points_3d, dtype=np.float64)
    R   = np.array(extrinsic['rotation'],    dtype=np.float64)
    t   = np.array(extrinsic['translation'], dtype=np.float64).reshape(3, 1)
    K   = np.array(intrinsic['K'],           dtype=np.float64)
    cam = np.linalg.pinv(R) @ (pts.T - t)
    cam = cam[:, cam[2] > 0.1]
    if cam.shape[1] < 2:
        return None
    px = K @ cam
    px /= px[2]
    return px[:2].T


def draw_projected_line(img, points_3d, intrinsic, extrinsic, color, thickness=2):
    pts = interp_arc(points_3d)
    if pts is None:
        return
    px = project(pts, intrinsic, extrinsic)
    if px is None:
        return
    ih, iw = img.shape[:2]
    valid = (px[:, 0] >= 0) & (px[:, 0] < iw) & (px[:, 1] >= 0) & (px[:, 1] < ih)
    prev = None
    for p, v in zip(px, valid):
        if v:
            if prev is not None:
                cv2.line(img, (int(prev[0]), int(prev[1])), (int(p[0]), int(p[1])),
                         color, thickness, cv2.LINE_AA)
            prev = p
        else:
            prev = None


# ── Annotation helpers ────────────────────────────────────────────────────────

def assign_attribute(ann):
    if not ann.get('topology_lste') or not ann.get('traffic_element'):
        for ls in ann['lane_segment']:
            ls['attributes'] = set()
        return ann
    topo = np.array(ann['topology_lste'], dtype=bool)
    for i, ls in enumerate(ann['lane_segment']):
        ls['attributes'] = set(
            ann['traffic_element'][j]['attribute']
            for j in range(topo.shape[1]) if i < topo.shape[0] and topo[i, j]
        )
    return ann


def assign_topology(ann):
    ann['topology'] = []
    if not ann.get('topology_lste') or not ann.get('traffic_element'):
        return ann
    topo = np.array(ann['topology_lste'], dtype=bool)
    for i in range(topo.shape[0]):
        for j in range(topo.shape[1]):
            if topo[i, j]:
                ann['topology'].append({
                    'lane_centerline': ann['lane_segment'][i]['centerline'],
                    'traffic_element': ann['traffic_element'][j]['points'],
                    'attribute':       ann['traffic_element'][j]['attribute'],
                })
    return ann


def draw_topology_curve(img, entry, intrinsic, extrinsic):
    te  = entry['traffic_element']
    src = np.array([(te[0][0] + te[1][0]) / 2, te[1][1]])
    pts = interp_arc(entry['lane_centerline'])
    if pts is None:
        return
    px = project(pts, intrinsic, extrinsic)
    if px is None:
        return
    dst   = px[len(px) // 2]
    color = ATTR_COLOR.get(entry['attribute'], ATTR_COLOR[0])
    mid   = ((dst[0] + src[0]) / 2, (dst[1] + src[1]) / 2 - 40)
    curve = np.array([src, mid, dst])
    try:
        fit   = np.polyfit(curve[:, 0], curve[:, 1], 2)
        xs    = np.linspace(curve[0, 0], curve[-1, 0], 60)
        ys    = np.polyval(fit, xs)
        pts2d = np.int32(np.stack([xs, ys], axis=1).reshape(1, -1, 2))
        cv2.polylines(img, pts2d, False, color, 2, cv2.LINE_AA)
    except Exception:
        cv2.line(img, tuple(src.astype(int)), tuple(dst.astype(int)), color, 2)


def overlay_lanes(img, ann, intrinsic, extrinsic, is_front=False):
    for ls in ann['lane_segment']:
        attrs  = ls.get('attributes', set())
        is_int = ls.get('is_intersection_or_connector', False)
        if attrs:
            cl_color, thick = ATTR_COLOR.get(max(attrs), C_REGULAR), 4
        elif is_int:
            cl_color, thick = C_INTERSECT, 3
        else:
            cl_color, thick = C_REGULAR, 3
        draw_projected_line(img, ls['left_laneline'],  intrinsic, extrinsic, C_LL, 2)
        draw_projected_line(img, ls['right_laneline'], intrinsic, extrinsic, C_LL, 2)
        draw_projected_line(img, ls['centerline'],     intrinsic, extrinsic, cl_color, thick)
    if is_front:
        for te in ann.get('traffic_element', []):
            p1    = (int(te['points'][0][0]) - 6, int(te['points'][0][1]) - 6)
            p2    = (int(te['points'][1][0]) + 6, int(te['points'][1][1]) + 6)
            color = ATTR_COLOR.get(te.get('attribute', 0), ATTR_COLOR[0])
            cv2.rectangle(img, p1, p2, color, 3)
        for topo in ann.get('topology', []):
            draw_topology_curve(img, topo, intrinsic, extrinsic)


# ── Camera grid (2-row surround layout) ────────────────────────────────────────

def load_annotated(scene_dir, cam_name, frame_id, sensor, ann):
    """Load image, draw lane overlay, return raw numpy (original resolution)."""
    img_path = scene_dir / cam_name / f'{frame_id}.jpg'
    if not img_path.exists():
        return None
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    s = sensor[cam_name]
    overlay_lanes(img, ann, s['intrinsic'], s['extrinsic'],
                  is_front=(cam_name == 'ring_front_center'))
    return img


def make_camera_grid(scene_dir, frame_id, sensor, ann, target_w, target_h):
    """
    Builds 2-row surround grid matching OpenLane-V2 _render_surround_img layout,
    then scales to (target_w, target_h).
    """
    div = np.full((1, DIVIDER_W, 3), C_GRID, dtype=np.uint8)  # placeholder

    # Load all 7 images
    imgs = [load_annotated(scene_dir, c, frame_id, sensor, ann) for c in CAMS]
    # Fallback blank (use landscape size)
    blank = lambda: np.full((1550, 2048, 3), C_BG, dtype=np.uint8)
    imgs = [i if i is not None else blank() for i in imgs]

    # ring_front_center: portrait → crop to square 1550×1550
    fc = imgs[0]
    fc = fc[FC_CROP_TOP:FC_CROP_BOTTOM, :]        # (1550, 1550, 3)

    # Row 1: FL | div | FC | div | FR  — all at original height (1550)
    h1    = imgs[1].shape[0]                       # 1550
    divv  = np.full((h1, DIVIDER_W, 3), C_GRID, dtype=np.uint8)
    row1  = np.concatenate([imgs[1], divv, fc, divv, imgs[2]], axis=1)

    # Row 2: SL | div | RL | div | RR | div | SR
    h2    = imgs[5].shape[0]
    divv2 = np.full((h2, DIVIDER_W, 3), C_GRID, dtype=np.uint8)
    row2  = np.concatenate([imgs[5], divv2, imgs[3], divv2, imgs[4], divv2, imgs[6]], axis=1)

    # Scale row2 width to match row1
    scale    = row1.shape[1] / row2.shape[1]
    new_w2   = row1.shape[1]
    new_h2   = int(round(row2.shape[0] * scale))
    row2     = cv2.resize(row2, (new_w2, new_h2))

    # Add horizontal divider between rows
    hdiv  = np.full((DIVIDER_W * 2, row1.shape[1], 3), C_GRID, dtype=np.uint8)
    grid  = np.concatenate([row1, hdiv, row2], axis=0)

    # Scale to target dimensions
    grid = cv2.resize(grid, (target_w, target_h))

    # Camera labels (drawn after resize)
    label_pos = []
    # Row 1 positions (approximate after scale)
    row1_h_scaled = int(round(h1 * target_h / grid.shape[0]))  # approx
    r1h = int(target_h * h1 / (h1 + DIVIDER_W * 2 + new_h2))
    label_pos += [
        ('front left',   5, r1h // 2 - 60),
        ('front center', target_w // 2 - 50, r1h // 2 - 60),
        ('front right',  target_w - 120,     r1h // 2 - 60),
    ]
    r2y0 = r1h + DIVIDER_W * 2
    r2h  = target_h - r2y0
    col4 = target_w // 4
    label_pos += [
        ('side left',   5,             r2y0 + 10),
        ('rear left',   col4 + 5,      r2y0 + 10),
        ('rear right',  col4 * 2 + 5,  r2y0 + 10),
        ('side right',  col4 * 3 + 5,  r2y0 + 10),
    ]
    for label, lx, ly in label_pos:
        cv2.putText(grid, label, (lx, ly + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_WHITE, 1, cv2.LINE_AA)

    return grid


# ── BEV ───────────────────────────────────────────────────────────────────────

def bev_px(x, y, w, h):
    px = int((x - BEV_X[0]) / (BEV_X[1] - BEV_X[0]) * w)
    py = int((BEV_Y[1] - y) / (BEV_Y[1] - BEV_Y[0]) * h)
    return px, py


def dashed_arrow(img, p1, p2, color, thickness=1, dash=8, gap=5):
    x1, y1 = p1; x2, y2 = p2
    L = max(1, int(np.hypot(x2 - x1, y2 - y1)))
    if L < 8: return
    dx, dy = (x2 - x1) / L, (y2 - y1) / L
    i, draw = 0, True
    while i < L - 6:
        xs, ys = int(x1 + dx * i), int(y1 + dy * i)
        ie     = min(i + (dash if draw else gap), L)
        xe, ye = int(x1 + dx * ie), int(y1 + dy * ie)
        if draw:
            cv2.line(img, (xs, ys), (xe, ye), color, thickness)
        i = ie; draw = not draw
    cv2.arrowedLine(img, (int(x2 - dx * 12), int(y2 - dy * 12)), (x2, y2),
                    color, thickness, tipLength=0.6)


def draw_bev(ann, w, h):
    bev = np.full((h, w, 3), C_BEV_BG, dtype=np.uint8)
    for x in range(int(BEV_X[0]), int(BEV_X[1]) + 1, 10):
        cv2.line(bev, bev_px(x, BEV_Y[0], w, h), bev_px(x, BEV_Y[1], w, h), C_BEV_GRID, 1)
    for y in range(int(BEV_Y[0]), int(BEV_Y[1]) + 1, 10):
        cv2.line(bev, bev_px(BEV_X[0], y, w, h), bev_px(BEV_X[1], y, w, h), C_BEV_GRID, 1)

    lsls      = ann.get('topology_lsls', [])
    centroids = []
    for idx, ls in enumerate(ann.get('lane_segment', [])):
        pts    = np.array(ls['centerline'])
        is_int = ls.get('is_intersection_or_connector', False)
        attrs  = ls.get('attributes', set())
        color  = ATTR_COLOR.get(max(attrs), C_LANE_REG) if attrs else (C_LANE_INT if is_int else C_LANE_REG)
        thick  = 3 if (attrs or is_int) else 2
        pxs    = [bev_px(p[0], p[1], w, h) for p in pts]
        for i in range(len(pxs) - 1):
            cv2.line(bev, pxs[i], pxs[i + 1], color, thick, cv2.LINE_AA)
        mid = pts[len(pts) // 2]
        cx, cy = bev_px(mid[0], mid[1], w, h)
        centroids.append((cx, cy))
        cv2.putText(bev, str(ls.get('id', idx)), (cx + 2, cy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 100), 1, cv2.LINE_AA)

    overlay = bev.copy()
    for i, row in enumerate(lsls):
        if i >= len(centroids): break
        for j, val in enumerate(row):
            if val and j < len(centroids) and i != j:
                dashed_arrow(overlay, centroids[i], centroids[j], C_BLUE, 1)
    cv2.addWeighted(overlay, 0.6, bev, 0.4, 0, bev)

    ex, ey = bev_px(0, 0, w, h)
    cv2.rectangle(bev, (ex - 5, ey - 9), (ex + 5, ey + 9), (180, 180, 180), -1)
    cv2.arrowedLine(bev, (ex, ey + 5), (ex, ey - 16), C_WHITE, 2, tipLength=0.5)

    lx, ly = 6, h - 92
    cv2.rectangle(bev, (lx - 2, ly - 4), (lx + 215, h - 4), (36, 40, 48), -1)
    for row_y, lbl, col in [(10, 'Regular lane', C_LANE_REG), (28, 'Intersection lane', C_LANE_INT),
                             (46, 'TE-connected lane', ATTR_COLOR[1])]:
        cv2.line(bev, (lx + 4, ly + row_y), (lx + 28, ly + row_y), col, 2)
        cv2.putText(bev, lbl, (lx + 34, ly + row_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_DIM, 1)
    dashed_arrow(bev, (lx + 4, ly + 64), (lx + 28, ly + 64), C_BLUE, 1)
    cv2.putText(bev, 'TOP_ll lane-lane', (lx + 34, ly + 69), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_DIM, 1)
    return bev


# ── Frame selection ────────────────────────────────────────────────────────────

def pick_frames(info_dir, n):
    files  = sorted(f.replace('-ls.json', '') for f in os.listdir(info_dir) if f.endswith('-ls.json'))
    scored = []
    for fid in files:
        d    = json.load(open(info_dir / f'{fid}-ls.json'))
        ann  = d['annotation']
        lste = sum(sum(r) for r in ann.get('topology_lste', []))
        lsls = sum(sum(r) for r in ann.get('topology_lsls', []))
        scored.append((lste * 10 + lsls, fid, d))
    scored.sort(reverse=True)
    return [(fid, d) for _, fid, d in sorted(scored[:n], key=lambda x: x[1])]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(OUT), fourcc, FPS, (FRAME_W, FRAME_H))

    for split, scene_id, label in SCENES:
        scene_dir = BASE / split / scene_id
        info_dir  = scene_dir / 'info'
        print(f'\n=== {label}  [{scene_id}] ===')

        for frame_id, data in pick_frames(info_dir, FRAMES_PER_SCENE):
            ann    = data['annotation']
            sensor = data['sensor']
            ann    = assign_attribute(ann)
            ann    = assign_topology(ann)

            n_ls   = len(ann['lane_segment'])
            n_te   = len(ann['traffic_element'])
            n_lsls = sum(sum(r) for r in ann.get('topology_lsls', []))
            n_lste = sum(sum(r) for r in ann.get('topology_lste', []))

            canvas = np.full((FRAME_H, FRAME_W, 3), C_BG, dtype=np.uint8)

            # Header
            cv2.rectangle(canvas, (0, 0), (FRAME_W, HEADER_H), C_HDR, -1)
            cv2.putText(canvas, 'CARLA-OpenLane  Subset-A', (14, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.88, C_WHITE, 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (530, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (160, 200, 255), 1, cv2.LINE_AA)
            stats = f'Lanes {n_ls}    TE {n_te}    TOP_ll {n_lsls}    TOP_lt {n_lste}'
            cv2.putText(canvas, stats, (FRAME_W - 560, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 220, 150), 1, cv2.LINE_AA)

            # BEV panel
            bev = draw_bev(ann, BEV_W, CONTENT_H)
            canvas[HEADER_H:, GRID_W + 6:GRID_W + 6 + BEV_W] = bev
            cv2.line(canvas, (GRID_W + 3, HEADER_H), (GRID_W + 3, FRAME_H), C_GRID, 1)
            cv2.putText(canvas, 'BEV  Lane Topology', (GRID_W + 6, HEADER_H - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 200, 255), 1, cv2.LINE_AA)

            # Camera grid
            grid = make_camera_grid(scene_dir, frame_id, sensor, ann, GRID_W, CONTENT_H)
            canvas[HEADER_H:, :GRID_W] = grid

            writer.write(canvas)
            print(f'  {frame_id}: lanes={n_ls}, TE={n_te}, lsls={n_lsls}, lste={n_lste}')

    writer.release()
    print(f'\nSaved -> {OUT}')


if __name__ == '__main__':
    main()
