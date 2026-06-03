"""
CARLA-OpenLane teaser video — 8 diverse Subset-B scenes
Conditions: clear / fog / mid_rain / hard_rain / soft_rain / cloudy / wet / wet_cloudy

Per frame:
  Left  (2/3): 6-camera grid with lane overlay
               - Lane centerline colored by TE attribute (assign_attribute)
               - Front camera: TE bbox + lane-TE topology curves (assign_topology)
  Right (1/3): BEV — lane centerlines + lsls arrows only

Output: demo_teaser_subsetB.mp4
"""
import cv2, json, os
import numpy as np
from pathlib import Path

# ── Scene list — 8 diverse weather/time conditions ─────────────────────────
BASE = Path('/home/user/Carla-OpenLane/LaneSegNet/data/Carla-OpenLane/subset_B')
OUT  = Path(__file__).parent / 'demo_teaser_subsetB.mp4'

SCENES = [
    # (split, scene_id, label)
    ('train', '0293', 'Clear  |  Sunset  |  Downtown'),
    ('train', '0331', 'Fog  |  Noon  |  Downtown'),
    ('train', '0130', 'Mid Rain  |  Sunset  |  Downtown'),
    ('train', '0315', 'Hard Rain  |  Sunset  |  Downtown'),
    ('train', '0125', 'Soft Rain  |  Sunset  |  Downtown'),
    ('train', '0106', 'Cloudy  |  Sunset  |  Downtown'),
    ('train', '0128', 'Wet  |  Sunset  |  Downtown'),
    ('train', '0112', 'Wet Cloudy  |  Noon  |  Downtown'),
]
FRAMES_PER_SCENE = 6   # pick top-N frames by topology richness

# ── Video config ────────────────────────────────────────────────────────────
FRAME_W, FRAME_H = 1920, 1080
HEADER_H = 52
FPS      = 4
BEV_W    = 680

CAM_GRID = [
    ['CAM_FRONT_LEFT', 'CAM_FRONT',  'CAM_FRONT_RIGHT'],
    ['CAM_BACK_LEFT',  'CAM_BACK',   'CAM_BACK_RIGHT'],
]

# ── Colors (BGR) ─────────────────────────────────────────────────────────────
C_BG   = (28, 32, 38)
C_HDR  = (38, 44, 56)
C_GRID = (65, 65, 65)
C_WHITE = (255, 255, 255)
C_DIM   = (150, 150, 150)
C_BLUE  = (200, 160, 40)   # lsls arrows in BEV (blue-ish)

# Lane centerline colors (camera overlay)
C_REGULAR    = (80,  200,  80)    # green — unconnected
C_INTERSECT  = (30,  165, 255)    # orange — intersection unconnected
C_LL         = (200, 200, 200)    # white — laneline boundaries
# TE-attribute colors for connected lanes (BGR)
#   attr 0 = unknown, 1 = red, 2 = green, 3 = yellow
ATTR_COLOR = {
    0: (120, 120, 120),   # gray
    1: (40,   40, 230),   # red light
    2: (40,  200,  40),   # green light
    3: (30,  220, 220),   # yellow light
}
INTERP_T = 150

# BEV
C_BEV_BG   = (22,  26, 32)
C_BEV_GRID = (42,  46, 52)
C_LANE_REG = (200, 200, 200)
C_LANE_INT = (50,  185, 255)

BEV_X = (-20, 50)
BEV_Y = (-25, 25)


# ── Utilities ────────────────────────────────────────────────────────────────

def interp_arc(points, t=INTERP_T):
    pts = []
    for p in points:
        p = list(p)
        if not pts or p != pts[-1]:
            pts.append(p)
    if len(pts) <= 1:
        return None
    pts = np.array(pts, dtype=np.float64)
    n = len(pts)
    eq = np.linspace(0, 1, t)
    chord = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = chord.sum()
    if total == 0:
        return None
    chord /= total
    cumarc = np.zeros(n)
    cumarc[1:] = np.cumsum(chord)
    bins = np.clip(np.digitize(eq, cumarc), 1, n-1)
    denom = chord[bins-1]
    denom = np.where(denom == 0, 1e-9, denom)
    s = (eq - cumarc[bins-1]) / denom
    return pts[bins-1] + (pts[bins] - pts[bins-1]) * s[:, None]


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


def draw_projected_line(img, points_3d, intrinsic, extrinsic, color, thickness=3):
    pts = interp_arc(points_3d)
    if pts is None:
        return
    px = project(pts, intrinsic, extrinsic)
    if px is None:
        return
    ih, iw = img.shape[:2]
    valid = (px[:,0] >= 0) & (px[:,0] < iw) & (px[:,1] >= 0) & (px[:,1] < ih)
    prev = None
    for p, v in zip(px, valid):
        if v:
            if prev is not None:
                cv2.line(img, (int(prev[0]),int(prev[1])), (int(p[0]),int(p[1])),
                         color, thickness, cv2.LINE_AA)
            prev = p
        else:
            prev = None


# ── Annotation preparation ───────────────────────────────────────────────────

def assign_attribute(ann):
    """Color each lane by its connected TE attribute (adapted from utils.py)."""
    if not ann.get('topology_lste') or not ann.get('traffic_element'):
        for ls in ann['lane_segment']:
            ls['attributes'] = set()
        return ann
    topo = np.array(ann['topology_lste'], dtype=bool)
    for i, ls in enumerate(ann['lane_segment']):
        if i < topo.shape[0]:
            ls['attributes'] = set(
                ann['traffic_element'][j]['attribute']
                for j in range(topo.shape[1]) if topo[i, j]
            )
        else:
            ls['attributes'] = set()
    return ann


def assign_topology(ann):
    """Build topology list for front-cam TE→lane curve drawing (from utils.py)."""
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


# ── Camera overlay ────────────────────────────────────────────────────────────

def draw_topology_curve(img, topology_entry, intrinsic, extrinsic):
    """Draw curved line from TE bbox center to lane midpoint (pv.py _draw_topology)."""
    te_pts = topology_entry['traffic_element']
    coord_from = np.array([
        (te_pts[0][0] + te_pts[1][0]) / 2,
        te_pts[1][1],   # bottom of bbox
    ])
    pts = interp_arc(topology_entry['lane_centerline'])
    if pts is None:
        return
    px = project(pts, intrinsic, extrinsic)
    if px is None:
        return
    coord_to = px[len(px) // 2]
    attr  = topology_entry['attribute']
    color = ATTR_COLOR.get(attr, ATTR_COLOR[0])

    # Fit quadratic curve through (from, mid, to)
    mid = ((coord_to[0]+coord_from[0])/2, (coord_to[1]+coord_from[1])/2 - 40)
    curve = np.array([coord_from, mid, coord_to])
    try:
        fit = np.polyfit(curve[:,0], curve[:,1], 2)
        xs  = np.linspace(curve[0,0], curve[-1,0], 60)
        ys  = np.polyval(fit, xs)
        pts2d = np.int32(np.stack([xs, ys], axis=1).reshape(1, -1, 2))
        cv2.polylines(img, pts2d, False, color, 2, cv2.LINE_AA)
    except Exception:
        cv2.line(img, tuple(coord_from.astype(int)), tuple(coord_to.astype(int)), color, 2)


def overlay_lanes(img, ann, intrinsic, extrinsic, is_front=False):
    for ls in ann['lane_segment']:
        attrs   = ls.get('attributes', set())
        is_int  = ls.get('is_intersection_or_connector', False)

        # Choose centerline color: TE-attribute > intersection > regular
        if attrs:
            cl_color = ATTR_COLOR.get(max(attrs), C_REGULAR)
            thickness = 4
        elif is_int:
            cl_color = C_INTERSECT
            thickness = 3
        else:
            cl_color = C_REGULAR
            thickness = 3

        draw_projected_line(img, ls['left_laneline'],  intrinsic, extrinsic, C_LL, 2)
        draw_projected_line(img, ls['right_laneline'], intrinsic, extrinsic, C_LL, 2)
        draw_projected_line(img, ls['centerline'],     intrinsic, extrinsic, cl_color, thickness)

    if is_front:
        # TE bounding boxes
        for te in ann.get('traffic_element', []):
            pt1   = (int(te['points'][0][0]), int(te['points'][0][1]))
            pt2   = (int(te['points'][1][0]), int(te['points'][1][1]))
            color = ATTR_COLOR.get(te.get('attribute', 0), ATTR_COLOR[0])
            # Expand bbox for visibility
            pad = 6
            pt1 = (pt1[0]-pad, pt1[1]-pad)
            pt2 = (pt2[0]+pad, pt2[1]+pad)
            cv2.rectangle(img, pt1, pt2, color, 3, cv2.LINE_AA)

        # TE→lane topology curves
        for topo in ann.get('topology', []):
            draw_topology_curve(img, topo, intrinsic, extrinsic)


# ── BEV ──────────────────────────────────────────────────────────────────────

def bev_px(x, y, w, h):
    px = int((x - BEV_X[0]) / (BEV_X[1]-BEV_X[0]) * w)
    py = int((BEV_Y[1] - y) / (BEV_Y[1]-BEV_Y[0]) * h)
    return px, py


def dashed_arrow(img, p1, p2, color, thickness=1, dash=8, gap=5):
    x1,y1 = p1; x2,y2 = p2
    L = max(1, int(np.hypot(x2-x1, y2-y1)))
    if L < 8: return
    dx,dy = (x2-x1)/L, (y2-y1)/L
    i, draw = 0, True
    while i < L-6:
        xs,ys = int(x1+dx*i), int(y1+dy*i)
        ie = min(i+(dash if draw else gap), L)
        xe,ye = int(x1+dx*ie), int(y1+dy*ie)
        if draw:
            cv2.line(img,(xs,ys),(xe,ye),color,thickness)
        i=ie; draw=not draw
    cv2.arrowedLine(img,(int(x2-dx*12),int(y2-dy*12)),(x2,y2),color,thickness,tipLength=0.6)


def draw_bev(ann, w, h):
    bev = np.full((h, w, 3), C_BEV_BG, dtype=np.uint8)
    for x in range(int(BEV_X[0]), int(BEV_X[1])+1, 10):
        cv2.line(bev, bev_px(x,BEV_Y[0],w,h), bev_px(x,BEV_Y[1],w,h), C_BEV_GRID, 1)
    for y in range(int(BEV_Y[0]), int(BEV_Y[1])+1, 10):
        cv2.line(bev, bev_px(BEV_X[0],y,w,h), bev_px(BEV_X[1],y,w,h), C_BEV_GRID, 1)

    lanes     = ann.get('lane_segment', [])
    lsls      = ann.get('topology_lsls', [])
    centroids = []
    for idx, ls in enumerate(lanes):
        pts    = np.array(ls['centerline'])
        is_int = ls.get('is_intersection_or_connector', False)
        attrs  = ls.get('attributes', set())
        if attrs:
            color = ATTR_COLOR.get(max(attrs), C_LANE_REG)
            thick = 3
        elif is_int:
            color = C_LANE_INT; thick = 3
        else:
            color = C_LANE_REG; thick = 2
        pxs = [bev_px(p[0],p[1],w,h) for p in pts]
        for i in range(len(pxs)-1):
            cv2.line(bev, pxs[i], pxs[i+1], color, thick, cv2.LINE_AA)
        mid = pts[len(pts)//2]
        cx, cy = bev_px(mid[0], mid[1], w, h)
        centroids.append((cx, cy))
        # Lane segment index label
        lane_id = ls.get('id', idx)
        cv2.putText(bev, str(lane_id), (cx+2, cy-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 100), 1, cv2.LINE_AA)

    overlay = bev.copy()
    for i,row in enumerate(lsls):
        if i >= len(centroids): break
        for j,val in enumerate(row):
            if val and j < len(centroids) and i != j:
                dashed_arrow(overlay, centroids[i], centroids[j], C_BLUE, 1)

    cv2.addWeighted(overlay, 0.6, bev, 0.4, 0, bev)

    ex,ey = bev_px(0,0,w,h)
    cv2.rectangle(bev,(ex-5,ey-9),(ex+5,ey+9),(180,180,180),-1)
    cv2.arrowedLine(bev,(ex,ey+5),(ex,ey-16),C_WHITE,2,tipLength=0.5)

    # legend
    lx,ly = 6, h-92
    cv2.rectangle(bev,(lx-2,ly-4),(lx+215,h-4),(36,40,48),-1)
    cv2.line(bev,(lx+4,ly+10),(lx+28,ly+10),C_LANE_REG,2)
    cv2.putText(bev,'Regular lane',(lx+34,ly+15),cv2.FONT_HERSHEY_SIMPLEX,0.38,C_DIM,1)
    cv2.line(bev,(lx+4,ly+28),(lx+28,ly+28),C_LANE_INT,2)
    cv2.putText(bev,'Intersection lane',(lx+34,ly+33),cv2.FONT_HERSHEY_SIMPLEX,0.38,C_DIM,1)
    cv2.line(bev,(lx+4,ly+46),(lx+28,ly+46),ATTR_COLOR[1],3)
    cv2.putText(bev,'TE-connected lane',(lx+34,ly+51),cv2.FONT_HERSHEY_SIMPLEX,0.38,C_DIM,1)
    dashed_arrow(bev,(lx+4,ly+64),(lx+28,ly+64),C_BLUE,1)
    cv2.putText(bev,'TOP_ll lane-lane',(lx+34,ly+69),cv2.FONT_HERSHEY_SIMPLEX,0.38,C_DIM,1)
    return bev


# ── Camera grid ───────────────────────────────────────────────────────────────

def make_camera_grid(scene_dir, frame_id, sensor_data, ann, grid_w, grid_h):
    cell_w = grid_w // 3
    cell_h = grid_h // 2
    grid   = np.full((grid_h, grid_w, 3), C_BG, dtype=np.uint8)
    for r, row_cams in enumerate(CAM_GRID):
        for c, cam in enumerate(row_cams):
            img_path = scene_dir / cam / f'{frame_id}.jpg'
            if not img_path.exists(): continue
            img = cv2.imread(str(img_path))
            if img is None: continue
            s = sensor_data[cam]
            overlay_lanes(img, ann, s['intrinsic'], s['extrinsic'], is_front=(cam=='CAM_FRONT'))
            img_r = cv2.resize(img, (cell_w, cell_h))
            y0, x0 = r*cell_h, c*cell_w
            grid[y0:y0+cell_h, x0:x0+cell_w] = img_r
            cv2.rectangle(grid,(x0,y0),(x0+cell_w-1,y0+cell_h-1),C_GRID,1)
            label = cam.replace('CAM_','').replace('_',' ').lower()
            cv2.putText(grid, label,(x0+5,y0+16),cv2.FONT_HERSHEY_SIMPLEX,0.38,C_WHITE,1,cv2.LINE_AA)
    return grid


# ── Frame selection ────────────────────────────────────────────────────────

def pick_frames(info_dir, n):
    """Pick n most topology-rich frames; fill with evenly-spaced if needed."""
    files = sorted(f.replace('-ls.json','') for f in os.listdir(info_dir) if f.endswith('-ls.json'))
    scored = []
    for fid in files:
        with open(info_dir / f'{fid}-ls.json') as f:
            d = json.load(f)
        ann = d['annotation']
        lste = sum(sum(r) for r in ann.get('topology_lste',[]))
        lsls = sum(sum(r) for r in ann.get('topology_lsls',[]))
        scored.append((lste*10 + lsls, fid, d))
    scored.sort(reverse=True)
    # take top-n by score, then sort chronologically
    chosen = sorted(scored[:n], key=lambda x: x[1])
    return [(fid, d) for _, fid, d in chosen]


# ── Title card ─────────────────────────────────────────────────────────────

def make_title_card(label, scene_id):
    card = np.full((FRAME_H, FRAME_W, 3), (20, 24, 30), dtype=np.uint8)
    cv2.putText(card, 'CARLA-OpenLane  Subset-B', (60, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, C_WHITE, 3, cv2.LINE_AA)
    cv2.putText(card, label, (60, 490),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (160, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(card, f'Scene {scene_id}', (60, 540),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_DIM, 1, cv2.LINE_AA)
    return card


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(OUT), fourcc, FPS, (FRAME_W, FRAME_H))

    content_h = FRAME_H - HEADER_H
    grid_w    = FRAME_W - BEV_W - 10

    for split, scene_id, label in SCENES:
        scene_dir = BASE / split / scene_id
        info_dir  = scene_dir / 'info'
        print(f'\n=== {label}  [{scene_id}] ===')

        frames = pick_frames(info_dir, FRAMES_PER_SCENE)
        for frame_id, data in frames:
            ann    = data['annotation']
            sensor = data['sensor']
            ann    = assign_attribute(ann)
            ann    = assign_topology(ann)

            n_ls   = len(ann['lane_segment'])
            n_te   = len(ann['traffic_element'])
            n_lsls = sum(sum(r) for r in ann.get('topology_lsls',[]))
            n_lste = sum(sum(r) for r in ann.get('topology_lste',[]))

            canvas = np.full((FRAME_H, FRAME_W, 3), C_BG, dtype=np.uint8)

            # Header
            cv2.rectangle(canvas,(0,0),(FRAME_W,HEADER_H),C_HDR,-1)
            cv2.putText(canvas,'CARLA-OpenLane  Subset-B',(14,34),
                        cv2.FONT_HERSHEY_SIMPLEX,0.88,C_WHITE,2,cv2.LINE_AA)
            cv2.putText(canvas, label,(570,34),
                        cv2.FONT_HERSHEY_SIMPLEX,0.60,(160,200,255),1,cv2.LINE_AA)
            stats = f'Lanes {n_ls}    TE {n_te}    TOP_ll {n_lsls}    TOP_lt {n_lste}'
            cv2.putText(canvas, stats,(FRAME_W-560,34),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(150,220,150),1,cv2.LINE_AA)

            grid = make_camera_grid(scene_dir, frame_id, sensor, ann, grid_w, content_h)
            canvas[HEADER_H:, :grid_w] = grid

            bev = draw_bev(ann, BEV_W, content_h)
            canvas[HEADER_H:, grid_w+6:grid_w+6+BEV_W] = bev
            cv2.line(canvas,(grid_w+3,HEADER_H),(grid_w+3,FRAME_H),C_GRID,1)
            cv2.putText(canvas,'BEV  Lane Topology',(grid_w+6,HEADER_H-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.44,(160,200,255),1,cv2.LINE_AA)

            writer.write(canvas)
            print(f'  {frame_id}: lanes={n_ls}, TE={n_te}, lsls={n_lsls}, lste={n_lste}')

    writer.release()
    print(f'\nSaved -> {OUT}')


if __name__ == '__main__':
    main()
