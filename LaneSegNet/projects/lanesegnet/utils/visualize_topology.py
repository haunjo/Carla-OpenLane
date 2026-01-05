"""
Topology Visualization for BEV (Bird's Eye View)

Adapted from TopoLogic visualization module.
Provides functions to render lane topology in BEV perspective.
"""

import numpy as np
import cv2


# Color scheme
GT_COLOR = (1, 152, 1)        # Green for ground truth
PRED_COLOR = (255, 63, 44)     # Red for predictions
TOPO_COLOR = (255, 63, 44)     # Red for topology connections


def visualize_bev_topology(gt_lanes=None,
                          pred_lanes=None,
                          gt_topology=None,
                          pred_topology=None,
                          gt_traffic=None,
                          pred_traffic=None,
                          mode='both',
                          map_size=[-51.2, 51.2, -25.6, 25.6],
                          scale=10,
                          line_width=None,
                          show_endpoints=True,
                          show_arrows=True):
    """
    Visualize lane topology in BEV (Bird's Eye View).

    Args:
        gt_lanes (list[np.ndarray]): Ground truth lanes, each [N, 3] (x, y, z)
        pred_lanes (list[np.ndarray]): Predicted lanes, each [N, 3] (x, y, z)
        gt_topology (np.ndarray): GT topology adjacency matrix [N_lanes, N_lanes]
        pred_topology (np.ndarray): Predicted topology adjacency matrix
        gt_traffic (list): Ground truth traffic elements (not implemented yet)
        pred_traffic (list): Predicted traffic elements (not implemented yet)
        mode (str): Visualization mode - 'gt', 'pred', or 'both'
        map_size (list): BEV range [x_min, x_max, y_min, y_max] in meters
        scale (int): Pixels per meter for rendering
        line_width (int): Line thickness. If None, auto-scaled
        show_endpoints (bool): Draw circles at lane start/end points
        show_arrows (bool): Draw arrows for topology connections

    Returns:
        np.ndarray: BEV visualization image [H, W, 3]
    """
    # Create white canvas
    height = int(scale * (map_size[1] - map_size[0]))
    width = int(scale * (map_size[3] - map_size[2]))
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Auto-scale line width if not specified
    if line_width is None:
        line_width = max(round(scale * 0.2), 1)

    # Render ground truth
    if (mode == 'both' or mode == 'gt') and gt_lanes is not None:
        image = _draw_lanes(
            image, gt_lanes, GT_COLOR, map_size, scale,
            line_width, show_endpoints
        )
        if gt_topology is not None and show_arrows:
            image = _draw_topology(
                image, gt_lanes, gt_topology, GT_COLOR,
                map_size, scale
            )

    # Render predictions
    if (mode == 'both' or mode == 'pred') and pred_lanes is not None:
        image = _draw_lanes(
            image, pred_lanes, PRED_COLOR, map_size, scale,
            line_width, show_endpoints
        )
        if pred_topology is not None and show_arrows:
            image = _draw_topology(
                image, pred_lanes, pred_topology, TOPO_COLOR,
                map_size, scale
            )

    return image


def _draw_lanes(image, lanes, color, map_size, scale, line_width, show_endpoints):
    """Draw lane polylines on BEV image."""
    for lane in lanes:
        # Convert 3D coordinates to BEV pixel coordinates
        # BEV coordinate system: x forward, y left
        # Image coordinate system: row (down), col (right)
        # Transform: (x, y) -> (-y + y_max, -x + x_max) for proper orientation
        draw_coords = (scale * (-lane[:, :2] + np.array([map_size[1], map_size[3]]))).astype(int)

        # Draw polyline (swap x,y for image coordinates: [row, col] = [y, x])
        image = cv2.polylines(
            image,
            [draw_coords[:, [1, 0]]],
            isClosed=False,
            color=color,
            thickness=line_width
        )

        # Draw start/end points if requested
        if show_endpoints:
            point_radius = max(2, round(scale * 0.5))
            # Start point
            image = cv2.circle(
                image,
                (draw_coords[0, 1], draw_coords[0, 0]),
                point_radius,
                color,
                -1  # Filled circle
            )
            # End point
            image = cv2.circle(
                image,
                (draw_coords[-1, 1], draw_coords[-1, 0]),
                point_radius,
                color,
                -1
            )

    return image


def _draw_topology(image, lanes, topology, color, map_size, scale):
    """Draw topology connections as arrows on BEV image."""
    arrow_thickness = max(round(scale * 0.15), 1)

    for l1_idx, connections in enumerate(topology):
        for l2_idx, is_connected in enumerate(connections):
            if is_connected:
                # Get midpoint of each lane for arrow placement
                l1 = lanes[l1_idx]
                l2 = lanes[l2_idx]
                l1_mid = len(l1) // 2
                l2_mid = len(l2) // 2

                # Convert to pixel coordinates
                p1 = (scale * (-l1[l1_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(int)
                p2 = (scale * (-l2[l2_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(int)

                # Draw arrow from l1 to l2
                image = cv2.arrowedLine(
                    image,
                    (p1[1], p1[0]),  # Start (col, row)
                    (p2[1], p2[0]),  # End (col, row)
                    color,
                    arrow_thickness,
                    tipLength=0.15
                )

    return image


def visualize_lane_traffic_topology(gt_lanes=None,
                                    pred_lanes=None,
                                    gt_traffic=None,
                                    pred_traffic=None,
                                    gt_lane_to_lane=None,
                                    pred_lane_to_lane=None,
                                    gt_lane_to_traffic=None,
                                    pred_lane_to_traffic=None,
                                    mode='both',
                                    map_size=[-51.2, 51.2, -25.6, 25.6],
                                    scale=10):
    """
    Visualize both lane-to-lane and lane-to-traffic topology in BEV.

    Args:
        gt_lanes (list[np.ndarray]): GT lanes [N_lanes, N_points, 3]
        pred_lanes (list[np.ndarray]): Predicted lanes
        gt_traffic (list[np.ndarray]): GT traffic elements [N_traffic, 4] (bbox)
        pred_traffic (list[np.ndarray]): Predicted traffic elements
        gt_lane_to_lane (np.ndarray): GT lane topology [N_lanes, N_lanes]
        pred_lane_to_lane (np.ndarray): Predicted lane topology
        gt_lane_to_traffic (np.ndarray): GT lane-traffic topology [N_lanes, N_traffic]
        pred_lane_to_traffic (np.ndarray): Predicted lane-traffic topology
        mode (str): 'gt', 'pred', or 'both'
        map_size (list): BEV range [x_min, x_max, y_min, y_max]
        scale (int): Pixels per meter

    Returns:
        np.ndarray: BEV visualization image
    """
    # Create white canvas
    height = int(scale * (map_size[1] - map_size[0]))
    width = int(scale * (map_size[3] - map_size[2]))
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    line_width = max(round(scale * 0.2), 1)

    # Draw lanes and lane-to-lane topology
    if mode in ['both', 'gt']:
        if gt_lanes is not None:
            image = _draw_lanes(image, gt_lanes, GT_COLOR, map_size, scale, line_width, True)
        if gt_lane_to_lane is not None and gt_lanes is not None:
            image = _draw_topology(image, gt_lanes, gt_lane_to_lane, GT_COLOR, map_size, scale)
        if gt_traffic is not None:
            image = _draw_traffic_elements(image, gt_traffic, GT_COLOR, map_size, scale)
        if gt_lane_to_traffic is not None and gt_lanes is not None and gt_traffic is not None:
            image = _draw_lane_to_traffic(
                image, gt_lanes, gt_traffic, gt_lane_to_traffic,
                GT_COLOR, map_size, scale
            )

    if mode in ['both', 'pred']:
        if pred_lanes is not None:
            image = _draw_lanes(image, pred_lanes, PRED_COLOR, map_size, scale, line_width, True)
        if pred_lane_to_lane is not None and pred_lanes is not None:
            image = _draw_topology(image, pred_lanes, pred_lane_to_lane, TOPO_COLOR, map_size, scale)
        if pred_traffic is not None:
            image = _draw_traffic_elements(image, pred_traffic, PRED_COLOR, map_size, scale)
        if pred_lane_to_traffic is not None and pred_lanes is not None and pred_traffic is not None:
            image = _draw_lane_to_traffic(
                image, pred_lanes, pred_traffic, pred_lane_to_traffic,
                TOPO_COLOR, map_size, scale
            )

    return image


def _draw_traffic_elements(image, traffic_elements, color, map_size, scale):
    """
    Draw traffic elements (e.g., traffic lights) as boxes in BEV.

    Args:
        traffic_elements (list): List of traffic element positions
        color (tuple): RGB color
    """
    # TODO: Implement traffic element visualization
    # This depends on how traffic elements are represented in your data
    return image


def _draw_lane_to_traffic(image, lanes, traffic, topology, color, map_size, scale):
    """
    Draw lane-to-traffic connections as arrows.

    Args:
        lanes (list[np.ndarray]): Lane polylines
        traffic (list): Traffic element positions
        topology (np.ndarray): [N_lanes, N_traffic] adjacency matrix
        color (tuple): RGB color
    """
    # TODO: Implement lane-to-traffic visualization
    # Need to determine traffic element representation first
    return image
