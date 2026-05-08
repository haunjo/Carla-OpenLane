"""
Example usage of topology visualization functions.

This script demonstrates how to use the BEV topology visualization module.
"""

import numpy as np
import cv2
from visualize_topology import visualize_bev_topology, visualize_lane_traffic_topology


def example_basic_usage():
    """Basic example: Visualize lanes and topology."""

    # Create dummy lane data (3 lanes)
    # Each lane: [N_points, 3] where columns are (x, y, z)
    lane1 = np.array([
        [0, 0, 0],
        [10, 0, 0],
        [20, 0, 0],
    ])

    lane2 = np.array([
        [20, 0, 0],
        [30, 5, 0],
        [40, 10, 0],
    ])

    lane3 = np.array([
        [0, -5, 0],
        [10, -5, 0],
        [20, -5, 0],
    ])

    gt_lanes = [lane1, lane2, lane3]

    # Create topology: lane1 connects to lane2
    gt_topology = np.array([
        [0, 1, 0],  # lane1 -> lane2
        [0, 0, 0],  # lane2 -> none
        [0, 1, 0],  # lane3 -> lane2
    ])

    # Visualize
    image = visualize_bev_topology(
        gt_lanes=gt_lanes,
        gt_topology=gt_topology,
        mode='gt',
        map_size=[-51.2, 51.2, -25.6, 25.6],  # OpenLane-V2 default
        scale=10,
        show_endpoints=True,
        show_arrows=True
    )

    # Save result
    cv2.imwrite('bev_topology_example.png', image)
    print("✓ Saved: bev_topology_example.png")


def example_gt_vs_pred():
    """Example: Compare ground truth vs predictions."""

    # GT lanes
    gt_lane1 = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]])
    gt_lane2 = np.array([[20, 0, 0], [30, 5, 0], [40, 10, 0]])
    gt_lanes = [gt_lane1, gt_lane2]

    # Predicted lanes (slightly different)
    pred_lane1 = np.array([[0, 0.5, 0], [10, 0.3, 0], [20, 0.2, 0]])
    pred_lane2 = np.array([[20, 0.2, 0], [30, 4.8, 0], [40, 9.5, 0]])
    pred_lanes = [pred_lane1, pred_lane2]

    # GT topology
    gt_topology = np.array([
        [0, 1],  # lane1 -> lane2
        [0, 0],
    ])

    # Predicted topology (same as GT in this example)
    pred_topology = gt_topology.copy()

    # Visualize both
    image = visualize_bev_topology(
        gt_lanes=gt_lanes,
        pred_lanes=pred_lanes,
        gt_topology=gt_topology,
        pred_topology=pred_topology,
        mode='both',  # Show both GT and predictions
        scale=10
    )

    cv2.imwrite('bev_gt_vs_pred.png', image)
    print("✓ Saved: bev_gt_vs_pred.png")


def example_from_model_output():
    """
    Example: Visualize results from model inference.

    This shows how to use the function with actual LaneSegNet output.
    """
    # Assume you have model outputs like this:
    # results = model.forward_test(...)
    # lane_results = results['lane_results']
    # topology_results = results['topology_results']

    # Example structure (replace with actual model output):
    # lane_results = {
    #     'lanes_3d': list of [N, 3] arrays,
    #     'scores': list of confidence scores,
    # }
    # topology_results = np.ndarray [N_lanes, N_lanes]

    # For demonstration, use dummy data
    pred_lanes = [
        np.random.randn(10, 3) * 10,  # Random lane 1
        np.random.randn(10, 3) * 10,  # Random lane 2
        np.random.randn(10, 3) * 10,  # Random lane 3
    ]

    # Predicted topology (adjacency matrix)
    pred_topology = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])

    # Visualize predictions only
    image = visualize_bev_topology(
        pred_lanes=pred_lanes,
        pred_topology=pred_topology,
        mode='pred',
        scale=8,  # Smaller scale for wider view
        line_width=2,
        show_endpoints=True,
        show_arrows=True
    )

    cv2.imwrite('bev_model_output.png', image)
    print("✓ Saved: bev_model_output.png")


def example_custom_settings():
    """Example: Custom visualization settings."""

    # Create some lanes
    lanes = [
        np.array([[i*5, j*2, 0] for i in range(10)])
        for j in range(5)
    ]

    # Dense topology (all lanes connect sequentially)
    N = len(lanes)
    topology = np.zeros((N, N))
    for i in range(N-1):
        topology[i, i+1] = 1

    # Custom settings
    image = visualize_bev_topology(
        gt_lanes=lanes,
        gt_topology=topology,
        mode='gt',
        map_size=[-60, 60, -30, 30],  # Custom BEV range
        scale=5,                        # Lower scale = wider view
        line_width=3,                   # Thicker lines
        show_endpoints=False,           # Hide endpoints
        show_arrows=True
    )

    cv2.imwrite('bev_custom_settings.png', image)
    print("✓ Saved: bev_custom_settings.png")


if __name__ == '__main__':
    print("Running topology visualization examples...\n")

    example_basic_usage()
    example_gt_vs_pred()
    example_from_model_output()
    example_custom_settings()

    print("\n✓ All examples completed!")
    print("Check the generated PNG files in the current directory.")
