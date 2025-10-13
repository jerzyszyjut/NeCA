import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import os
import cc3d
from typing import Tuple


def load_data(gt_path: str, pred_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ground truth and prediction numpy arrays."""
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    gt = np.load(gt_path)
    pred = np.load(pred_path)

    print(
        f"GT shape: {gt.shape}, dtype: {gt.dtype}, range: [{gt.min():.4f}, {gt.max():.4f}]"
    )
    print(
        f"Pred shape: {pred.shape}, dtype: {pred.dtype}, range: [{pred.min():.4f}, {pred.max():.4f}]"
    )

    return gt, pred


def apply_connected_components_filter(
    volume: np.ndarray, threshold: float = 0.5, min_voxels: int = 25
) -> np.ndarray:
    """
    Apply connected components filtering to remove small disconnected objects.

    Args:
        volume: 3D numpy array
        threshold: threshold to binarize the volume
        min_voxels: minimum number of voxels for a component to be kept

    Returns:
        Filtered binary volume
    """
    # Binarize the volume
    binary_volume = (volume > threshold).astype(np.uint8)

    # Apply connected components analysis
    labels = cc3d.connected_components(binary_volume, connectivity=26)

    # Remove components with less than min_voxels
    filtered_labels = cc3d.dust(labels, threshold=min_voxels, connectivity=26)

    # Convert back to binary
    filtered_binary = (filtered_labels > 0).astype(np.float32)

    # For prediction data, preserve original values in filtered regions
    if volume.max() <= 1.0 and volume.min() >= 0.0:  # Sigmoid output
        filtered_volume = volume * filtered_binary
    else:  # Binary ground truth
        filtered_volume = filtered_binary

    return filtered_volume


def volume_to_points(
    volume: np.ndarray,
    threshold: float = 0.5,
    apply_cc_filter: bool = True,
    min_voxels: int = 25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert 3D volume to point coordinates where values exceed threshold."""
    indices = np.where(volume > threshold)
    x, y, z = indices
    values = volume[indices]

    if apply_cc_filter:
        values = apply_connected_components_filter(values, threshold, min_voxels)

    return x, y, z, values


def create_interactive_visualization(
    gt: np.ndarray,
    pred: np.ndarray,
    output_path: str = "pointcloud_comparison.html",
    apply_cc_filter: bool = True,
    min_voxels: int = 25,
):
    """Create interactive HTML visualization with threshold slider."""

    # Get GT points (threshold = 0.5 for binary data)
    gt_x, gt_y, gt_z, gt_values = volume_to_points(
        gt, threshold=0.5, apply_cc_filter=apply_cc_filter, min_voxels=min_voxels
    )

    # Calculate percentile thresholds for pred
    percentiles = [50, 75, 90, 95, 99]
    thresholds = [np.percentile(pred[pred > 0], p) for p in percentiles]
    default_threshold = thresholds[2]  # 90th percentile

    print(
        f"Prediction thresholds at percentiles {percentiles}: {[f'{t:.4f}' for t in thresholds]}"
    )
    print(f"Using default threshold: {default_threshold:.4f} (90th percentile)")

    # Get initial pred points
    pred_x, pred_y, pred_z, pred_values = volume_to_points(
        pred,
        threshold=default_threshold,
        apply_cc_filter=apply_cc_filter,
        min_voxels=min_voxels,
    )

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Ground Truth",
            f"Prediction (threshold: {default_threshold:.4f})",
        ),
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    )

    # Add GT points
    fig.add_trace(
        go.Scatter3d(
            x=gt_x,
            y=gt_y,
            z=gt_z,
            mode="markers",
            marker=dict(size=3, color="red", opacity=0.6),
            name=f"GT ({len(gt_x)} points)",
            hovertemplate="<b>Ground Truth</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add pred points
    fig.add_trace(
        go.Scatter3d(
            x=pred_x,
            y=pred_y,
            z=pred_z,
            mode="markers",
            marker=dict(
                size=3,
                color=pred_values,
                colorscale="Viridis",
                opacity=0.6,
                colorbar=dict(title="Prediction Value", x=1.02),
            ),
            name=f"Pred ({len(pred_x)} points)",
            hovertemplate="<b>Prediction</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<br>Value: %{marker.color:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text="Point Cloud Comparison: Ground Truth vs Prediction",
            x=0.5,
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        scene2=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        height=600,
        showlegend=True,
    )

    # Add slider for threshold control
    steps = []
    threshold_values = np.linspace(0.1, 0.95, 20)

    for i, threshold in enumerate(threshold_values):
        pred_x_thresh, pred_y_thresh, pred_z_thresh, pred_values_thresh = (
            volume_to_points(
                pred,
                threshold=threshold,
                apply_cc_filter=apply_cc_filter,
                min_voxels=min_voxels,
            )
        )

        step = dict(
            method="restyle",
            args=[
                {
                    "x": [gt_x, pred_x_thresh],
                    "y": [gt_y, pred_y_thresh],
                    "z": [gt_z, pred_z_thresh],
                    "marker.color": [None, pred_values_thresh],
                    "name": [
                        f"GT ({len(gt_x)} points)",
                        f"Pred ({len(pred_x_thresh)} points)",
                    ],
                }
            ],
            label=f"{threshold:.3f}",
        )
        steps.append(step)

    sliders = [
        dict(
            active=10,  # Default to middle value
            currentvalue={"prefix": "Threshold: "},
            pad={"t": 50},
            steps=steps,
        )
    ]

    fig.update_layout(sliders=sliders)

    # Save HTML
    fig.write_html(output_path)
    print(f"Interactive visualization saved to: {output_path}")

    return fig


def create_static_visualization(
    gt: np.ndarray,
    pred: np.ndarray,
    threshold_percentile: float = 90,
    output_path: str = "pointcloud_comparison.png",
    apply_cc_filter: bool = True,
    min_voxels: int = 25,
):
    """Create static image visualization with fixed threshold."""

    # Get GT points
    gt_x, gt_y, gt_z, _ = volume_to_points(
        gt, threshold=0.5, apply_cc_filter=apply_cc_filter, min_voxels=min_voxels
    )

    # Calculate threshold from percentile
    threshold = np.percentile(pred[pred > 0], threshold_percentile)
    pred_x, pred_y, pred_z, pred_values = volume_to_points(
        pred,
        threshold=threshold,
        apply_cc_filter=apply_cc_filter,
        min_voxels=min_voxels,
    )

    print(f"Using {threshold_percentile}th percentile threshold: {threshold:.4f}")
    print(f"GT points: {len(gt_x)}, Pred points: {len(pred_x)}")

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Ground Truth ({len(gt_x)} points)",
            f"Prediction ({len(pred_x)} points, threshold: {threshold:.4f})",
        ),
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    )

    # Add GT points
    fig.add_trace(
        go.Scatter3d(
            x=gt_x,
            y=gt_y,
            z=gt_z,
            mode="markers",
            marker=dict(size=4, color="red", opacity=0.7),
            name="Ground Truth",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Add pred points
    fig.add_trace(
        go.Scatter3d(
            x=pred_x,
            y=pred_y,
            z=pred_z,
            mode="markers",
            marker=dict(
                size=4,
                color=pred_values,
                colorscale="Viridis",
                opacity=0.7,
                colorbar=dict(title="Prediction Value"),
            ),
            name="Prediction",
            showlegend=True,
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Point Cloud Comparison (Threshold: {threshold:.4f} at {threshold_percentile}th percentile)",
            x=0.5,
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        scene2=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        height=600,
        width=1200,
    )

    # Save static image
    fig.write_image(output_path, width=1200, height=600, scale=2)
    print(f"Static visualization saved to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize NeCA point clouds")
    parser.add_argument(
        "--gt", default="data/neca/0_1/gt.npy", help="Path to ground truth .npy file"
    )
    parser.add_argument(
        "--pred", default="data/neca/0_1/pred.npy", help="Path to prediction .npy file"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "static", "both"],
        default="both",
        help="Visualization mode",
    )
    parser.add_argument(
        "--output-html",
        default="pointcloud_comparison.html",
        help="Output HTML file for interactive mode",
    )
    parser.add_argument(
        "--output-image",
        default="pointcloud_comparison.png",
        help="Output image file for static mode",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99,
        help="Percentile threshold for prediction (static mode)",
    )
    parser.add_argument(
        "--no-cc-filter",
        action="store_true",
        help="Disable connected components filtering (removes objects < 25 voxels by default)",
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=25,
        help="Minimum number of voxels for connected components filtering",
    )

    args = parser.parse_args()

    # Load data
    try:
        gt, pred = load_data(args.gt, args.pred)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Set connected components parameters
    apply_cc_filter = not args.no_cc_filter
    min_voxels = args.min_voxels

    if apply_cc_filter:
        print(
            f"Connected components filtering enabled: removing objects < {min_voxels} voxels"
        )
    else:
        print("Connected components filtering disabled")

    # Create visualizations
    if args.mode in ["interactive", "both"]:
        try:
            create_interactive_visualization(
                gt, pred, args.output_html, apply_cc_filter, min_voxels
            )
        except Exception as e:
            print(f"Error creating interactive visualization: {e}")

    if args.mode in ["static", "both"]:
        try:
            create_static_visualization(
                gt,
                pred,
                args.percentile,
                args.output_image,
                apply_cc_filter,
                min_voxels,
            )
        except Exception as e:
            print(f"Error creating static visualization: {e}")
            print("Note: Static mode requires kaleido package for image export")
            print("Install with: pip install kaleido")


if __name__ == "__main__":
    main()
