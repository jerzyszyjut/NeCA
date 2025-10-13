import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import argparse
from tqdm import tqdm
import cc3d


def filter_connected_components(data, threshold, min_voxels=25):
    binary_data = (data > threshold).astype(np.int32)
    labels = cc3d.connected_components(binary_data, connectivity=26)
    stats = cc3d.statistics(labels)

    filtered_data = data.copy()
    for label_id in range(1, len(stats["voxel_counts"])):
        if stats["voxel_counts"][label_id] < min_voxels:
            filtered_data[labels == label_id] = 0

    return filtered_data


def create_comparison_visualization(pred_data, gt_data, title):
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=["Prediction (Thresholded)", "Ground Truth"],
    )

    # threshold = np.percentile(pred_data[pred_data > 0], 99)
    threshold = 0.2
    pred_filtered = filter_connected_components(pred_data, threshold)
    pred_coords = np.where(pred_filtered > threshold)
    pred_values = pred_filtered[pred_filtered > threshold]

    if len(pred_values) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=pred_coords[0],
                y=pred_coords[1],
                z=pred_coords[2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=pred_values,
                    colorscale="Viridis",
                    opacity=0.8,
                    colorbar=dict(title="Prediction", x=0.45),
                ),
                name="Prediction",
            ),
            row=1,
            col=1,
        )

    gt_coords = np.where(gt_data > 0)
    gt_values = gt_data[gt_data > 0]

    if len(gt_values) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=gt_coords[0],
                y=gt_coords[1],
                z=gt_coords[2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=gt_values,
                    colorscale="Plasma",
                    opacity=0.8,
                    colorbar=dict(title="Ground Truth", x=1.0),
                ),
                name="Ground Truth",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title=title,
        width=1600,
        height=800,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        scene2=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )

    print(f"Prediction: {len(pred_values)}, GT: {len(gt_values)}")
    return fig


def visualize_prediction(pred_file):
    pred_data = np.load(pred_file)

    gt_file = pred_file.parent / "gt.npy"
    if not gt_file.exists():
        return

    gt_data = np.load(gt_file)
    title = f"{pred_file.parent.name} - Prediction vs Ground Truth"

    fig = create_comparison_visualization(pred_data, gt_data, title)
    output_file = pred_file.parent / "visualization.html"
    fig.write_html(str(output_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/neca")
    args = parser.parse_args()

    pred_files = list(Path(args.data_dir).glob("**/pred.npy"))

    for pred_file in tqdm(pred_files):
        visualize_prediction(pred_file)


if __name__ == "__main__":
    main()
