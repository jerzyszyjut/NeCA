import torch
import numpy as np
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

from src.config.configloading import load_config
from src.render import run_network
from src.trainer import Trainer


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="configs file path (if not set, runs all in ./data/neca/)",
    )
    return parser


parser = config_parser()
args = parser.parse_args()


def run_training_for_config(config_path, idx=0):
    print(f"\n[INFO] Running training for config: {config_path}")
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class BasicTrainer(Trainer):
        def __init__(self):
            super().__init__(cfg, device)  # type: ignore
            print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")
            self.l2_loss = torch.nn.MSELoss(reduction="mean")

        def compute_loss(self, data, global_step, idx_epoch):
            loss = {"loss": 0.0}
            projs = data.projs
            image_pred = run_network(self.voxels, self.net, self.netchunk)
            train_output = image_pred.squeeze()[None, ...]
            train_projs_one = self.ct_projector_first.forward_project(train_output)
            train_projs_two = self.ct_projector_second.forward_project(train_output)
            train_projs = torch.cat((train_projs_one, train_projs_two), 1)
            loss["loss"] = self.l2_loss(train_projs, projs)
            return loss

    trainer = BasicTrainer()
    trainer.start()

    # Run visualization after training completes
    print(f"[INFO] Training completed for {config_path}. Running visualization...")
    try:
        # Get the data directory from config path (e.g., data/neca/0_0/config.yaml -> data/neca/0_0)
        config_dir = os.path.dirname(config_path)
        gt_path = os.path.join(config_dir, "gt.npy")
        pred_path = os.path.join(config_dir, "pred.npy")

        # Check if the required files exist
        if os.path.exists(gt_path) and os.path.exists(pred_path):
            # Create output paths based on config directory
            html_output = f"./data/pointcloud_comparison_{idx}.html"

            # Run the visualization script
            viz_cmd = [
                sys.executable,
                "visualize_pointclouds.py",
                "--gt",
                gt_path,
                "--pred",
                pred_path,
                "--mode",
                "interactive",
                "--output-html",
                html_output,
            ]

            result = subprocess.run(
                viz_cmd, capture_output=True, text=True, cwd=os.getcwd()
            )

            if result.returncode == 0:
                print(f"[SUCCESS] Visualization saved to: {html_output}")
            else:
                print(f"[ERROR] Visualization failed: {result.stderr}")
        else:
            print(
                f"[WARN] Missing gt.npy or pred.npy in {config_dir}, skipping visualization"
            )

    except Exception as e:
        print(f"[ERROR] Failed to run visualization: {e}")


if args.config is not None:
    # Single config mode
    run_training_for_config(args.config)
else:
    # Autodiscover all config files in ./data/neca/
    config_files = sorted(glob.glob(os.path.join("data", "neca", "**/config.yaml")))
    if not config_files:
        print("[WARN] No config files found in ./data/neca/")
    for idx, config_path in enumerate(config_files):
        run_training_for_config(config_path, idx)
