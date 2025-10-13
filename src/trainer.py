import os
import os.path as osp
import json
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from shutil import copyfile
import numpy as np
import random
import math
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .dataset import TIGREDataset as Dataset
from .network import get_network
from .encoder import get_encoder
from src.render import run_network

from src.render.ct_geometry_projector import ConeBeam3DProjector
from odl.tomo.util.utility import axis_rotation, rotation_matrix_from_to


def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rotation_matrix_to_axis_angle(m):
    angle = np.arccos((m[0, 0] + m[1, 1] + m[2, 2] - 1) / 2)

    x = (m[2, 1] - m[1, 2]) / math.sqrt(
        (m[2, 1] - m[1, 2]) ** 2 + (m[0, 2] - m[2, 0]) ** 2 + (m[1, 0] - m[0, 1]) ** 2
    )
    y = (m[0, 2] - m[2, 0]) / math.sqrt(
        (m[2, 1] - m[1, 2]) ** 2 + (m[0, 2] - m[2, 0]) ** 2 + (m[1, 0] - m[0, 1]) ** 2
    )
    z = (m[1, 0] - m[0, 1]) / math.sqrt(
        (m[2, 1] - m[1, 2]) ** 2 + (m[0, 2] - m[2, 0]) ** 2 + (m[1, 0] - m[0, 1]) ** 2
    )
    axis = (x, y, z)

    return axis, angle


class Trainer:
    def __init__(self, cfg, device="cuda"):
        # Set seed for reproducibility
        if "seed" in cfg["train"]:
            set_seed(cfg["train"]["seed"])
            print(f"Seed set to: {cfg['train']['seed']}")

        # Args
        self.global_step = 0
        self.conf = cfg
        self.n_fine = cfg["render"]["n_fine"]
        self.epochs = cfg["train"]["epoch"]
        self.i_eval = cfg["log"]["i_eval"]
        self.i_save = cfg["log"]["i_save"]
        self.netchunk = cfg["render"]["netchunk"]

        # Log direcotry
        self.expdir = osp.join(cfg["exp"]["expdir"], cfg["exp"]["expname"])
        self.ckptdir = osp.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.evaldir = osp.join(self.expdir, "eval")
        os.makedirs(self.evaldir, exist_ok=True)

        #######################################
        configPath = cfg["exp"]["dataconfig"]
        with open(configPath, "r") as handle:
            data = yaml.safe_load(handle)

        self.data_dir = data["datadir"]

        data["projections"] = np.load(self.data_dir)

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------

        self.ct_projector_first = ConeBeam3DProjector(
            data["projections_settings"][0]["alpha"],
            data["projections_settings"][0]["beta"],
            data["projections_settings"][0]["sid"],
            data["projections_settings"][0]["sod"],
            data["global"]["grid_spacing"],
            data["global"]["grid_resolution"],
            data["global"]["image_spacing"],
            data["global"]["image_resolution"],
        )

        self.ct_projector_second = ConeBeam3DProjector(
            data["projections_settings"][1]["alpha"],
            data["projections_settings"][1]["beta"],
            data["projections_settings"][1]["sid"],
            data["projections_settings"][1]["sod"],
            data["global"]["grid_spacing"],
            data["global"]["grid_resolution"],
            data["global"]["image_spacing"],
            data["global"]["image_resolution"],
        )

        # Dataset
        self.dataconfig = data
        self.train_dset = Dataset(data, device)
        self.voxels = self.train_dset.voxels

        # Network
        network = get_network(cfg["network"]["net_type"])
        net_type = cfg["network"].pop("net_type", None)
        encoder = get_encoder(**cfg["encoder"])
        self.net = network(encoder, **cfg["network"]).to(device)
        self.grad_vars = list(self.net.parameters())
        self.net_fine = None
        if self.n_fine > 0:
            self.net_fine = network(encoder, **cfg["network"]).to(device)
            self.grad_vars += list(self.net_fine.parameters())
        cfg["network"]["net_type"] = net_type

        # Optimizer
        self.optimizer = torch.optim.Adam(
            params=self.grad_vars, lr=cfg["train"]["lrate"], betas=(0.9, 0.999)
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=cfg["train"]["lrate_step"],
            gamma=cfg["train"]["lrate_gamma"],
        )

        # Load checkpoints
        self.epoch_start = 0
        if cfg["train"]["resume"] and osp.exists(self.ckptdir):
            print(f"Load checkpoints from {self.ckptdir}.")
            ckpt = torch.load(self.ckptdir)
            self.epoch_start = ckpt["epoch"] + 1
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.global_step = self.epoch_start  # * len(self.train_dloader)
            self.net.load_state_dict(ckpt["network"])
            if self.n_fine > 0 and self.net_fine is not None:
                self.net_fine.load_state_dict(ckpt["network_fine"])

        # Summary writer
        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text("parameters", self.args2string(cfg), global_step=0)

    def args2string(self, hp):
        """
        Transfer args to string.
        """
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

    def start(self):
        """
        Main loop.
        """

        iter_per_epoch = 1  # len(self.train_dloader)
        pbar = tqdm(total=iter_per_epoch * self.epochs, leave=True)
        if self.epoch_start > 0:
            pbar.update(self.epoch_start * iter_per_epoch)

        for idx_epoch in range(self.epoch_start, self.epochs + 1):
            # Evaluate
            if (
                idx_epoch % self.i_eval == 0 or idx_epoch == self.epochs
            ) and self.i_eval > 0:
                self.net.eval()
                with torch.no_grad():
                    # print(self.voxels.shape) #torch.Size([128, 128, 128, 3])
                    image_pred = run_network(
                        self.voxels,
                        self.net_fine if self.net_fine is not None else self.net,
                        self.netchunk,
                    )
                    image_pred = (image_pred.squeeze()).detach().cpu().numpy()

                    np.save(self.evaldir + "/" + str(idx_epoch), image_pred)

            # Train
            # for data in self.train_dloader:
            self.global_step += 1
            # Train
            self.net.train()
            loss_train = self.train_step(
                self.train_dset, global_step=self.global_step, idx_epoch=idx_epoch
            )
            pbar.set_description(
                f"epoch={idx_epoch}/{self.epochs}, {loss_train['loss']}, lr={self.optimizer.param_groups[0]['lr']:.3g}"
            )
            pbar.update(1)

            # Save
            if (
                (idx_epoch % self.i_save == 0 or idx_epoch == self.epochs)
                and self.i_save > 0
                and idx_epoch > 0
                and self.net_fine is not None
            ):
                if osp.exists(self.ckptdir):
                    copyfile(self.ckptdir, self.ckptdir_backup)
                tqdm.write(
                    f"[SAVE] epoch: {idx_epoch}/{self.epochs}, path: {self.ckptdir}"
                )
                torch.save(
                    {
                        "epoch": idx_epoch,
                        "network": self.net.state_dict(),
                        "network_fine": self.net_fine.state_dict()
                        if self.n_fine > 0
                        else None,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    self.ckptdir,
                )

            # Update lrate
            self.writer.add_scalar(
                "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
            )
            self.lr_scheduler.step()

        image_pred = run_network(
            self.voxels,
            self.net_fine if self.net_fine is not None else self.net,
            self.netchunk,
        )

        train_output = image_pred.squeeze()[None, ...]  # .transpose(1,4).squeeze(4)

        projs_one = self.ct_projector_first.forward_project(train_output)
        projs_two = self.ct_projector_second.forward_project(train_output)

        proj_one_np = projs_one.squeeze().detach().cpu().numpy()
        proj_two_np = projs_two.squeeze().detach().cpu().numpy()

        output_dir = Path(self.data_dir).parent

        proj_one_norm = (
            (proj_one_np - proj_one_np.min())
            / (proj_one_np.max() - proj_one_np.min())
            * 255
        ).astype(np.uint8)
        proj_two_norm = (
            (proj_two_np - proj_two_np.min())
            / (proj_two_np.max() - proj_two_np.min())
            * 255
        ).astype(np.uint8)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(proj_one_norm, cmap="gray")
        plt.title("Projection 1")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(proj_two_norm, cmap="gray")
        plt.title("Projection 2")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(
            output_dir / "projections_combined.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        np.save(output_dir / "projection_0.npy", proj_one_np)
        np.save(output_dir / "projection_1.npy", proj_two_np)

        image_pred = (image_pred.squeeze()).detach().cpu().numpy()

        np.save(Path(self.data_dir).parent / "pred.npy", image_pred)

        tqdm.write(f"Training complete! See logs in {self.expdir}")

    def train_step(self, data, global_step, idx_epoch):
        """
        Training step
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, global_step, idx_epoch)
        loss["loss"].backward()
        self.optimizer.step()
        return loss

    def compute_loss(self, data, global_step, idx_epoch):
        """
        Training step
        """
        raise NotImplementedError()
