import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import odl
from odl.contrib import torch as odl_torch
from odl.tomo.util.utility import axis_rotation, rotation_matrix_from_to


def _source_params(alpha, beta, sod):
    def rotation_matrix_to_axis_angle(m):
        import math

        angle = np.arccos((m[0, 0] + m[1, 1] + m[2, 2] - 1) / 2)
        x = (m[2, 1] - m[1, 2]) / math.sqrt(
            (m[2, 1] - m[1, 2]) ** 2
            + (m[0, 2] - m[2, 0]) ** 2
            + (m[1, 0] - m[0, 1]) ** 2
        )
        y = (m[0, 2] - m[2, 0]) / math.sqrt(
            (m[2, 1] - m[1, 2]) ** 2
            + (m[0, 2] - m[2, 0]) ** 2
            + (m[1, 0] - m[0, 1]) ** 2
        )
        z = (m[1, 0] - m[0, 1]) / math.sqrt(
            (m[2, 1] - m[1, 2]) ** 2
            + (m[0, 2] - m[2, 0]) ** 2
            + (m[1, 0] - m[0, 1]) ** 2
        )
        axis = (x, y, z)
        return axis, angle

    from_source_vec = (0, -sod, 0)
    alpha = -alpha
    beta = -beta
    to_source_vec = axis_rotation((0, 0, 1), alpha, from_source_vec)
    to_source_vec = axis_rotation((1, 0, 0), -beta, to_source_vec)
    rot_mat = rotation_matrix_from_to(from_source_vec, to_source_vec[0])
    proj_axis, proj_angle = rotation_matrix_to_axis_angle(rot_mat)
    return proj_axis, proj_angle


def build_camera_model(
    alpha, beta, sid, sod, grid_spacing, grid_size, img_spacing, img_size
):
    proj_axis, proj_angle = _source_params(alpha, beta, sod)

    cube_width = grid_spacing * grid_size[0]
    cube_height = grid_spacing * grid_size[1]
    cube_depth = grid_spacing * grid_size[2]
    img_width = img_spacing * img_size[0]
    img_height = img_spacing * img_size[1]

    reco_space = odl.uniform_discr(
        min_pt=[-cube_width / 2, -cube_height / 2, -cube_depth / 2],
        max_pt=[cube_width / 2, cube_height / 2, cube_depth / 2],
        shape=grid_size,
        dtype="float32",
    )
    angle_partition = odl.uniform_partition(min_pt=0, max_pt=proj_angle * 2, shape=[1])
    detector_partition = odl.uniform_partition(
        min_pt=[-img_width / 2, -img_height / 2],
        max_pt=[img_width / 2, img_height / 2],
        shape=img_size,
    )

    geometry = odl.tomo.ConeBeamGeometry(
        apart=angle_partition,
        dpart=detector_partition,
        src_radius=sod,
        det_radius=sid - sod,
        src_to_det_init=(0, 1, 0),
        det_axes_init=[(1, 0, 0), (0, 0, 1)],
        axis=proj_axis,
    )
    ray_trafo = odl.tomo.RayTransform(
        vol_space=reco_space, geometry=geometry, impl="astra_cuda"
    )

    return ray_trafo


class Projection_ConeBeam(nn.Module):
    def __init__(
        self, alpha, beta, sid, sod, grid_spacing, grid_size, img_spacing, img_size
    ):
        super(Projection_ConeBeam, self).__init__()
        ray_trafo = build_camera_model(
            alpha, beta, sid, sod, grid_spacing, grid_size, img_spacing, img_size
        )
        self.trafo = odl_torch.OperatorModule(ray_trafo)
        self.back_projector = odl_torch.OperatorModule(ray_trafo.adjoint)

    def forward(self, x):
        x = self.trafo(x)
        # x = x / self.reso
        return x

    def back_projection(self, x):
        x = self.back_projector(x)
        return x


class ConeBeam3DProjector:
    def __init__(
        self, alpha, beta, sid, sod, grid_spacing, grid_size, img_spacing, img_size
    ):
        self.forward_projector = Projection_ConeBeam(
            alpha, beta, sid, sod, grid_spacing, grid_size, img_spacing, img_size
        )

    def forward_project(self, volume):
        """
        Arguments:
            volume: torch tensor with input size (B, C, img_x, img_y, img_z)
        """

        proj_data = self.forward_projector(volume)

        return proj_data
