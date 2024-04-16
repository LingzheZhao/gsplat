"""Python bindings for 3D gaussian projection"""

from typing import Optional, Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C


def project_gaussians(
    means3d: Float[Tensor, "*batch 3"],
    scales: Float[Tensor, "*batch 3"],
    glob_scale: float,
    quats: Float[Tensor, "*batch 4"],
    viewmat: Float[Tensor, "4 4"],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_height: int,
    img_width: int,
    block_width: int,
    clip_thresh: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Note:
        This function is differentiable w.r.t the means3d, scales and quats inputs.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       quats (Tensor): rotations in normalized quaternion [w,x,y,z] format.
       viewmat (Tensor): view matrix for rendering.
       fx (float): focal length x.
       fy (float): focal length y.
       cx (float): principal point x.
       cy (float): principal point y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       block_width (int): side length of tiles inside projection/rasterization in pixels (always square). 16 is a good default value, must be between 2 and 16 inclusive.
       clip_thresh (float): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **compensation** (Tensor): the density compensation for blurring 2D kernel
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
        - **cov3d** (Tensor): 3D covariances.
    """
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    assert (quats.norm(dim=-1) - 1 < 1e-6).all(), "quats must be normalized"
    return _ProjectGaussians.apply(
        means3d.contiguous(),
        scales.contiguous(),
        glob_scale,
        quats.contiguous(),
        viewmat.contiguous(),
        fx,
        fy,
        cx,
        cy,
        img_height,
        img_width,
        block_width,
        clip_thresh,
    )


class _ProjectGaussians(Function):
    """Project 3D gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means3d: Float[Tensor, "*batch 3"],
        scales: Float[Tensor, "*batch 3"],
        glob_scale: float,
        quats: Float[Tensor, "*batch 4"],
        viewmat: Float[Tensor, "4 4"],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        img_height: int,
        img_width: int,
        block_width: int,
        clip_thresh: float = 0.01,
    ):
        num_points = means3d.shape[-2]
        if num_points < 1 or means3d.shape[-1] != 3:
            raise ValueError(f"Invalid shape for means3d: {means3d.shape}")

        (
            cov3d,
            xys,
            depths,
            radii,
            conics,
            compensation,
            num_tiles_hit,
        ) = _C.project_gaussians_forward(
            num_points,
            means3d,
            scales,
            glob_scale,
            quats,
            viewmat,
            fx,
            fy,
            cx,
            cy,
            img_height,
            img_width,
            block_width,
            clip_thresh,
        )

        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points
        ctx.glob_scale = glob_scale
        ctx.fx = fx
        ctx.fy = fy
        ctx.cx = cx
        ctx.cy = cy

        # Save tensors.
        ctx.save_for_backward(
            means3d,
            scales,
            quats,
            viewmat,
            cov3d,
            radii,
            conics,
            compensation,
        )

        return (xys, depths, radii, conics, compensation, num_tiles_hit, cov3d)

    @staticmethod
    def backward(
        ctx,
        v_xys,
        v_depths,
        v_radii,
        v_conics,
        v_compensation,
        v_num_tiles_hit,
        v_cov3d,
    ):
        (
            means3d,
            scales,
            quats,
            viewmat,
            cov3d,
            radii,
            conics,
            compensation,
        ) = ctx.saved_tensors

        (v_cov2d, v_cov3d, v_mean3d, v_scale, v_quat) = _C.project_gaussians_backward(
            ctx.num_points,
            means3d,
            scales,
            ctx.glob_scale,
            quats,
            viewmat,
            ctx.fx,
            ctx.fy,
            ctx.cx,
            ctx.cy,
            ctx.img_height,
            ctx.img_width,
            cov3d,
            radii,
            conics,
            compensation,
            v_xys,
            v_depths,
            v_conics,
            v_compensation,
        )

        if viewmat.requires_grad:
            v_viewmat = torch.zeros_like(viewmat)
            R = viewmat[..., :3, :3]
            t = viewmat[..., :3, 3]

            # Denote:
            #  - viewmat = [R, t]
            #  - mean3d_cam: mean3d in camera space
            # We have:
            # $$mean3d_cam = R @ means3d + t$$
            # $$\frac{\partial mean3d_cam}{\partial means3d} = R$$
            # Thus Jacobian of means3d w.r.t. mean3d_cam is $R^T$.
            # Now we have the gradient of mean3d_cam:
            v_mean3d_cam = torch.matmul(v_mean3d, R.transpose(-1, -2))

            # Jacobian of mean3d_cam w.r.t. viewmat is:
            # $$I(3) \otimes [means3d^T 1]$$
            # where $\otimes$ is the Kronecker product.
            # We have the naive implementation below:

            # means3d_hom = torch.cat([means3d, torch.ones_like(means3d[..., :1]], dim=-1)  # N, 4
            # means3d_hom_T = means3d_hom.unsqueeze(-2)  # N, 1, 4
            # v_pose_cam = torch.kron(
            #     torch.eye(3, device=means3d.device), means3d_hom_T
            # ).view(-1, 3, 12)  # N, 3, 12
            # v_viewmat_ = v_mean3d_cam.unsqueeze(-2) @ v_pose_cam  # N, 1, 12
            # v_viewmat = v_viewmat_.sum(0).view(3, 4)

            # We can simplify the above code block as follows:
            # gradient w.r.t. view matrix translation
            v_viewmat[..., :3, 3] = v_mean3d_cam.sum(-2)
            # gradent w.r.t. view matrix rotation
            v_viewmat[..., :3, :3] = torch.bmm(
                v_mean3d_cam.unsqueeze(-1),
                means3d.unsqueeze(-2)
            ).sum(-3)

        else:
            v_viewmat = None

        # Return a gradient for each input.
        return (
            # means3d: Float[Tensor, "*batch 3"],
            v_mean3d,
            # scales: Float[Tensor, "*batch 3"],
            v_scale,
            # glob_scale: float,
            None,
            # quats: Float[Tensor, "*batch 4"],
            v_quat,
            # viewmat: Float[Tensor, "4 4"],
            v_viewmat,
            # fx: float,
            None,
            # fy: float,
            None,
            # cx: float,
            None,
            # cy: float,
            None,
            # img_height: int,
            None,
            # img_width: int,
            None,
            # block_width: int,
            None,
            # clip_thresh,
            None,
        )
