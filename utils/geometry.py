import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / torch.norm(vec1)).view(3), (vec2 / torch.norm(vec2)).view(3)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    s = torch.norm(v)
    kmat = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if s == 0:
        rotation_matrix = torch.eye(3)
    else:
        rotation_matrix = torch.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix


def find_rigid_transform(true_points, mapping_points):
    # align one fragment to the other and find rigid transformation with Kabsch algorithm
    # mapping_points @ R.T + t = true_points
    t1 = true_points.mean(0)
    t2 = mapping_points.mean(0)
    x1 = true_points - t1
    x2 = mapping_points - t2

    h = x2.T @ x1
    u, s, vt = np.linalg.svd(h)
    v = vt.T
    d = np.linalg.det(v @ u.T)
    e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])
    R = v @ e @ u.T
    t = -R @ t2.T + t1.T
    return R, t


def get_pca_axes(coords, weights=None):
    """Computes the (weighted) PCA of the given coordinates.
    Args:
        coords (np.ndarray): (N ,D)
        weights (np.ndarray, optional): (N). Defaults to None.
    Returns:
        Tuple[np.ndarray, np.ndarray]: (D), (D, D)
    """
    if weights is None:
        weights = np.ones((*coords.shape[:-1],))
    weights /= weights.sum()

    mean = (weights[..., None] * coords).mean()
    centered = coords - mean
    cov = centered.T @ np.diag(weights) @ centered

    s, vecs = np.linalg.eigh(cov)
    return s, vecs


def find_axes(atoms, charges):
    """Generates equivariant axes based on PCA.
    Args:
        atoms (np.ndarray): (..., M, 3)
        charges (np.ndarray): (M)
    Returns:
        np.ndarray: (3, 3)
    """
    atoms, charges = deepcopy(atoms), deepcopy(charges)
    # First compute the axes by PCA
    atoms = atoms - atoms.mean(-2, keepdims=True)
    s, axes = get_pca_axes(atoms, charges)
    # Let's check whether we have identical eigenvalues
    # if that's the case we need to work with soem pseudo positions
    # to get unique atoms.
    unique_values = np.zeros_like(s)
    v, uindex = np.unique(s, return_index=True)
    unique_values[uindex] = v
    is_ambiguous = np.count_nonzero(unique_values) < np.count_nonzero(s)
    # We always compute the pseudo coordinates because it causes some compile errors
    # for some unknown reason on A100 cards with jax.lax.cond.
    # Compute pseudo coordiantes based on the vector inducing the largest coulomb energy.
    distances = atoms[None] - atoms[..., None, :]
    dist_norm = np.linalg.norm(distances, axis=-1)
    coulomb = charges[None] * charges[:, None] / (dist_norm + 1e-20)
    off_diag_mask = ~np.eye(atoms.shape[0], dtype=bool)
    coulomb, distances = coulomb[off_diag_mask], distances[off_diag_mask]
    idx = np.argmax(coulomb)
    scale_vec = distances[idx]
    scale_vec /= np.linalg.norm(scale_vec)
    # Projected atom positions
    proj = atoms @ scale_vec[..., None] * scale_vec
    diff = atoms - proj
    pseudo_atoms = proj * (1 + 1e-4) + diff

    pseudo_s, pseudo_axes = get_pca_axes(pseudo_atoms, charges)

    # Select pseudo axes if it is ambiguous
    s = np.where(is_ambiguous, pseudo_s, s)
    axes = np.where(is_ambiguous, pseudo_axes, axes)

    order = np.argsort(s)[::-1]
    axes = axes[:, order]

    # Compute an equivariant vector
    distances = np.linalg.norm(atoms[None] - atoms[..., None, :], axis=-1)
    weights = distances.sum(-1)
    equi_vec = ((weights * charges)[..., None] * atoms).mean(0)

    ve = equi_vec @ axes
    flips = ve < 0
    axes = np.where(flips[None], -axes, axes)

    right_hand = np.stack(
        [axes[:, 0], axes[:, 1], np.cross(axes[:, 0], axes[:, 1])], axis=1)
    # axes = np.where(np.abs(ve[-1]) < 1e-7, right_hand, axes)
    return right_hand


def local_to_global(R, t, p):
    """
    Description:
        Convert local (internal) coordinates to global (external) coordinates q.
        q <- Rp + t
    Args:
        R:  (F, 3, 3).
        t:  (F, 3).
        p:  Local coordinates, (F, ..., 3).
    Returns:
        q:  Global coordinates, (F, ..., 3).
    """
    assert p.size(-1) == 3
    assert R.ndim - 1 == t.ndim
    squeeze_dim = False
    if R.ndim == 2:
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
        p = p.unsqueeze(0)
        squeeze_dim = True

    p_size = p.size()
    num_frags = p_size[0]

    p = p.view(num_frags, -1, 3).transpose(-1, -2)  # (F, *, 3) -> (F, 3, *)
    q = torch.matmul(R, p) + t.unsqueeze(-1)  # (F, 3, *)
    q = q.transpose(-1, -2).reshape(p_size)  # (F, 3, *) -> (F, *, 3) -> (F, ..., 3)
    if squeeze_dim:
        q = q.squeeze(0)
    return q


def global_to_local(R, t, q):
    """
    Description:
        Convert global (external) coordinates q to local (internal) coordinates p.
        p <- R^{T}(q - t)
    Args:
        R:  (F, 3, 3).
        t:  (F, 3).
        q:  Global coordinates, (F, ..., 3).
    Returns:
        p:  Local coordinates, (F, ..., 3).
    """
    assert q.size(-1) == 3
    assert R.ndim - 1 == t.ndim
    squeeze_dim = False
    if R.ndim == 2:
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
        q = q.unsqueeze(0)
        squeeze_dim = True

    q_size = q.size()
    num_frags = q_size[0]

    q = q.reshape(num_frags, -1, 3).transpose(-1, -2)  # (F, *, 3) -> (F, 3, *)
    p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))  # (F, 3, *)
    p = p.transpose(-1, -2).reshape(q_size)  # (F, 3, *) -> (F, *, 3) -> (F, ..., 3)
    if squeeze_dim:
        p = p.squeeze(0)
    return p



# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
def quaternion_to_rotation_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    quaternions = F.normalize(quaternions, dim=-1)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
BSD License

For PyTorch3D software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Meta nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def quaternion_1ijk_to_rotation_matrix(q):
    """
    (1 + ai + bj + ck) -> R
    Args:
        q:  (..., 3)
    """
    b, c, d = torch.unbind(q, dim=-1)
    s = torch.sqrt(1 + b**2 + c**2 + d**2)
    a, b, c, d = 1/s, b/s, c/s, d/s

    o = torch.stack(
        (
            a**2 + b**2 - c**2 - d**2,  2*b*c - 2*a*d,  2*b*d + 2*a*c,
            2*b*c + 2*a*d,  a**2 - b**2 + c**2 - d**2,  2*c*d - 2*a*b,
            2*b*d - 2*a*c,  2*c*d + 2*a*b,  a**2 - b**2 - c**2 + d**2,
        ),
        -1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


def apply_rotation_to_vector(R, p):
    return local_to_global(R, torch.zeros_like(p), p)


def axis_angle_to_matrix(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_rotation_matrix(axis_angle_to_quaternion(axis_angle))


def axis_angle_to_quaternion(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions
