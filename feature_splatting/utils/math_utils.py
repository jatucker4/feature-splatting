import numpy as np
import torch
import torch.nn.functional as F

def point_to_plane_distance(point, plane):
    """
    Compute distance from point to plane
    :param point: point (x, y, z)
    :param plane: plane (A, B, C, D)
    :return: distance from point to plane
    """
    x, y, z = point
    A, B, C, D = plane
    numerator = np.abs(A*x + B*y + C*z + D)
    denominator = np.sqrt(A**2 + B**2 + C**2)
    distance = numerator / denominator
    return distance

def vector_angle(vec_a, vec_b):
    """
    Calculate angle between two vectors
    :param vec_a: vector a
    :param vec_b: vector b
    :return: angle between two vectors
    """
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    cos_theta = dot / (norm_a * norm_b)
    theta = np.arccos(cos_theta)
    return theta

# def velocity_to_optical_flow(velocity, means2d, W, H):
#     """
#     Convert velocity to a 2D optical flow field in pixel space.

#     Args:
#         velocity: Tensor of shape (N, 3) representing the 3D velocities (vx, vy, vz) of each point.
#         means2d: Tensor of shape (N, 2) representing the 2D pixel coordinates (x, y) of each point from rasterization.
#         W: Width of the image.
#         H: Height of the image.

#     Returns:
#         optical_flow: Tensor of shape (H, W, 2) representing the optical flow field (u, v) in pixel space.
#     """
#     # Extract the 2D velocity components (u, v) from the 3D velocity
#     velocity_2d = velocity[:, :2]  # (N, 2)
    
#     # Create an empty optical flow field
#     optical_flow = torch.zeros((H, W, 2), device=velocity.device)
    
#     # Map the velocities to the optical flow field
#     for i in range(means2d.shape[0]):
#         x, y = means2d[i].long()
#         if 0 <= x < W and 0 <= y < H:
#             optical_flow[y, x, :] = velocity_2d[i]
    
#     return optical_flow

def velocity_to_optical_flow(velocity, means2d, W, H):
    """
    Convert velocity to a 2D optical flow field in pixel space.

    Args:
        velocity: Tensor of shape (N, 3) representing the 3D velocities (vx, vy, vz) of each point.
        means2d: Tensor of shape (N, 2) representing the 2D pixel coordinates (x, y) of each point from rasterization.
        W: Width of the image.
        H: Height of the image.

    Returns:
        optical_flow: Tensor of shape (H, W, 2) representing the optical flow field (u, v) in pixel space.
    """
    # Extract the 2D velocity components (u, v)
    velocity_2d = velocity[:, :2]  # Shape: (N, 2)

    # Get x and y coordinates and ensure they are long tensors
    x = means2d[:, 0].long()
    y = means2d[:, 1].long()

    # Create a mask to filter out points outside the image boundaries
    mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)

    # Apply the mask to x, y, and velocity_2d
    x = x[mask]
    y = y[mask]
    velocity_2d = velocity_2d[mask]

    # Initialize the optical flow field
    optical_flow = torch.zeros((H, W, 2), device=velocity.device)

    # Map the velocities to the optical flow field using advanced indexing
    optical_flow[y, x] = velocity_2d

    return optical_flow
