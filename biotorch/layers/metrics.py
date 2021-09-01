import math
import torch


def compute_matrix_angle(A, B):
    with torch.no_grad():
        # Flatten and normalize matrices
        flat_A = torch.reshape(A, (-1, ))
        normalized_flat_A = flat_A / torch.norm(flat_A)
        flat_B = torch.reshape(B, (-1, ))
        normalized_flat_B = flat_B / torch.norm(flat_B)
        # Compute angle
        angle = (180.0 / math.pi) * torch.arccos(torch.clip(torch.dot(normalized_flat_A, normalized_flat_B), -1.0, 1.0))

    return angle
