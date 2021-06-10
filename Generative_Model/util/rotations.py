import torch
from torch import Tensor


class Rotations:

    @staticmethod
    def rotate_2d(tensor: Tensor):
        rotate_90 = tensor.transpose(0, 1).flip(0)
        rotate_180 = rotate_90.transpose(0, 1).flip(0)  # rotate_180 = tensor.flip(0).flip(1) #(0=HR, 1=VR)
        rotate_270 = rotate_180.transpose(0, 1).flip(0)

        return rotate_90, rotate_180, rotate_270

    @staticmethod
    def rotate_3d(tensor: Tensor):
        # Rotations are counter clockwise looking toward the origin from a positive position along the axis

        # Z Axis Rotations
        z90 = tensor.transpose(1, 2).flip(1)
        z180 = z90.transpose(1, 2).flip(1)
        z270 = z180.transpose(1, 2).flip(1)

        # Y Axis Rotations
        y90 = tensor.transpose(0, 2).flip(2)
        y180 = y90.transpose(0, 2).flip(2)
        y270 = y180.transpose(0, 2).flip(2)

        # X Axis Rotations
        x90 = tensor.transpose(0, 1).flip(1)
        x180 = x90.transpose(0, 1).flip(1)
        x270 = x180.transpose(0, 1).flip(1)

        # return x90, x180, x270, tensor, y90, y180, y270, tensor, z90, z180, z270, tensor
        return x90, x180, x270, y90, y180, y270, z90, z180, z270


def main():
    t = torch.tensor([[[1, 2, 3, 4],
                       [5, 0, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]],
                      [[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]],
                      [[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]]])
    print(t)
    for r in Rotations.rotate_3d(t):
        print(r)


if __name__ == '__main__':
    main()
