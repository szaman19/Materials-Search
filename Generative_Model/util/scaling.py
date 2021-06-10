from typing import Union, Tuple

import torch
from torch import Tensor
from torch.nn import functional
import torch.nn.functional


class ScaleUtil:

    @staticmethod
    def resize_2d(t: Tensor, size: Union[int, Tuple[int, int, int]]) -> Tensor:
        t = t.unsqueeze(0).unsqueeze(0)  # Input format is Batch x Channels x Dims
        return functional.interpolate(t, size=size, mode='bilinear', align_corners=False)[0][0]

    @staticmethod
    def resize_3d(t: Tensor, size: Union[int, Tuple[int, int, int]]) -> Tensor:
        t = t.unsqueeze(0).unsqueeze(0)  # Input format is Batch x Channels x Dims
        return functional.interpolate(t, size=size, mode='trilinear', align_corners=False)[0][0]


def main():
    t2 = torch.tensor([[1, 2, 3, 4],
                       [5, 0, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]]).float()

    t3 = torch.tensor([[[1, 2, 3, 4],
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
                        [13, 14, 15, 16]]]).float()

    t = t2 if True else t3
    print("BEFORE:", t.shape)
    result = ScaleUtil.resize_2d(t, 5)
    print(result)
    print("AFTER:", result.shape)


if __name__ == '__main__':
    main()
