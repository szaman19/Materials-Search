import torch
from torch import Tensor
from torch.nn import Module, functional
from torch.nn.modules.utils import _ntuple


class PeriodicPadNd(Module):
    def forward(self, x: Tensor) -> Tensor:
        return functional.pad(x, self.padding, 'circular')

    def extra_repr(self) -> str:
        return '{}'.format(self.padding)


class PeriodicPad2d(PeriodicPadNd):
    def __init__(self, padding) -> None:
        super(PeriodicPad2d, self).__init__()
        self.padding = _ntuple(4)(padding)


class PeriodicPad3d(PeriodicPadNd):
    def __init__(self, padding) -> None:
        super(PeriodicPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)


def main():
    p = PeriodicPad2d(2)
    x = torch.arange(9).float().reshape(1, 1, 3, 3)
    print(x)
    y = p(x)
    print(y)


if __name__ == '__main__':
    main()
