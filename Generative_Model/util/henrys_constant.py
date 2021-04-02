import math
from typing import List

GAS_CONSTANT = 8.31446


class PropertyCalculations:

    @staticmethod
    def get_henrys_constant(grid: List[List[List]], temperature=77):
        grid_size = 32

        temp = 0
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    energy_value = grid[i][j][k]
                    temp += math.exp(-energy_value / temperature)
        return temp / (GAS_CONSTANT * temperature * (grid_size ** 3))
