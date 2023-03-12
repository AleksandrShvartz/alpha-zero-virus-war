from collections import defaultdict
from itertools import product

import numpy as np
from enum import IntEnum
from math import factorial


class Cell(IntEnum):
    E = 0  # empty
    AI = 1  # alive, mine
    AN = -1  # alive, enemy
    EI = 2  # eaten, mine (I ate enemy)
    EN = -2  # eaten, enemy (Enemy ate mine)


class Board:
    __dirs = ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))

    def __init__(self, n):
        """

        :param n: board size
        """
        self._n = n
        self._board = np.zeros((n, n))

    @classmethod
    def from_numpy(cls, arr):
        cl = cls(arr.shape[0])
        cl._board = arr.copy()
        return cl

    def __index__(self, idx):
        return self._board[idx]

    def __repr__(self):
        return f"{self._board}"

    def _is_in_bounds(self, pos):
        return all(0 <= e < self._n for e in pos)

    def _get_connected_components(self):
        print(*Cell)
        comps = defaultdict(list)
        to_visit = set(product(range(self._n), repeat=2))
        while to_visit:
            cpos = to_visit.pop()
            if self._board[cpos] != Cell.E:
                comps[self._board[cpos]].append(set())
                _set = comps[self._board[cpos]][~0]
                _set.add(cpos)
                curr_comp = {cpos}
                c_type = self._board[cpos]
                while curr_comp:
                    ccx, ccy = curr_comp.pop()
                    for pos in ((ccx + dx, ccy + dy) for dx, dy in self.__dirs):
                        if self._is_in_bounds(pos) and pos in to_visit and self._board[pos] == c_type:
                            _set.add(pos)
                            curr_comp.add(pos)
                            to_visit.remove(pos)
        return comps

    def _get_base_moves(self):
        pass
        # for i, row in enumerate(self._board):
        #     for j, el in enumerate(row):

    def _get_available_moves(self):
        pass


if __name__ == "__main__":
    print(Board(5))
    a = np.ones((5, 5))

    b = np.array([
        [1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2],
        [0, 0, 0, 2, 0],
        [-1, -1, -2, -2, -2],
        [1, -1, 2, -2, 0],
    ])

    print(Board.from_numpy(a))
    print(b)
    comps = Board.from_numpy(b)._get_connected_components()
    print(comps)
    # for k, v in comps.items():
    #     print(f"{k}, len: {len(v)}")
