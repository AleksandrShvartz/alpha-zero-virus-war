from collections import defaultdict
from itertools import product, chain

import numpy as np
from enum import IntEnum
from math import factorial


class Cell(IntEnum):
    E = 0  # empty
    A = 1  # alive, mine; enemy's with minus
    EA = 2  # eaten, mine (I ate enemy (was -1, I ate, became 2)); enemy's with minus
    ED = 3  # eaten, mine, dead; enemy's with minus


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

    def __is_in_bounds(self, pos):
        return all(0 <= e < self._n for e in pos)

    def __get_neighbours(self, pos):
        return filter(lambda p: self.__is_in_bounds(p), ((pos[0] + dx, pos[1] + dy) for dx, dy in self.__dirs))

    def _get_connected_components(self, board):
        board = board.copy()
        comps = defaultdict(list)
        # global container to iterate over
        to_visit = set(product(range(self._n), repeat=2))
        while to_visit:
            cpos = to_visit.pop()
            if self._board[cpos] != Cell.E:
                # add the current component (set of positions) to the array (.append)
                # of components of similar type (comps[c_type])
                c_type = board[cpos]
                comps[c_type].append(set())
                # get the set, appended just now
                _set = comps[c_type][~0]
                _set.add(cpos)
                # curr_comp is a container to iterate over (will be popped, so _set is not suitable)
                curr_comp = {cpos}
                while curr_comp:
                    curr_pos = curr_comp.pop()
                    for pos in self.__get_neighbours(curr_pos):
                        # if not yet visited and of the same type
                        if pos in to_visit and board[pos] == c_type:
                            _set.add(pos)
                            curr_comp.add(pos)
                            to_visit.remove(pos)
        return comps

    def _add_alive(self, pos, pl, board, comps):
        board = board.copy()
        comps = comps.copy()
        board[pos] = Cell.A * pl
        for nbour in self.__get_neighbours(pos):
            # dead, mine, but the newly added cell makes it alive
            if board[nbour] == Cell.ED * pl:
                # search for its component
                for comp in comps[Cell.ED * pl]:
                    if nbour in comp:
                        # xs, ys = zip(*list(comp))
                        # make all the elements in component alive
                        board[*zip(*tuple(comp))] = Cell.EA * pl
        # recalculate
        comps = self._get_connected_components(board)
        return board, comps

    def _is_chain_alive(self, pl, board, comp):
        # iterate over all elements in the component
        for pos in comp:
            for nbour in self.__get_neighbours(pos):
                # if the neighbour is an alive cell
                # the chain is alive, yahoo
                if board[nbour] == Cell.A * pl:
                    return True
        return False

    def _mark_eaten(self, pos, pl, board, comps):
        # mark eaten by enemy
        board = board.copy()
        comps = comps.copy()
        # mark eaten, mine (was -1, ate, became 2)
        board[pos] = (abs(board[pos]) + 1) * pl
        for nbour in self.__get_neighbours(pos):
            # enemy's alive cell was eaten, check if it was connected
            # to the eaten chain
            if board[nbour] == Cell.EA * -pl:
                # find the right component
                for comp in comps[Cell.EA * -pl]:
                    # if the eaten cell was the only one, connected
                    # to the eaten chain
                    if nbour in comp and not self._is_chain_alive(-pl, board, comp):
                        # mark all the cells as eaten dead
                        board[*zip(*list(comp))] = Cell.ED * -pl
        # recalculate
        comps = self._get_connected_components(board)
        return board, comps

    # alternatively, one can "rotate" the board depending on the player
    # then, probably, one will not need to pass "pl" parameter to all subsequent
    # funcs
    def _get_moves(self, k, pl):
        # get all possible positions from which subsequent moves are available
        made_moves = [(pos,) for pos in
                      zip(*np.where(np.logical_or(self._board == Cell.A * pl, self._board == Cell.EA * pl)))]
        comps = self._get_connected_components(self._board)
        # copy as the same board will be related to different moves (their ancestors will modify their board,
        # which will result in modifications of others, hence copy)
        moves = {move: (self._board.copy(), comps.copy()) for move in made_moves}
        print("base case before any moves")
        self.print(self._board)
        print()

        action_table = {
            Cell.E: self._add_alive,  # occupy the empty cell
            Cell.A * -pl: self._mark_eaten  # eat the enemy's cell
        }

        for _ in range(k):
            # made_moves is updated. In order to avoid infinite loop
            # iterate over the part, which was added on the previous "level"
            moves_num = len(made_moves)
            for made_move in made_moves[:moves_num]:
                # there is only one state from each pos
                state, comps = moves[made_move]
                *_, last_move = made_move
                # check all positions
                for cpos in self.__get_neighbours(last_move):
                    move = tuple((*made_move, cpos))
                    action = action_table.get(state[cpos], None)
                    if action is not None:
                        moves[move] = action(cpos, pl, state, comps)
                        print(f"move: {move}")
                        self.print(moves[move][0])
                        print()
                        made_moves.append(move)
                print()
            # remove moves, added on the previous level
            # leave added on the current one
            made_moves = made_moves[moves_num:]
        # return only the moves of the "last" layer
        # remove the starting point from them (so only the made moves are returned)
        return (key[1:] for key in moves.keys() if len(key) == (k + 1))

    def _apply_move(self, moves, pl, board):
        action_table = {
            Cell.E: self._add_alive,  # occupy the empty cell
            Cell.A * -pl: self._mark_eaten  # eat the enemy's cell
        }
        self.print(board)
        print()
        comps = self._get_connected_components(board)
        for move in moves:
            action = action_table.get(board[move], None)
            if action is not None:
                board, comps = action(move, pl, board, comps)
            else:
                raise RuntimeError(f"Illegal move: trying to place onto {board[move]}, which is prohibited")
            self.print(board)
            print()
        return board

    def print(self, board):
        trt = dict(zip(range(-3, 4), "@¥o·x$#"))
        for row in board:
            print(" ".join(trt[e] for e in row))

    def _is_ended(self, k, player, board):
        self._board = board.copy()
        return not [*self._get_moves(k, player)]

    def _get_all_moves(self, k):
        return {**self._get_moves(k, 1), **self._get_moves(k, -1)}

    # TODO
    #   smarten things up
    #   instead of returning and storing all the moves in _get_moves
    #       return the last level, store the current and previous levels


if __name__ == "__main__":
    # print(Board(5))
    a = np.ones((5, 5))

    # b = np.array([
    #     [1, 1, 1, 1, 1],
    #     [1, 2, 2, 2, 2],
    #     [0, 0, 3, 2, 0],
    #     [-1, -2, -2, -2, -2],
    #     [1, -1, 2, -2, 0],
    # ])
    b = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 2, -2, -1, ],
        [0, 0, 3, 3, 3],
        [-3, -3, -3, 3, -3],
        [1, 1, -3, 3, 3],
    ])

    c = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -2, -1, 0],
        [0, 0, 0, 0, -1],
    ])
    print((Board.from_numpy(c)._get_moves(3, -1)).keys())
