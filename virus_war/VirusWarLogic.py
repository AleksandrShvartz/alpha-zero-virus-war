from collections import defaultdict
from itertools import product, chain
from pathlib import Path

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

class Board:
    __dirs = ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))
    __n = 0  # caller must set it via `Board._Board__n = `
    __start_pos = (0, 0)
    __dummy_pos = (-1, -1)
    __dtype = np.int8
    __file = None

    @staticmethod
    def get_init():
        return np.zeros((Board.__n, Board.__n), dtype=Board.__dtype)

    @staticmethod
    def __is_in_bounds(pos):
        return all(0 <= e < Board.__n for e in pos)

    @staticmethod
    def __get_neighbours(pos):
        return filter(lambda p: Board.__is_in_bounds(p), ((pos[0] + dx, pos[1] + dy) for dx, dy in Board.__dirs))

    @staticmethod
    def __get_connected_components(board):
        # TODO
        #   make mapping from each board point to its component (every point of component points to the same instance)
        comps = defaultdict(list)
        # global container to iterate over
        to_visit = set(zip(*np.where(board == Cell.E)))

        while to_visit:
            cpos = to_visit.pop()
            # add the current component (set of positions) to the array (.append)
            # of components of similar type (comps[cell_type])
            cell_type = board[cpos]
            comp_set = {cpos}
            comps[cell_type].append(comp_set)
            # curr_comp is a container to iterate over (will be popped, so comp_set is not suitable)
            curr_comp = {cpos}
            while curr_comp:
                curr_pos = curr_comp.pop()
                for pos in Board.__get_neighbours(curr_pos):
                    # if not yet visited and of the same type
                    if pos in to_visit and board[pos] == cell_type:
                        comp_set.add(pos)
                        curr_comp.add(pos)
                        to_visit.remove(pos)
        return comps

    @staticmethod
    def __generic_marking(pos, board, comps, cell1, cell2, cell3, func):
        board = board.copy()
        board[pos] = cell1
        for nbour in Board.__get_neighbours(pos):
            # check whether the type1 connected the dead chain, making it alive
            # or ate the alive cell from the chain, making it dead
            if board[nbour] == cell2:
                # search for its component
                for comp in comps[cell2]:
                    if nbour in comp and func(board, comp):
                        # make all the elements in component of type3 (alive or dead)
                        board[*zip(*tuple(comp))] = cell3
        # recalculate
        comps = Board.__get_connected_components(board)
        return board, comps

    @staticmethod
    def __add_alive(pos, pl, board, comps):
        return Board.__generic_marking(pos, board, comps, Cell.A * pl, Cell.ED * pl, Cell.EA * pl,
                                       lambda b, c: True)

    @staticmethod
    def __mark_eaten(pos, pl, board, comps):
        return Board.__generic_marking(pos, board, comps, (abs(board[pos]) + 1) * pl, Cell.EA * -pl, Cell.ED * -pl,
                                       lambda b, c: not Board._is_chain_alive(-pl, b, c))

    @staticmethod
    def _is_chain_alive(pl, board, comp):
        # iterate over all elements in the component
        for pos in comp:
            for nbour in Board.__get_neighbours(pos):
                # if the neighbour is an alive cell
                # the chain is alive, yahoo
                if board[nbour] == Cell.A * pl:
                    return True
        return False

    # alternatively, one can "rotate" the board depending on the player
    # then, probably, one will not need to pass "pl" parameter to all subsequent
    # funcsBoard
    @staticmethod
    def get_moves(k, pl, board):
        # TODO
        #   remove duplicates:
        #       ((0, 1), (1, 0)) and ((1, 0), (0, 1)) are considered duplicates
        #       duplicates don't make much difference in moves' diversity, but greatly increase the computational cost
        #       one has to store the points of the last level in order to explore
        #
        #       create class pt (point) and implement hash
        #       use Counter(moves) == Counter(moves2) to determine of they are equal
        #       this can be done without custom class
        board = board.copy()
        if board[Board.__start_pos] == Cell.E:
            # the board is empty, start from the dummy pos
            made_moves = [(Board.__dummy_pos,)]
        else:
            # get all possible positions from which subsequent moves are available
            made_moves = [(pos,) for pos in
                          zip(*np.where(np.logical_or(board == Cell.A * pl, board == Cell.EA * pl)))]
        comps = Board.__get_connected_components(board)
        # copy as the same board will be related to different moves (their ancestors will modify their board,
        # which will result in modifications of others, hence copy)
        moves = {move: (board.copy(), comps.copy()) for move in made_moves}
        Board._print(board, move="basic")
        n_added_moves = 0

        for _ in range(k):
            # moves is updated. In order to avoid infinite loop
            # freeze its current keys using tuple, iterate over them
            for made_move in tuple(moves.keys()):
                # there is only one state from each pos
                state, comps = moves[made_move]
                *_, last_move = made_move
                # check all positions
                for cpos in Board.__get_neighbours(last_move):
                    move = tuple((*made_move, cpos))
                    if state[cpos] in {Cell.E, Cell.A * -pl}:
                        # no need to apply moves on the last "level"
                        # storing only the moves themselves is sufficient
                        # if _ == k-1:
                        #   ans.append(move[1:])
                        # else:
                        moves[move] = Board.apply_move((cpos,), pl, state, comps=comps)
                        n_added_moves += 1
                        Board._print(moves[move][0], move=move)
            # remove moves, added on the previous level
            # leave added on the current one
            *_, = map(moves.pop, tuple(moves.keys())[:len(moves) - n_added_moves])
            n_added_moves = 0
        # return only the moves of the "last" layer
        # remove the starting point from them (so only the made moves are returned)
        return (key[1:] for key in moves.keys()), bool(moves)
        # return (key[1:] for key in moves.keys() if len(key) == (k + 1))

    @staticmethod
    def apply_move(moves, pl, board, *, comps=None):
        action_table = {
            Cell.E: Board.__add_alive,  # occupy the empty cell
            Cell.A * -pl: Board.__mark_eaten  # eat the enemy's cell
        }
        # Board._print(board)
        if comps is None:
            comps = Board.__get_connected_components(board)

        for move in moves:
            action = action_table.get(board[move], None)
            if action is not None:
                board, comps = action(move, pl, board, comps)
            else:
                raise RuntimeError(f"Illegal move: trying to place onto {board[move]}, which is prohibited")
            # Board._print(board)
        return board, comps

    @staticmethod
    def _print(board, *, move: str | tuple = None, newline: bool = True):
        board = board.astype(dtype=Board.__dtype)
        trt = "·x$#@¥o"
        pref, header = ("", "") if move is None else (" " * 4, f"move: {move}\n")
        suff = "\n" * newline

        out = f"""{header}{chr(10).join(pref + ' '.join(map(trt.__getitem__, row)) for row in board)}{suff}"""
        if Board.__file is None:
            print(out)
        else:
            with Board.__file.open("a") as f:
                f.write(out)

    @staticmethod
    def is_ended(k, player, board):
        return not Board.get_moves(k, player, board)[1]

    # TODO
    #   smarten things up
    #   instead of returning and storing all the moves in _get_moves
    #       return the last level, store the current and previous levels


if __name__ == "__main__":
    # print(Board(5))
    a = np.ones((5, 5))

    Board._Board__n = a.shape[0]
    print(Board._Board__n)
    # Board._Board__file = Path("../myfile.txt")

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
