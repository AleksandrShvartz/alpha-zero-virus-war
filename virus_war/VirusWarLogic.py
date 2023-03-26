from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np


class Cell(IntEnum):
    E = 0  # empty
    A = 1  # alive, mine; enemy's with minus
    EA = 2  # eaten, mine (I ate enemy (was -1, I ate, became 2)); enemy's with minus
    ED = 3  # eaten, mine, dead; enemy's with minus


@dataclass
class Data:
    explored: set
    board: np.array
    comps: dict

    def __iter__(self): return iter((self.explored, self.board, self.comps))


def cache(func):
    _cache = {}

    def wrapper(moves, pl, board, *a, **kw):
        h = hash(moves) + hash(Board.to_str(board))
        if h in _cache:
            wrapper.hit += 1
        else:
            wrapper.miss += 1
            _cache[h] = func(moves, pl, board, *a, **kw)
        return _cache[h]

    wrapper.miss = 0
    wrapper.hit = 0
    return wrapper


class Board:
    __dirs = ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))
    __n = 0  # caller must set it via `Board._Board__n = `
    __start_pos = (0, 0)
    __dummy_pos = (-1, -1)
    __dtype = np.int8
    __file = None
    __apl_cnt = 0
    __ign_cnt = 0

    @staticmethod
    def get_init():
        return np.zeros((Board.__n, Board.__n), dtype=Board.__dtype)

    @staticmethod
    def to_str(board):
        return np.array2string(board)

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
        to_visit = set(zip(*np.where(board != Cell.E)))

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
        # doesn't allow duplicates:
        #       ((0, 1), (1, 0)) and ((1, 0), (0, 1)) are considered duplicates
        #       duplicates don't make much difference in moves' diversity, but greatly increase the computational cost
        #       one has to store the points of the last level in order to explore
        # some duplicates are still possible:
        #   if x . x, two moves are possible, that will make the board state the same,
        #   though the moves to achieve it were different
        #   semi-solution is to hash the board to avoid calculating the same moves on the same board
        board = board.copy()
        if board[Board.__start_pos] == Cell.E:
            # the board is empty, start from the dummy pos
            made_moves = [(Board.__dummy_pos,)]
        else:
            # get all possible positions from which subsequent moves are available
            made_moves = [(pos,) for pos in
                          zip(*np.where(np.logical_or(board == Cell.A * pl, board == Cell.EA * pl)))]
        comps = Board.__get_connected_components(board)
        perms = {move: Data(set(), board.copy(), comps.copy()) for move in made_moves}

        # Board._print(board, move="basic")
        n_added_moves = 0

        for cnt in range(k):
            # moves is updated. In order to avoid infinite loop
            # freeze its current keys using tuple, iterate over them
            n_added_moves = 0
            for made_move in tuple(perms.keys()):
                _, state, comps = perms[made_move]
                *_, last_move = made_move
                # check all positions
                for cpos in Board.__get_neighbours(last_move):
                    move = tuple((*made_move, cpos))
                    s_move = tuple(sorted(move))
                    if state[cpos] in {Cell.E, Cell.A * -pl}:
                        if s_move not in perms:
                            # no need to apply moves on the last "level"
                            # storing only the moves themselves is sufficient
                            board_, comps_ = Board.apply_move((cpos,), pl, state, comps=comps) if cnt < k - 1 else (
                                None, None)
                            perms[s_move] = Data({move}, board_, comps_)
                            n_added_moves += 1
                        else:
                            perms[s_move].explored.add(move)
                            Board.__ign_cnt += 1
                        # Board._print(perms[s_move].board, move=move)

            if cnt == k - 1:
                break
            # remove all the dummy points (starting points)
            # also unify some sequences of moves (no different root for the same seq)
            # a
            # . c d      will result in 2 seqs: (a c d) and (b c d)
            # b
            # removing the not so important starter point will reduce them to (c d)
            perms = {new_move[cnt == 0:]: Data(set(), board.copy(), comps.copy())
                     for moves, board, comps in tuple(perms.values())[-n_added_moves:] for new_move in moves}
        # return only the moves of the "last" layer
        r = tuple(perms)[len(perms) - n_added_moves:]
        return r, bool(r)

    @staticmethod
    @cache
    def apply_move(moves, pl, board, *, comps=None):
        Board.__apl_cnt += 1
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
        [-1, 0, -2, -1, 0],
        [0, 0, 0, 0, -1],
    ])
    # print(Board.apply_move(((2, 0), (3, 0), (3, 1)), 1, c))
    d = np.array([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    # print((Board.from_numpy(c)._get_moves(3, -1)).keys())
    moves, _ = Board.get_moves(3, 1, d.astype(np.int8))
    moves = [*moves]
    # print(*sorted(moves), sep=" ")
    from pprint import pprint

    # pprint(sorted(moves))
    print(f"my moves len: {len(moves)}")
    s = set([frozenset(el) for el in moves])
    ls = sorted(tuple(sorted(el)) for el in s)
    # print(ls)
    print(f"frozenset len: {len(s)}")
    print(len(set(moves)), len(set(ls)))
    from collections import Counter
    from pprint import pprint

    pprint(Counter(moves))
    print(f"len of counter {len(Counter(moves))}")
    print(f"len of mod counter {len(Counter([move[1:] for move in moves]))}")
    # print(set(moves) - set(ls))
    # print(len(set(moves)-set(ls)))
    print(f"appended number {Board._Board__apl_cnt}")
    print(f"ignored number {Board._Board__ign_cnt}")
    print(f"cache missed {Board.apply_move.miss}, cache hit {Board.apply_move.hit}, "
          f"cache cnt {Board.apply_move.miss + Board.apply_move.hit}")
