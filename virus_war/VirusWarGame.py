import numpy as np
from Game import Game
from VirusWarLogic import Board
from math import factorial as fac


class VirusWarGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self, n, k):
        """

        :param n: board size
        :param k: number of moves
        """
        super().__init__()
        self._n = n
        self._k = k
        self._bsize = n * n

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return Board(self._n)

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return self._n, self._n

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        # + 1 is for "cannot move"
        return fac(self._bsize) / fac(self._bsize - self._k) + 1

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        nextBoard = Board.from_numpy(board)._apply_moves(action, player, board)
        return nextBoard, -player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        valid_moves = Board.from_numpy(board)._get_moves(self._k, player)
        all_moves = np.zeros(int(self.getActionSize()))
        to_vec_pos = lambda move: sum(
            e * self._bsize * i for i, e in enumerate(map(lambda m: m[0] * self._n + m[1], move)))
        *v_moves_pos, = map(to_vec_pos, valid_moves)
        all_moves[v_moves_pos] = 1
        return all_moves

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        return Board.from_numpy(board)._is_ended(self._k, player, board)

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return board * player

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """

        # copied from OthelloGame
        pi_board = np.reshape(pi[:-1], (self._n, self._n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.to_string()


if __name__ == "__main__":
    g = VirusWarGame(5, 3)
    a = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [-1, 0, -2, -1, 0],
        [0, 0, 0, 0, -1],
    ])
    o = np.ones((5,5))
    # o[0, :3] = 0
    v_moves = g.getGameEnded(o, 1)
    # print(len(v_moves[v_moves==1.0]))
    print(v_moves)