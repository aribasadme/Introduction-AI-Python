"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    nX = 0
    nO = 0
    for i in range(3):
        nX += board[i].count(X)
        nO += board[i].count(O)
        
    return O if nX > nO else X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    if terminal(board):
        return
    
    a = set()
    for i in range(3):
        for j in range(3):
            if (board[i][j] == EMPTY):
                a.add((i,j))
    return a    


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if board[action[0]][action[1]] != EMPTY:
        raise NameError('Not a valid action!')
    
    b = copy.deepcopy(board)
    b[action[0]][action[1]] = player(board)
    return b


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Rows
    for i in range(3):
        if (board[i].count(X) == 3):
            return X
        elif (board[i].count(O) == 3):
            return O
    
    # Columns
    for i in range(3):    
        if (board[0][i] == board[1][i] == board[2][i]):
            if (board[0][i] == X):
                return X
            elif (board[0][i] == O):
                return O

    # Diagonals 
    if (board[0][0] == board[1][1] == board[2][2]):
        if (board[0][0] == X):
            return X
        elif (board[0][0] == O):
            return O
    if (board[2][0] == board[1][1] == board[0][2]):
        if (board[2][0] == X):
            return X
        elif (board[2][0] == O):
            return O

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if (winner(board)):
        return True
    else:
        nEmpty = 0
        for row in board:
            nEmpty += row.count(EMPTY)
        return nEmpty == 0
        # return not any(EMPTY in row for row in board)


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if (winner(board) == X):
        return 1
    elif (winner(board) == O):
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if (terminal(board)):
        return None

    if (player(board) == X):
        v = -math.inf
        for action in actions(board):
            score = min_value(result(board, action))
            if score > v:
                v = score
                best_move = action
    elif (player(board) == O):
        v = math.inf
        for action in actions(board):
            score = max_value(result(board, action))
            if score < v:
                v = score
                best_move = action
    return best_move


def max_value(board):
    """
    Returns the action that maximizes the value of the board.
    """
    if terminal(board):
        return utility(board)

    v = -math.inf
    for a in actions(board):
        v = max(v, min_alpha_beta(result(board, a)))
    return v


def min_value(board):
    """
    Returns the action that minimizes the value of the board.
    """
    if terminal(board):
        return utility(board)
        
    v = math.inf
    for a in actions(board):
        v = min(v, max_alpha_beta(result(board, a)))
    return v


def alphabeta(board):
    """
    Returns the optimal action for the current player on the board.
    """
    alpha = -math.inf
    beta = math.inf
    
    if (terminal(board)):
        return None

    if (player(board) == X):
        v = -math.inf
        for action in actions(board):
            score = min_alpha_beta(result(board, action), alpha, beta)
            if score > v:
                v = score
                best_move = action
    elif (player(board) == O):
        v = math.inf
        for action in actions(board):
            score = max_alpha_beta(result(board, action), alpha, beta)
            if score < v:
                v = score
                best_move = action
    return best_move


def min_alpha_beta(board, alpha, beta):
    if terminal(board):
        return utility(board)
        
    v = math.inf
    for a in actions(board):
        v = min(v, max_alpha_beta(result(board, a), alpha, beta))
        beta = min(beta, v)
        if beta <= alpha:
            break
    return v


def max_alpha_beta(board, alpha, beta):
    if terminal(board):
        return utility(board)

    v = -math.inf
    for a in actions(board):
        v = max(v, min_alpha_beta(result(board, a), alpha, beta))
        alpha = max(alpha, v)
        if alpha >= beta:
            break    
    return v