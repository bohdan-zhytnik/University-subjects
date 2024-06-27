import random
import time
import sys
import copy


# positional advantage board for 6x6 8x8 10x10
# corners are valued the most while tiles that give the opponent access to corners the least
positional_advantage = [[
    [60, -10, 10, 10, -10, 60],
    [-10, -20, -2, -2, -20, -10],
    [10, -2, 5, 5, -2, 10],
    [10, -2, 5, 5, -2, 10],
    [-10, -20, -2, -2, -20, -10],
    [60, -10, 10, 10, -10, 60]
],
    [
    [120, -20, 20, 5, 5, 20, -20, 120],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [120, -20, 20, 5, 5, 20, -20, 120]
],
    [
    [120, -20, 20, 5, 5, 5, 5, 20, -20, 120],
    [-20, -40, -5, -5, -5, -5, -5, -5, -40, -20],
    [20, -5, 15, 3, 3, 3, 3, 15, -5, 20],
    [5, -5, 3, 3, 3, 3, 3, 3, -5, 5],
    [5, -5, 3, 3, 3, 3, 3, 3, -5, 5],
    [5, -5, 3, 3, 3, 3, 3, 3, -5, 5],
    [5, -5, 3, 3, 3, 3, 3, 3, -5, 5],
    [20, -5, 15, 3, 3, 3, 3, 15, -5, 20],
    [-20, -40, -5, -5, -5, -5, -5, -5, -40, -20],
    [120, -20, 20, 5, 5, 5, 5, 20, -20, 120]
]]


class MyPlayer():
    '''minimax with alpha beta prouning, evaluation based on positional advantage'''

    def __init__(self, my_color, opp_color, board_size=8):
        self.name = 'Marchell0'
        self.my_color = my_color
        self.opp_color = opp_color
        self.board_size = board_size
        self.board_idx = (self.board_size // 2) - 3

    def move(self, board):
        # setting starting time and depth for the first iteration
        start_time = time.time()
        max_depth = 4
        best_move = None

        possible_moves = self.get_valid_moves(board, self.my_color)
        if possible_moves:
            if len(possible_moves) == 1:
                return (possible_moves[0][0], possible_moves[0][1])
            else:
                while True:
                    # if the algorithm is going to deep return the best move
                    if max_depth > self.board_size**2:
                        return (best_move[0], best_move[1])

                    # initialize main variables
                    max_eval = float('-inf')
                    alpha = float("-inf")
                    beta = float("inf")

                    for move in possible_moves:
                        # make a copy of the board and make a move on it
                        tmp_board = copy.deepcopy(board)
                        tmp_board = self.make_move(
                            tmp_board, move, self.my_color)

                        # call a recursive minimax function on it
                        eval = self.minimax(
                            tmp_board, max_depth, False, alpha, beta, start_time)

                        # if minimax returned out of time, return the best move
                        if eval == "out of time":
                            return (best_move[0], best_move[1])

                        # if minimax evaluation of the move is better than the previous best change it
                        if eval > max_eval:
                            max_eval = eval
                            temp_best_move = move

                    # after an iteration ended change the actual best move
                    best_move = temp_best_move

                    # and iterate with depth+1
                    max_depth += 1
        else:
            return (best_move[0], best_move[1])

    def minimax(self, board, depth, maximizing_player, alpha, beta, start_time):

        # for every state evaluation check if theres enough time
        if time.time() - start_time > 4.900:
            return "out of time"

        # if allowed depth is reached or its the ending node return static evaluation
        if depth == 0 or self.is_game_over(board):
            return self.evaluate(board)

        # same as the outer move function
        if maximizing_player:
            max_eval = float('-inf')

            possible_moves = self.get_valid_moves(board, self.my_color)
            if possible_moves:
                for move in possible_moves:
                    temp_board = copy.deepcopy(board)
                    temp_board = self.make_move(board, move, self.my_color)

                    eval = self.minimax(
                        temp_board, depth - 1, False, alpha, beta, start_time)

                    if eval == 'out of time':
                        return eval

                    max_eval = max(max_eval, eval)

                    # alpha beta prouning part
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break

                return max_eval

            # if there are no available moves call another player to play
            else:
                return self.minimax(board, depth-1, False, alpha, beta, start_time)

        # same as the outer move function but for minimazing player
        else:
            min_eval = float('inf')

            possible_moves = self.get_valid_moves(board, self.opp_color)
            if possible_moves:
                for move in possible_moves:
                    temp_board = copy.deepcopy(board)
                    temp_board = self.make_move(board, move, self.opp_color)

                    eval = self.minimax(
                        temp_board, depth - 1, True, alpha, beta, start_time)

                    if eval == 'out of time':
                        return eval

                    min_eval = min(min_eval, eval)

                    # alpha beta prouning part
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break

                return min_eval

            # if there are no available moves call another player to play
            else:
                return self.minimax(board, depth-1, True, alpha, beta, start_time)

    def is_game_over(self, board):
        # if no valid moves for both players return True
        if not self.get_valid_moves(board, 0) and not self.get_valid_moves(board, 1):
            return True
        else:
            return False

    def evaluate(self, board):
        board_eval = 0

        # multitply the tile value by the assigned value on the positional advantage board
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] != -1:
                    if board[i][j] == 0:
                        board_eval -= positional_advantage[self.board_idx][i][j]
                    else:
                        board_eval += positional_advantage[self.board_idx][i][j]

        if self.my_color == 0:
            return -board_eval
        else:
            return board_eval

    def make_move(self, board, move, player_color):
        temp_board = copy.deepcopy(board)
        temp_board[move[0]][move[1]] = player_color
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]
        for i in range(len(dx)):
            if self.__confirm_direction(temp_board, move, dx[i], dy[i], player_color):
                self.change_stones_in_direction(
                    temp_board, move, dx[i], dy[i], player_color)
        return temp_board

    def get_valid_moves(self, board, player_color):

        valid_moves = []

        for x in range(self.board_size):
            for y in range(self.board_size):
                if (board[x][y] == -1) and self.__is_correct_move([x, y], board, player_color):
                    valid_moves.append(
                        (x, y, positional_advantage[self.board_idx][x][y]))

        if len(valid_moves) <= 0:
            return None

        # return available moves sorted by their positional advantage
        return sorted(valid_moves, key=lambda x: x[2], reverse=True)

    def __confirm_direction(self, board, move, dx, dy, player_color):

        if player_color == 0:
            opponents_color = 1
        else:
            opponents_color = 0

        posx = move[0]+dx
        posy = move[1]+dy
        if (posx >= 0) and (posx < self.board_size) and (posy >= 0) and (posy < self.board_size):
            if board[posx][posy] == opponents_color:
                while (posx >= 0) and (posx < self.board_size) and (posy >= 0) and (posy < self.board_size):
                    posx += dx
                    posy += dy
                    if (posx >= 0) and (posx < self.board_size) and (posy >= 0) and (posy < self.board_size):
                        if board[posx][posy] == -1:
                            return False
                        if board[posx][posy] == player_color:
                            return True

        return False

    def change_stones_in_direction(self, board, move, dx, dy, player_color):
        posx = move[0]+dx
        posy = move[1]+dy
        while (not (board[posx][posy] == player_color)):
            board[posx][posy] = player_color
            posx += dx
            posy += dy

    def __is_correct_move(self, move, board, player_color):
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]
        for i in range(len(dx)):
            if self.__confirm_direction(board, move, dx[i], dy[i], player_color):
                return True,
        return False


if __name__ == "__main__":

    board = [
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 1, -1, -1, -1],
        [-1, -1, -1, 1, 1, -1, -1, -1],
        [-1, -1, -1, 0, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ]

    player = MyPlayer(0, 1, 8)
    move = player.move(board)
    print(move)
