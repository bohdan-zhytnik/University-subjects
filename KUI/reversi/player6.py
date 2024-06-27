import copy
import time
import numpy as np


grid_6x6 = np.array([
    [200,-3,11,11,-3,200],
    [-3,-7,-4,-4,-7,-3],
    [11,-4, 2, 2,-4,11],
    [11,-4, 2, 2,-4,11],
    [-3,-7,-4,-4,-7,-3],
    [200,-3,11,11,-3,200],
])
grid_8x8 = np.array([
    [200, -3, 11, 8, 8, 11, -3, 200],
    [-3, -7, -4, 1, 1, -4, -7, -3],
    [11, -4,  2, 2, 2,  2, -4, 11],
    [8,   1,  2,-3,-3,  2,  1, 8 ],
    [8,   1,  2,-3,-3,  2,  1, 8 ],
    [11, -4,  2, 2, 2,  2, -4, 11],
    [-3, -7, -4, 1, 1, -4, -7, -3],
    [200, -3, 11, 8, 8, 11, -3, 200],
])
grid10x10 = ([
    [200,-3,11,8, 6, 6, 8,11,-3,200],
    [-3,-7,-4,1, 0, 0, 1,-4,-7,-3],
    [11,-4, 3,0, 3, 3, 0, 3,-4,11],
    [8,  1, 0,2, 2, 2, 2, 0, 1, 8],
    [6,  0, 3,2,-3,-3, 2, 3, 0, 6],
    [6,  0, 3,2,-3,-3, 2, 3, 0, 6],
    [8,  1, 0,2, 2, 2, 2, 0, 1, 8],
    [11,-4, 3,0, 3, 3, 0, 3,-4,11],
    [-3,-7,-4,1, 0, 0, 1,-4,-7,-3],
    [200,-3,11,8, 6, 6, 8,11,-3,200]
])
    

class MyPlayer:
    """Template Docstring for MyPlayer, look at the TODOs"""  # TODO a short description of your player

    def __init__(self, my_color, opponent_color, board_size=8):
        self.name = "username"  # TODO: fill in your username
        self.my_color = my_color
        self.opponent_color = opponent_color
        self.board_size = board_size

        if self.board_size == 6:
            self.grid_board = grid_6x6#Grid.grid_6x6
        elif self.board_size == 8:
            self.grid_board = grid_8x8#Grid.grid_8x8
        else:
            self.grid_board = grid10x10#Grid.grid10x10

        self.corners = [[0,0],[0, self.board_size-1],[self.board_size-1, 0],[self.board_size-1, self.board_size-1]]    

    def move(self, board):
        RECURSION_LEN = 6
        BOARD_AREA = self.board_size * self.board_size  # is used as max score of evaluation
        ALPHA = float("-inf")
        BETA = float("inf")

        corners = [[0,0],[0, self.board_size-1],[self.board_size-1, 0],[self.board_size-1, self.board_size-1]]

        def ab_best_of_moves(board):
            cost = float("-inf")
            maxcost = float("-inf")
            bestmove = None
            moves = self.get_all_valid_moves(board)

            num_of_moves = len(moves)

            if num_of_moves == 1:
                return moves[0]

            for move in moves:
                if move in corners:
                    #print("fucking here --->", move)
                    return move
                print("here")    
                start_time = time.time()
                new_board = copy.deepcopy(board)
                new_board = play_my_move(new_board, move, self.my_color)#<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<
                cost = max(cost, minimax(new_board, RECURSION_LEN, start_time, move, ALPHA, BETA, False))
                #cost = max(cost, minimax(new_board, RECURSION_LEN, start_time, move, ALPHA, BETA, True))

                if cost > maxcost:
                    maxcost = cost
                    bestmove = move

            return bestmove

        def minimax(current_board, rec_len, start_time, move, alpha, beta, maxPlayer):
            if is_terminal(current_board) or rec_len == 0 or (time.time()-start_time >= 0.1):
                print("rec len =",RECURSION_LEN-rec_len)
                color = maxPlayer
                return utility(current_board, color, move) # returns an actual cost of the gameboard
            moves = self.get_all_valid_moves(current_board)
            if maxPlayer:  # MAX player
                cost = float("-inf")
                for move in moves:
                    new_board = copy.deepcopy(current_board)
                    new_board = play_my_move(new_board, move, self.my_color)#----<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<
                    #current_board[move[0]][move[1]] = self.my_color  # make possible move
                    cost = max(cost, minimax(new_board, rec_len - 1, start_time, move, alpha, beta, False))
                    #alpha = max(alpha, cost)
                    if beta <= alpha:
                        #current_board[move[0]][move[1]] = -1  # clear possible move
                        break
                    alpha = max(alpha, cost)
                    #current_board[move[0]][move[1]] = -1  # clear possible move
            else:  # MIN player
                cost = float("inf")
                for move in moves:
                    new_board = copy.deepcopy(current_board)
                    new_board = play_my_move(new_board, move, self.opponent_color)#----<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<
                    #current_board[move[0]][move[1]] = self.opponent_color  # make possible move
                    cost = min(cost, minimax(new_board, rec_len - 1, start_time, move, alpha, beta, True))
                    #beta = min(beta, cost)
                    # if cost < alpha:
                    if beta <= alpha:
                        #current_board[move[0]][move[1]] = -1  # clear possible move
                        break
                    beta = min(beta, cost)
                    #current_board[move[0]][move[1]] = -1  # clear possible move
            return cost

        def utility(board, color, move):
            start = time.process_time()
            # mobility -
            # mobility of the stones
            # side borders -
            # stability of stones - is it easy to capture my stones

            #w1 = 0.2#0.25  # difference and finish
            #w2 = 0.4#0.25  # corners
            #w3 = 0#0.25  # possible moves
            #w4 = 0.4#0.25  # sides

            w1 = 0.3
            w2 = 0.3
            w3 = 0.3

            def difference_and_finish(board, color, move):
                mine_num = 0
                not_mine_num = 0
                finish = 0
                #grid_sum = 0
                for line in board:
                    for cell_color in line:
                        if cell_color == color:
                            mine_num += 1
                            #grid_sum += self.grid_board[move[0]][move[1]]
                        elif cell_color == (color+1)%2:
                            not_mine_num += 1
                            #grid_sum -= self.grid_board[move[0]][move[1]]
                difference = mine_num - not_mine_num
                if mine_num + not_mine_num is BOARD_AREA:
                    if difference > 0:
                        finish = BOARD_AREA
                    elif difference < 0:
                        finish = -BOARD_AREA
                return 0.35*difference + finish# + 0.01*grid_sum

            dif_f = difference_and_finish(board, color, move)

            def grid_move(move):
                return self.grid_board[move[0]][move[1]]

            grdm = grid_move(move)

            def num_of_possible_moves(board):
                moves = self.get_all_valid_moves(board)
                if moves == None:
                    return 0
                else:
                    return 2*len(moves)

            npm = num_of_possible_moves(board)

            print(time.process_time() - start)
            return w1*dif_f + w2*grdm + w3*npm
            """
            def corners(board, color):
                my_occupied = 0
                opp_occupied = 0
                corners = [
                    [0, 0],
                    [0, self.board_size - 1],
                    [self.board_size - 1, 0],
                    [self.board_size - 1, self.board_size - 1],
                ]
                for corner in corners:
                    if board[corner[0]][corner[1]] is color:
                        my_occupied += 1
                    elif board[corner[0]][corner[1]] is (color + 1) % 2:
                        opp_occupied += 1
                return (my_occupied - opp_occupied)*100

            def num_of_possible_moves(board):
                moves = self.get_all_valid_moves(board)
                if moves == None:
                    return 0
                else:
                    return len(moves)

            npm = num_of_possible_moves(board)    

            cor = corners(board, color)

            def sides(board, color):
                num_on_sides = 0
                for cell in board[0]:
                    if cell == color:
                        num_on_sides += 1
                    elif cell == (color + 1) % 2:
                        num_on_sides -= 1
                for cell in board[self.board_size - 1]:
                    if cell == color:
                        num_on_sides += 1
                    elif cell == (color + 1) % 2:
                        num_on_sides -= 1
                for i in range(1, self.board_size - 2):
                    for j in [0, self.board_size - 1]:
                        if board[i][j] == color:
                            num_on_sides += 1
                        elif board[i][j] == (color + 1) % 2:
                            num_on_sides -= 1
                return num_on_sides*100

            side = sides(board, color)
            #print(f"dif = {dif_f}, corners = {cor}, sides = {side}")
            #return w1*dif_f + w2*cor + w3*npm + w4*side
            return w1*dif_f + w2*cor + w4*side    
            """

        def play_my_move(board, move, color):
            board[move[0]][move[1]] = color
            dx = [-1,-1,-1,0,1,1,1,0]
            dy = [-1,0,1,1,1,0,-1,-1]
            for i in range(len(dx)):
                if self.__confirm_direction(move, dx[i], dy[i], board, color)[0]: ##############################
                    posx = move[0]+dx[i]
                    posy = move[1]+dy[i]
                    while (not(board[posx][posy] == color)):
                        board[posx][posy] = color
                        posx += dx[i]
                        posy += dy[i]
            return board            


        def is_terminal(board):
            if self.get_all_valid_moves(board) == None:
                return True
            else:
                return False

        return ab_best_of_moves(board)

    def __is_correct_move(self, move, board):
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]
        for i in range(len(dx)):
            if self.__confirm_direction(move, dx[i], dy[i], board, self.my_color)[0]:
            #if self.__confirm_direction(move, dx[i], dy[i], board)[0]:
                return (True,)
        return False

    def __confirm_direction(self, move, dx, dy, board, color):
    #def __confirm_direction(self, move, dx, dy, board):
        posx = move[0] + dx
        posy = move[1] + dy
        opp_stones_inverted = 0
        if (
            (posx >= 0)
            and (posx < self.board_size)
            and (posy >= 0)
            and (posy < self.board_size)
        ):
            if board[posx][posy] == (color+1)%2:
            #if board[posx][posy] == self.opponent_color:
                opp_stones_inverted += 1
                while (
                    (posx >= 0)
                    and (posx <= (self.board_size - 1))
                    and (posy >= 0)
                    and (posy <= (self.board_size - 1))
                ):
                    posx += dx
                    posy += dy
                    if (
                        (posx >= 0)
                        and (posx < self.board_size)
                        and (posy >= 0)
                        and (posy < self.board_size)
                    ):
                        if board[posx][posy] == -1:
                            return False, 0
                        if board[posx][posy] == color:
                        #if board[posx][posy] == self.my_color:
                            return True, opp_stones_inverted
                    opp_stones_inverted += 1

        return False, 0

    def get_all_valid_moves(self, board):
        valid_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (board[x][y] == -1) and self.__is_correct_move([x, y], board):
                    valid_moves.append((x, y))

        if len(valid_moves) <= 0:
            print("No possible move!")
            return None
        return valid_moves
