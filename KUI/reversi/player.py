# I used the pseudocode from this link
# https://pastebin.com/rZg1Mz9G


import copy
import time

positional_advantage = [
[ 
    [50, -20,  5,  5,  -20,  50],
    [-20,-30,  1,  1,  -30, -20],
    [10,   1,  3,  3,    1,  10],
    [10,   1,  3,  2,    2,   3],
    [-20,-30,  1,  1,  -30, -20],
    [50, -20,  5,  5,  -20,  50]
],
[
    [100, -20, 20, 5, 5, 20, -20, 100],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [100, -20, 20, 5, 5, 20, -20, 100]
],
[
    [150, -50,  20,  10,  10,  10,  10,  20, -50, 150],
    [-50, -70, -10,  -5,  -5,  -5,  -5, -10, -70, -50],
    [ 20, -10,   20,   3,   3,   3,   3,   20, -10,  20],
    [ 10,  -5,   3,   3,   3,   3,   3,   3,  -5,  10],
    [ 10,  -5,   3,   3,   3,   3,   3,   3,  -5,  10],
    [ 10,  -5,   3,   3,   3,   3,   3,   3,  -5,  10],
    [ 10,  -5,   3,   3,   3,   3,   3,   3,  -5,  10],
    [ 20, -10,   20,   3,   3,   3,   3,   20, -10,  20],
    [-50, -70, -10,  -5,  -5,  -5,  -5, -10, -70, -50],
    [150, -50,  20,  10,  10,  10,  10,  20, -50, 150]
]]



class MyPlayer():
    ''' 
The eval. f. considers the number of stones and the value of positions.

    ''' 

    def __init__(self, my_color,opponent_color, board_size=8):
        self.name = 'zhytnboh'
        self.my_color = my_color
        self.opponent_color = opponent_color
        self.board_size = board_size
        self.max_depth=4     
        self.empty_color=-1

    def evaluate_max_depth(self):
        if self.board_size==10:
            self.max_depth=4
        elif self.board_size==8:
            self.max_depth=5
        elif self.board_size==6:
            self.max_depth=6

    def move(self,board):
        global start_time
        start_time = time.time()
        self.evaluate_max_depth()
        inf=float('inf')
        my_player=True
        # this value is needed so that mini_max starts with the max search
        max_value,best_move=self.mini_max_A_B(board,self.my_color,self.max_depth,-inf,inf,my_player)
        return best_move

    def mini_max_A_B(self,board0,color,depth,alpha,beta,my_player):
    # This method recursively evaluates moves

        now_time=time.time()
        running_time = now_time-start_time
        # print('running_time',running_time)
        if running_time>=4.6:
            depth=0

        if my_player:
            color=self.my_color
        else:
            color=self.opponent_color

        all_valid_moves= self.get_all_valid_moves(board0,color)
        best_move=None
        if (all_valid_moves is None ) or (depth == 0):
            return self.evaluation(board0),(0,0)                                        
        if my_player:
            # This part of the function is looking for the best move for my_player
            max_value=-float('inf')
            for move in all_valid_moves:
                board=copy.deepcopy(board0)
                self.play_move(board,move,color)
                eval,some_move=self.mini_max_A_B(board,color,depth-1,alpha,beta,False)
                if max_value<eval:
                    max_value=eval
                    best_move=move
                alpha=max(alpha,eval)
                if beta<=alpha:
                    break    
            return max_value,best_move
        

        else:
            # This part of the function looks for the worst possible move for me by my opponent
            min_value=float('inf')
            for move in all_valid_moves:
                board=copy.deepcopy(board0)
                self.play_move(board,move,color)
                eval,some_move=self.mini_max_A_B(board,color,depth-1,alpha,beta,True)
                if min_value>eval:
                    min_value=eval
                    best_move=move
                beta=min(beta,eval)
                if beta<=alpha:
                    break    

            return min_value,best_move


    def evaluation(self,board):
        
        # this method returns an estimate of the move by:

        # the difference in the number of my stones and the opponent's stones
        # the difference in the number of my possible moves and the opponent's possible moves
        # how advantageous are the positions of my stones
        
        # Here I look for the difference in the number of my stones and the opponent's stones
        stones = [0 , 0]
        for x in range(len(board)):
            for y in range(len(board)):
                if board[x][y] == self.my_color:
                    stones[0] += 1
                if board[x][y] == self.opponent_color:
                    stones[1] += 1
        diff_num=(stones[0]-stones[1])

        # Here I look for the difference in the number of my possible moves and the opponent's possible moves
        diff_moves=0
        my_moves=self.get_all_valid_moves(board,self.my_color)
        opponent_moves=self.get_all_valid_moves(board,self.opponent_color)
        if my_moves is not None and opponent_moves is not None:
            diff_moves=(len(my_moves)-len(opponent_moves))*5

        # Еhis part of method explore how advantageous are the positions of my stones

        sum_grid=0
        if self.board_size==6:
           number_board=0
        elif self.board_size==8:
           number_board=1
        else:
           number_board=2

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j]==self.my_color:
                    sum_grid+=positional_advantage[number_board][i][j]
                if board[i][j]==self.opponent_color:
                    sum_grid-=positional_advantage[number_board][i][j]

        move_evaluation = diff_num*2 + diff_moves + sum_grid
        # Final heuristic value of this move
        return move_evaluation
    

    def play_move(self,board,move,players_color):

        board[move[0]][move[1]] = players_color
        dx = [-1,-1,-1,0,1,1,1,0]
        dy = [-1,0,1,1,1,0,-1,-1]
        for i in range(len(dx)):
            if self.confirm_direction(board,move,dx[i],dy[i],players_color):
                self.change_stones_in_direction(board,move,dx[i],dy[i],players_color)

    def confirm_direction(self,board,move,dx,dy,players_color):
        if players_color == self.my_color:
            opponents_color = self.opponent_color
        else:
            opponents_color = self.my_color
        posx = move[0]+dx
        posy = move[1]+dy
        if (posx>=0) and (posx<self.board_size) and (posy>=0) and (posy<self.board_size):
            if board[posx][posy] == opponents_color:
                while (posx>=0) and (posx<self.board_size) and (posy>=0) and (posy<self.board_size):
                    posx += dx
                    posy += dy
                    if (posx>=0) and (posx<self.board_size) and (posy>=0) and (posy<self.board_size):
                        if board[posx][posy] == self.empty_color:
                            return False
                        if board[posx][posy] == players_color:
                            return True

        return False

    def change_stones_in_direction(self,board,move,dx,dy,players_color):
        posx = move[0]+dx
        posy = move[1]+dy
        while (not(board[posx][posy] == players_color)):
            board[posx][posy] = players_color
            posx += dx
            posy += dy

    def __is_correct_move(self, move, board, active_color):
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]    
        permission = False
        sum_opp_stones_inverted=0
        for i in range(len(dx)):
            check=self.__confirm_direction(move, dx[i], dy[i], board, active_color)
            if check==True:
                permission=True
        if permission == True:
            return True
        else:
            return False

    def __confirm_direction(self, move, dx, dy, board, active_color):
        posx = move[0]+dx
        posy = move[1]+dy
        opp_stones_inverted = 0
        if (posx >= 0) and (posx < self.board_size) and (posy >= 0) and (posy < self.board_size):
            if board[posx][posy] == (active_color+1)%2:
                opp_stones_inverted += 1
                while (posx >= 0) and (posx <= (self.board_size-1)) and (posy >= 0) and (posy <= (self.board_size-1)):
                    posx += dx
                    posy += dy
                    if (posx >= 0) and (posx < self.board_size) and (posy >= 0) and (posy < self.board_size):
                        if board[posx][posy] == -1:
                            return False
                        if board[posx][posy] == active_color:
                            return True
                    opp_stones_inverted += 1

        return False

    def get_all_valid_moves(self, board,active_color):
        valid_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                check=self.__is_correct_move([x, y], board,active_color)
                if (board[x][y] == -1) and check==True :
                    valid_moves.append( (x, y) )

        if len(valid_moves) <= 0:
            print('No possible move!')
            return None
        return valid_moves
