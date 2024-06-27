#!/usr/bin/python3
'''
Very simple example how to use gym_wrapper and BaseAgent class for state space search 
@author: Zdeněk Rozsypálek, and the KUI-2019 team
@contact: svobodat@fel.cvut.cz
@copyright: (c) 2017, 2018, 2019
'''

import time
import kuimaze
import os
import random
import operator

class Agent(kuimaze.BaseAgent):
    '''
    Simple example of agent class that inherits kuimaze.BaseAgent class 
    '''
    def __init__(self, environment):
        self.environment = environment

    def evaluate_h(self, position,goal):
        h=(((goal[0]-position[0])**2)+((goal[1]-position[1])**2))**(1/2)
        return(h)
        
    

    def find_path(self):
        '''
        Method that must be implemented by you. 
        Expects to return a path as a list of positions [(x1, y1), (x2, y2), ... ].
        '''
        observation = self.environment.reset() # must be called first, it is necessary for maze initialization
        #print('self.environment',self.environment)
        print('observation',observation)
        goal = observation[1][0:2]
        position = observation[0][0:2]                               # initial state (x, y)
        # print('observation[0][2])]',observation[0][2])
        h_position = self.evaluate_h(position,goal)
        queue = [(position, [position],observation[0][2],h_position+observation[0][2])]
        new_positions_for_loop=[]
        BigPath=[]
        point_cost={}
        # new_positions = self.environment.expand(position)
        # print('new_positions',new_positions)
        # for i in range(len(new_positions)):
        #     new_positions_for_loop.append(new_positions[i][0])
        # print('new_positions_for_loop',set(new_positions_for_loop))
        while queue:
            # print('queue',queue)
            # sorted(queue, key=itemgetter(3))
            queue=sorted(queue, key= operator.itemgetter(3))
            # print('queue_A_S',queue)
            new_positions_for_loop.clear()
            (vertex, pathBFS,cost,f) = queue.pop(0)
            # print('cost',cost)
            # print('vertex, pathBfS',vertex, pathBFS)

            # print('queue',queue)
            # print('position',position)
            new_positions = self.environment.expand(vertex)
            for i in range(len(new_positions)):
                h=self.evaluate_h(new_positions[i][0],goal)
                new_positions[i].append(h+new_positions[i][1])
            new_positions=sorted(new_positions, key= operator.itemgetter(2))
            # new_positions = self.environment.expand(vertex)
            print('new_positions',new_positions)
            for i in range(len(new_positions)):
                new_positions_for_loop.append(new_positions[i][0])
                point_cost[new_positions[i][0]]=new_positions[i][1]
            # print('point_cost',point_cost)
            # print('new_positions_for_loop',(new_positions_for_loop))
            BigPath.extend(pathBFS)
            # print('BigPath',set(BigPath))
            # (vertex, pathBFS) = queue.pop(0)
            # print('pathBFS',set(pathBFS))
            # print('set(new_positions_for_loop) - set(pathBFS)',set(new_positions_for_loop) - set(pathBFS))
            for next in set(new_positions_for_loop) - set(BigPath):
                BigPath.append(next)
                print('next',next)
                if point_cost[next]!=1:
                    print(point_cost[next])
                # print('next',next)
                if next == goal:
                    print('............................')
                    return pathBFS +[next],cost
                else:
                    h=self.evaluate_h(next,goal)
                    print((next, pathBFS + [next], point_cost[next]+cost,(h/5)+(cost*2)))
                    # queue.append((next, pathBFS + [next], point_cost[next]+cost,h+(cost/4)))
                    queue.append((next, pathBFS + [next], point_cost[next]+cost,(h)+(cost)+point_cost[next]))
                    # print(queue)
        # while True:
        #     new_positions = self.environment.expand(position)         # [[(x1, y1), cost], [(x2, y2), cost], ... ]
        #     print('new_positions',new_positions)
        #     position = random.choice(new_positions)[0]                # select next at random, ignore the cost infor
        #     print(position)
        #     if position == goal:                    # break the loop when the goal position is reached
        #         print("goal reached")
        #         break
            self.environment.render()               # show enviroment's GUI       DO NOT FORGET TO COMMENT THIS LINE BEFORE FINAL SUBMISSION!      
            # time.sleep(0.2)                         # sleep for demonstartion     DO NOT FORGET TO COMMENT THIS LINE BEFORE FINAL SUBMISSION! 

        # path = [(4,0),(4,1)]        # create path as list of tuples in format: [(x1, y1), (x2, y2), ... ] 
        # return path


if __name__ == '__main__':

    MAP = 'maps/normal/normal12.bmp'
    MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
    GRAD = (0, 0)
    SAVE_PATH = False
    SAVE_EPS = False

    env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)       # For using random map set: map_image=None
    agent = Agent(env) 

    path,cost = agent.find_path()
    print('path',path)
    print('cost',cost)
    env.set_path(path)          # set path it should go from the init state to the goal state
    if SAVE_PATH:
        env.save_path()         # save path of agent to current directory
    if SAVE_EPS:
        env.save_eps()          # save rendered image to eps
    env.render(mode='human')
    time.sleep(60)
