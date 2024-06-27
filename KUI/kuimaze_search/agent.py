import time
import kuimaze
import os
import random
import operator

class Agent(kuimaze.BaseAgent):
    def __init__(self, environment):
        self.environment = environment
                                                                                    #f=h+g
                                                                                    #g - path cost
    def evaluate_h(self, position,goal):                                            #h value for A* algorithm
        h=(((goal[0]-position[0])**2)+((goal[1]-position[1])**2))**(1/2)              
        return(h)
        
    def find_path(self):
        observation = self.environment.reset()
        goal = observation[1][0:2]
        position = observation[0][0:2]                                              # starting position coordinates
        h_position = self.evaluate_h(position,goal)
        queue = [(position, [position],observation[0][2],h_position+observation[0][2])]        #queue=[(vertex,[path],path cost,f value)]
        new_positions_for_next=[]                                                   # elements of new_positions without path cost
        Visited=[]                                                                  # Visited point
        point_cost={}
        while queue:
            queue=sorted(queue, key= operator.itemgetter(3))                        # sort queue by value f
            new_positions_for_next.clear()
            (vertex, pathBFS,cost,f) = queue.pop(0)
            new_positions = self.environment.expand(vertex)

            for i in range(len(new_positions)):                                     # add to the element new_positions its valuse h
                h=self.evaluate_h(new_positions[i][0],goal)
                new_positions[i].append(h+new_positions[i][1])

            new_positions=sorted(new_positions, key= operator.itemgetter(2))        # sort new_positions by value f

            for i in range(len(new_positions)): 
                new_positions_for_next.append(new_positions[i][0])
                point_cost[new_positions[i][0]]=new_positions[i][1]

            Visited.extend(pathBFS)

            for next in set(new_positions_for_next) - set(Visited):
                Visited.append(next)
                if next == goal:
                    return pathBFS +[next]
                else:
                    h=self.evaluate_h(next,goal)
                    queue.append((next, pathBFS + [next], point_cost[next]+cost,(h)+(cost)+point_cost[next]))


if __name__ == '__main__':

    MAP = 'maps/easy/easy3.bmp'
    MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
    GRAD = (0, 0)
    SAVE_PATH = False
    SAVE_EPS = False

    env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)       # For using random map set: map_image=None
    agent = Agent(env) 

    path= agent.find_path()
    print('path',path)
    env.set_path(path)          # set path it should go from the init state to the goal state
    if SAVE_PATH:
        env.save_path()         # save path of agent to current directory
    if SAVE_EPS:
        env.save_eps()          # save rendered image to eps
    env.render(mode='human')
    # time.sleep(60)
