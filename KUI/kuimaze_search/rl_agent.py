import kuimaze
from kuimaze import HardMaze
import numpy as np
import time


def learn_policy(env):
    start_time = time.time()
    obv = env.reset()

    alpha=0.2
    #this is the parameter for the Q-function

    all_states=env.get_all_states()
    q_table = np.zeros([(all_states[-1][0])+1, (all_states[-1][1])+1, 4], dtype=float)
    for state in all_states:
        for action in range(4):
            q_table[state[0],state[1],action]=0

    while(True):
        obv = env.reset()
        my_state=obv[0:2]
        done = False
        #done is True if state is terminal
        while not done:
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)
            new_state=observation[0:2]

            max_temporary_Q = float("-inf")
            # Q-function variable

            for act in range(4):
                max_temporary_Q=max(max_temporary_Q,q_table[new_state[0],new_state[1],act])

            q_table[my_state[0],my_state[1],action]=(q_table[my_state[0],my_state[1],action])
            q_table[my_state[0],my_state[1],action]+=alpha*(reward+(max_temporary_Q)-q_table[my_state[0],my_state[1],action])

            my_state=new_state
            end_time = time.time()
            if (end_time-start_time)>=5:
                break

        end_time = time.time()
        if (end_time-start_time)>5:
            policy={}
            for state in all_states:
                best_policy=0
                max_Q=float("-inf")
                for action in range(4):
                    if max_Q<q_table[state[0],state[1],action]:
                        max_Q=q_table[state[0],state[1],action]
                        best_policy=action
                policy[state]=best_policy
            print(q_table)
                
            break
    return q_table
