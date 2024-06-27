#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
A sandbox for playing with the HardMaze
@author: Tomas Svoboda
@contact: svobodat@fel.cvut.cz
@copyright: (c) 2017, 2018
"""

import time
import numpy as np
import sys
import os
import gym

import kuimaze
from kuimaze import keyboard
import rl_agent


# MAP = 'maps/normal/normal3.bmp'
MAP = "maps/easy/easy3.bmp"
MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
# PROBS = [0.8, 0.1, 0.1, 0]
PROBS = [1, 0, 0, 0]
GRAD = (0, 0)
keyboard.SKIP = False
VERBOSITY = 2

GRID_WORLD3 = [
    [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0]],
    [[255, 255, 255], [0, 0, 0], [255, 255, 255], [0, 255, 0]],
    [[0, 0, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
]

# MAP = GRID_WORLD3


def get_visualisation(table):
    ret = []
    for i in range(len(table[0])):
        for j in range(len(table)):
            ret.append(
                {
                    "x": j,
                    "y": i,
                    "value": [
                        table[j][i][0],
                        table[j][i][1],
                        table[j][i][2],
                        table[j][i][3],
                    ],
                }
            )
    return ret


def walk_randomly(env):
    # use gym.spaces.prng.seen(number) if you want to fix the randomness between experiments
    # gym.spaces.prng.seed()  # without parameter it uses machine tima
    action_space = env.action_space
    action_space.seed()
    obv = env.reset()
    state = obv[0:2]
    total_reward = 0
    is_done = False
    MAX_T = 1000  # max trials (for one episode)
    t = 0
    while not is_done and t < MAX_T:
        t += 1
        action = env.action_space.sample()

        obv, reward, is_done, _ = env.step(action)
        nextstate = obv[0:2]
        total_reward += reward

        # this is perhaps not correct, just to show something
        q_table[state[0]][state[1]][action] = reward
        # another silly idea
        # q_table[state[0]][state[1]][action] = t

        if VERBOSITY > 0:
            print(state, action, nextstate, reward)
            env.visualise(get_visualisation(q_table))
            env.render()
            keyboard.wait_n_or_s()

        state = nextstate

    if not is_done:
        print("Timed out")

    print("total_reward:", total_reward)


# def learn_policy(env):


if __name__ == "__main__":
    # Initialize the maze environment
    env = kuimaze.HardMaze(map_image=MAP, probs=PROBS, grad=GRAD)

    if VERBOSITY > 0:
        print("====================")
        print("works only in terminal! NOT in IDE!")
        print("press n - next")
        print("press s - skip to end")
        print("====================")

    """
    Define constants:
    """
    # Maze size
    x_dims = env.observation_space.spaces[0].n
    y_dims = env.observation_space.spaces[1].n
    maze_size = tuple((x_dims, y_dims))

    # Number of discrete actions
    num_actions = env.action_space.n
    # print(num_actions)
    # Q-table:
    q_table = np.zeros([maze_size[0], maze_size[1], num_actions], dtype=float)
    
    # print(maze_size[0],'.', maze_size[1])
    if VERBOSITY > 0:
        env.visualise(get_visualisation(q_table))
        env.render()
    # time.sleep(5)


    q_table=rl_agent.learn_policy(env)


    # walk_randomly(env)

    if VERBOSITY > 0:
        keyboard.SKIP = False
        env.visualise(get_visualisation(q_table))
        time.sleep(5)
        env.render()
        keyboard.wait_n_or_s()

        env.save_path()
        env.save_eps()
