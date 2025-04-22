#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-09-21
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for RRT motion planning algorithm."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from copy import deepcopy

from robotics_toolbox.core import SE3, SE2
from robotics_toolbox.robots.robot_base import RobotBase


class RRT:
    def __init__(self, robot: RobotBase, delta_q=0.2, p_sample_goal=0.5) -> None:
        """RRT planner for a given robot.
        Args:
            robot: robot used to sample configuration and check collisions
            delta_q: maximum distance between two configurations
            p_sample_goal: probability of sampling goal as q_rand
        """
        self.p_sample_goal = p_sample_goal
        self.robot = robot
        self.delta_q = delta_q


        self.nodes=[]
        self.parents=[]

    def plan(
        self,
        q_start: ArrayLike | SE2 | SE3,
        q_goal: ArrayLike | SE2 | SE3,
        max_iterations: int = 10000,
    ) -> list[ArrayLike | SE2 | SE3]:
        """RRT algorithm for motion planning."""
        assert not self.robot.set_configuration(q_start).in_collision()
        assert not self.robot.set_configuration(q_goal).in_collision()
        # todo: hw06opt implement RRT

        self.nodes=[np.array(q_start)]
        self.parents=[None]
        for iteration in range (max_iterations):
            if np.random.rand() < self.p_sample_goal:
                q_rand =np.array( q_goal)
            else:
                q_rand = self.robot.sample_configuration()
            q_nearest_index = self.nearest_node(q_rand)
            q_nearest = self.nodes[q_nearest_index]
            q_new = self.get_new_node(q_rand, q_nearest)
            if not self.robot.set_configuration(q_new).in_collision():
                self.nodes.append(q_new)
                self.parents.append(q_nearest_index)

                if np.linalg.norm(q_goal - q_new) < self.delta_q:
                    self.nodes.append(q_goal)
                    self.parents.append(len(self.nodes)-2)
                    print(f"Goal reached in {iteration + 1} iterations.")
                    return self.full_path()
        return []

    def nearest_node(self, q_rand: ArrayLike) -> int:
        min_dist = float('inf')
        min_dist_index=0
        for i in range(len(self.nodes)):
            distance = np.linalg.norm(q_rand - self.nodes[i])
            if distance < min_dist:
                min_dist = distance
                min_dist_index = i
        return min_dist_index
    
    def get_new_node(self,q_rand: ArrayLike, q_nearest: ArrayLike):
        distance = np.linalg.norm(q_rand-q_nearest)
        if distance < self.delta_q:
            return q_rand
        else:
            return q_nearest + ((q_rand - q_nearest) / distance ) * self.delta_q
     
    def full_path(self):
        path=[]
        index = len(self.nodes)-1
        while index is not None:
            path.append(self.nodes[index])
            index=self.parents[index]
        path.reverse()
        return path


    def random_shortcut(
        self, path: list[np.ndarray | SE2 | SE3], max_iterations=100
    ) -> list[np.ndarray | SE2 | SE3]:
        """Random shortcut algorithm that pick two points on the path randomly and tries
        to interpolate between them. If collision free interpolation exists,
        the path between selected points is replaced by the interpolation."""
        # todo: hw06opt implement random shortcut algorithm
        out = deepcopy(path)
        len_out=len(out)
        for _ in range(max_iterations):
            outer_break = False
            if len_out < 3:
                break
            i = np.random.randint(0,len_out-2)
            j = np.random.randint(i+2,len_out)
            q_i = out[i]
            q_j = out[j]
            n = int(np.ceil(np.linalg.norm(q_j - q_i) / (self.delta_q * 0.5)))
            new_segment=[]
            for k in range(n):
                q_k =q_i + (q_j-q_i) *k / n
                if not self.robot.set_configuration(q_k).in_collision():
                    new_segment.append(q_k)
                else:
                    outer_break = True
                    break
            if outer_break == True:
                continue
            else:
                out = out[:i+1] + new_segment[1:-1] + out[j:]
                len_out=len(out)
        
        return out
