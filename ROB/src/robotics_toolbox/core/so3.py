#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 3D rotation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class SO3:
    """This class represents an SO3 rotations internally represented by rotation
    matrix."""

    def __init__(self, rotation_matrix: ArrayLike | None = None) -> None:
        """Creates a rotation transformation from rot_vector."""
        super().__init__()
        self.rot: np.ndarray = (
            np.asarray(rotation_matrix) if rotation_matrix is not None else np.eye(3)
        )

    @staticmethod
    def exp(rot_vector: ArrayLike) -> SO3:
        """Compute SO3 transformation from a given rotation vector, i.e. exponential
        representation of the rotation."""
        v = np.asarray(rot_vector)
        assert v.shape == (3,)
        t = SO3()
        theta = np.linalg.norm(v)
        omega=v/theta
        Skew_symetric=np.array([
        [0, -omega[2],omega[1]],
        [omega[2],0,-omega[0]],
        [-omega[1],omega[0],0]
                                ])

        # todo HW01: implement Rodrigues' formula, t.rot = ...
        t.rot=np.eye(3)+np.sin(theta) * Skew_symetric+(1-np.cos(theta)) * np.dot(Skew_symetric,Skew_symetric)
        return t

    def log(self) -> np.ndarray:
        """Compute rotation vector from this SO3"""
        # todo HW01: implement computation of rotation vector from this SO3
        v = np.zeros(3)
        omega = np.zeros(3)
        if not np.allclose(self.rot,np.eye(3)):
            if np.isclose(np.trace(self.rot),-1) : 
                if self.rot[2,2]!=-1:
                    omega=(1/(np.sqrt(2*(1+self.rot[2,2]))))*np.array([self.rot[0,2],self.rot[1,2],self.rot[2,2]+1])
                elif self.rot[1,1]!=-1:
                    omega=(1/(np.sqrt(2*(1+self.rot[1,1]))))*np.array([self.rot[0,1],self.rot[1,1]+1,self.rot[2,1]])
                elif self.rot[0,0]!=-1:
                    omega=(1/(np.sqrt(2*(1+self.rot[0,0]))))*np.array([self.rot[0,0]+1,self.rot[1,0],self.rot[2,0]])
                v=omega*np.pi
            else:
                theta=np.arccos(0.5*(np.trace(self.rot)-1))
                Skew_symetric=(self.rot-self.rot.T)/(2*np.sin(theta))
                omega[0]=Skew_symetric[2,1]
                omega[1]=Skew_symetric[0,2]
                omega[2]=Skew_symetric[1,0]
                v=omega*theta
                
        
        return v

    def __mul__(self, other: SO3) -> SO3:
        """Compose two rotations, i.e., self * other"""
        # todo: HW01: implement composition of two rotation.
        result=SO3()
        result.rot=np.dot(self.rot,other.rot)
        return result

    def inverse(self) -> SO3:
        """Return inverse of the transformation."""
        # todo: HW01: implement inverse, do not use np.linalg.inverse()
        result=SO3()
        result.rot=self.rot.T
        return result    

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given vector by this transformation."""
        v = np.asarray(vector)
        assert v.shape == (3,)
        return self.rot @ v

    def __eq__(self, other: SO3) -> bool:
        """Returns true if two transformations are almost equal."""
        return np.allclose(self.rot, other.rot)

    @staticmethod
    def rx(angle: float) -> SO3:
        """Return rotation matrix around x axis."""
        # todo: HW1opt: implement rx
        rx=SO3()
        rot=np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
            ])
        rx.rot=rot
        return rx

    @staticmethod
    def ry(angle: float) -> SO3:
        """Return rotation matrix around y axis."""
        # todo: HW1opt: implement ry
        ry=SO3()
        rot=np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0,1,0],
            [-np.sin(angle), 0, np.cos(angle)]
            ])
        ry.rot=rot
        return ry

    @staticmethod
    def rz(angle: float) -> SO3:
        """Return rotation matrix around z axis."""
        # todo: HW1opt: implement rz
        rz=SO3()
        rot=np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),np.cos(angle),0],
            [0, 0, 1]
            ])
        rz.rot=rot
        return rz

    @staticmethod
    def from_quaternion(q: ArrayLike) -> SO3:
        """Compute rotation from quaternion in a form [qx, qy, qz, qw]."""
        # todo: HW1opt: implement from quaternio    n
        q = np.asarray(q, dtype=float)
        q = q / np.linalg.norm(q)
        qx, qy, qz, qw = q
        rot=np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])

        return SO3(rot)

    def to_quaternion(self) -> np.ndarray:
        """Compute quaternion from self."""
        R = self.rot
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        return np.array([qx, qy, qz, qw])
    @staticmethod
    def from_angle_axis(angle: float, axis: ArrayLike) -> SO3:
        """Compute rotation from angle axis representation."""
        # todo: HW1opt: implement from angle axis
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        kx, ky, kz = axis
        skew_symmetric = np.array([
            [0, -kz, ky],
            [kz, 0, -kx],
            [-ky, kx, 0]
        ])
        I = np.eye(3)
        R = I + np.sin(angle) * skew_symmetric + \
            (1 - np.cos(angle)) * np.dot(skew_symmetric, skew_symmetric)
        return SO3(R)

    def to_angle_axis(self) -> tuple[float, np.ndarray]:
        """Compute angle axis representation from self."""
        R = self.rot
        angle = np.arccos((np.trace(R) - 1) / 2)
        if np.isclose(angle, np.pi):
            R_plus = (R + np.eye(3)) / 2
            axis = np.sqrt(np.diagonal(R_plus))
            axis = axis / np.linalg.norm(axis)
        else:
            axis = np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ]) / (2 * np.sin(angle))
        return angle, axis

    @staticmethod
    def from_euler_angles(angles: ArrayLike, seq: list[str]) -> SO3:
        """Compute rotation from Euler angles defined by a given sequence.
        angles: is a three-dimensional array of angles
        seq: is a list of axis around which angles rotate, e.g. ['x', 'y', 'z'], ['x', 'z', 'x'], etc.
        """
        angles = np.asarray(angles, dtype=float)
        R = SO3()
        R.rot = np.eye(3)
        for angle, axis in zip(angles, seq):
            if axis.lower() == 'x':
                R_axis = SO3.rx(angle)
            elif axis.lower() == 'y':
                R_axis = SO3.ry(angle)
            elif axis.lower() == 'z':
                R_axis = SO3.rz(angle)
            R = R * R_axis
        return R

    def __hash__(self):
        return id(self)
