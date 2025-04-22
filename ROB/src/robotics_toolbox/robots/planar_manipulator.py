#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-08-21
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing planar manipulator."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from shapely import MultiPolygon, LineString, MultiLineString

from robotics_toolbox.core import SE2, SE3
from robotics_toolbox.robots.robot_base import RobotBase


class PlanarManipulator(RobotBase):
    def __init__(
        self,
        link_parameters: ArrayLike | None = None,
        structure: list[str] | str | None = None,
        base_pose: SE2 | None = None,
        gripper_length: float = 0.2,
    ) -> None:
        """
        Creates a planar manipulator composed by rotational and prismatic joints.

        The manipulator kinematics is defined by following kinematics chain:
         T_flange = (T_base) T(q_0) T(q_1) ... T_n(q_n),
        where
         T_i describes the pose of the next link w.r.t. the previous link computed as:
         T_i = R(q_i) Tx(l_i) if joint is revolute,
         T_i = R(l_i) Tx(q_i) if joint is prismatic,
        with
         l_i is taken from @param link_parameters;
         type of joint is defined by the @param structure.

        Args:
            link_parameters: either the lengths of links attached to revolute joints
             in [m] or initial rotation of prismatic joint [rad].
            structure: sequence of joint types, either R or P, [R]*n by default
            base_pose: mounting of the robot, identity by default
            gripper_length: length of the gripper measured from the flange
        """
        super().__init__()
        self.link_parameters: np.ndarray = np.asarray(
            [0.5] * 3 if link_parameters is None else link_parameters
        )
        n = len(self.link_parameters)
        self.base_pose = SE2() if base_pose is None else base_pose
        self.structure = ["R"] * n if structure is None else structure
        assert len(self.structure) == len(self.link_parameters)
        self.gripper_length = gripper_length

        # Robot configuration:
        self.q = np.array([np.pi / 8] * n)
        self.gripper_opening = 0.2

        # Configuration space
        self.q_min = np.array([-np.pi] * n)
        self.q_max = np.array([np.pi] * n)

        # Additional obstacles for collision checking function
        self.obstacles: MultiPolygon | None = None

    @property
    def dof(self):
        """Return number of degrees of freedom."""
        return len(self.q)

    def sample_configuration(self):
        """Sample robot configuration inside the configuration space. Will change
        internal state."""
        return np.random.uniform(self.q_min, self.q_max)

    def set_configuration(self, configuration: np.ndarray | SE2 | SE3):
        """Set configuration of the robot, return self for chaining."""
        self.q = configuration
        return self

    def configuration(self) -> np.ndarray | SE2 | SE3:
        """Get the robot configuration."""
        return self.q

    def flange_pose(self) -> SE2:
        """Return the pose of the flange in the reference frame."""
        # todo HW02: implement fk for the flange
        result=self.base_pose
        for i in range(len(self.link_parameters)):
            if self.structure[i] == 'R':
                result=result*SE2(rotation=self.q[i])*SE2(translation=[self.link_parameters[i],0])
            else:
                result=result*SE2(rotation=self.link_parameters[i])*SE2(translation=[self.q[i],0])

        return result


    def fk_all_links(self) -> list[SE2]:
        """Compute FK for frames that are attached to the links of the robot.
        The first frame is base_frame, the next frames are described in the constructor.
        """
        # todo HW02: implement fk
        frames = []
        result=self.base_pose
        frames.append(result)
        for i in range(len(self.link_parameters)):
            if self.structure[i] == 'R':
                result=result*SE2(rotation=self.q[i])*SE2(translation=[self.link_parameters[i],0])
            else:
                result=result*SE2(rotation=self.link_parameters[i])*SE2(translation=[self.q[i],0])
            frames.append(result)
        return frames

    def _gripper_lines(self, flange: SE2):
        """Return tuple of lines (start-end point) that are used to plot gripper
        attached to the flange frame."""
        gripper_opening = self.gripper_opening / 2.0
        return (
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([0, +gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([self.gripper_length, -gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, +gripper_opening])).translation,
                (flange * SE2([self.gripper_length, +  gripper_opening ])).translation,
            ),
        )

    def jacobian(self) -> np.ndarray:
        """Computes jacobian of the manipulator for the given structure and
        configuration."""
        jac = np.zeros((3, len(self.q)))
        # todo: HW03 implement jacobian computation
        phi = self.base_pose.rotation.angle
        angles=[]
        length=[]
        for i in range(len(self.q)):
            if self.structure[i] == 'R':
                phi+=self.q[i]
                length.append(self.link_parameters[i])

            else:
                phi+=self.link_parameters[i]
                length.append(self.q[i])
            angles.append(phi)
        
        for i in range(len(self.q)):
            if self.structure[i]=='R':
                for j in range(i,len(self.q)):
                    jac[0,i]+=-length[j]*np.sin(angles[j])
                    jac[1,i]+=length[j]*np.cos(angles[j])
                    jac[2,i]=1
            else:
                jac[0,i]=np.cos(angles[i])
                jac[1,i]=np.sin(angles[i])
                jac[2,i]=0
        return jac


    def jacobian_finite_difference(self, delta=1e-5) -> np.ndarray:
        jac = np.zeros((3, len(self.q))) 
        # todo: HW03 implement jacobian computation 
        q_original=self.q.copy()
        FK_original=np.array([self.flange_pose().translation[0],
                              self.flange_pose().translation[1],
                              self.flange_pose().rotation.angle])
        for j in range(jac.shape[1]):
            q_delta = q_original.copy()
            q_delta[j]=q_delta[j] + delta
            self.q = q_delta

            FK_flange=self.flange_pose()
            FK_delta=np.array([ FK_flange.translation[0],
                               FK_flange.translation[1],
                               FK_flange.rotation.angle])
            delta_f=FK_delta-FK_original
            delta_f[2]=(delta_f[2] + np.pi) % (2 * np.pi) - np.pi
            jac[:,j]=delta_f/delta

        self.q=q_original
        return jac


    def ik_numerical(
        self,
        flange_pose_desired: SE2,
        max_iterations=1000,
        acceptable_err=1e-4,
        e=np.array([0,0,0])
    ) -> bool:
        """Compute IK numerically. Value self.q is used as an initial guess and updated
        to solution of IK. Returns True if converged, False otherwise."""
        # todo: HW04 implement numerical IK
        for i in range(max_iterations):
            flange_pose=self.flange_pose()
            e=[flange_pose_desired.translation[0]-flange_pose.translation[0],
               flange_pose_desired.translation[1]-flange_pose.translation[1],
               flange_pose_desired.rotation.angle-flange_pose.rotation.angle]
            e[2]=( e[2] + np.pi) % (2 * np.pi) - np.pi
            if np.linalg.norm(e) < acceptable_err:
                return True
            jacobian=self.jacobian_finite_difference(delta=1e-5)
            pseudoinv_jac=np.linalg.pinv(jacobian)
            delta_q=pseudoinv_jac @ e
            self.q+=delta_q


        return False

    def ik_analytical(self, flange_pose_desired: SE2) -> list[np.ndarray]:
        """Compute IK analytically, return all solutions for joint limits being
        from -pi to pi for revolute joints -inf to inf for prismatic joints."""
        assert self.structure in (
            "RRR",
            "PRR",
        ), "Only RRR or PRR structure is supported"

        # todo: HW04 implement analytical IK for RRR manipulator
        # todo: HW04 optional implement analytical IK for PRR manipulator

        if self.structure == "RRR":
            solution = []

            l1 = self.link_parameters[0]
            l2 = self.link_parameters[1]

            phi_desired = flange_pose_desired.rotation.angle
        
            T_sec=SE2()
            T_sec=flange_pose_desired * SE2(translation=[-self.link_parameters[2],0])
            x_wrist=T_sec.translation[0]
            y_wrist=T_sec.translation[1]
            x_base=self.base_pose.translation[0]
            y_base=self.base_pose.translation[1]



            x0, y0, r0 = np.array([x_base, y_base, l1], dtype=float)
            x1, y1, r1 = np.array([x_wrist, y_wrist, l2], dtype=float)
            dx = x1 - x0
            dy = y1 - y0
            d = np.hypot(dx, dy)
            a = (r0**2 - r1**2 + d**2) / (2 * d)
            h_sq = r0**2 - a**2
            h_sq = np.maximum(h_sq, 0)
            h = np.sqrt(h_sq)
            x2 = x0 + a * dx / d
            y2 = y0 + a * dy / d
            rx = -dy * (h / d)
            ry = dx * (h / d)
            intersection1 = np.array([x2 + rx, y2 + ry])
            intersection2 = np.array([x2 - rx, y2 - ry])
            print(intersection1,intersection2)
            
            T_base_R=self.base_pose.inverse()
            T_intersec1_R=SE2(translation=intersection1)
            T_intersec1_base=T_base_R * T_intersec1_R
            theta11=np.arctan2(T_intersec1_base.translation[1],T_intersec1_base.translation[0])
            theta11=( theta11 + np.pi) % (2 * np.pi) - np.pi
            T_R_A=self.base_pose * SE2(rotation=theta11) * SE2(translation=[l1,0])
            T_A_R=T_R_A.inverse()
            T_sec_A=T_A_R * T_sec
            theta21=np.arctan2(T_sec_A.translation[1],T_sec_A.translation[0])
            theta21=( theta21 + np.pi) % (2 * np.pi) - np.pi
            theta31=phi_desired-self.base_pose.rotation.angle - theta11 - theta21
            theta31=( theta31 + np.pi) % (2 * np.pi) - np.pi
            q_sol1=np.array([theta11,theta21,theta31])
            solution.append(q_sol1)



            T_intersec2_R=SE2(translation=intersection2)
            T_intersec2_base=T_base_R * T_intersec2_R
            theta12=np.arctan2(T_intersec2_base.translation[1],T_intersec2_base.translation[0])
            theta12=( theta12 + np.pi) % (2 * np.pi) - np.pi
            T_R_A_=self.base_pose * SE2(rotation=theta12) * SE2(translation=[l1,0])
            T_A_R_=T_R_A_.inverse()
            T_sec_A_=T_A_R_ * T_sec
            theta22=np.arctan2(T_sec_A_.translation[1],T_sec_A_.translation[0])
            theta22=( theta22 + np.pi) % (2 * np.pi) - np.pi
            theta32=phi_desired-self.base_pose.rotation.angle - theta12 - theta22
            theta32=( theta32 + np.pi) % (2 * np.pi) - np.pi
            q_sol2=np.array([theta12,theta22,theta32])
            solution.append(q_sol2)
            return solution
        if self.structure == "PRR":

            
            
            solution = []
            theta1=self.link_parameters[0]
            l2 = self.link_parameters[1]
            l3 = self.link_parameters[2]

            phi_desired = flange_pose_desired.rotation.angle

            T_C_R=SE2()
            T_C_R=flange_pose_desired * SE2(translation=[-l3,0])
            x_C_R=T_C_R.translation[0]
            y_C_R=T_C_R.translation[1]
            x_base=self.base_pose.translation[0]
            y_base=self.base_pose.translation[1]
            
            T_base_=self.base_pose*SE2(rotation=theta1)*SE2(translation=[1,0])
            x_base_=T_base_.translation[0]
            y_base_=T_base_.translation[1]


            # print([x_base,y_base])
            # print([x_base_,y_base_])
            # print(np.hypot(x_base_-x_base,y_base_-y_base))
            # T_base___=self.base_pose.inverse()*T_base_
            # T_base___=self.base_pose*SE2(rotation=theta1)
            # T_base___=T_base___.inverse()*T_base_
            # print([T_base___.translation[0],T_base___.translation[1]])

            intersections =self._line_circle_intersection(x_base,y_base,x_base_,y_base_,x_C_R,y_C_R,l2)
            for point in intersections:
                x_intersect, y_intersect = point
                d1 = np.hypot(x_intersect - x_base, y_intersect - y_base)
                # print('d1',d1)

                # T_B_R = self.base_pose.inverse()

                # T_A_B=T_B_R * SE2(rotation=theta1) * SE2(translation=point)

                T_B_R = self.base_pose* SE2(rotation=theta1)
                T_B_R = T_B_R.inverse()
                T_A_B=T_B_R  * SE2(translation=point)

                print(T_A_B.translation[0],T_A_B.translation[1]) 
                if T_A_B.translation[0] < 0 :
                    d1=-d1              
                print('d1',d1)
                T_point=self.base_pose*SE2(rotation=theta1) * SE2(translation=[d1,0])
                T_C_point=T_point.inverse() * T_C_R
                theta_2=np.arctan2(T_C_point.translation[1],T_C_point.translation[0])
                # print(T_C_point.translation[1],T_C_point.translation[0])
                theta_2=( theta_2 + np.pi) % (2 * np.pi) - np.pi
                theta_3=phi_desired-theta1-theta_2-self.base_pose.rotation.angle
                theta_3=( theta_3 + np.pi) % (2 * np.pi) - np.pi
                solution.append(np.array([d1,theta_2,theta_3]))
                print('theta_2',theta_2)
                print('theta_3',theta_3)

                result=self.base_pose
                result = result * SE2(rotation=theta1)*SE2(translation=[d1,0]) *SE2(rotation=theta_2) * SE2(translation=[l2,0])*SE2(rotation=theta_3) * SE2(translation=[l3,0])
                print("res",result.translation[0],result.translation[1])
            return solution




    def _line_circle_intersection(self, x1, y1, x2, y2, xc, yc, r):
        p1 = np.array([x1, y1], dtype=float)
        p2 = np.array([x2, y2], dtype=float)
        center = np.array([xc, yc], dtype=float)

        # Direction vector of the line
        d = p2 - p1

        # Shift line to circle's coordinate system
        f = p1 - center

        # Quadratic coefficients
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - r**2

        # Discriminant
        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            # No intersection
            return None
        elif np.isclose(discriminant, 0):
            # One intersection (tangent)
            t = -b / (2 * a)
            intersection = p1 + t * d
            return [tuple(intersection)]
        else:
            # Two intersections
            sqrt_discriminant = np.sqrt(discriminant)
            t1 = (-b + sqrt_discriminant) / (2 * a)
            t2 = (-b - sqrt_discriminant) / (2 * a)
            intersection1 = p1 + t1 * d
            intersection2 = p1 + t2 * d
            return [tuple(intersection1), tuple(intersection2)]

    def in_collision(self) -> bool:
        """Check if robot in its current pose is in collision."""
        frames = self.fk_all_links()
        points = [f.translation for f in frames]
        gripper_lines = self._gripper_lines(frames[-1])

        links = [LineString([a, b]) for a, b in zip(points[:-2], points[1:-1])]
        links += [MultiLineString((*gripper_lines, (points[-2], points[-1])))]
        for i in range(len(links)):
            for j in range(i + 2, len(links)):
                if links[i].intersects(links[j]):
                    return True
        return MultiLineString(
            (*gripper_lines, *zip(points[:-1], points[1:]))
        ).intersects(self.obstacles)
