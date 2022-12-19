import rospy
import rospkg
import numpy as np
import argparse
import time
from graic_msgs.msg import ObstacleList, ObstacleInfo
from graic_msgs.msg import LocationInfo, WaypointInfo
from ackermann_msgs.msg import AckermannDrive
from carla_msgs.msg import CarlaEgoVehicleControl
from graic_msgs.msg import LaneList
from graic_msgs.msg import LaneInfo
import math


class VehicleDecision():
    def __init__(self):
        self.vehicle_state = 'straight'
        self.lane_state = 0
        self.counter = 0

        self.lane_marker = None
        self.target_x = None
        self.target_y = None
        self.change_lane = False
        self.change_lane_wp_idx = 0
        self.detect_dist = 60
        self.speed = 40
        self.target_lane = 4
        self.reachEnd = False
        self.rrt_counter = 0
        self.waypoints_rrt = []
        self.velocity_multiplier = 1




    def get_borders(self, lane_marker):
        left_borders = []
        right_borders = []

        for i in range(1, len(lane_marker.lane_markers_center.location)):
            curr = lane_marker.lane_markers_center.location[i]
            prev = lane_marker.lane_markers_center.location[i-1]
            left_border, right_border = self.calc_normal(curr, prev)
            left_borders.append(left_border)
            right_borders.append(right_border)

        return np.array(left_borders), np.array(right_borders)

    def lengthSquare(self, X, Y):
        xDiff = X.x - Y.x
        yDiff = X.y - Y.y
        return xDiff * xDiff + yDiff * yDiff

    def printAngle(self, A, B, C):

        # Square of lengths be a2, b2, c2
        a2 = self.lengthSquare(B, C)
        b2 = self.lengthSquare(A, C)
        c2 = self.lengthSquare(A, B)

        # length of sides be a, b, c
        a = math.sqrt(a2);
        b = math.sqrt(b2);
        c = math.sqrt(c2);

        # From Cosine law
        alpha = math.acos((b2 + c2 - a2) /
                             (2 * b * c));
        betta = math.acos((a2 + c2 - b2) /
                             (2 * a * c));
        gamma = math.acos((a2 + b2 - c2) /
                             (2 * a * b));

        # Converting to degree
        alpha = alpha * 180 / math.pi;
        betta = betta * 180 / math.pi;
        gamma = gamma * 180 / math.pi;

        print("alpha : %f" %(alpha))
        print("betta : %f" %(betta))
        print("gamma : %f" %(gamma))
        return alpha, betta, gamma

    def get_ref_state(self, currState, obstacleList, lane_marker, waypoint):
        """
            Get the reference state for the vehicle according to the current state and result from perception module
            Inputs:
                currState: [Loaction, Rotation, Velocity] the current state of vehicle
                obstacleList: List of obstacles
            Outputs: reference state position and velocity of the vehicle
        """
        # self.reachEnd = waypoint.reachedFinal
        p_last = lane_marker.lane_markers_center.location[-1]


        self.lane_marker = p_last
        self.lane_state = lane_marker.lane_state
        self.velocity_multiplier = 1

        if self.target_lane == self.lane_state:
            self.change_lane = False

        ##CHANGE
        if not self.target_x or not self.target_y:
            self.target_x = self.lane_marker.x
            self.target_y = self.lane_marker.y
        if self.reachEnd:
            return None
        # print("Reach end: ", self.reachEnd)

        curr_x = currState[0][0]
        curr_y = currState[0][1]

        # Check whether any obstacles are in the front of the vehicle
        obs_front = False
        obs_left = False
        obs_right = False
        front_dist = np.inf

        if obstacleList and not self.change_lane:
            for obs in obstacleList:
                for vertex in obs.vertices_locations:
                    dy = vertex.vertex_location.y - curr_y
                    dx = vertex.vertex_location.x - curr_x
                    yaw = currState[1][2]
                    rx = np.cos(-yaw) * dx - np.sin(-yaw) * dy
                    ry = np.cos(-yaw) * dy + np.sin(-yaw) * dx

                    psi = np.arctan(ry / rx)
                    x = np.sqrt(dy * dy + dx * dx)

                    front_dist = min(front_dist, x)
                    # if rx > 0:
                    #     print("detected object is at {} away and {} radians".format(x, psi))
                    #     if psi < 0.2 and psi > -0.2:
                    #         obs_front = True
                    #     elif psi > 0.2:
                    #         obs_right = True
                    #     elif psi < -0.2:
                    #         obs_left = True
                    if x < self.detect_dist and rx>0:
                        alpha, beta, gamma = self.printAngle(vertex.vertex_location, lane_marker.lane_markers_left.location[-1], lane_marker.lane_markers_right.location[-1])
                        print("Point found")
                        if gamma > 90:
                            obs_right = True
                        elif alpha > 90:
                            obs_left = True
                        else:
                            obs_front = True

        if self.target_lane > self.lane_state:
            self.vehicle_state = "turn-right"
        elif self.target_lane < self.lane_state:
            self.vehicle_state = "turn-left"

        # prev_vehicle_state = self.vehicle_state
        if self.lane_state == LaneInfo.LEFT_LANE and not self.change_lane:
            if front_dist <= self.detect_dist and obs_front:
                if not obs_right:
                    self.vehicle_state = "turn-right"
                    self.change_lane = True
                    self.target_lane += 1
                else:
                    self.vehicle_state = "stop"
            else:
                self.vehicle_state = "straight"

        elif self.lane_state == LaneInfo.RIGHT_LANE and not self.change_lane:
            if front_dist <= self.detect_dist and obs_front:
                if not obs_left:
                    self.vehicle_state = "turn-left"
                    self.change_lane = True
                    self.target_lane -= 1
                else:
                    self.vehicle_state = "stop"
            else:
                self.vehicle_state = "straight"

        elif self.lane_state == LaneInfo.CENTER_LANE and not self.change_lane:
            if front_dist > self.detect_dist:
                self.vehicle_state = "straight"
            else:
                if not obs_front:
                    self.vehicle_state = "straight"
                elif not obs_left:
                    self.vehicle_state = "turn-left"
                    self.change_lane = True
                    self.target_lane -= 1
                elif not obs_right:
                    self.vehicle_state = "turn-right"
                    self.change_lane = True
                    self.target_lane += 1
                else:
                    self.vehicle_state = "stop"

        # if obstacleList and len(obstacleList) < 3:
        #     self.velocity_multiplier = min(0.5, self.velocity_multiplier)
        # if obstacleList and len(obstacleList) > 2 and front_dist < self.detect_dist/2:
        #     self.velocity_multiplier = min(0.025, self.velocity_multiplier)
        # if obstacleList and len(obstacleList) > 2 and front_dist > self.detect_dist/2:
        #     print("ObstacleList >2?????", len(obstacleList))
        #     self.velocity_multiplier = min(0.05, self.velocity_multiplier)
        # if self.vehicle_state =="straight":
        #     self.velocity_multiplier = min(1, self.velocity_multiplier)
        # if front_dist > self.detect_dist/2 and self.target_lane == self.lane_state and self.vehicle_state!="straight":
        #     print("We are here!!", self.vehicle_state, self.vehicle_state!="straight")
        #     self.velocity_multiplier = min(0.25, self.velocity_multiplier)
        #     self.vehicle_state = "straight"
        velocity = currState[2][0]*currState[2][0] + currState[2][1]*currState[2][1]
        if velocity >20 and len(obstacleList)>3:
            print("STARTING TO STOP")
            self.vehicle_state = "stop"
        if self.vehicle_state == "stop":
            self.speed = 0
        else:
            self.speed  = 40
        if self.change_lane:
            self.velocity_multiplier = 0.2
        # else:
        #     self.speed = 40


        print("Vehicle State: ", self.vehicle_state, "Vehicle Lane: ", self.lane_state,"Target Lane: ", self.target_lane)

        # print(front_dist, self.lane_state, self.vehicle_state, obs_front, obs_left, obs_right)

        distToTargetX = abs(self.target_x - curr_x)
        distToTargetY = abs(self.target_y - curr_y)
        print("distToTargetX", distToTargetX, "distToTargetY", distToTargetY)
        if ((distToTargetX < 5 and distToTargetY < 5)):

            prev_target_x = self.target_x
            prev_target_y = self.target_y

            self.target_x = self.lane_marker.x
            self.target_y = self.lane_marker.y

            target_orientation = np.arctan2(self.target_y - prev_target_y,
                                            self.target_x - prev_target_x)

            if self.vehicle_state == "turn-right":
                # self.change_lane = False
                tmp_x = 4.5
                tmp_y = 0
                x_offset = np.cos(target_orientation + np.pi /
                                  2) * tmp_x - np.sin(target_orientation +
                                                      np.pi / 2) * tmp_y
                y_offset = np.sin(target_orientation + np.pi /
                                  2) * tmp_x + np.cos(target_orientation +
                                                      np.pi / 2) * tmp_y
                self.target_x = self.target_x + x_offset
                self.target_y = self.target_y + y_offset
            elif self.vehicle_state == "turn-left":
                # self.change_lane = False
                tmp_x = 4.5
                tmp_y = 0
                x_offset = np.cos(target_orientation - np.pi /
                                  2) * tmp_x - np.sin(target_orientation -
                                                      np.pi / 2) * tmp_y
                y_offset = np.sin(target_orientation - np.pi /
                                  2) * tmp_x + np.cos(target_orientation -
                                                      np.pi / 2) * tmp_y
                self.target_x = self.target_x + x_offset
                self.target_y = self.target_y + y_offset

        else:
            print("distToTargetX is not defined???")
            self.counter += 1

        return [self.target_x, self.target_y, self.speed, self.velocity_multiplier]


class VehicleController():
    def stop(self, delta = 0):
        print("PRINTING STOP")
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.acceleration = -20
        newAckermannCmd.speed = 0
        newAckermannCmd.steering_angle = delta
        return newAckermannCmd

    def execute(self, currentPose, targetPose):
        """
            This function takes the current state of the vehicle and
            the target state to compute low-level control input to the vehicle
            Inputs:
                currentPose: ModelState, the current state of vehicle
                targetPose: The desired state of the vehicle
        """

        currentEuler = currentPose[1]
        curr_x = currentPose[0][0]
        curr_y = currentPose[0][1]

        target_x = targetPose[0]
        target_y = targetPose[1]
        target_v = targetPose[2]
        velocity_multiplier = targetPose[3]

        k_s = 0.1
        k_ds = 1
        k_n = 0.1
        k_theta = 1
        k_speed = 0.5
        newAckermannCmd = AckermannDrive()

        # compute errors
        dx = target_x - curr_x
        dy = target_y - curr_y
        xError = (target_x - curr_x) * np.cos(
            currentEuler[2]) + (target_y - curr_y) * np.sin(currentEuler[2])
        yError = -(target_x - curr_x) * np.sin(
            currentEuler[2]) + (target_y - curr_y) * np.cos(currentEuler[2])
        curr_v = np.sqrt(currentPose[2][0]**2 + currentPose[2][1]**2)
        delta = k_n * yError
        max_v = 20
        min_v = 10
        tmp_v = (min_v + 1/(k_speed*abs(yError) + 1/(max_v-min_v)))*velocity_multiplier
        vError = tmp_v - curr_v
        # delta = min(delta, delta*(20/curr_v))
        # Checking if the vehicle need to stop
        # print("Length: ", len(lane_marker.lane_markers_center.location), len(lane_marker.lane_markers_center.location[:-10]))


        if target_v > 0:
            # v = xError * k_s + vError * k_ds
            v = tmp_v
            print("velocity_multiplier: ", velocity_multiplier)
            print("Target_v: ", v)
            #Send computed control input to vehicle
            newAckermannCmd.speed = v
            # newAckermannCmd.acceleration = 0
            newAckermannCmd.steering_angle = delta
            return newAckermannCmd
        else:
            return self.stop(delta)


class Controller(object):
    """docstring for Controller"""
    def __init__(self):
        super(Controller, self).__init__()
        self.decisionModule = VehicleDecision()
        self.controlModule = VehicleController()

    def stop(self):
        return self.controlModule.stop()

    def execute(self, currState, obstacleList, lane_marker, waypoint):
        # Get the target state from decision module
        refState = self.decisionModule.get_ref_state(currState, obstacleList,
                                                     lane_marker, waypoint)
        if not refState:
            return None
        return self.controlModule.execute(currState, refState)
