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
import purepursuit
from RRT import RRT, Obstacle
from threading import Thread


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
        self.speed = 30
        self.target_lane = 4
        self.reachEnd = False
        self.rrt_counter = 0
        self.waypoints_rrt = []
        self.var = [1,2,3,4,5]

    def calc_dist(self,l1, l2):
        return np.sqrt((l1.x-l2.x)**2 +(l1.y-l2.y)**2), l1.x-l2.x, l1.y-l2.y

    def modify_variable(self, rrtsearch):
        path = rrtsearch.run()
        self.lock.acquire()
        self.waypoints_rrt = []
        for wp in path[40:]:
            self.waypoints_rrt.append([wp[0], wp[1]])
        self.waypoints_rrt = np.array(self.waypoints_rrt)
        self.lock.release()


    def rrt_main(self, currState, left_borders, right_borders, obstacle_list, left_extended, right_extended):
        # theta = np.arctan(currState[2][1]/(currState[2][0]+1e-9))
        theta = currState[1][2]
        # print("Rotation vector: ", currState[1])
        # print("RRT theta: ", theta, 180*theta)
        if theta<0:
            theta = 2*np.pi + theta
        start = np.array([currState[0][0], currState[0][1], theta])
        x_min = min(left_extended[0], left_borders[0][0], right_borders[0][0], right_extended[0], currState[0][0])
        x_max = max(left_extended[0], left_borders[0][0], right_borders[0][0], right_extended[0], currState[0][0])
        y_min = min(left_extended[1], left_borders[0][1], right_borders[0][1], right_extended[1], currState[0][1])
        y_max = max(left_extended[1], left_borders[0][1], right_borders[0][1], right_extended[1], currState[0][1])

        search_space = np.array([[x_min, x_max],[y_min,y_max],[0, 2*np.pi]])
        goal = (np.array(left_extended)+np.array(right_extended))/2
        goal = np.hstack((goal,theta))
        # print(goal)
        obstacles = []
        for obs in obstacle_list:
            center = np.array([obs.location.x, obs.location.y])
            radius = -np.inf
            for i in range(0, len(obs.vertices_locations),2):
                point = np.array([obs.vertices_locations[i].vertex_location.x, obs.vertices_locations[i].vertex_location.y])

                radius = max(radius, np.linalg.norm(center-point))
                obstacles.append(Obstacle(center, radius))
        rrtsearch = RRT(start, goal, obstacles, search_space, 0.01, 500, 0.03, '')
        t = Thread(target = self.modify_variable, args=(rrtsearch,))
        t.start()


    def calc_normal(self, curr, prev):
        if self.lane_state == 4:
            perp_left_factor = 3
        elif self.lane_state == 3:
            perp_left_factor = 1
        else:
            perp_left_factor = 5
        perp_right_factor = 6-perp_left_factor
        lane_vector = [curr.x - prev.x, curr.y - prev.y]
        norm_const = np.sqrt(lane_vector[0]**2 + lane_vector[1]**2)
        lane_left_normal = [lane_vector[1]/norm_const, -lane_vector[0]/norm_const]
        lane_right_normal = [-lane_vector[1]/norm_const, lane_vector[0]/norm_const]

        left_point = [lane_left_normal[0]*2.5*perp_left_factor + curr.x, lane_left_normal[1]*2.5*perp_left_factor + curr.y]
        right_point = [lane_right_normal[0]*2.5*perp_right_factor + curr.x, lane_right_normal[1]*2.5*perp_right_factor + curr.y]
        return left_point, right_point

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
        p_last_prev = lane_marker.lane_markers_center.location[-2]
        waypoint_angle = (p_last.y - p_last_prev.y)/(p_last.x - p_last_prev.x)
        velo_angle = (currState[2][1]/(currState[2][0]+1e-9))

        left_borders, right_borders = self.get_borders(lane_marker)
        left_last, left_last_prev = left_borders[-1], left_borders[-2]
        right_last, right_last_prev = right_borders[-1], right_borders[-2]
        left_border_vector = [left_last[0] - left_last_prev[0], left_last[1] - left_last_prev[1]]
        right_border_vector = [right_last[0] - right_last_prev[0], right_last[1] - right_last_prev[1]]
        left_extended = [left_last[0] + left_border_vector[0]*20, left_last[1] + left_border_vector[1]*20]
        right_extended = [right_last[0] + right_border_vector[0]*20, right_last[1] + right_border_vector[1]*20]

        self.lane_marker = p_last
        self.lane_state = lane_marker.lane_state
        if self.rrt_counter %50==0:
            self.rrt_main(currState, left_borders, right_borders, obstacleList, left_extended, right_extended)

        self.rrt_counter += 1
                # print("Lane_markers distance", self.calc_dist(lane_marker.lane_markers_center.location[-1], lane_marker.lane_markers_left.location[-1]),self.calc_dist(lane_marker.lane_markers_center.location[-1], lane_marker.lane_markers_right.location[-1]))
        #Lane distance = 2.5 units
        if self.target_lane == self.lane_state:
            self.change_lane = False
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
        front_dist = 20
        if obstacleList:
            obs_count = 0
            for obs in obstacleList:
                print("Obstacle number: ", obs_count)
                print("Obstacle: ", obs)
        front_dist = np.inf
        if obstacleList and self.target_lane == self.lane_state:
            for obs in obstacleList:
                for vertex in obs.vertices_locations:
                    dy = vertex.vertex_location.y - curr_y
                    dx = vertex.vertex_location.x - curr_x
                    yaw = currState[1][2]
                    rx = np.cos(-yaw) * dx - np.sin(-yaw) * dy
                    ry = np.cos(-yaw) * dy + np.sin(-yaw) * dx

                    psi = np.arctan(ry / rx)
                    if rx > 0:
                        front_dist = min(front_dist, np.sqrt(dy * dy + dx * dx))
                        # print("detected object is at {} away and {} radians".format(front_dist, psi))
                        if psi < 0.2 and psi > -0.2:
                            obs_front = True
                        elif psi > 0.2:
                            obs_right = True
                        elif psi < -0.2:
                            obs_left = True

        if self.target_lane > self.lane_state:
            self.vehicle_state = "turn-right"
        elif self.target_lane < self.lane_state:
            self.vehicle_state = "turn-left"

        # prev_vehicle_state = self.vehicle_state
        if self.lane_state == LaneInfo.LEFT_LANE and not self.change_lane:
            if front_dist <= self.detect_dist and obs_front:
                if not obs_right:
                    self.vehicle_state = "turn-right"
                    if front_dist < self.detect_dist/2:
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
                    if front_dist < self.detect_dist/2:
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
                    if front_dist < self.detect_dist/2:
                        self.change_lane = True
                        self.target_lane -= 1
                elif not obs_right:
                    self.vehicle_state = "turn-right"
                    if front_dist < self.detect_dist/2:
                        self.change_lane = True
                        self.target_lane += 1
                else:
                    self.vehicle_state = "stop"
        if self.vehicle_state =="straight" :
            self.speed = 30
        elif front_dist > self.detect_dist/2:
            self.speed = 20
            self.vehicle_state = "straight"
        elif self.vehicle_state == "stop":
            self.speed = 5
        print("Vehicle State: ", self.vehicle_state, "Vehicle Lane: ", self.lane_state,"Target Lane: ", self.target_lane)

        # print(front_dist, self.lane_state, self.vehicle_state, obs_front, obs_left, obs_right)

        while not self.target_x or not self.target_y:
            continue

        distToTargetX = abs(self.target_x - curr_x)
        distToTargetY = abs(self.target_y - curr_y)

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
            self.counter += 1

        return [self.target_x, self.target_y, self.speed]


class VehicleController():
    def stop(self):
        print("PRINTING STOP")
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.acceleration = -20
        newAckermannCmd.speed = 0
        newAckermannCmd.steering_angle = 0
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

        k_s = 0.1
        k_ds = 1
        k_n = 0.1
        k_theta = 1
        k_speed = 1

        # compute errors
        dx = target_x - curr_x
        dy = target_y - curr_y
        xError = (target_x - curr_x) * np.cos(
            currentEuler[2]) + (target_y - curr_y) * np.sin(currentEuler[2])
        yError = -(target_x - curr_x) * np.sin(
            currentEuler[2]) + (target_y - curr_y) * np.cos(currentEuler[2])
        curr_v = np.sqrt(currentPose[2][0]**2 + currentPose[2][1]**2)
        vError = target_v - curr_v
        print("Target_v", target_v)
        delta = k_n * yError
        # delta = min(delta, delta*(20/curr_v))
        # Checking if the vehicle need to stop

        if target_v > 0:
            # v = xError * k_s + vError * k_ds
            v = 10 + 1/(k_speed*yError + 1/30)
            #Send computed control input to vehicle
            newAckermannCmd = AckermannDrive()
            newAckermannCmd.speed = v
            newAckermannCmd.acceleration = 0
            newAckermannCmd.steering_angle = delta
            return newAckermannCmd
        else:
            return self.stop()


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
