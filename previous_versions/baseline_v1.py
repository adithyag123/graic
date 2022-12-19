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
from numba import njit

"""
Planner Helpers
"""
@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
        print(i, dists[i])
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = 4.0
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

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

    def calc_dist(self,l1, l2):
        return np.sqrt((l1.x-l2.x)**2 +(l1.y-l2.y)**2), l1.x-l2.x, l1.y-l2.y

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
        path = rrtsearch.run()
        return path

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

    def get_ref_state(self, currState, obstacleList, lane_marker, waypoint, planner):
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

            path = self.rrt_main(currState, left_borders, right_borders, obstacleList, left_extended, right_extended)
            print(len(path))
            self.waypoints_rrt = []
            for wp in path[40:]:
                self.waypoints_rrt.append([wp[0], wp[1]])
            self.waypoints_rrt = np.array(self.waypoints_rrt)
        _, steer_angle_rrt = planner.plan(currState[0][0], currState[0][1], currState[1][2], 2.82461887897713965, 0.90338203837889, self.waypoints_rrt)
        steer_angle_rrt/=2
        self.rrt_counter +=1

        print("Steering angle Pure Pursuit + RRT: ", steer_angle_rrt)

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
                        front_dist = np.sqrt(dy * dy + dx * dx)
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

        return [self.target_x, self.target_y, self.speed, steer_angle_rrt]

class PurePursuitPlanner:
    """
    Example Planner
    """
    def __init__(self, wb):
        self.wheelbase = wb
        # self.conf = conf
        # self.load_waypoints(conf)
        self.max_reacquire = 20.

        self.drawn_waypoints = []

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        #points = self.waypoints

        points = np.vstack((self.waypoints[:, 0], self.waypoints[:, 1])).T

        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack((waypoints[:, 0], waypoints[:, 1])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        # print("nearest_point, nearest_dist",nearest_point, nearest_dist, t , i)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            # current_waypoint = np.empty((3, ))
            current_waypoint = np.empty((2, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            # current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return wpts[i, :]
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain, waypoints):
        """
        gives actuation given observation
        """

        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(waypoints, lookahead_distance, position, pose_theta)
        print("lookahead_point:", lookahead_point)
        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed


        return speed, steering_angle


class VehicleController():
    def stop(self):
        print("PRINTING STOP")
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.acceleration = -20
        newAckermannCmd.speed = 0
        newAckermannCmd.steering_angle = 0
        return newAckermannCmd

    def execute(self, currentPose, targetPose, planner, lane_marker):
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
        print("Length: ", len(lane_marker.lane_markers_center.location), len(lane_marker.lane_markers_center.location[:-10]))
        waypoints = []
        for wp in lane_marker.lane_markers_center.location[-10:]:
            waypoints.append([wp.x, wp.y])
        waypoints = np.array(waypoints)
        _, pp_steer = planner.plan(curr_x, curr_y, currentPose[1][2], 2.82461887897713965, 0.90338203837889, waypoints)

        print("Steering angle Pure pursuit: ", pp_steer)
        print("Steering angle baseline: ", delta)
        print("Steering ratio: ", delta/pp_steer)

        if target_v > 0:
            # v = xError * k_s + vError * k_ds
            v = 10 + 1/(k_speed*pp_steer + 1/30)
            #Send computed control input to vehicle
            newAckermannCmd = AckermannDrive()
            newAckermannCmd.speed = v
            newAckermannCmd.acceleration = 0
            newAckermannCmd.steering_angle = targetPose[3]
            return newAckermannCmd
        else:
            return self.stop()


class Controller(object):
    """docstring for Controller"""
    def __init__(self):
        super(Controller, self).__init__()
        self.decisionModule = VehicleDecision()
        self.controlModule = VehicleController()
        self.planner = PurePursuitPlanner(0.17145+0.15875)

    def stop(self):
        return self.controlModule.stop()

    def execute(self, currState, obstacleList, lane_marker, waypoint):
        # Get the target state from decision module
        refState = self.decisionModule.get_ref_state(currState, obstacleList,
                                                     lane_marker, waypoint, self.planner)
        if not refState:
            return None
        return self.controlModule.execute(currState, refState, self.planner, lane_marker)
