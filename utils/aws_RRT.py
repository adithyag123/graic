import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import random
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.path import Path
import time

from scipy import stats, integrate


class Artists:
    'artists for animating tree search'

    def __init__(self, ax):
        self.artist_list = []
        self.ax = ax
        self.rand_pt_marker, = ax.plot([], [], '--o', color='lime', lw=1, zorder=1)
        self.artist_list.append(self.rand_pt_marker)

        self.goal_pt_marker, = ax.plot([], [], '--o', color='red', lw=1, zorder=2)
        self.artist_list.append(self.goal_pt_marker)

        self.nearest_node_marker, = ax.plot([], [], '--o', color='red', lw=1, zorder=2)
        self.artist_list.append(self.nearest_node_marker)

        self.root_pt_marker, = ax.plot([], [], '--o', color='blue', lw=1, zorder=2)
        self.artist_list.append(self.root_pt_marker)

        self.obs_solid_lines = LineCollection([], lw=2, animated=True, color='k', zorder=1)
        ax.add_collection(self.obs_solid_lines)
        self.artist_list.append(self.obs_solid_lines)

        self.resteer_solid_lines = LineCollection([], lw=2, animated=True, color='blue', zorder=3)
        ax.add_collection(self.resteer_solid_lines)
        self.artist_list.append(self.resteer_solid_lines)

        self.path_to_goal_lines = LineCollection([], lw=2, animated=True, color='blue', zorder=1)
        ax.add_collection(self.path_to_goal_lines)
        self.artist_list.append(self.path_to_goal_lines)

    def update_rand_pt_marker(self, rand_pt):
        'update random point marker'

        xs = [rand_pt[0]]
        ys = [rand_pt[1]]

        self.rand_pt_marker.set_data(xs, ys)

    def update_goal_pt_marker(self, goal_pt):
        'update goal point marker'

        xs = [goal_pt[0]]
        ys = [goal_pt[1]]

        self.goal_pt_marker.set_data(xs, ys)

    def update_nearest_node_marker(self, nearest_node):
        'update goal point marker'

        xs = [nearest_node[0]]
        ys = [nearest_node[1]]

        self.nearest_node_marker.set_data(xs, ys)

    def update_root_pt_marker(self, root_pt):
        'update root point marker'

        xs = [root_pt[0]]
        ys = [root_pt[1]]

        self.root_pt_marker.set_data(xs, ys)

    def update_obs_solid_lines(self, old_pt, new_pt):
        codes = [Path.MOVETO, Path.LINETO]
        verts = [(new_pt.pos[0], new_pt.pos[1]), (old_pt.pos[0], old_pt.pos[1])]
        obs_solid_paths = self.obs_solid_lines.get_paths()
        obs_solid_paths.append(Path(verts, codes))

    def update_resteer_solid_lines(self, old_pt, new_pt):
        codes = [Path.MOVETO, Path.LINETO]
        verts = [(new_pt[0], new_pt[1]), (old_pt[0], old_pt[1])]
        resteer_solid_paths = self.resteer_solid_lines.get_paths()
        resteer_solid_paths.append(Path(verts, codes))

    def clear_resteer_solid_lines(self):
        self.resteer_solid_lines.get_paths().clear()

    def update_circles(self, obstacles):
        print('update circles')
        for obs in obstacles:
            circle1 = plt.Circle((obs.center[0], obs.center[1]),obs.radius, color='r')
            self.ax.add_patch(circle1)
            self.artist_list.append(circle1)

    def update_path_to_goal(self, path):
        'update artist list'
        for i in range(len(path) - 1):
            codes = [Path.MOVETO, Path.LINETO]
            verts = [(path[i][0], path[i][1]), (path[i + 1][0], path[i + 1][1])]
            path_to_goal_paths = self.path_to_goal_lines.get_paths()
            path_to_goal_paths.append(Path(verts, codes))


class TreeNode:
    def __init__(self, pos, parent, cmd_from_parent=None):
        self.pos = pos
        self.parent = parent
        self.children = []
        self.cost = 0
        self.path_cost = 0
        self.total_cost = 0
        self.cmd_from_parent = cmd_from_parent

class Obstacle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def is_collision(self, pos):
        dist = np.linalg.norm(self.center - pos)
        if dist < self.radius:
            return True
        else:
            return False



class RRT:
    'RRT algorithm'

    def __init__(self, start, goal, obstacle_list, rand_area, step_size, max_iter, tolerance, filename, test_points=None):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacle_list = obstacle_list
        self.rand_area = rand_area
        self.step_size = step_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.d = len(start)

        self.path = []
        self.path_found = False
        self.path_length = 0
        self.time_taken = 0
        self.root = TreeNode(self.start, None)
        self.normal_distribution_array = []
        self.current_node = None

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(rand_area[0][0], rand_area[0][1]), ylim=(rand_area[1][0], rand_area[1][1]))
        self.artists = Artists(self.ax)
        self.node_list = [self.root]
        self.anim = None
        self.artists.update_root_pt_marker(self.start)
        if goal is not None:
            self.artists.update_goal_pt_marker(self.goal)
        self.artists.update_circles(self.obstacle_list)
        self.i = 0
        self.filename = filename
        self.test_points = test_points
        self.cnt = 0
        self.test_points_found_count = []
        self.mc_integrate_results = []

        self.actual_iterations_count = 0
        self.actual_iterations_lst = []
        self.action_list = self.get_action_list()
        print(self.filename)

    def get_random_point(self):
        'generate random point in search space'

        random_pt = np.empty(self.d)
        for dim in range(self.d):
            x_min, x_max = self.rand_area[dim]
            x = random.uniform(x_min, x_max)
            random_pt[dim] = x

        # random_pt = np.array([100,110,0])
        return random_pt

    def get_nearest_node(self, node):
        'get nearest node in tree'

        min_dist = float('inf')
        nearest_node = None

        for n in self.node_list:

            dist = np.linalg.norm(node - n.pos)

            if dist < min_dist:
                min_dist = dist
                nearest_node = n

        return nearest_node

    def get_nearest_node_goal(self):
        'get nearest node to the goal in tree'

        min_dist = float('inf')
        nearest_node = None

        for n in self.node_list:

            dist = np.linalg.norm(self.goal[:2] - n.pos[:2])

            if dist < min_dist:
                min_dist = dist
                nearest_node = n

        return nearest_node

    def steer(self, from_node, to_node):
        'steer from from_node to to_node'

        self.actual_iterations_count += 1
        dist = np.linalg.norm(to_node - from_node.pos)
        unit_vector = (to_node - from_node.pos) / dist
        cmd_from_parent = (unit_vector, dist)
        if dist < self.step_size:
            new_node = TreeNode(to_node, from_node, cmd_from_parent)
        else:
            action = self.find_optimal_action(from_node, to_node)
            if action == None:
                return
            new_node = from_node
            prev_node = new_node
            for _ in range(10):
                state_dot = self.model(new_node.pos, np.array(action))
                new_node = TreeNode(new_node.pos + self.step_size * state_dot, prev_node, action)
                self.artists.update_obs_solid_lines(prev_node, new_node)
                prev_node = new_node
                self.node_list.append(new_node)
                if np.linalg.norm(new_node.pos[0:2] - self.goal[0:2]) <2.3:
                    self.path_found = True
                    return new_node
        return new_node


    def find_optimal_action(self, from_node, to_node):
        'find optimal action'

        min_cost = float('inf')
        min_action = None
        # print("to_node: ", to_node, "from_node: ", from_node.pos, len(action_list))
        for action in self.action_list:
            current_node = from_node.pos
            points = [current_node]
            for _ in range(10):
                current_node = current_node + self.step_size * self.model(current_node, np.array(action))
                if not self.collision_check(current_node):
                    print("collision between :", current_node)
                    points = []
                    break
                points.append(current_node)
            cost = np.linalg.norm(current_node[0:2] - to_node[0:2])
            # print("cost: ", cost, "action: ", action, "new_node: ", current_node)
            if points and cost < min_cost:
                min_cost = cost
                min_action = action
        # print("min_cost: ", min_cost, "min_action: ", min_action, "new_node: ", current_node)
        return min_action

    def get_action_list(self):
        action_list = []
        max_speed = 10
        max_angle = np.pi / 4
        angle = - max_angle
        speed = 1
        num_of_actions = 10
        while speed <= max_speed:
            while angle <= max_angle:
                action_list.append([speed, angle])
                angle += 2 * max_angle / num_of_actions
            speed +=  max_speed / num_of_actions
            angle = - max_angle
        return action_list

    def collision_check(self, node):
        'check if node is in collision'

        for obs in self.obstacle_list:
            if np.linalg.norm(node[0:2] - obs.center[0:2]) < obs.radius:
                return False
        return True

    def iterate(self):
        'iterate RRT algorithm'
        random_pt = self.get_random_point()
        nearest_node = self.get_nearest_node(random_pt)
        # if nearest_node is not self.current_node:
        #     self.steerFromRoot(nearest_node)
        new_node = self.steer(nearest_node, random_pt)
        self.current_node = new_node
        return random_pt, nearest_node, new_node

    def steerFromRoot(self, node):
        '''Find the commands from root recursively that created this node and use those commands to steer to this node'''
        cmd_lst = []
        while node is not None and node is not self.root:
            cmd_lst.append(node.cmd_from_parent)
            node = node.parent

        current_pos = self.root.pos
        for i in range(len(cmd_lst) - 1, -1, -1):
            self.actual_iterations_count += 1
            old_pos = current_pos
            (unit_vector, dist) = cmd_lst[i]
            if dist < self.step_size:
                current_pos = current_pos + dist * unit_vector
            else:
                current_pos = current_pos + self.step_size * unit_vector
            self.artists.update_resteer_solid_lines(old_pos, current_pos)

    def update_path(self, new_node):
        self.path_found = True
        self.path = self.get_path(new_node)
        self.path_length = len(self.path)
        self.time_taken = time.time() - self.start_time
        self.total_nodes = len(self.node_list)
        print('path found ', self.path_length)
        print('Time taken: ', self.time_taken)
        print('Total Nodes explored', self.total_nodes)
        self.artists.update_path_to_goal(self.path)

    def return_results(self):
        return self.path_found, self.path_length, self.time_taken, self.total_nodes

    def get_dist(self, node, goal):
        return np.linalg.norm(node.pos - goal)

    def get_path(self, node):
        'get path from root to node'
        path = [node.pos]
        while node.parent is not None:
            node = node.parent
            path.append(node.pos)
        return path[::-1]

    def animate(self, i):
        'animation function'
        if i == self.max_iter - 1 or self.path_found:
            self.artists.clear_resteer_solid_lines()
            self.artists.update_rand_pt_marker([0, 0])
            self.path_found = True
            nearest_goal_node = self.get_nearest_node_goal()
            path = np.array(self.get_path(nearest_goal_node))
            self.artists.update_path_to_goal(path)
            print('iteration: ', i, 'test_pts found: ', self.cnt, 'actual_iterations: ', self.actual_iterations_count)

        if not self.path_found:
            # self.artists.clear_resteer_solid_lines()
            random_pt, nearest_node, new_node = self.iterate()
            self.artists.update_rand_pt_marker(random_pt)
            self.artists.update_nearest_node_marker(nearest_node.pos)
            # print("new_node: ", new_node.pos)
            # if new_node and nearest_node:
            #     self.artists.update_obs_solid_lines(nearest_node, new_node)
        return self.artists.artist_list

    def run(self):
        'run RRT algorithm'
        # plot root point (not animated)
        # self.ax.plot([self.root.pos[0]], [self.root.pos[1]], 'ko', ms=5)
        for i in range(self.max_iter):
            # print('iteration: ', i, 'test_pts found: ', self.cnt, 'actual_iterations: ', self.actual_iterations_count)
            random_pt, nearest_node, new_node = self.iterate()
            if self.path_found:
                break
        #     print("Path_found: ",i)



        # nearest_goal_node = self.get_nearest_node(self.goal)
        # path = np.array(self.get_path(nearest_goal_node))



        # self.anim = animation.FuncAnimation(self.fig, self.animate, frames=self.max_iter, interval=1, blit=True)
        #
        # plt.show()
        nearest_goal_node = self.get_nearest_node_goal()
        path = np.array(self.get_path(nearest_goal_node))
        return path
        # self.anim.save(self.filename, writer=animation.FFMpegWriter(fps=30))

    def plot_results_area(self, ax, color='r-'):
        'plot results'
        ax.plot(np.arange(0, len(self.actual_iterations_lst)), self.mc_integrate_results, color)

    def plot_results(self, ax, color='r-'):
        'plot results'
        ax.plot(self.actual_iterations_lst, np.arange(0, len(self.actual_iterations_lst)), color)


    def max_value_normal_distribution_function(self, args):
        val = 0.0
        for distribution in self.normal_distribution_array:
            val = max(val, distribution.pdf(args))
        return val

    def create_normal_distribution(self, mean, std):
        'create normal distribution'
        return stats.multivariate_normal(mean, std)

    def mc_integrate(self, func, a, b, dim, n=1000):
        # Monte Carlo integration of given function over domain from a to b (for each parameter)
        # dim: dimensions of function

        x_list = np.random.uniform(a, b, (n, dim))
        sm = 0
        for i in x_list:
            sm += func(i)

        y_mean = sm / n
        domain = np.power(b - a, dim)

        integ = domain * y_mean

        return integ

    def model(self, state, action):
        'model for motion'
        n, u = len(state), len(action)
        state_dot = np.zeros(n)
        speed, car_length = 1, 3
        state_dot[0] = action[0] * np.cos(state[2]) * speed
        state_dot[1] = action[0] * np.sin(state[2]) * speed
        state_dot[2] = np.tan(action[1]) * speed / car_length

        return state_dot


class RRT_Opt(RRT):

    def __init__(self, start, goal, obstacle_list, rand_area, step_size, max_iter, tolerance, filename, test_points):
        super().__init__(start, goal, obstacle_list, rand_area, step_size, max_iter, tolerance, filename, test_points)
        self.current_rand_pt = None
        self.old_node = self.root

    def iterate(self):
        if self.current_rand_pt is None or self.current_node is None or self.get_dist(self.current_node,
                                                                                      self.current_rand_pt) < self.tolerance / 3:
            self.current_rand_pt = self.get_random_point()
            nearest_node = self.get_nearest_node(self.current_rand_pt)
            if nearest_node != self.current_node:
                self.steerFromRoot(nearest_node)
            self.current_node = nearest_node
        self.old_node = self.current_node
        self.current_node = self.steer(self.current_node, self.current_rand_pt)

        return self.current_rand_pt, self.current_node, self.old_node


def generate_random_point(search_space, d):
    'generate random point in search space'

    random_pt = np.empty(d)
    for dim in range(d):
        x_min, x_max = search_space[dim]
        x = random.uniform(x_min, x_max)
        random_pt[dim] = x

    return random_pt


def collision_check(node, obstacle_list):
    'check if node is in collision'

    for obs in obstacle_list:
        if np.linalg.norm(node[0:2] - obs.center) < obs.radius:
            return False
    return True


if __name__ == '__main__':
    search_space = np.array([[0, 110], [0, 150], [0, 6.28]])
    obstacles = [Obstacle(np.array([100, 120]), 3)]
    # obstacles = [(300, 300,0), (300, 200,0),(300, 600,0), (300, 500,0),(300, 400,0), (300, 100,0), (100, 300,0), (100, 200,0),(100, 600,0), (100, 500,0),(100, 400,0), (100, 100,0)]
    # obstacles = []
    path_found_rrt = []
    path_lengths_rrt = []
    time_taken_rrt = []
    total_nodes_rrt = []

    path_found_rrt_opt = []
    path_lengths_rrt_opt = []
    time_taken_rrt_opt = []
    total_nodes_rrt_opt = []
    test_points = []
    d = len(search_space)


    for test in range(1):
        start = generate_random_point(search_space, d)
        goal = generate_random_point(search_space, d)
        start = np.array([100,90,np.pi])
        # goal = np.array([100,105, np.pi/2])
        print("start: ", start, "goal: ", goal)
        while not collision_check(start, obstacles):
            print('start in collision')
            start = generate_random_point(search_space, d)
        while not collision_check(goal, obstacles):
            print('goal in collision')
            goal = generate_random_point(search_space, d)
        print('Distance between start and goal ', np.linalg.norm(start - goal))
        rrtsearch = RRT(start, goal, obstacles, search_space, 0.01, 1000, 0.03, 'cache/rrt_' + str(test) + '.png',
                        test_points)
        rrtsearch.run()

        # rrtsearch2 = RRT_Opt(start, None, [], search_space, 0.01, 3000, 0.03, 'cache/rrt_opt_' + str(test) + '.png',
        #                      test_points)
        # rrtsearch2.run()

        # fig,ax = plt.subplots()
        # ax.set_xlim(0,10000)
        # ax.set_ylim(0,3000)

        # ax.set_ylabel('Nodes found')
        # ax.set_xlabel('Simulation Time')
        # rrtsearch.plot_results(ax, 'r-')
        # rrtsearch2.plot_results(ax, 'b-')
        # ax.legend(['RRT', 'RRT_No_Save'])

        # ax1 = plt.subplot(212)
        # ax1.set_xlabel('Nodes found')
        # ax1.set_ylabel('Area under the curve')
        # rrtsearch.plot_results_area(ax1, 'r-')
        # rrtsearch2.plot_results_area(ax1, 'b-')
        # ax1.legend(['RRT', 'RRT_No_Save'])
