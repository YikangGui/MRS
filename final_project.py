import rps.robotarium as robotarium
from rps.utilities.barrier_certificates import *
from rps.utilities.controllers import *
from scipy.spatial.distance import cdist
import numpy as np
from dataclasses import dataclass
from enum import Enum
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.pyplot as plt
from copy import deepcopy


np.random.seed = 1024
# TODO add expiration
# TODO treasure picking up mechanism


class TaskType(Enum):
    IDLE = 0
    GO_TO_PICK = 1
    GO_TO_COLLECTION = 2
    GO_TO_RECHARGE = 3
    RECHARGING = 4
    WAIT_FOR_RECHARGING = 5
    DEAD = 6
    DEADLOCK = 7


@dataclass
class TaskReceipt(object):
    task_type: TaskType
    task_location: list
    robot_id: int
    prev_task_location: list
    timestamp: int = -1
    priority: int = 1  # 0 for expiration, 1 for default, 2 for itself
    task_no: int = 0
    distance_to_goal: int = 0

    def copy(self):
        return deepcopy(self)


def loc_init(locations, no):
    loc_list = np.array([locations[i] for i in range(no)])
    loc2id, id2loc = dict(), dict()
    for i in range(no):
        id2loc[i] = locations[i]
        loc2id[locations[i]] = i
    return loc_list, loc2id, id2loc


NULL = 1234

# Visualization
ROBOT_COLOR = {0: "#FF0000", 1: "#FFA500", 2: "#FFFF00", 3: "#008000", 4: "#0000FF"}
ROBOT_COLOR_NO = {0: "RED", 1: "ORANGE", 2: "YELLOW", 3: "GREEN", 4: "BLUE"}
DEFAULT_COLOR = "000000"


# Locations
TREASURE_NO = 2
RECHARGER_NO = 2
COLLECTION_NO = 1
TREASURE_LOC = [(0, 0), (0, 0.8), (0, -0.8), (0.8, 0), (-0.8, 0)]
RECHARGER_LOC = [(-1.3, 0.8), (1.3, 0.8)]
COLLECTION_LOC = [(-0.5, -0.7), (-1.3, -0.7), (-1.1, -0.7), (-0.9, -0.7), (-0.7, -0.7)]
TREASURE_LOC, TREASURE_LOC2ID, TREASURE_ID2LOC = loc_init(TREASURE_LOC, TREASURE_NO)
RECHARGER_LOC, RECHARGER_LOC2ID, RECHARGER_ID2LOC = loc_init(RECHARGER_LOC, RECHARGER_NO)
COLLECTION_LOC, COLLECTION_LOC2ID, COLLECTION_ID2LOC = loc_init(COLLECTION_LOC, COLLECTION_NO)

# Robotarium
ROBOTARIUM_SHOW_FIGURE = True
ROBOTARIUM_SIM_IN_REAL_TIME = False
BARRIER_CERTIFICATES = False
ROBOTARIUM_ITERATIONS = 10000
ROBOT_NO = 5
UPDATE_FIG = False

# other settings
DIST_THRESHOLD = 0.1
SENSE_RANGE = 0.5
EXPIRATION = 300

# Battery
ALPHA = 0.01
BETA = 0.1
GAMMA = 0.02
DELTA = 0.1
ENERGY_LEVEL = [50.0] * ROBOT_NO


class TaskController(object):
    """
    Each robot will have its own task controller. The task controller will maintain the information about itself and
    other robots. When conflicts happen, the validation is checked by comparing the self._task_schedule
    """
    def __init__(self, robot_no, robot_id, recharger_no=RECHARGER_NO, treasure_no=TREASURE_NO):
        self._robot_no = robot_no
        self._robot_id = robot_id
        self._recharger_no = recharger_no
        self._treasure_no = treasure_no
        self._recharger_status = []
        self._treasure_status = []
        self._current_timestamps = [0] * self._robot_no
        self._current_tasks = [TaskType.IDLE] * self._robot_no
        self._task_schedule = [TaskReceipt(task_type=TaskType.IDLE, task_location=[NULL, NULL], prev_task_location=[NULL, NULL], robot_id=i) for i in range(self._robot_no)]

    @property
    def robot_id(self):
        return self._robot_id

    def get_sim_status(self):
        """
        The robot status manager should call get_sim_status to update the robots' status when the task table has been modified.
        :return:
        """
        self._recharger_status = np.array([0] * self._recharger_no)
        self._treasure_status = np.array([0] * self._treasure_no)
        current_timestamps = []
        current_tasks = []
        for task in self._task_schedule:
            if task.task_type == TaskType.GO_TO_RECHARGE or task.task_type == TaskType.RECHARGING or task.task_type == TaskType.WAIT_FOR_RECHARGING:
                self._recharger_status[RECHARGER_LOC2ID[tuple(task.task_location)]] += 1
            # elif task.task_type == TaskType.GO_TO_PICK or task.task_type == TaskType.GO_TO_COLLECTION:
            elif task.task_type == TaskType.GO_TO_PICK:
                self._treasure_status[TREASURE_LOC2ID[tuple(task.prev_task_location)]] += 1
                # TODO: when assign GO_TO_PICK, make sure set the prev_task_location to the treasure location
            current_timestamps.append(task.timestamp)
            current_tasks.append(task.task_type)
        self._current_timestamps = current_timestamps
        self._current_tasks = current_tasks
        return self._current_timestamps, self._current_tasks, self._recharger_status, self._treasure_status

    def get_task_schedule(self):
        return self._task_schedule

    def get_task_receipt(self, robot_id):
        task_receipt = self._task_schedule[robot_id]
        assert task_receipt.robot_id == robot_id
        return task_receipt

    def set_task_receipt(self, task_type, task_location, robot_id, timestamp, prev_task_location=None):
        self._task_schedule[robot_id].task_type = task_type
        self._task_schedule[robot_id].task_location = task_location
        self._task_schedule[robot_id].timestamp = timestamp
        if prev_task_location is not None:
            self._task_schedule[robot_id].prev_task_location = prev_task_location

    def set_task_schedule(self, task_schedule):
        self._task_schedule = task_schedule


class RobotStatusManager(object):
    def __init__(self, energy_level, robot_id, robot_no=ROBOT_NO):
        self.total_dist = 0
        self.robot_no = robot_no
        self.robot_id = robot_id

        self.task_controller = TaskController(robot_no, robot_id)
        self.current_timestamps, self.current_tasks, self.recharger_status, self.treasure_status = self.task_controller.get_sim_status()
        self.energy_level = energy_level
        self.treasure_picked = False
        self.alive = True
        self.recharging = False

    def energy_loss(self, dist, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        energy_level = self.energy_level - (alpha + beta * dist + gamma * self.treasure_picked)
        self.energy_level = 0 if energy_level < 0 else energy_level

    def energy_gain(self, delta=DELTA):
        self.energy_level += delta

    def dist_update(self, dist):
        self.total_dist += dist

    def get_task(self, robot_id):
        return self.task_controller.get_task_receipt(robot_id)

    def set_task(self, task_type, robot_id, timestamp, task_location=None):
        if task_type == TaskType.GO_TO_PICK:
            assert task_location is not None
            self.task_controller.set_task_receipt(task_type, task_location, robot_id, timestamp, task_location)
        elif task_type == TaskType.GO_TO_COLLECTION:
            assert task_location is not None
            self.task_controller.set_task_receipt(task_type, task_location, robot_id, timestamp)
        else:
            if task_location is None:
                task_location = [NULL, NULL]
            self.task_controller.set_task_receipt(task_type, task_location, robot_id, timestamp, task_location)

    def set_task_schedule(self, task_schedule):
        self.task_controller.set_task_schedule(task_schedule)

    def get_sim_status(self):
        return self.task_controller.get_sim_status()

    def get_task_table(self):
        return self.task_controller.get_task_schedule()


class Simulator(object):
    def __init__(self):
        self.robot_number = ROBOT_NO

        # robotarium settings
        self.robotarium = robotarium.Robotarium(number_of_robots=self.robot_number, show_figure=ROBOTARIUM_SHOW_FIGURE,
                                                sim_in_real_time=ROBOTARIUM_SIM_IN_REAL_TIME)
        # , initial_conditions = np.array([[0.6, -0.7], [0, 0.1], [0, 0]])
        self.iterations = ROBOTARIUM_ITERATIONS
        self.uni_barrier_cert = create_unicycle_barrier_certificate_with_boundary()
        self.unicycle_position_controller = create_clf_unicycle_position_controller()

        # adjacency list
        self.adjacency_list = {i: [] for i in range(ROBOT_NO)}

        # task controller, each robot has its own task controller
        self.robots_status_manager = [RobotStatusManager(ENERGY_LEVEL[i], i) for i in range(self.robot_number)]
        self.battery = ENERGY_LEVEL.copy()

        # statistics
        self.recharging_time = 0
        self.goto_recharging_time = 0
        self.wait_recharging_time = 0
        self.deadlock_time = 0
        self.total_time = 1
        self.timestamp = 0
        self.idle_time = 0
        self.total_dist = 0

        """simulator status (can be accessed only by the simulator instead of robots)"""
        # number of treasure collected
        self.sim_no_treasure_completed = 0

        # recharger availability, the value represents the current active assigned job, i.e there are n robots using recharger
        self.sim_recharger_status = np.array([0] * RECHARGER_NO)

        # status of task availability, the value represents the current active assigned job, i.e there are n robots assigning this treasure
        self.sim_treasure_status = np.array([0] * TREASURE_NO)

        self.edges = set(self.get_all_edges())

        for i in range(ROBOT_NO):
            self.robotarium.chassis_patches[i].set_facecolor(ROBOT_COLOR[i])
        self.treasure_points = self.visualization_init(TREASURE_LOC, TREASURE_NO)
        self.recharger_points = self.visualization_init(RECHARGER_LOC, RECHARGER_NO)
        self.collection_points = self.visualization_init(COLLECTION_LOC, COLLECTION_NO)

        self.line = self.init_connectivity()

        self.fig = plt.figure(figsize=(10, 4.8))
        self.ax_battery = self.fig.add_subplot(121)
        self.ax_battery.set_ylim([0, 100])
        self.bar_battery = self.ax_battery.bar(range(self.robot_number), self.battery)
        self.bar_text = self.auto_label(self.bar_battery)

    def auto_label(self, bar_plot):
        texts = []
        for idx, rect in enumerate(bar_plot):
            rect.set_facecolor(ROBOT_COLOR[idx])
            height = rect.get_height()
            texts.append(self.ax_battery.text(rect.get_x() + rect.get_width() / 2., height + 3,
                                              self.battery[idx], ha='center', va='bottom', rotation=0))
        return texts

    def fig_update(self):
        for index, level in enumerate(self.battery):
            self.bar_battery[index].set_height(level)
            self.bar_text[index].set_y(level + 3)
            self.bar_text[index].set_text(str(round(level, 1)))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def get_all_edges(self):
        edges = []
        for tail in range(1, self.robot_number):
            for head in range(tail):
                edges.append((head, tail))
        return tuple(edges)

    def visualization_init(self, locations, no, radius=0.025, color="000000"):
        collection_points = []
        for i in range(no):
            collection_point = patches.Circle(locations[i], radius=radius, color=color)
            collection_points.append(collection_point)
            self.robotarium.axes.add_patch(collection_point)
        return collection_points

    def get_connected_robots(self, pose):
        dist = cdist(pose[:2, :].T, pose[:2, :].T)
        dist_updated = np.where(dist < SENSE_RANGE, dist, 0)
        edge_list = np.unique(np.sort(np.transpose(np.where(dist_updated > 0))), axis=0)
        self.adjacency_list = {i: [] for i in range(ROBOT_NO)}
        edges = set()
        for edge in edge_list:
            assert edge[0] < edge[1]
            edges.add((edge[0], edge[1]))
            self.adjacency_list[edge[0]].append(edge[1])
            self.adjacency_list[edge[1]].append(edge[0])
        return edges

    def is_same_task_schedule(self, timestamp, task, task_loc, prev_task_loc, robot_id):
        """

        :param robot_id:
        :param prev_task_loc:
        :param self:
        :param timestamp:
        :param task:
        :param task_loc:
        :return:
        """
        assert len(timestamp) == len(task) == 2
        if timestamp[0] == timestamp[1] and task[0] == task[1]:
            return [], True

        for i in range(self.robot_number):
            if (timestamp[0][i], task[0][i], task_loc[0][i], prev_task_loc[0][i]) == (timestamp[1][i], task[1][i], task_loc[1][i], prev_task_loc[1][i]):
                pass
            else:
                if task[0][i] == TaskType.GO_TO_PICK or task[0][i] == TaskType.GO_TO_COLLECTION:
                    indexes = np.where((np.array(prev_task_loc[1]) == prev_task_loc[0][i]).all(axis=1))[0]
                    indexes = np.delete(indexes, np.where(indexes == i))
                    if len(indexes) > 0:  # if there are two or more robots having the same treasure task, the one with newer timestamp must abandon the task and set to idle.
                        for index in indexes:
                            if task[0][i] == TaskType.GO_TO_PICK:
                                if task[1][index] == TaskType.GO_TO_PICK:
                                    if timestamp[0][i] < timestamp[1][index]:  # robot 0 got the treasure earlier, robot 1 has to give up.
                                        timestamp[1][index] = self.timestamp
                                        task[1][index] = TaskType.IDLE
                                        task_loc[1][index] = [NULL, NULL]
                                        prev_task_loc[1][index] = [NULL, NULL]

                                        timestamp[0][index] = self.timestamp
                                        task[0][index] = TaskType.IDLE
                                        task_loc[0][index] = [NULL, NULL]
                                        prev_task_loc[0][index] = [NULL, NULL]

                                        timestamp[1][i] = timestamp[0][i]
                                        task[1][i] = task[0][i]
                                        task_loc[1][i] = task_loc[0][i].copy()
                                        prev_task_loc[1][i] = prev_task_loc[0][i].copy()

                                    else:
                                        timestamp[0][i] = self.timestamp
                                        task[0][i] = TaskType.IDLE
                                        task_loc[0][i] = [NULL, NULL]
                                        prev_task_loc[0][i] = [NULL, NULL]

                                        timestamp[1][i] = self.timestamp
                                        task[1][i] = TaskType.IDLE
                                        task_loc[1][i] = [NULL, NULL]
                                        prev_task_loc[1][i] = [NULL, NULL]

                                        timestamp[0][index] = timestamp[1][index]
                                        task[0][index] = task[1][index]
                                        task_loc[0][index] = task_loc[1][index]
                                        prev_task_loc[0][index] = prev_task_loc[1][index]
                                elif task[1][index] == TaskType.GO_TO_COLLECTION:
                                    timestamp[0][i] = self.timestamp
                                    task[0][i] = TaskType.IDLE
                                    task_loc[0][i] = [NULL, NULL]
                                    prev_task_loc[0][i] = [NULL, NULL]

                                    timestamp[1][i] = self.timestamp
                                    task[1][i] = TaskType.IDLE
                                    task_loc[1][i] = [NULL, NULL]
                                    prev_task_loc[1][i] = [NULL, NULL]

                                    timestamp[0][index] = timestamp[1][index]
                                    task[0][index] = task[1][index]
                                    task_loc[0][index] = task_loc[1][index]
                                    prev_task_loc[0][index] = prev_task_loc[1][index]
                                else:
                                    raise ValueError
                            elif task[0][i] == TaskType.GO_TO_COLLECTION:
                                if task[1][index] == TaskType.GO_TO_PICK:
                                    timestamp[1][index] = self.timestamp
                                    task[1][index] = TaskType.IDLE
                                    task_loc[1][index] = [NULL, NULL]
                                    prev_task_loc[1][index] = [NULL, NULL]

                                    timestamp[0][index] = self.timestamp
                                    task[0][index] = TaskType.IDLE
                                    task_loc[0][index] = [NULL, NULL]
                                    prev_task_loc[0][index] = [NULL, NULL]

                                    timestamp[1][i] = timestamp[0][i]
                                    task[1][i] = task[0][i]
                                    task_loc[1][i] = task_loc[0][i].copy()
                                    prev_task_loc[1][i] = prev_task_loc[0][i].copy()
                                elif task[1][index] == TaskType.GO_TO_COLLECTION:
                                    pass
                                else:
                                    raise ValueError

                if timestamp[0][i] != timestamp[1][i]:
                    if timestamp[0][i] > timestamp[1][i]:
                        timestamp[1][i] = timestamp[0][i]
                        task[1][i] = task[0][i]
                        task_loc[1][i] = task_loc[0][i].copy()
                        prev_task_loc[1][i] = prev_task_loc[0][i].copy()
                    else:
                        timestamp[0][i] = timestamp[1][i]
                        task[0][i] = task[1][i]
                        task_loc[0][i] = task_loc[1][i].copy()
                        prev_task_loc[0][i] = prev_task_loc[1][i].copy()

        robot0_task = []
        robot1_task = []
        for i in range(self.robot_number):
            robot0_task.append(TaskReceipt(task[0][i], task_loc[0][i], robot_id[0][i], prev_task_loc[0][i], timestamp[0][i]))
            robot1_task.append(TaskReceipt(task[1][i], task_loc[1][i], robot_id[1][i], prev_task_loc[1][i], timestamp[1][i]))
        return [robot0_task, robot1_task], False

    def check_task_schedule_by_distance(self, task_schedule_a, task_schedule_b):
        assert len(task_schedule_a) == len(task_schedule_b) == ROBOT_NO
        result_task_schedule_a = task_schedule_a.copy()
        result_task_schedule_b = task_schedule_b.copy()
        target_treasure_id_a = []
        target_treasure_id_b = []

        for i in range(ROBOT_NO):
            receipt_a = task_schedule_a[i].copy()
            receipt_b = task_schedule_b[i].copy()

            if receipt_a.priority == 0:
                if receipt_b.priority >= 1:
                    result_task_schedule_a[i] = receipt_b
            elif receipt_b.priority == 0:
                if receipt_a.priority >= 1:
                    result_task_schedule_b[i] = receipt_a
            else:
                assert receipt_a.priority >= receipt_b.priority >= 1
                if receipt_a.timestamp > receipt_b.timestamp:
                    result_task_schedule_b[i] = receipt_a
                else:
                    result_task_schedule_a[i] = receipt_b

            if result_task_schedule_a[i].task_type == TaskType.GO_TO_PICK:
                target_treasure_id_a.append(TREASURE_LOC2ID[tuple(result_task_schedule_a[i].prev_task_location.copy())])
            else:
                target_treasure_id_a.append(-1)

            if result_task_schedule_b[i].task_type == TaskType.GO_TO_PICK:
                target_treasure_id_b.append(TREASURE_LOC2ID[tuple(result_task_schedule_b[i].prev_task_location.copy())])
            else:
                target_treasure_id_b.append(-1)

        for i in range(ROBOT_NO):
            current_treasure_id = target_treasure_id_a[i]
            receipt_a = result_task_schedule_a[i].copy()
            if current_treasure_id != -1:
                same_treasure_robot_id = np.where(np.array(target_treasure_id_b) == current_treasure_id)[0]
                for j in same_treasure_robot_id:
                    if j == i:
                        continue
                    receipt_b = result_task_schedule_b[j].copy()
                    if receipt_a.distance_to_goal > receipt_b.distance_to_goal:
                        result_task_schedule_a[i] = TaskReceipt(TaskType.IDLE, [NULL, NULL], receipt_a.robot_id, [NULL, NULL], self.timestamp)
                        result_task_schedule_b[i] = TaskReceipt(TaskType.IDLE, [NULL, NULL], receipt_a.robot_id, [NULL, NULL], self.timestamp)
                        target_treasure_id_a[i] = -1
                        target_treasure_id_b[i] = -1
                    else:
                        result_task_schedule_a[j] = TaskReceipt(TaskType.IDLE, [NULL, NULL], receipt_b.robot_id, [NULL, NULL], self.timestamp)
                        result_task_schedule_b[j] = TaskReceipt(TaskType.IDLE, [NULL, NULL], receipt_b.robot_id, [NULL, NULL], self.timestamp)
                        target_treasure_id_a[j] = -1
                        target_treasure_id_b[j] = -1
        return result_task_schedule_a, result_task_schedule_b

    def sim_status_update(self):
        self.sim_treasure_status = [0] * TREASURE_NO
        self.sim_recharger_status = [0] * RECHARGER_NO
        for robot_id in range(self.robot_number):
            robot = self.robots_status_manager[robot_id]
            robot_task = robot.get_task(robot_id)
            # if robot_task.task_type == TaskType.GO_TO_COLLECTION:
            if robot_task.task_type == TaskType.GO_TO_PICK:
                treasure_loc = robot_task.prev_task_location
                treasure_id = TREASURE_LOC2ID[tuple(treasure_loc)]
                self.sim_treasure_status[treasure_id] += 1
            elif robot_task.task_type == TaskType.RECHARGING:
                recharger_loc = robot_task.task_location
                recharger_id = RECHARGER_LOC2ID[tuple(recharger_loc)]
                self.sim_recharger_status[recharger_id] += 1

    def init_connectivity(self):
        edge_dict = {}
        for tail in range(1, self.robot_number):
            for head in range(tail):
                path_patch = path.Path(np.array([[0, 0], [0, 0.0]]))
                line = patches.PathPatch(path_patch)
                edge_dict[(head, tail)] = line
                self.robotarium.axes.add_patch(line)
        return edge_dict

    def update_connectivity(self, connected_edges, x):
        all_edges = self.edges.copy()
        for edge in connected_edges:
            all_edges.remove(edge)
            path_patch = patches.Path(np.array([x[:2, edge[0]], x[:2, edge[1]]]))
            self.line[edge].set_path(path_patch)
            self.line[edge].set_linewidth(1)
        for edge in all_edges:
            path_patch = patches.Path(np.array([x[:2, edge[0]], x[:2, edge[1]]]))
            self.line[edge].set_path(path_patch)
            self.line[edge].set_linewidth(0)

    def check_expiration(self, task_schedule, robot_id):
        new_task_schedule = task_schedule.copy()
        for i, task in enumerate(task_schedule):
            if abs(task.timestamp - self.timestamp) > EXPIRATION and task.priority > 0 and robot_id != i:
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                new_task_schedule[i] = TaskReceipt(TaskType.IDLE, [NULL, NULL], task.robot_id, [NULL, NULL], timestamp=-1, priority=0)
        return new_task_schedule

    def strategy(self):
        prev_x = self.robotarium.get_poses().copy()
        print(prev_x)
        self.robotarium.step()
        for iteration in range(self.iterations):
            x = self.robotarium.get_poses()
            dist_travel = np.linalg.norm(x[:2, :] - prev_x[:2, :], axis=0)
            self.total_dist += np.sum(dist_travel)
            connected_edges = self.get_connected_robots(x.copy())
            self.update_connectivity(connected_edges, x)
            if iteration == 401:
                print()
            target_loc = x[:2, :].T.copy()
            for robot_id in range(self.robot_number):
                robot = self.robots_status_manager[robot_id]
                robot_task = robot.get_task(robot_id)
                robot_task_table = robot.get_task_table().copy()
                # TODO: add lowest priority to expiration receipt
                robot_task_schedule = self.check_expiration(robot_task_table, robot_id)
                robot.set_task_schedule(robot_task_schedule)
            for robot_id in range(self.robot_number):
                dist = dist_travel[robot_id]
                robot = self.robots_status_manager[robot_id]
                robot_task = robot.get_task(robot_id)
                robot_task_table = robot.get_task_table().copy()
                dist_to_goal = np.linalg.norm(x[:2, robot_id] - robot_task.task_location)
                # print(robot_id, dist_to_goal)

                # TODO: update task receipt given adjacency list (fixed?)
                for j in self.adjacency_list[robot_id]:

                    # if robot is dead, its target location will be its current location
                    if robot_task.task_type == TaskType.DEAD:
                        # target_loc[robot_id] = x[:2, robot_id].copy()
                        robot_task.task_location = x[:2, robot_id].copy()
                        continue

                    # timestamp1, timestamp2 = [], []
                    # task1, task2 = [], []
                    # task_loc1, task_loc2 = [], []
                    # prev_task_loc1, prev_task_loc2 = [], []
                    # robot_id1, robot_id2 = [], []

                    robot_task = robot.get_task(robot_id)
                    robot_task_table = robot.get_task_table().copy()
                    robot_adjacent = self.robots_status_manager[j]
                    robot_adjacent_task_table = robot_adjacent.get_task_table()
                    assert j != robot_id
                    assert robot_adjacent_task_table is not robot_task_table

                    updated_robot_task_schedule_a, updated_robot_task_schedule_b = self.check_task_schedule_by_distance(robot_task_table, robot_adjacent_task_table)
                    robot.set_task_schedule(updated_robot_task_schedule_a.copy())
                    robot_adjacent.set_task_schedule(updated_robot_task_schedule_b.copy())
                self.sim_status_update()

                    # for k in range(self.robot_number):
                    #     timestamp1.append(robot_task_table[k].timestamp)
                    #     timestamp2.append(robot_adjacent_task_table[k].timestamp)
                    #     task1.append(robot_task_table[k].task_type)
                    #     task2.append(robot_adjacent_task_table[k].task_type)
                    #     task_loc1.append(robot_task_table[k].task_location)
                    #     task_loc2.append(robot_adjacent_task_table[k].task_location)
                    #     prev_task_loc1.append(robot_task_table[k].prev_task_location)
                    #     prev_task_loc2.append(robot_adjacent_task_table[k].prev_task_location)
                    #     robot_id1.append(robot_task_table[k].robot_id)
                    #     robot_id2.append(robot_adjacent_task_table[k].robot_id)
                    # updated_robot_task_schedule, flag = self.is_same_task_schedule([timestamp1, timestamp2], [task1, task2], [task_loc1, task_loc2], [prev_task_loc1, prev_task_loc2], [robot_id1, robot_id2])
                    # if not flag:
                    #     robot.set_task_schedule(updated_robot_task_schedule[0].copy())
                    #     robot_adjacent.set_task_schedule(updated_robot_task_schedule[1].copy())
                    #     self.sim_status_update()

                robot_task = robot.get_task(robot_id)
                robot_task_table = robot.get_task_table()

                robot.energy_loss(dist)
                robot.dist_update(dist)

                if robot.energy_level <= 0:
                    # TODO: give up current task, update simulator status
                    assert robot_task.task_type != TaskType.RECHARGING
                    if robot_task.task_type == TaskType.GO_TO_PICK or robot_task.task_type == TaskType.GO_TO_COLLECTION:
                        treasure_loc = robot_task.prev_task_location
                        treasure_id = TREASURE_LOC2ID[tuple(treasure_loc)]
                        assert self.sim_treasure_status[treasure_id] > 0
                        # self.sim_treasure_status[treasure_id] -= 1
                    elif robot_task.task_type == TaskType.GO_TO_RECHARGE or robot_task.task_type == TaskType.WAIT_FOR_RECHARGING:
                        recharger_loc = robot_task.task_location
                        recharger_id = RECHARGER_LOC2ID[tuple(recharger_loc)]
                        assert self.sim_recharger_status[recharger_id] > 0
                        # self.sim_recharger_status[recharger_id] -= 1
                    # update robot status
                    robot.set_task(TaskType.DEAD, robot_id, self.timestamp)
                    robot.alive = False
                    self.sim_status_update()

                if robot_task.task_type == TaskType.GO_TO_COLLECTION:
                    if dist_to_goal < DIST_THRESHOLD:
                        print(f'robot {robot_id} finished treasure {robot_task.prev_task_location}, dropped at {robot_task.task_location}')
                        treasure_loc = robot_task.prev_task_location
                        treasure_id = TREASURE_LOC2ID[tuple(treasure_loc)]
                        # self.sim_treasure_status[treasure_id] -= 1
                        assert self.sim_treasure_status[treasure_id] >= 0
                        robot.set_task(TaskType.IDLE, robot_id, self.timestamp)
                        self.sim_status_update()
                        robot.treasure_picked = False
                        self.sim_no_treasure_completed += 1
                        # target_loc[robot_id] = x[:2, robot_id].copy()
                    else:
                        target_loc[robot_id] = robot_task.task_location

                # if robot goes to pick treasure
                elif robot_task.task_type == TaskType.GO_TO_PICK:
                    treasure_loc = robot_task.prev_task_location
                    treasure_id = TREASURE_LOC2ID[tuple(treasure_loc)]
                    # if dist_to_goal < DIST_THRESHOLD and self.sim_treasure_status[treasure_id] == 0:
                    if dist_to_goal < DIST_THRESHOLD:
                        # TODO collection selection
                        print(f'robot {robot_id} picked treasure at {robot_task.task_location}, going to collection ###')
                        robot.set_task(TaskType.GO_TO_COLLECTION, robot_id, self.timestamp, [-0.5, -0.7])
                        # self.sim_treasure_status[treasure_id] += 1
                        self.sim_status_update()
                        target_loc[robot_id] = [-0.5, -0.7]
                        robot.treasure_picked = True
                    target_loc[robot_id] = robot_task.task_location.copy()

                elif robot_task.task_type == TaskType.WAIT_FOR_RECHARGING:
                    recharger_loc = robot_task.task_location
                    recharger_id = RECHARGER_LOC2ID[tuple(recharger_loc)]
                    if self.sim_recharger_status[recharger_id] == 0:
                        robot.set_task(TaskType.RECHARGING, robot_id, self.timestamp, robot_task.task_location.copy())
                        # self.sim_recharger_status[recharger_id] += 1
                        self.sim_status_update()
                    else:
                        self.wait_recharging_time += 1
                    # target_loc[robot_id] = x[:2, robot_id].copy()

                elif robot_task.task_type == TaskType.GO_TO_RECHARGE:
                    recharger_loc = robot_task.task_location
                    recharger_id = RECHARGER_LOC2ID[tuple(recharger_loc)]
                    if dist_to_goal < DIST_THRESHOLD:
                        if self.sim_recharger_status[recharger_id] == 0:
                            robot.set_task(TaskType.RECHARGING, robot_id, self.timestamp, robot_task.task_location.copy())
                            # self.sim_recharger_status[recharger_id] += 1
                            self.sim_status_update()
                        else:
                            robot.set_task(TaskType.WAIT_FOR_RECHARGING, robot_id, self.timestamp, robot_task.task_location.copy())
                        # target_loc[robot_id] = x[:2, robot_id].copy()
                    else:
                        target_loc[robot_id] = robot_task.task_location
                        self.goto_recharging_time += 1

                elif robot_task.task_type == TaskType.RECHARGING:
                    robot.energy_gain()
                    if robot.energy_level > 80:
                        recharger_loc = robot_task.task_location
                        recharger_id = RECHARGER_LOC2ID[tuple(recharger_loc)]
                        # self.sim_recharger_status[recharger_id] -= 1
                        self.sim_status_update()
                        robot.set_task(TaskType.IDLE, robot_id, self.timestamp)
                    else:
                        self.recharging_time += 1

                elif robot_task.task_type == TaskType.IDLE:
                    _, _, recharger_status, treasure_status = robot.get_sim_status()
                    if robot.energy_level < 40:  # assign recharge
                        recharger_id = int(np.argmin(recharger_status))
                        robot.set_task(TaskType.GO_TO_RECHARGE, robot_id, self.timestamp, list(RECHARGER_ID2LOC[recharger_id]))
                        target_loc[robot_id] = RECHARGER_ID2LOC[recharger_id]
                    else:  # assign treasure
                        treasure_id = int(np.argmin(treasure_status))
                        if treasure_status[treasure_id] < 1:
                            robot.set_task(TaskType.GO_TO_PICK, robot_id, self.timestamp, list(TREASURE_ID2LOC[treasure_id]))
                            target_loc[robot_id] = TREASURE_ID2LOC[treasure_id]
                        else:
                            self.idle_time += 1
                # print()
                self.battery[robot_id] = self.robots_status_manager[robot_id].energy_level
                robot_task.distance_to_goal = np.linalg.norm(x[:2, robot_id] - target_loc[robot_id].T)
                robot_task.timestamp = self.timestamp

            if UPDATE_FIG:
                self.fig_update()

            prev_x = x.copy()
            # print(self.robots_status_manager[0].get_task_table())
            # print(self.robots_status_manager[1].get_task_table())
            # print()
            # print(self.timestamp)
            self.timestamp += 1
            dxu = self.unicycle_position_controller(x, target_loc.T)
            if BARRIER_CERTIFICATES:
                dxu = self.uni_barrier_cert(dxu, x)
            self.robotarium.set_velocities(np.arange(ROBOT_NO), dxu)
            self.robotarium.step()

        # Call at end of script to print debug information and for your script to run on the Robotarium server properly
        self.robotarium.call_at_scripts_end()


if __name__ == '__main__':
    simulator = Simulator()
    simulator.strategy()

    with open(f'./data/dist.txt', 'a') as f:
        f.write(str(np.sum(simulator.total_dist)) + '\n')

    with open(f'./data/goto.txt', 'a') as f:
        f.write(str(simulator.goto_recharging_time) + '\n')

    with open(f'./data/recharging.txt', 'a') as f:
        f.write(str(simulator.recharging_time) + '\n')

    with open(f'./data/wait.txt', 'a') as f:
        f.write(str(simulator.wait_recharging_time) + '\n')

    with open(f'./data/treasure.txt', 'a') as f:
        f.write(str(simulator.sim_no_treasure_completed) + '\n')
