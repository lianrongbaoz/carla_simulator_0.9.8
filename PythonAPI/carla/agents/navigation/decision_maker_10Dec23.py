from collections import deque
import random
from enum import Enum

import carla
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import draw_waypoints

"""
State Machines
"""
# Lateral
# FOLLOW_LANE = 1
# CHANGE_LANE_LEFT = 2
# CHANGE_LANE_RIGHT = 3
# Longitudinal
# ACCELERATE = 21
# DECELERATE = 22
# MAINTAIN_SPEED = 23

class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    FOLLOW_LANE = 1
    CHANGE_LANE_LEFT = 2
    CHANGE_LANE_RIGHT = 3

class DecisionMaker(object):
    def __init__(self, vehicle, opt_dict=None):
        self._state = RoadOption.FOLLOW_LANE
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()

        self._dt = None
        self._target_speed = None
        self._min_distance = None
        self._MIN_DISTANCE_PERCENTAGE = 0.9
        self._initial_population = 100

        self._min_waypoint_interval = 5
        self._current_waypoint = None
        self._buffer_size = 5
        self._target_waypoint = None
        self._next_waypoints = deque(maxlen=2000)  #double-ended queue with max size limit to auto delete front entries
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        self._options = [RoadOption.FOLLOW_LANE, RoadOption.CHANGE_LANE_LEFT, RoadOption.CHANGE_LANE_RIGHT]
        self._init_controller(opt_dict)

    def _state_transition(self, k=200):
        # for _ in range(k):
            # if len(self._next_waypoints) == 1:
            #     last_waypoint = self._next_waypoints[0][0]
            # else:
            #     last_waypoint = self._next_waypoints[-1][0]
            # print(self._current_waypoint)
            # self._next_waypoints.append((self._current_waypoint.next_until_lane_end(3), RoadOption.FOLLOW_LANE))
        # print(len(self._next_waypoints))
        # # print(self._next_waypoints[-1])
        # print('------------------------------------------------')
        last_waypoint = self._next_waypoints[-1][0][0]
        print(last_waypoint)
        if self._state == RoadOption.FOLLOW_LANE:
            distance = random.randint(50, 100)
            available_entries = self._next_waypoints.maxlen - len(self._next_waypoints)
            distance = min(available_entries, distance)
            # print(f'Following lane of distance: {distance}')         
            for _ in range(distance):
                self._next_waypoints.append((last_waypoint.next(self._min_waypoint_interval)[0], RoadOption.FOLLOW_LANE))
            # self._state = random.choice(self._options)

        # elif self._state == RoadOption.CHANGE_LANE_LEFT:
        #     next_waypoint = last_waypoint.get_left_lane().next(3)
        #     if next_waypoint:
        #         self._next_waypoints.append((next_waypoint, RoadOption.CHANGE_LANE_LEFT))
        #     else:
        #         while self._state == RoadOption.CHANGE_LANE_LEFT:
        #             self._state = random.choice(self._options)
        
        # elif self._state == RoadOption.CHANGE_LANE_RIGHT:
        #     next_waypoint = last_waypoint.get_right_lane().next(3)
        #     if next_waypoint:
        #         self._next_waypoints.append((next_waypoint, RoadOption.CHANGE_LANE_RIGHT))
        #     else:
        #         while self._state == RoadOption.CHANGE_LANE_RIGHT:
        #             self._state = random.choice(self._options)

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon
        self._min_distance = self._sampling_radius * self._MIN_DISTANCE_PERCENTAGE
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 1,
            'dt': self._dt}

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = self._target_speed * \
                                        opt_dict['sampling_radius'] / 3.6
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict)
        self._next_waypoints.append((self._current_waypoint.next(self._initial_population), RoadOption.FOLLOW_LANE))
        # for _ in range(self._initial_population):
        #     self._next_waypoints.append((self._current_waypoint.next(3), RoadOption.FOLLOW_LANE))
        # self._state_transition(k=200)

    def run_step(self, debug=True):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # # not enough waypoints in the horizon? => add more!
        # if not self._global_plan and len(self._next_waypoints) < int(self._next_waypoints.maxlen * 0.5):
        #     self._compute_next_waypoints(k=100)
        if len(self._next_waypoints) < int(self._next_waypoints.maxlen * 0.5):
            self._state_transition()

        if len(self._next_waypoints) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        # #   Buffering the waypoints
        # if not self._waypoint_buffer:
        #     for i in range(self._buffer_size):
        #         if self._next_waypoints:
        #             self._waypoint_buffer.append(
        #                 self._next_waypoints.popleft())
        #         else:
        #             break

        # current vehicle waypoint
        vehicle_transform = self._vehicle.get_transform()
        self._current_waypoint = self._map.get_waypoint(vehicle_transform.location)
        # target waypoint
        self._target_waypoint, road_option = self._next_waypoints[0]
        print(self._target_waypoint)
        # move using PID controllers
        control = self._vehicle_controller.run_step(self._target_speed, self._target_waypoint)

        # # purge the queue of obsolete waypoints
        # max_index = -1

        # for i, (waypoint, _) in enumerate(self._waypoint_buffer):
        #     if waypoint.transform.location.distance(vehicle_transform.location) < self._min_distance:
        #         max_index = i
        # if max_index >= 0:
        #     for i in range(max_index + 1):
        #         self._next_waypoints.popleft()

        if debug:
            draw_waypoints(self._vehicle.get_world(), [self._target_waypoint], self._vehicle.get_location().z + 1.0)

        return control
    
    # def state_transition(self, k):
    #     for _ in range(k):
    #         last_waypoint = self._waypoints_queue[-1][0]
    #         print(last_waypoint.lane_change)
    #         # print(last_waypoint)
    #         # print(len(list(last_waypoint.next(2))))
    #         if not last_waypoint.next(3):
    #             while self._state == RoadOption.FOLLOW_LANE:
    #                 self._state = random.choice(self._options)

    #         # print(len(self._waypoints_queue), last_waypoint)

    #         if self._state == RoadOption.FOLLOW_LANE:
    #             new_waypoints = list(last_waypoint.next(3))[0]
    #             if new_waypoints:
    #                 self._waypoints_queue.append((new_waypoints, RoadOption.FOLLOW_LANE))
    #             else:
    #                 self._waypoints_queue.pop()
    #                 last_waypoint = self._waypoints_queue[-1][0]
    #             self._state = random.choice(self._options)

    #         elif self._state == RoadOption.CHANGE_LANE_LEFT:
    #             if last_waypoint.get_left_lane():
    #                 self._waypoints_queue.pop()
    #                 self.get_intermediate_waypoints(self._waypoints_queue[-1][0], last_waypoint.get_left_lane())
    #                 self._waypoints_queue.append((last_waypoint.get_left_lane(), RoadOption.CHANGE_LANE_LEFT))
    #                 self._state = RoadOption.FOLLOW_LANE
    #             else:
    #                 while self._state == RoadOption.CHANGE_LANE_LEFT:
    #                     self._state = random.choice(self._options)

    #         elif self._state == RoadOption.CHANGE_LANE_RIGHT:
    #             if last_waypoint.get_right_lane():
    #                 self._waypoints_queue.pop()
    #                 self.get_intermediate_waypoints(self._waypoints_queue[-1][0], last_waypoint.get_right_lane())
    #                 self._waypoints_queue.append((last_waypoint.get_right_lane(), RoadOption.CHANGE_LANE_RIGHT))
    #                 self._state = RoadOption.FOLLOW_LANE
    #             else:
    #                 while self._state == RoadOption.CHANGE_LANE_RIGHT:
    #                     self._state = random.choice(self._options)