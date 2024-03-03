from enum import Enum
from collections import deque
import random

import carla
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import draw_waypoints
from agents.tools.misc import get_speed

import numpy as np
import math


class RoadOption(Enum):
    """`
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    FOLLOW_LANE = 1
    CHANGE_LANE_LEFT = 2
    CHANGE_LANE_RIGHT = 3

    MAINTAIN_SPEED = 11
    ACCELERATE = 12
    DECELERATE = 13

class DecisionMaker(object):
    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict=None):

        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()
        self._state = RoadOption.FOLLOW_LANE
        self._speed_state = None

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_controller = None

        self._waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        self._counter = 0

        self._options = [RoadOption.FOLLOW_LANE, RoadOption.CHANGE_LANE_LEFT, RoadOption.CHANGE_LANE_RIGHT]

        # initializing controller
        self._init_controller(opt_dict)

    def __del__(self):
        if self._vehicle:
            self._vehicle.destroy()
        print("Destroying ego-vehicle!")

    def reset_vehicle(self):
        self._vehicle = None
        print("Resetting ego-vehicle!")

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
        # self._sampling_radius = 1.0
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
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

        self._global_plan = False

        # compute initial waypoints
        for k in range(10):
            self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius+(k*3))[0], RoadOption.FOLLOW_LANE))
        # print(self._waypoints_queue[-1])
        self._target_road_option = RoadOption.FOLLOW_LANE
        # fill waypoint trajectory queue
        # self._compute_next_waypoints(k=200)
        self.state_transition(k=100)

    def set_speed(self, speed):
        """
        Request new target speed.

        :param speed: new target speed in km/h
        :return:
        """
        self._target_speed = speed

    def get_intermediate_waypoints(self, initial_waypoint, end_waypoint, k=3, option=RoadOption):
        """
        Adds intermediate waypoints in between the new waypoint on the adjacent lane and the last waypoint of the current lane.
        Also adds 10 waypoints for the new lane to ensure that the ego vehicle stays on the new lane for a while.

        :param initial_waypoint: last waypoint in the current lane
        :param end_waypoint: first waypoint in the adjacent lane
        :param k: number of intermediate waypoints to be added.
        :param option: the RoadOption for the lane change
        :return:
        """
        # print(end_waypoint, type(end_waypoint))
        initial_coordinate = np.array((initial_waypoint.transform.location.x, initial_waypoint.transform.location.y, initial_waypoint.transform.location.z))
        end_coordinate = np.array((end_waypoint.transform.location.x, end_waypoint.transform.location.y, end_waypoint.transform.location.z))

        difference = (end_coordinate - initial_coordinate) / k
    
        # Generate intermediate waypoints
        for i in range(0, k + 1):
            intermediate_coordinate = initial_coordinate + difference*i
            # print(intermediate_coordinate)
            intermediate_coordinate = carla.Location(intermediate_coordinate[0], intermediate_coordinate[1], intermediate_coordinate[2])
            # print(intermediate_coordinate)
            self._waypoints_queue.append((self._map.get_waypoint(intermediate_coordinate), option))
            # print(self._waypoints_queue[-1][0])
        
        for _ in range(10):
            last_waypoint = self._waypoints_queue[-1][0]
            if last_waypoint.next(self._sampling_radius) == []:
                break
            else:
                new_waypoints = list(last_waypoint.next(self._sampling_radius))[0]
                self._waypoints_queue.append((new_waypoints, RoadOption.FOLLOW_LANE))

    def available_lane_changes(self, last_waypoint):
        """
        Retrieves the available lane changes that the ego vehicle can take based on its last waypoint

        :param last_waypoint: last waypoint in the deque
        :return:
        """
        options = last_waypoint.lane_change

        if options == carla.LaneChange.Both:
            self._state = random.choice(self._options)
            if (self._state == RoadOption.CHANGE_LANE_RIGHT and last_waypoint.get_right_lane()) or (self._state == RoadOption.CHANGE_LANE_LEFT and last_waypoint.get_left_lane()):
                return
            else:
                self._state = RoadOption.FOLLOW_LANE

        elif options == carla.LaneChange.Left and last_waypoint.get_left_lane():
            # print("LEFT")
            self._state = RoadOption.CHANGE_LANE_LEFT

        elif options == carla.LaneChange.Right and last_waypoint.get_right_lane():
            # print("RIGHT")
            self._state = RoadOption.CHANGE_LANE_RIGHT
            
        else:
            self._state = RoadOption.FOLLOW_LANE

    def speed_check(self, speed_limit, speed):
        """
        Checks if the ego vehicle is at the speed limit of the road and sets the self._speed_state accordingly.

        :param speed_limit: speed limit of the road the ego vehicle is on
        :return:
        """
        # speed
        # print(f'Checking speed with speed limit {speed_limit} and speed {speed}')

        if (speed_limit - 2) <= speed <= (speed_limit + 2):
            # print("Maintain")
            self._speed_state = RoadOption.MAINTAIN_SPEED
            return
        elif speed < speed_limit:
            self._speed_state = RoadOption.ACCELERATE
            return
        elif speed > speed_limit:
            self._speed_state = RoadOption.DECELERATE
            return

    def state_transition(self):
        """
        Adds 'k' number of waypoints to the deque while checking for available lateral and longitudinal manoevres at each iteration

        :param k: number of waypoints to be added
        :return:
        """
        # for i in range(k):
        last_waypoint = self._waypoints_queue[-1][0]
        # print(len(self._waypoints_queue), self._waypoints_queue[-1])
        self.lateral_manoeuvres(last_waypoint)
        last_waypoint = self._waypoints_queue[-1][0]
        # print(i)
        self.longitudinal_manoevres(last_waypoint)

    def longitudinal_manoevres(self, last_waypoint):
        """
        Checks for the speed limit of the road the ego vehicle is on and adjusts the ego speed accordingly.

        :param last_waypoint: last waypoint in the deque
        :return:
        """
        # print('--------------------------------------------------------------------------------')
        speed_limit = self._vehicle.get_speed_limit()
        # v = self._vehicle.get_velocity()
        # speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        speed = get_speed(self._vehicle)
        self.speed_check(speed_limit, speed)
        # print(f'Entering longitudinal next iteration: {self._speed_state} with {speed_limit} at {speed}')
        # print(speed_limit, speed)

        # print('Speed:   % 15.0f km/h' % speed)

        if self._speed_state == RoadOption.ACCELERATE:
            # speed_limit = self._vehicle.get_speed_limit()

            self._target_speed = speed_limit
            # print(f'Speeding up from {speed} going for {speed_limit}; {last_waypoint}')
            # print(f'Target speed up from decision maker: {self._target_speed}')
            # control.throttle = 1
            # control = self._vehicle_controller.run_step(self._target_speed, last_waypoint)
            # self._vehicle.apply_control(control)
            # print("Done applying control for speed up")
            # speed = get_speed(self._vehicle)
            # self.speed_check(speed_limit, speed)
            # print(self._speed_state)
        
        elif self._speed_state == RoadOption.DECELERATE:
            # speed_limit = self._vehicle.get_speed_limit()
            # self.set_speed(speed_limit)
            self._target_speed = speed_limit
            # print(f'Slowing down from {speed} going for {speed_limit}')
            # control = self._vehicle_controller.run_step(self._target_speed, last_waypoint)
            # control.brake = 0.2
            # self._vehicle.apply_control(control)

            # print("Done applying control for slow down")
            # speed = get_speed(self._vehicle)

            # self.speed_check(speed_limit, speed)


        else:
            # print("Maintaining")
            pass

    def lateral_manoeuvres(self, last_waypoint):
        """
        Checks for the available lane changes at the last waypoint and adjusts the self._state accordingly.
        Intermediate waypoints are added between lane changing to ensure smooth transition.

        :param last_waypoint: last waypoint in the deque
        :return:
        """

        self.available_lane_changes(last_waypoint) # finds the lane changes available for that waypoint and adjusts the self._state accordingly

        if last_waypoint.next(self._sampling_radius) == []: # when last_waypoint.next() returns NONE (i.e. no waypoints ahead in the self._sampling radius distance)
            while self._state == RoadOption.FOLLOW_LANE:
                self._state = random.choice(self._options)

        if self._state == RoadOption.FOLLOW_LANE:
            new_waypoints = list(last_waypoint.next(self._sampling_radius))[0]
            self._waypoints_queue.append((new_waypoints, RoadOption.FOLLOW_LANE))

        elif self._state == RoadOption.CHANGE_LANE_LEFT:
            for _ in range(min(int(self._sampling_radius*1.5), len(self._waypoints_queue)-1)):
                self._waypoints_queue.pop() # removes the last few waypoints before adding the waypoint on the adjacent lane --> prevents two directly adjacent waypoints from causing awkward ego manoevres
            self.get_intermediate_waypoints(self._waypoints_queue[-1][0], last_waypoint.get_left_lane(), option=RoadOption.CHANGE_LANE_LEFT)    # to add waypoints in between lane transitions to ensure smoothness

        elif self._state == RoadOption.CHANGE_LANE_RIGHT:
            for _ in range(min(int(self._sampling_radius*1.5), len(self._waypoints_queue)-1)):
                self._waypoints_queue.pop()
            self.get_intermediate_waypoints(self._waypoints_queue[-1][0], last_waypoint.get_right_lane(), option=RoadOption.CHANGE_LANE_RIGHT)

    def run_step(self, debug=True):
        """
        Execute one step of decision making which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # not enough waypoints in the horizon? => add more!
        if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            # self._compute_next_waypoints(k=100)
            for _ in range(100):
                self.state_transition()

        if len(self._waypoints_queue) == 0 and len(self._waypoint_buffer) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        #   Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # current vehicle waypoint
        vehicle_transform = self._vehicle.get_transform()
        self._current_waypoint = self._map.get_waypoint(vehicle_transform.location)
        # target waypoint
        self.target_waypoint, self._target_road_option = self._waypoint_buffer[0]
        # move using PID controllers
        # print(self._target_speed)
        control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)
        print(f'Throttle: {control.throttle}    |   Brake: {control.brake}   |   Current speed = {get_speed(self._vehicle)}     |    Target speed: {self._target_speed}     |    Speed limit: {self._vehicle.get_speed_limit()}     |    Current speed state: {self._speed_state}')

        # purge the queue of obsolete waypoints
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if waypoint.transform.location.distance(vehicle_transform.location) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        if debug:
            draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], self._vehicle.get_location().z + 1.0)

        return control