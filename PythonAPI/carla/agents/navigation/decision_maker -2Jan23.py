from enum import Enum
from collections import deque
import random

import carla
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import draw_waypoints, get_speed, is_within_distance_ahead, compute_magnitude_angle
# from agents.navigation.agent import Agent

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

    STOPPED_BY_TRAFFIC_LIGHT = 21
    VEHICLE_AHEAD = 22
    STOP_SIGN = 23

class DecisionMaker(object):
    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict=None):

        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()
        self._state = RoadOption.FOLLOW_LANE
        self._speed_state = None
        self._target_vehicle_speed = None

        self._stop_sign_counter = 0
        self._previous_stop = None

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

        self._last_traffic_light = None
        self._proximity_threshold = 7.0  # meters
        self._slowing_down_threshold = 50.0 # meters

        self._waypoints_and_speed = []

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

        self.state_transition(k=100)

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
        initial_coordinate = np.array((initial_waypoint.transform.location.x, initial_waypoint.transform.location.y, initial_waypoint.transform.location.z))
        end_coordinate = np.array((end_waypoint.transform.location.x, end_waypoint.transform.location.y, end_waypoint.transform.location.z))

        difference = (end_coordinate - initial_coordinate) / k
    
        # Generate intermediate waypoints
        for i in range(0, k + 1):
            intermediate_coordinate = initial_coordinate + difference*i
            intermediate_coordinate = carla.Location(intermediate_coordinate[0], intermediate_coordinate[1], intermediate_coordinate[2])
            self._waypoints_queue.append((self._map.get_waypoint(intermediate_coordinate), option))
        
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

        if (speed > speed_limit) or self._state == RoadOption.VEHICLE_AHEAD:
            self._speed_state = RoadOption.DECELERATE
            return
        
        elif (speed_limit - 2) <= speed <= (speed_limit + 2):
            # print("Maintain")
            self._speed_state = RoadOption.MAINTAIN_SPEED
            return
        elif speed < speed_limit:
            self._speed_state = RoadOption.ACCELERATE
            return

    def state_transition(self, k=100):
        """
        Adds 'k' number of waypoints to the deque while checking for available lateral manoevres at each iteration

        :param k: number of waypoints to be added
        :return:
        """
        for i in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            self.lateral_manoeuvres(last_waypoint)

    def longitudinal_manoevres(self):
        """
        Checks for the speed limit of the road the ego vehicle is on and adjusts the ego speed accordingly.

        :param:
        :return:
        """
        speed_limit = self._vehicle.get_speed_limit()
        speed = get_speed(self._vehicle)
        self.speed_check(speed_limit, speed)

        if self._speed_state == RoadOption.ACCELERATE:
            self._target_speed = speed_limit
        
        elif self._speed_state == RoadOption.DECELERATE:
            if self._state == RoadOption.VEHICLE_AHEAD:
                self._target_speed = max(10.0, self._target_vehicle_speed)
            else:
                self._target_speed = speed_limit

        else:
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

    def _is_light_red(self, lights_list):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        if self._map.name == 'Town01' or self._map.name == 'Town02':
            return self._is_light_red_europe_style(lights_list)
        else:
            return self._is_light_red_us_style(lights_list)

    def _is_light_red_europe_style(self, lights_list):
        """
        This method is specialized to check European style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                  affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_waypoint = self._map.get_waypoint(traffic_light.get_location())
            if object_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    object_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            if is_within_distance_ahead(traffic_light.get_transform(),
                                        self._vehicle.get_transform(),
                                        self._proximity_threshold):
                if traffic_light.state == carla.TrafficLightState.Red:
                    return (True, traffic_light)

        return (False, None)

    def _is_light_red_us_style(self, lights_list, debug=False):
        """
        This method is specialized to check US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        if ego_vehicle_waypoint.is_junction:
            # It is too late. Do not block the intersection! Keep going!
            return (False, None)

        if self.target_waypoint is not None:
            if self.target_waypoint.is_junction:
                min_angle = 180.0
                sel_magnitude = 0.0
                sel_traffic_light = None
                for traffic_light in lights_list:
                    loc = traffic_light.get_location()
                    magnitude, angle = compute_magnitude_angle(loc,
                                                               ego_vehicle_location,
                                                               self._vehicle.get_transform().rotation.yaw)
                    if magnitude < 60.0 and angle < min(25.0, min_angle):
                        sel_magnitude = magnitude
                        sel_traffic_light = traffic_light
                        min_angle = angle

                if sel_traffic_light is not None:
                    if debug:
                        print('=== Magnitude = {} | Angle = {} | ID = {}'.format(
                            sel_magnitude, min_angle, sel_traffic_light.id))

                    if self._last_traffic_light is None:
                        self._last_traffic_light = sel_traffic_light

                    if self._last_traffic_light.state == carla.TrafficLightState.Red:
                        return (True, self._last_traffic_light)
                else:
                    self._last_traffic_light = None

        return (False, None)    

    def _is_vehicle_hazard(self, vehicle_list):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle.

        WARNING: This method is an approximation that could fail for very large
         vehicles, which center is actually on a different lane but their
         extension falls within the ego vehicle lane.

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        self._target_vehicle_speed = self._vehicle.get_speed_limit()

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            if is_within_distance_ahead(target_vehicle.get_transform(),
                                        self._vehicle.get_transform(),
                                        self._proximity_threshold):
                return (True, target_vehicle)

            if is_within_distance_ahead(target_vehicle.get_transform(),
                                        self._vehicle.get_transform(),
                                        self._slowing_down_threshold):
                self._target_vehicle_speed = get_speed(target_vehicle)
                self._state = RoadOption.VEHICLE_AHEAD
                return (False, target_vehicle)
            
        return (False, None)

    def _is_stop_sign(self, stop_list, debug=False):
        # ego_vehicle_location = self._vehicle.get_location()
        # # ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        # self._target_vehicle_speed = self._vehicle.get_speed_limit()

        # for target_stop in stop_list:

        #     # if the object is not in our lane it's not an obstacle
        #     target_stop_waypoint = self._map.get_waypoint(target_stop.get_location())
        #     if target_stop_waypoint.road_id != self.target_waypoint.road_id or \
        #             target_stop_waypoint.lane_id != self.target_waypoint.lane_id:
        #         continue

        #     if is_within_distance_ahead(target_stop.get_location(),
        #                                 ego_vehicle_location,
        #                                 self._proximity_threshold):
        #         return (True, target_stop)
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for stop_sign in stop_list:
            stop_sign_waypoint = self._map.get_waypoint(stop_sign.get_location())
            if stop_sign_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    stop_sign_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            if is_within_distance_ahead(stop_sign.get_transform(),
                                        self._vehicle.get_transform(),
                                        self._proximity_threshold):
                    return (True, stop_sign)

        return (False, None)        
    
    # if self.target_waypoint is not None:
    #         min_angle = 180.0
    #         self_magnitude = 0.0
    #         self_stop_sign = None
    #         for stop_sign in stop_list:
    #             loc = stop_sign.get_location()
    #             magnitude, angle = compute_magnitude_angle(loc,
    #                                                         ego_vehicle_location,
    #                                                         self._vehicle.get_transform().rotation.yaw)
    #             if magnitude < 60.0 and angle < min(25.0, min_angle):
    #                 self_magnitude = magnitude
    #                 self_stop_sign = stop_sign
    #                 min_angle = angle

    #             if self_stop_sign is not None:
    #                 if debug:
    #                     print('=== Magnitude = {} | Angle = {} | ID = {}'.format(
    #                         self_magnitude, min_angle, self_stop_sign.id))

    #                 if self_stop_sign is None:
    #                     self_last_stop_sign = self_stop_sign

    #                 return (True, self_last_stop_sign)
    #             else:
    #                 self_last_stop_sign = None

    #     return (False, None)
    
    def hazards(self, debug=True):
        """
        Checks for the conditions that the ego vehicle is at the traffic light.

        :param debug: boolean flag to print indicator of red light if ego vehicle encounters one
        :return:
        """
        # is there an obstacle in front of us?
        self.hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._vehicle.get_world().get_actors()

        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")
        stop_list = actor_list.filter("*stop*")
        # stop_list.extend(actor_list.filter("*stop*"))
        # print(stop_list)
        self._slowing_down_threshold = max(30.0, get_speed(self._vehicle)/2)

        # # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)

        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD !!!')

            self._state = RoadOption.VEHICLE_AHEAD
            self.hazard_detected = True
            return
        
        if not vehicle_state and self._state == RoadOption.VEHICLE_AHEAD:
            if debug:
                print('!!! KEEPING SAFE DISTANCE !!!')
            self.longitudinal_manoevres()

        # check for the state of the traffic lights
        # Agent._is_light_red()
        light_state, traffic_light = self._is_light_red(lights_list)
        if light_state:
            if debug:
                print('=== RED LIGHT AHEAD ===')

            self._state = RoadOption.STOPPED_BY_TRAFFIC_LIGHT
            self.hazard_detected = True
            return
        
        stop_state, stop_sign = self._is_stop_sign(stop_list)
        # print(stop_sign)
        # print(self._previous_stop)
        if stop_state:
            if self._previous_stop == stop_sign.id:
                print("EXIT!!!")
                # print(self._stop_sign_counter)
                return
            if debug:
                print('=== STOP SIGN AHEAD ===')
            self._stop_sign_counter += 1
            if self._stop_sign_counter >= 300:
                # print(self._stop_sign_counter)
                self.hazard_detected = False
                self._previous_stop = stop_sign.id
                self._stop_sign_counter = 0
                return
            self._state = RoadOption.STOP_SIGN
            self.hazard_detected = True
            return
        
        else:
            self.hazard_detected = False
    
    def emergency_stop(self):   
        """
        Causes the ego vehicle to come to a complete stop in the event of an emergency (e.g. hazard ahead)

        :param:
        :return:
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        control.manual_gear_shift = False

        return control

    def run_step(self, debug=True):
        """
        Execute one step of decision making which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # adds more waypoints if there is not enough
        if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self.state_transition(100)

        if len(self._waypoints_queue) == 0 and len(self._waypoint_buffer) == 0:
            self.emergency_stop()
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
        self.longitudinal_manoevres()
        # print(self._state)
        self.hazards()
        # move using PID controllers
        if self.hazard_detected == True:
            control = self.emergency_stop()
            return control
        else:
            # print(self.target_waypoint.get_landmarks(distance=500.0))
            control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)
            print(f'Throttle: {control.throttle}    |   Brake: {control.brake}     |    Current speed: {get_speed(self._vehicle)}    |    Target speed: {self._target_speed}     |    Speed limit: {self._vehicle.get_speed_limit()}    |     Current speed state: {self._speed_state}')

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