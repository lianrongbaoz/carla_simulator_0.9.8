#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

from agents.navigation.agent import Agent, AgentState
# from agents.navigation.local_planner import LocalPlanner
# from agents.navigation.decision_maker_fyp import DecisionMaker
from agents.navigation.decision_maker import DecisionMaker

class RoamingAgent(Agent):
    """
    RoamingAgent implements a basic agent that navigates scenes making random
    choices when facing an intersection.

    This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(RoamingAgent, self).__init__(vehicle)
        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        self._decision_maker = DecisionMaker(self._vehicle)

    def run_step(self, debug=True):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """
        control = self._decision_maker.run_step()

        return control
