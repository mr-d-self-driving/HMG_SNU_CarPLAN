import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.batch_kinematic_bicycle import (
    BatchKinematicBicycleModel,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.batch_lqr import (
    BatchLQRTracker,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)
import torch

class PDMSimulator:
    """
    Re-implementation of nuPlan's simulation pipeline. Enables batch-wise simulation.
    """

    def __init__(self, proposal_sampling: TrajectorySampling, batch_size=16):
        """
        Constructor of PDMSimulator.
        :param proposal_sampling: Sampling parameters for proposals
        """

        # time parameters
        self._proposal_sampling = proposal_sampling

        # simulation objects
        self._motion_model = BatchKinematicBicycleModel()
        self._tracker = BatchLQRTracker()
        self._motion_models = [BatchKinematicBicycleModel() for i in range(batch_size)]

    def simulate_proposals(
        self, states: npt.NDArray[np.float64], initial_ego_state: EgoState
    ) -> npt.NDArray[np.float64]:
        """
        Simulate all proposals over batch-dim
        :param initial_ego_state: ego-vehicle state at current iteration
        :param states: proposal states as array
        :return: simulated proposal states as array
        """

        # TODO: find cleaner way to load parameters
        # set parameters of motion model and tracker
        self._motion_model._vehicle = initial_ego_state.car_footprint.vehicle_parameters
        self._tracker._discretization_time = self._proposal_sampling.interval_length

        proposal_states = states[:, : self._proposal_sampling.num_poses + 1]
        self._tracker.update(proposal_states)

        # state array representation for simulated vehicle states
        simulated_states = np.zeros(proposal_states.shape, dtype=np.float64)
        simulated_states[:, 0] = ego_state_to_state_array(initial_ego_state)

        # timing objects
        current_time_point = initial_ego_state.time_point
        delta_time_point = TimeDuration.from_s(self._proposal_sampling.interval_length)

        current_iteration = SimulationIteration(current_time_point, 0)
        next_iteration = SimulationIteration(current_time_point + delta_time_point, 1)

        for time_idx in range(1, self._proposal_sampling.num_poses + 1):
            sampling_time: TimePoint = (
                next_iteration.time_point - current_iteration.time_point
            )

            command_states = self._tracker.track_trajectory(
                current_iteration,
                next_iteration,
                simulated_states[:, time_idx - 1],
            )

            simulated_states[:, time_idx] = self._motion_model.propagate_state(
                states=simulated_states[:, time_idx - 1],
                command_states=command_states,
                sampling_time=sampling_time,
            )

            current_iteration = next_iteration
            next_iteration = SimulationIteration(
                current_iteration.time_point + delta_time_point, 1 + time_idx
            )

        return simulated_states
    
    def simulate_proposals_for_batch(
        self, states: npt.NDArray[np.float64], initial_ego_states: EgoState
    ) -> npt.NDArray[np.float64]:
        """
        Simulate all proposals over batch-dim
        :param initial_ego_state: ego-vehicle state at current iteration
        :param states: proposal states as array
        :return: simulated proposal states as array
        """

        # TODO: find cleaner way to load parameters
        # set parameters of motion model and tracker
        # self._motion_model._vehicle = initial_ego_state.car_footprint.vehicle_parameters
        # self._motion_model._vehicle = initial_ego_states[0].car_footprint.vehicle_parameters
        self._motion_model._vehicle = initial_ego_states[0].car_footprint.vehicle_parameters
        for i in range(len(initial_ego_states)):
            if i != 0:
                if self._motion_model._vehicle != initial_ego_states[i].car_footprint.vehicle_parameters:
                    NameError('There is other type ego_vehicle in simulate_proposals_for_batch')

        
        self._tracker._discretization_time = self._proposal_sampling.interval_length
        states = np.concatenate((states.detach().cpu().numpy(), np.zeros((states.shape[0], states.shape[1], 8))), axis=-1)

        proposal_states = states[:, : self._proposal_sampling.num_poses + 1]
        self._tracker.update(proposal_states)

        # state array representation for simulated vehicle states
        simulated_states = np.zeros(proposal_states.shape, dtype=np.float64)
        simulated_states[:, 0] = np.array([ego_state_to_state_array(initial_ego_state) for initial_ego_state in initial_ego_states])

        # timing objects
        current_time_point = initial_ego_states[0].time_point
        delta_time_point = TimeDuration.from_s(self._proposal_sampling.interval_length)

        current_iteration = SimulationIteration(current_time_point, 0)
        next_iteration = SimulationIteration(current_time_point + delta_time_point, 1)

        for time_idx in range(1, self._proposal_sampling.num_poses + 1):
            sampling_time: TimePoint = (
                next_iteration.time_point - current_iteration.time_point
            )

            command_states = self._tracker.track_trajectory(
                current_iteration,
                next_iteration,
                simulated_states[:, time_idx - 1],
            )

            # simulated_states[:, time_idx] = [_motion_model.propagate_state(states=simulated_states[idx, time_idx - 1],
            #                                                                 command_states=command_states[idx],
            #                                                                 sampling_time=sampling_time) for idx, _motion_model in enumerate(self._motion_models)]

            simulated_states[:, time_idx] = self._motion_model.propagate_state(
                states=simulated_states[:, time_idx - 1],
                command_states=command_states,
                sampling_time=sampling_time)

            current_iteration = next_iteration
            next_iteration = SimulationIteration(
                current_iteration.time_point + delta_time_point, 1 + time_idx
            )

        return simulated_states
