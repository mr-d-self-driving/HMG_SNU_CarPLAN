#   Heavily borrowed from:
#   https://github.com/autonomousvision/tuplan_garage (Apache License 2.0)
# & https://github.com/motional/nuplan-devkit (Apache License 2.0)

import warnings
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import shapely.creation
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from shapely import LineString

from src.scenario_manager.occupancy_map import OccupancyMap

from .common.enum import (
    CollisionType,
    EgoAreaIndex,
)
from .common.geometry import (
    compute_agents_vertices,
    ego_rear_to_center,
    get_collision_type,
)
from .forward_simulation.forward_simulator import ForwardSimulator
from .observation.world_from_prediction import WorldFromPrediction



class TrajectoryEvaluator:
    def __init__(
        self,
        dt: float = 0.1,
        num_frames: int = 40,
    ) -> None:
        assert dt * num_frames <= 8, "dt * num_frames should be less than 8s"

        self._dt = dt
        self._num_frames = num_frames

        self._route_lane_dict = None
        self._drivable_area_map: Optional[OccupancyMap] = None
        self._world = WorldFromPrediction(dt, num_frames)
        self._forward_simulator = ForwardSimulator(dt, num_frames)

        self._init_ego_state: Optional[EgoState] = None
        self._ego_rollout: Optional[np.ndarray[np.float64]] = None
        self._ego_polygons: Optional[np.ndarray[np.object_]] = None
        self._ego_footprints: Optional[np.ndarray[np.object_]] = None
        self._ego_parameters = get_pacifica_parameters()
        self._ego_shape = np.array(
            [self._ego_parameters.width, self._ego_parameters.length],
            dtype=np.float64,
        )
        self._at_fault_collision_time: Optional[np.ndarray[np.float64]] = None

    def time_to_at_fault_collision(self, rollout_idx: int) -> float:
        return self._at_fault_collision_time[rollout_idx]

    def evaluate(
        self,
        candidate_trajectories: np.ndarray,
        init_ego_state: EgoState,
        detections: DetectionsTracks,
        traffic_light_data: List[TrafficLightStatusData],
        agents_info: Dict[str, np.ndarray],
        route_traffic_dict: Dict[str, LaneGraphEdgeMapObject],
        route_roadblocks_ids : None, 
        drivable_area_map: Optional[OccupancyMap],
    ):
        self._reset(
            candidate_trajectories=candidate_trajectories,
            init_ego_state=init_ego_state,
            detections=detections,
            traffic_light_data=traffic_light_data,
            agents_info=agents_info,
            route_traffic_dict=route_traffic_dict,
            route_roadblocks_ids = route_roadblocks_ids,
            drivable_area_map=drivable_area_map,
        )
        self.route_roadblocks_ids = route_roadblocks_ids
        self._update_ego_footprints()
        self._evaluate_no_at_fault_collisions()
        return self._at_fault_collision_time[0]

    def _reset(
        self,
        candidate_trajectories: np.ndarray,
        init_ego_state: EgoState,
        detections: DetectionsTracks,
        traffic_light_data: List[TrafficLightStatusData],
        agents_info: Dict[str, np.ndarray],
        route_traffic_dict: Dict[str, LaneGraphEdgeMapObject],
        route_roadblocks_ids: None,
        drivable_area_map: Optional[OccupancyMap],
    ):
        self._num_candidates = len(candidate_trajectories)
        self.route_traffic_dict = route_traffic_dict
        self._drivable_area_map = drivable_area_map
        self._world.drivable_area = drivable_area_map
        self._route_lane_ids = None

        self._update_ego_rollout(
            init_ego_state=init_ego_state,
            candidate_trajectories=candidate_trajectories,
        )
        self._world.update(
            ego_state=init_ego_state,
            detections=detections,
            traffic_light_data=traffic_light_data,
            agents_info=agents_info,
            route_traffic_dict=route_traffic_dict,
            route_roadblocks_ids = route_roadblocks_ids,
        )

        self._at_fault_collision_time = np.full((self._num_candidates,), np.inf)


    def _update_ego_rollout(
        self, init_ego_state: EgoState, candidate_trajectories: np.ndarray
    ):
        rollout_states = self._forward_simulator.forward(
            candidate_trajectories, init_ego_state
        )
        N, T, _ = rollout_states.shape
        vertices = compute_agents_vertices(
            center=ego_rear_to_center(rollout_states[..., :2], rollout_states[..., 2]),
            angle=rollout_states[..., 2],
            shape=self._ego_shape[None, :].repeat(N, axis=0),
        )
        self._ego_rollout = rollout_states
        self._ego_vertices = vertices
        self._ego_polygons = shapely.creation.polygons(vertices)
        self._ego_footprints = np.zeros((N, T, 3), dtype=bool)

    def _update_ego_footprints(self) -> None:
        keypoints = np.concatenate(
            [self._ego_vertices, self._ego_rollout[:, :, None, :2]], axis=-2
        )

        N, T, P, _ = keypoints.shape
        keypoints = keypoints.reshape(N * T * P, 2)

        (
            in_polygons,
            speed_limit,
        ) = self._drivable_area_map.points_in_polygons_with_attribute(
            keypoints, "speed_limit"
        )

        # (N, T, num_polygons, num_points)
        in_polygons = in_polygons.reshape(
            len(self._drivable_area_map), N, T, P
        ).transpose(1, 2, 0, 3)
        speed_limit = speed_limit.reshape(
            len(self._drivable_area_map), N, T, P
        ).transpose(1, 2, 0, 3)[..., 4]

        da_on_route_idx: List[int] = [
            idx
            for idx, token in enumerate(self._drivable_area_map.tokens)
            if token in self.route_roadblocks_ids
        ]

        corners_in_polygon, center_in_polygon = (
            in_polygons[..., :4],
            in_polygons[..., 4],
        )

        on_multi_lane_mask = corners_in_polygon.any(-1).sum(-1) > 1
        on_single_lane_mask = corners_in_polygon.all(-1).any(-1)
        on_multi_lane_mask &= ~on_single_lane_mask
        out_drivable_area_mask = (corners_in_polygon.sum(-2) > 0).sum(-1) < 4
        oncoming_traffic_mask = ~center_in_polygon[..., da_on_route_idx].any(-1)
        speed_limit[~center_in_polygon] = 0.0

        self._ego_footprints[on_multi_lane_mask, EgoAreaIndex.MULTIPLE_LANES] = True
        self._ego_footprints[out_drivable_area_mask, EgoAreaIndex.NON_DRIVABLE_AREA] = (
            True
        )
        self._ego_footprints[oncoming_traffic_mask, EgoAreaIndex.ONCOMING_TRAFFIC] = (
            True
        )

    def _evaluate_no_at_fault_collisions(self):
        collided_tokens = {
            i: deepcopy(self._world.collided_tokens)
            for i in range(self._num_candidates)
        }
        for i in range(1, self._num_frames + 1):
            ego_polygons = self._ego_polygons[:, i]
            intersect_indices = self._world[i].query(ego_polygons, "intersects")

            if len(intersect_indices) == 0:
                continue
            for rollout_idx, obj_idx in zip(intersect_indices[0], intersect_indices[1]):
                token = self._world[i].tokens[obj_idx]
                if token.startswith(self._world.red_light_prefix):
                    self._at_fault_collision_time[rollout_idx] = min(
                        i * self._dt, self._at_fault_collision_time[rollout_idx]
                    )
                    continue
                elif token in collided_tokens[rollout_idx]:
                    continue

                ego_in_multiple_lanes_or_nondrivable_area = (
                    self._ego_footprints[rollout_idx, i, EgoAreaIndex.MULTIPLE_LANES]
                    or self._ego_footprints[
                        rollout_idx, i, EgoAreaIndex.NON_DRIVABLE_AREA
                    ]
                )

                object_info = self._world.get_object_at_frame(token, i)

                collision_type = get_collision_type(
                    state=self._ego_rollout[rollout_idx, i],
                    ego_polygon=ego_polygons[rollout_idx],
                    object_info=object_info,
                )

                collisions_at_stopped_track_or_active_front = collision_type in [
                    CollisionType.ACTIVE_FRONT_COLLISION,
                    CollisionType.STOPPED_TRACK_COLLISION,
                ]
                collision_at_lateral = (
                    collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
                )

                if collisions_at_stopped_track_or_active_front or (
                    ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
                ):
                    collided_tokens[rollout_idx].append(token)
                    self._at_fault_collision_time[rollout_idx] = min(
                        i * self._dt, self._at_fault_collision_time[rollout_idx]
                    )
