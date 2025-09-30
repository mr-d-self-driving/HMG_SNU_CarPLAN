from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely.creation
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import (
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import Polygon

from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_object_manager import (
    PDMObjectManager,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
)


class PDMObservation:
    """PDM's observation class for forecasted occupancy maps."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        map_radius: float,
        observation_sample_res: int = 2,
    ):
        """
        Constructor of PDMObservation
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param map_radius: radius around ego to consider, defaults to 50
        :param observation_sample_res: sample resolution of forecast, defaults to 2
        """
        assert (
            trajectory_sampling.interval_length == proposal_sampling.interval_length
        ), "PDMObservation: Proposals and Trajectory must have equal interval length!"

        # observation needs length of trajectory horizon or proposal horizon +1s (for TTC metric)
        self._sample_interval: float = trajectory_sampling.interval_length  # [s]

        self._observation_samples: int = (
            proposal_sampling.num_poses + int(1 / self._sample_interval)
            if proposal_sampling.num_poses + int(1 / self._sample_interval)
            > trajectory_sampling.num_poses
            else trajectory_sampling.num_poses
        )

        self._map_radius: float = map_radius
        self._observation_sample_res: int = observation_sample_res

        # useful things
        self._global_to_local_idcs = [
            idx // observation_sample_res
            for idx in range(self._observation_samples + observation_sample_res)
        ]
        self._collided_track_ids: List[str] = []
        self._red_light_token = "red_light"

        # lazy loaded (during update)
        self._occupancy_maps: Optional[List[PDMOccupancyMap]] = None
        self._occupancy_maps_object_dynamic:  Optional[List[PDMOccupancyMap]] = []
        self._occupancy_maps_object_col_dynamic:  Optional[List[PDMOccupancyMap]] = []
        self._occupancy_maps_object_static:  Optional[List[PDMOccupancyMap]] = []
        self._occupancy_maps_traffic:  Optional[List[PDMOccupancyMap]] = []
        self._object_manager: Optional[PDMObjectManager] = None

        self._initialized: bool = False

    def __getitem__(self, time_idx) -> PDMOccupancyMap:
        """
        Retrieves occupancy map for time_idx and adapt temporal resolution.
        :param time_idx: index for future simulation iterations [10Hz]
        :return: occupancy map
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        assert (
            0 <= time_idx < len(self._global_to_local_idcs)
        ), f"PDMObservation: index {time_idx} out of range!"

        local_idx = self._global_to_local_idcs[time_idx]
        return self._occupancy_maps[local_idx]

    @property
    def collided_track_ids(self) -> List[str]:
        """
        Getter for past collided track tokens.
        :return: list of tokens
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._collided_track_ids

    @property
    def red_light_token(self) -> str:
        """
        Getter for red light token indicator
        :return: string
        """
        return self._red_light_token

    @property
    def unique_objects(self) -> Dict[str, TrackedObject]:
        """
        Getter for unique tracked objects
        :return: dictionary of tokens, tracked objects
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._object_manager.unique_objects

    def update(
        self,
        ego_state: EgoState,
        observation: Observation,
        traffic_light_data: List[TrafficLightStatusData],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ) -> None:
        """
        Update & lazy loads information  of PDMObservation.
        :param ego_state: state of ego vehicle
        :param observation: input observation of nuPlan
        :param traffic_light_data: list of traffic light states
        :param route_lane_dict: dictionary of on-route lanes
        :param map_api: map object of nuPlan
        """

        self._occupancy_maps: List[PDMOccupancyMap] = []
        self._occupancy_maps_object_dynamic: List[PDMOccupancyMap] = []
        self._occupancy_maps_object_col_dynamic: List[PDMOccupancyMap] = []
        self._occupancy_maps_object_static: List[PDMOccupancyMap] = []
        self._occupancy_maps_traffic: List[PDMOccupancyMap] = []

        self._object_manager, self._object_manager_collided = self._get_object_manager(ego_state, observation)

        (
            traffic_light_tokens,
            traffic_light_polygons,
        ) = self._get_traffic_light_geometries(traffic_light_data, route_lane_dict)

        (
            static_object_tokens,
            static_object_coords,
            dynamic_object_tokens,
            dynamic_object_coords,
            dynamic_object_dxy,
        ) = self._object_manager.get_nearest_objects(ego_state.center.point)

        (
            static_col_object_tokens,
            static_col_object_coords,
            dynamic_col_object_tokens,
            dynamic_col_object_coords,
            dynamic_col_object_dxy,
        ) = self._object_manager_collided.get_nearest_objects(ego_state.center.point)

        has_static_object, has_dynamic_object = (
            len(static_object_tokens) > 0,
            len(dynamic_object_tokens) > 0,
        )

        has_dynamic_col_object = len(dynamic_col_object_tokens) > 0

        if has_static_object and static_object_coords.ndim == 1:
            static_object_coords = static_object_coords[None, ...]

        if has_dynamic_object and dynamic_object_coords.ndim == 1:
            dynamic_object_coords = dynamic_object_coords[None, ...]
            dynamic_object_dxy = dynamic_object_dxy[None, ...]

        if has_dynamic_col_object and dynamic_col_object_coords.ndim == 1:
            dynamic_col_object_coords = dynamic_col_object_coords[None, ...]
            dynamic_col_object_dxy = dynamic_col_object_dxy[None, ...]

        if has_static_object:
            static_object_coords[..., BBCoordsIndex.CENTER, :] = static_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
            static_object_polygons = shapely.creation.polygons(static_object_coords)

        else:
            static_object_polygons = np.array([], dtype=np.object_)

        if has_dynamic_object:
            dynamic_object_coords[..., BBCoordsIndex.CENTER, :] = dynamic_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
        else:
            dynamic_object_polygons = np.array([], dtype=np.object_)
            dynamic_object_tokens = []

        if has_dynamic_col_object:
            dynamic_col_object_coords[..., BBCoordsIndex.CENTER, :] = dynamic_col_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
        else:
            dynamic_col_object_polygons = np.array([], dtype=np.object_)
            dynamic_col_object_tokens = []

        traffic_light_polygons = np.array(traffic_light_polygons, dtype=np.object_)

        for sample in np.arange(
            0,
            self._observation_samples + self._observation_sample_res, # self._observation_sample = 80 / self._observation_sample_res = 2
            self._observation_sample_res, 
        ): 
            # 8초의 future에 대한 occupancy map을 작성하는 부분
            # 0.2s의 샘플링 주기를 지님
            # Static object의 경우 위치를 고정
            # Traffic light polygon? lane connector polygon을 가져옴.. 에바지, 신호는 고정 
            # Dynamic object의 경우 현재 속도를 통해 미래 위치를 추정(lane 정보같은건 안씀), 위치를 바꾸어가며 polygon을 만들어 사용함
            if has_dynamic_object:
                delta_t = float(sample) * self._sample_interval
                dynamic_object_coords_t = (
                    dynamic_object_coords + delta_t * dynamic_object_dxy[:, None]
                )
                dynamic_object_polygons = shapely.creation.polygons(
                    dynamic_object_coords_t
                )
            
            if has_dynamic_col_object:
                delta_t = float(sample) * self._sample_interval
                dynamic_col_object_coords_t = (
                    dynamic_col_object_coords + delta_t * dynamic_col_object_dxy[:, None]
                )
                dynamic_col_object_polygons = shapely.creation.polygons(
                    dynamic_col_object_coords_t
                )
            

            all_polygons = np.concatenate(
                [
                    static_object_polygons,
                    dynamic_object_polygons,
                    traffic_light_polygons,
                ],
                axis=0,
            )

            occupancy_map = PDMOccupancyMap(
                static_object_tokens + dynamic_object_tokens + traffic_light_tokens,
                all_polygons,
            )
            self._occupancy_maps.append(occupancy_map)


            # object_polygons = np.concatenate([static_object_polygons,
            #                                 dynamic_object_polygons,],axis=0)
            # occupancy_object = PDMOccupancyMap(static_object_tokens + dynamic_object_tokens,object_polygons)
            # self._occupancy_maps_object.append(occupancy_object)

            object_polygons_dynamic = dynamic_object_polygons
            occupancy_object_dynamic = PDMOccupancyMap(dynamic_object_tokens,object_polygons_dynamic)
            self._occupancy_maps_object_dynamic.append(occupancy_object_dynamic)

            object_col_polygons_dynamic = dynamic_col_object_polygons
            occupancy_object_col_dynamic = PDMOccupancyMap(dynamic_col_object_tokens,object_col_polygons_dynamic)
            self._occupancy_maps_object_col_dynamic.append(occupancy_object_col_dynamic)

            object_polygons_static = static_object_polygons
            occupancy_object_static = PDMOccupancyMap(static_object_tokens,object_polygons_static)
            self._occupancy_maps_object_static.append(occupancy_object_static)

            traffic_polygons = traffic_light_polygons
            occupancy_traffic = PDMOccupancyMap(traffic_light_tokens,traffic_polygons)
            self._occupancy_maps_traffic.append(occupancy_traffic)


        # save collided objects to ignore in the future
        ego_polygon: Polygon = ego_state.car_footprint.geometry
        intersecting_obstacles = self._occupancy_maps[0].intersects(ego_polygon)
        new_collided_track_ids = []

        for intersecting_obstacle in intersecting_obstacles:
            if self._red_light_token in intersecting_obstacle:
                within = ego_polygon.within(
                    self._occupancy_maps[0][intersecting_obstacle]
                )
                if not within:
                    continue
            new_collided_track_ids.append(intersecting_obstacle)

        self._collided_track_ids = self._collided_track_ids + new_collided_track_ids
        self._initialized = True

    def update_w_predict(
        self,
        ego_state: EgoState,
        observation: Observation,
        traffic_light_data: List[TrafficLightStatusData],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        g_predictions_,
        tokens_pluto_pred,
        pred_agent_shapes,
        pred_agent_current_bboxes) -> None:
        """
        Update & lazy loads information  of PDMObservation.
        :param ego_state: state of ego vehicle
        :param observation: input observation of nuPlan
        :param traffic_light_data: list of traffic light states
        :param route_lane_dict: dictionary of on-route lanes
        :param map_api: map object of nuPlan
        """
        
        # 딕셔너리로 묶기
        g_prediction_dict = {}
        g_shape_dict = {}
        for agent_shape, g_pred, token in zip(pred_agent_shapes, g_predictions_, tokens_pluto_pred):
            g_prediction_dict[token] = g_pred
            g_shape_dict[token] = agent_shape
        self._occupancy_maps: List[PDMOccupancyMap] = []
        self._occupancy_maps_object_dynamic: List[PDMOccupancyMap] = []
        self._occupancy_maps_object_col_dynamic: List[PDMOccupancyMap] = []
        self._occupancy_maps_object_static: List[PDMOccupancyMap] = []
        self._occupancy_maps_traffic: List[PDMOccupancyMap] = []

        self._object_manager, self._object_manager_collided = self._get_object_manager(ego_state, observation)

        (
            traffic_light_tokens,
            traffic_light_polygons,
        ) = self._get_traffic_light_geometries(traffic_light_data, route_lane_dict)

        (
            static_object_tokens,
            static_object_coords,
            dynamic_object_tokens,
            dynamic_object_coords,
            dynamic_object_dxy,
        ) = self._object_manager.get_nearest_objects(ego_state.center.point)
                
        (
            static_col_object_tokens,
            static_col_object_coords,
            dynamic_col_object_tokens,
            dynamic_col_object_coords,
            dynamic_col_object_dxy,
        ) = self._object_manager_collided.get_nearest_objects(ego_state.center.point)
        
        # all_dynamic_tokens = list((set(dynamic_object_tokens) & set(tokens_pluto_pred))) # pluto_token과 pdm에서 감지한 object 합치기, 이때 col_object_token은 제외하기
        all_dynamic_tokens = dynamic_object_tokens

        has_static_object, has_dynamic_object = (
            len(static_object_tokens) > 0,
            len(all_dynamic_tokens) > 0,
        )

        has_dynamic_col_object = len(dynamic_col_object_tokens) > 0

        if has_static_object and static_object_coords.ndim == 1:
            static_object_coords = static_object_coords[None, ...]

        if has_dynamic_object and dynamic_object_coords.ndim == 1:
            dynamic_object_coords = dynamic_object_coords[None, ...]
            dynamic_object_dxy = dynamic_object_dxy[None, ...]

        if has_dynamic_col_object and dynamic_col_object_coords.ndim == 1:
            dynamic_col_object_coords = dynamic_col_object_coords[None, ...]
            dynamic_col_object_dxy = dynamic_col_object_dxy[None, ...]

        if has_static_object:
            static_object_coords[..., BBCoordsIndex.CENTER, :] = static_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
            static_object_polygons = shapely.creation.polygons(static_object_coords)

        else:
            static_object_polygons = np.array([], dtype=np.object_)

        if has_dynamic_object:
            dynamic_object_coords[..., BBCoordsIndex.CENTER, :] = dynamic_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
        else:
            dynamic_object_polygons = np.array([], dtype=np.object_)
            dynamic_object_tokens = []

        if has_dynamic_col_object:
            dynamic_col_object_coords[..., BBCoordsIndex.CENTER, :] = dynamic_col_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
        else:
            dynamic_col_object_polygons = np.array([], dtype=np.object_)
            dynamic_col_object_tokens = []

        traffic_light_polygons = np.array(traffic_light_polygons, dtype=np.object_)

        for sample in np.arange(
            0,
            self._observation_samples + self._observation_sample_res,
            self._observation_sample_res,
        ):
            
            if has_dynamic_object:
                polygon_lists = []
                for dynamic_token in all_dynamic_tokens:
                    if dynamic_token in tokens_pluto_pred:
                        idx = tokens_pluto_pred.index(dynamic_token)
                        ## observation의 bbox는 center_point 기준
                        # ~/tuplan_garage/tuplan_garage/planning/simulation/planner/pdm_planner/observation/pdm_object_manager.py
                        ## pluto의 prediction은 rear_axle 기준
                        # planning에서 center_axle와 rear_axle는 하늘과 땅 끝 차이
                        if sample != 0:
                            if sample < g_prediction_dict[dynamic_token].shape[0]:
                                d_loc = g_prediction_dict[dynamic_token][sample, :] - g_prediction_dict[dynamic_token][0, :]# pluto는 rear_axle 기준으로 global_location을 넘겨줌
                            else:
                                d_loc = g_prediction_dict[dynamic_token][-1, :] + (g_prediction_dict[dynamic_token][-1, :] - g_prediction_dict[dynamic_token][-2, :])
                        else:
                            d_loc = np.array([0,0,0])
                        agent_w, agent_l = list(g_shape_dict[dynamic_token])
                        # agent_current_bbox = pred_agent_current_bboxes[dynamic_token].geometry.bounds
                        x_, y_ = pred_agent_current_bboxes[dynamic_token].geometry.boundary.xy
                        agent_bbox = np.array([x_, y_]).T
                        new_agent_bbox = self.transform_bounding_box(agent_bbox, list(d_loc[:2]), d_loc[-1])
                        polygon_lists.append(new_agent_bbox)
                        polygons_array_ = np.array(polygon_lists)
                        dynamic_object_polygons = shapely.creation.polygons(polygons_array_)
                    else:
                        token_index = dynamic_object_tokens.index(dynamic_token)
                        agent_bbox = dynamic_object_coords[token_index]
                        delta_t = float(sample) * self._sample_interval
                        agent_bbox_t = (agent_bbox + delta_t * dynamic_object_dxy[token_index, None])
                        polygon_lists.append(agent_bbox_t)
                        polygons_array_ = np.array(polygon_lists)
                        dynamic_object_polygons = shapely.creation.polygons(polygons_array_)
            
            # col_object는 refinement 코드상 없는 객체로 취급됨
            if has_dynamic_col_object:
                delta_t = float(sample) * self._sample_interval
                dynamic_col_object_coords_t = (
                    dynamic_col_object_coords + delta_t * dynamic_col_object_dxy[:, None]
                )
                dynamic_col_object_polygons = shapely.creation.polygons(
                    dynamic_col_object_coords_t
                )
            

            all_polygons = np.concatenate(
                [
                    static_object_polygons,
                    dynamic_object_polygons,
                    traffic_light_polygons,
                ],
                axis=0,
            )
            try:
                occupancy_map = PDMOccupancyMap(
                    static_object_tokens + all_dynamic_tokens + traffic_light_tokens,
                    all_polygons,
                )
                self._occupancy_maps.append(occupancy_map)
            except Exception as e:
                print()

            # object_polygons = np.concatenate([static_object_polygons,
            #                                 dynamic_object_polygons,],axis=0)
            # occupancy_object = PDMOccupancyMap(static_object_tokens + dynamic_object_tokens,object_polygons)
            # self._occupancy_maps_object.append(occupancy_object)

            object_polygons_dynamic = dynamic_object_polygons
            occupancy_object_dynamic = PDMOccupancyMap(all_dynamic_tokens,object_polygons_dynamic)
            self._occupancy_maps_object_dynamic.append(occupancy_object_dynamic)

            object_col_polygons_dynamic = dynamic_col_object_polygons
            occupancy_object_col_dynamic = PDMOccupancyMap(dynamic_col_object_tokens,object_col_polygons_dynamic)
            self._occupancy_maps_object_col_dynamic.append(occupancy_object_col_dynamic)

            object_polygons_static = static_object_polygons
            occupancy_object_static = PDMOccupancyMap(static_object_tokens,object_polygons_static)
            self._occupancy_maps_object_static.append(occupancy_object_static)

            traffic_polygons = traffic_light_polygons
            occupancy_traffic = PDMOccupancyMap(traffic_light_tokens,traffic_polygons)
            self._occupancy_maps_traffic.append(occupancy_traffic)


        # save collided objects to ignore in the future
        ego_polygon: Polygon = ego_state.car_footprint.geometry
        intersecting_obstacles = self._occupancy_maps[0].intersects(ego_polygon)
        new_collided_track_ids = []

        for intersecting_obstacle in intersecting_obstacles:
            if self._red_light_token in intersecting_obstacle:
                within = ego_polygon.within(
                    self._occupancy_maps[0][intersecting_obstacle]
                )
                if not within:
                    continue
            new_collided_track_ids.append(intersecting_obstacle)

        self._collided_track_ids = self._collided_track_ids + new_collided_track_ids
        self._initialized = True
        
    def transform_bounding_box(self, bbox, translation, heading):
        translation_x, translation_y = translation

        # Compute center of the bounding box
        center_x = np.mean(bbox[:, 0])
        center_y = np.mean(bbox[:, 1])

        # Define relative corners of the bounding box
        relative_corners = bbox - np.array([center_x, center_y])

        # Rotation matrix
        rotation_matrix = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading),  np.cos(heading)],
        ])

        # Apply rotation and translation
        rotated_translated_corners = (
            np.dot(relative_corners, rotation_matrix.T)
            + [center_x + translation_x, center_y + translation_y]
        )

        return rotated_translated_corners


    def _get_object_manager(
        self, ego_state: EgoState, observation: Observation
    ) -> PDMObjectManager:
        """
        Creates object manager class, but adding valid tracked objects.
        :param ego_state: state of ego-vehicle
        :param observation: input observation of nuPlan
        :return: PDMObjectManager class
        """
        object_manager = PDMObjectManager()
        object_manager_collided = PDMObjectManager() # bh가 추가한 객체

        for object in observation.tracked_objects:
            if (
                (object.tracked_object_type == TrackedObjectType.EGO)
                or (
                    self._map_radius
                    and ego_state.center.distance_to(object.center) > self._map_radius
                )
                or (object.track_token in self._collided_track_ids)
            ):
                ## BH
                if object.track_token in self._collided_track_ids:
                    object_manager_collided.add_object(object)
                else:
                    continue
                ##BH
                #continue # bh 수정전


            object_manager.add_object(object)

        return object_manager, object_manager_collided

    def _get_traffic_light_geometries(
        self,
        traffic_light_data: List[TrafficLightStatusData],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ) -> Tuple[List[str], List[Polygon]]:
        """
        Collects red traffic lights along ego's route.
        :param traffic_light_data: list of traffic light states
        :param route_lane_dict: dictionary of on-route lanes
        :return: tuple of tokens and polygons of red traffic lights
        """
        traffic_light_tokens, traffic_light_polygons = [], []

        for data in traffic_light_data:
            lane_connector_id = str(data.lane_connector_id)

            if (data.status == TrafficLightStatusType.RED) and (
                lane_connector_id in route_lane_dict.keys()
            ):
                lane_connector = route_lane_dict[lane_connector_id]
                traffic_light_tokens.append(
                    f"{self._red_light_token}_{lane_connector_id}"
                )
                traffic_light_polygons.append(lane_connector.polygon)

        return traffic_light_tokens, traffic_light_polygons
