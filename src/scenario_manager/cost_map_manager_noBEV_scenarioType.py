import itertools
from typing import Dict, List, Set

import cv2
import numpy as np
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from scipy import ndimage
from shapely import Polygon

DA = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
# DA2 = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR, SemanticMapLayer.CARPARK_AREA, SemanticMapLayer.WALKWAYS, SemanticMapLayer.CROSSWALK]


#JY
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

class CostMapManager:
    def __init__(
        self,
        origin: np.ndarray,
        angle: float,
        map_api: AbstractMap,
        height: int = 500,
        width: int = 500,
        resolution: float = 0.2,
    ) -> None:
        self.map_api = map_api
        self.height = height
        self.width = width
        self.resolution = resolution
        self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
        self.origin = origin
        self.angle = angle
        self.offset = np.array([height / 2, width / 2], dtype=np.float32)
        self.rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float64,
        )

    @classmethod
    def from_scenario(cls, scenario: AbstractScenario):
        ego_state = scenario.initial_ego_state
        origin = ego_state.rear_axle.point.array
        angle = ego_state.rear_axle.heading

        return cls(origin=origin, angle=angle, map_api=scenario.map_api)

    def build_cost_maps(
        self,
        static_objects: list[StaticObject],
        agents: Dict[str, np.ndarray] = None,
        agents_polygon: List[Polygon] = None,
        route_roadblock_ids: Set[str] = None,
    ):
        drivable_area_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        speed_limit_mask = np.zeros((self.height, self.width), dtype=np.float32)

        radius = max(self.height, self.width) * self.resolution / 2
        da_objects_dict = self.map_api.get_proximal_map_objects(
            Point2D(*self.origin), radius, DA
        )
        da_objects = itertools.chain.from_iterable(da_objects_dict.values())

        for obj in da_objects:
            self.fill_polygon(drivable_area_mask, obj.polygon, value=1)

            speed_limit_mps = obj.speed_limit_mps if obj.speed_limit_mps else 50
            self.fill_polygon(speed_limit_mask, obj.polygon, value=speed_limit_mps)

        for static_ojb in static_objects:
            if np.linalg.norm(static_ojb.center.array - self.origin, axis=-1) > radius:
                continue
            self.fill_convex_polygon(
                drivable_area_mask, static_ojb.box.geometry, value=0
            )
        
        if agents is not None:
            # parking vehicles as static obstacles
            position = agents["position"]
            valid_mask = agents["valid_mask"]
            for pos, mask, polygon in zip(position, valid_mask, agents_polygon):
                if mask.sum() < 50:
                    continue
                pos = pos[mask]
                displacement = np.linalg.norm(pos[-1] - pos[0])
                if displacement < 1.0:
                    self.fill_convex_polygon(drivable_area_mask, polygon, value=0)
        
        # import matplotlib.pyplot as plt
        # fig1, axes1 = plt.subplots(1, 1, figsize=(10, 10))
        # axes1.imshow(drivable_area_mask, cmap="binary")
        # plt.tight_layout() 
        # plt.savefig("/media/hdd/jyyun/ai/pluto/vis/drivable_mask.png")

        distance = ndimage.distance_transform_edt(drivable_area_mask)
        inv_distance = ndimage.distance_transform_edt(1 - drivable_area_mask)
        drivable_area_sdf = distance - inv_distance
        drivable_area_sdf *= self.resolution

        return {
            "cost_maps": drivable_area_sdf[:, :, None].astype(np.float16),  # (H, W. C)
        }

    #JY
    def get_drivable_area_bev_map(
        self,
        static_objects: list[StaticObject],
        agents: Dict[str, np.ndarray] = None,
        agents_polygon: List[Polygon] = None,
        occ_size = 128,
        occ_range = 50,
        occ_size_low = 128,
        occ_range_low = 50,
        is_make_extra_res=False,
        is_front2rear=False,
        radius=120
    ):
        
        if is_front2rear:
            resolution = occ_range / occ_size
        
            self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
            self.offset = np.array([51, 128], dtype=np.float32)
        else:
            resolution = occ_range / (occ_size//2)
            
            self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
            self.offset = np.array([occ_size / 2, occ_size / 2], dtype=np.float32)
        
        drivable_area_mask = np.zeros((occ_size, occ_size), dtype=np.uint8)
        
        da_objects_dict = self.map_api.get_proximal_map_objects(
            Point2D(*self.origin), radius, DA
        )        
        da_objects = itertools.chain.from_iterable(da_objects_dict.values())

        for obj in da_objects:
            self.fill_polygon(drivable_area_mask, obj.polygon, value=1)

        for static_ojb in static_objects:
            if np.linalg.norm(static_ojb.center.array - self.origin, axis=-1) > occ_range:
                continue
            self.fill_convex_polygon(
                drivable_area_mask, static_ojb.box.geometry, value=0
            )
        
        drivable_area_mask_noAgent = drivable_area_mask.copy()
        
        if agents is not None:
            # parking vehicles as static obstacles
            position = agents["position"]
            valid_mask = agents["valid_mask"]
            for pos, mask, polygon in zip(position, valid_mask, agents_polygon):
                if mask.sum() < ((mask.shape[0]-1)//2):
                    continue
                pos = pos[mask]
                displacement = np.linalg.norm(pos[-1] - pos[0])
                if displacement < 1.0:
                    self.fill_convex_polygon(drivable_area_mask, polygon, value=0)
        
        if is_make_extra_res:
            if is_front2rear:
                resolution = occ_range_low / occ_size_low
            
                self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
                self.offset = np.array([51, 128], dtype=np.float32)
            else:
                resolution = occ_range_low / (occ_size_low//2)
                
                self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
                self.offset = np.array([occ_size_low / 2, occ_size_low / 2], dtype=np.float32)
            
            drivable_area_mask_low = np.zeros((occ_size_low, occ_size_low), dtype=np.uint8)
            
            da_objects = itertools.chain.from_iterable(da_objects_dict.values())
            
            for obj in da_objects:
                self.fill_polygon(drivable_area_mask_low, obj.polygon, value=1)

            for static_ojb in static_objects:
                if np.linalg.norm(static_ojb.center.array - self.origin, axis=-1) > occ_range_low:
                    continue
                self.fill_convex_polygon(
                    drivable_area_mask_low, static_ojb.box.geometry, value=0
                )
            
            drivable_area_mask_noAgent_low = drivable_area_mask_low.copy()
            
            if agents is not None:
                # parking vehicles as static obstacles
                position = agents["position"]
                valid_mask = agents["valid_mask"]
                for pos, mask, polygon in zip(position, valid_mask, agents_polygon):
                    if mask.sum() < ((mask.shape[0]-1)//2):
                        continue
                    pos = pos[mask]
                    displacement = np.linalg.norm(pos[-1] - pos[0])
                    if displacement < 1.0:
                        self.fill_convex_polygon(drivable_area_mask_low, polygon, value=0)
        else:
            drivable_area_mask_low, drivable_area_mask_noAgent_low = None, None

        # import matplotlib.pyplot as plt
        # fig1, axes1 = plt.subplots(1, 1, figsize=(10, 10))
        # axes1.imshow(drivable_area_mask_noAgent, cmap="binary")
        # plt.tight_layout() 
        # plt.savefig("/home/jyyun/workshop/spa_pluto_CVPR/tmp/drivable_area_mask_noAgent.png")

        return drivable_area_mask, drivable_area_mask_noAgent, drivable_area_mask_low, drivable_area_mask_noAgent_low

    #JY
    def get_route_bev_map(
        self,
        scenario_manager,
        occ_size,
        occ_range,
        occ_size_low,
        occ_range_low,
        is_make_extra_res,
        is_front2rear
    ):
        
        current_lane = scenario_manager._route_manager._get_starting_lane(scenario_manager._ego_state) #JY #현재 위치에 있는 current lane
        roadblocks = list(scenario_manager._route_manager._route_roadblock_dict.values()) 
        roadblock_ids = list(scenario_manager._route_manager._route_roadblock_dict.keys())
        
        # find current roadblock index
        start_idx = np.argmax(
            np.array(roadblock_ids) == current_lane.get_roadblock_id()
        )
        roadblock_window = roadblocks[start_idx :]
        
        if is_front2rear:
            resolution = occ_range / occ_size
        
            self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
            self.offset = np.array([51, 128], dtype=np.float32)
        else:
            resolution = occ_range / (occ_size//2)
            
            self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
            self.offset = np.array([occ_size / 2, occ_size / 2], dtype=np.float32)
        
        route_mask = np.zeros((occ_size, occ_size), dtype=np.uint8)
        route_lanes_mask = np.zeros((occ_size, occ_size), dtype=np.uint8)

        for obj in roadblock_window:
            self.fill_polygon(route_mask, obj.polygon, value=1)
            
        for route in roadblock_window:
            for lane in route.interior_edges:
                self.fill_polyline(route_lanes_mask, np.stack(lane.baseline_path.linestring.coords.xy, axis=1), value=1)
                
        if is_make_extra_res:
            if is_front2rear:
                resolution = occ_range_low / occ_size_low
            
                self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
                self.offset = np.array([51, 128], dtype=np.float32)
            else:
                resolution = occ_range_low / (occ_size_low//2)
                
                self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
                self.offset = np.array([occ_size_low / 2, occ_size_low / 2], dtype=np.float32)
            
            route_mask_low = np.zeros((occ_size_low, occ_size_low), dtype=np.uint8)
            route_lanes_mask_low = np.zeros((occ_size_low, occ_size_low), dtype=np.uint8)

            for obj in roadblock_window:
                self.fill_polygon(route_mask_low, obj.polygon, value=1)
                
            for route in roadblock_window:
                for lane in route.interior_edges:
                    self.fill_polyline(route_lanes_mask_low, np.stack(lane.baseline_path.linestring.coords.xy, axis=1), value=1)
        else:
            route_mask_low, route_lanes_mask_low = None, None
            
        # import matplotlib.pyplot as plt
        # fig1, axes1 = plt.subplots(1, 1, figsize=(10, 10))
        # axes1.imshow(route_mask, cmap="binary")
        # plt.tight_layout() 
        # plt.savefig("/home/jyyun/workshop/spa_pluto_CVPR/tmp/route_mask.png")
        
        return route_mask, route_lanes_mask, route_mask_low, route_lanes_mask_low
    
    #JY
    def get_reference_lane_bev_map(
        self,
        merged_polygons,
        occ_size,
        occ_range,
        occ_size_low,
        occ_range_low,
        is_make_extra_res,
        is_front2rear
    ):
        
        if is_front2rear:
            resolution = occ_range / occ_size
        
            self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
            self.offset = np.array([51, 128], dtype=np.float32)
        else:
            resolution = occ_range / (occ_size//2)
            
            self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
            self.offset = np.array([occ_size / 2, occ_size / 2], dtype=np.float32)
        
        reference_lane_mask = np.zeros((len(merged_polygons), occ_size, occ_size), dtype=np.uint8)

        for idx, merged_polygon in enumerate(merged_polygons):
            for obj in merged_polygon:
                self.fill_polygon(reference_lane_mask[idx], obj, value=1)
                
        if is_make_extra_res:
            if is_front2rear:
                resolution = occ_range_low / occ_size_low
            
                self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
                self.offset = np.array([51, 128], dtype=np.float32)
            else:
                resolution = occ_range_low / (occ_size_low//2)
                
                self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
                self.offset = np.array([occ_size_low / 2, occ_size_low / 2], dtype=np.float32)
            
            reference_lane_mask_low = np.zeros((len(merged_polygons), occ_size_low, occ_size_low), dtype=np.uint8)

            for idx, merged_polygon in enumerate(merged_polygons):
                for obj in merged_polygon:
                    self.fill_polygon(reference_lane_mask_low[idx], obj, value=1)
        else:
            reference_lane_mask_low = None
            
        # import matplotlib.pyplot as plt
        # for i in range(len(merged_polygons)):  
        #     fig1, axes1 = plt.subplots(1, 1, figsize=(10, 10))
        #     axes1.imshow(reference_lane_mask[i], cmap="binary")
        #     plt.tight_layout() 
        #     plt.savefig(f"/home/jyyun/workshop/spa_pluto_CVPR/tmp/reference_lane_mask_{i}.png")

        return reference_lane_mask, reference_lane_mask_low
    
    #JY
    def get_AgentAV_bev_map(
        self,
        av_states,
        agent_states,
        query_xy,
        present_idx,
        occ_size,
        occ_range,
        is_make_extra_res,
        is_front2rear,
        max_agents
    ):
        if is_front2rear:
            resolution = occ_range / occ_size
        
            self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
            self.offset = np.array([51, 128], dtype=np.float32)
        else:
            resolution = occ_range / (occ_size//2)
            
            self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
            self.offset = np.array([occ_size / 2, occ_size / 2], dtype=np.float32)
        
        interested_objects_types = [
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]
        
        T = len(av_states)

        av_temporal_mask = np.zeros((T, occ_size, occ_size, 3), dtype=np.uint8)
        agent_temporal_mask = np.zeros((T, occ_size, occ_size, 3), dtype=np.uint8)
        vehicle_temporal_mask = np.zeros((T, occ_size, occ_size, 3), dtype=np.uint8)
        pedestrian_temporal_mask = np.zeros((T, occ_size, occ_size, 3), dtype=np.uint8)
        cyclist_temporal_mask = np.zeros((T, occ_size, occ_size, 3), dtype=np.uint8)
        occupancy_av_pixel_indx = np.ones((1, occ_size, occ_size), dtype=np.float32)
        occupancy_av_pixel_indx_padding = np.zeros((1, occ_size, occ_size), dtype=np.bool)
        occupancy_agent_pixel_indx = np.zeros((1, occ_size, occ_size), dtype=np.float32)
        occupancy_agent_pixel_indx_padding = np.zeros((1, occ_size, occ_size), dtype=np.bool)
        
        for t, state in enumerate(av_states):
            polygon = state.agent.box.geometry
            self.fill_polygon(av_temporal_mask[t], polygon, value=(1, 2, 0))
            
            #JY
            av_pixel_idx = np.where(av_temporal_mask[t, :, :, 1] == 2)
            if t != present_idx:
                pass
            else:
                occupancy_av_pixel_indx[0, av_pixel_idx[0], av_pixel_idx[1]] = 0 * np.ones(av_pixel_idx[0].shape)
                occupancy_av_pixel_indx_padding[0, av_pixel_idx[0], av_pixel_idx[1]] = True
                
            av_temporal_mask[t, av_pixel_idx[0], av_pixel_idx[1], 1] = 0
            
        present_tracked_objects = agent_states[present_idx]
        present_agents = present_tracked_objects.get_tracked_objects_of_types(
            interested_objects_types
        )
        
        N, T = min(len(present_agents), max_agents), len(agent_states)
        if N == 0:
            return av_temporal_mask[:, :, :, 0], agent_temporal_mask[:, :, :, 0], vehicle_temporal_mask[:, :, :, 0], pedestrian_temporal_mask[:, :, :, 0], cyclist_temporal_mask[:, :, :, 0], \
                occupancy_av_pixel_indx, occupancy_av_pixel_indx_padding, occupancy_agent_pixel_indx, occupancy_agent_pixel_indx_padding
                
        if is_front2rear:
            resolution = occ_range / occ_size
        
            self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
            self.offset = np.array([51, 128], dtype=np.float32)
        else:
            resolution = occ_range / (occ_size//2)
            
            self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
            self.offset = np.array([occ_size / 2, occ_size / 2], dtype=np.float32)
            
        agent_ids = np.array([agent.track_token for agent in present_agents])
        agent_cur_pos = np.array([agent.center.array for agent in present_agents])
        distance = np.linalg.norm(agent_cur_pos - query_xy.array[None, :], axis=1)
        agent_ids_sorted = ["ego"] + list(agent_ids[np.argsort(distance)[: max_agents]]) #JY 거리순으로 최대 48개의 주변 agent 정보 가져오기
        agent_ids_dict = {agent_id: i for i, agent_id in enumerate(agent_ids_sorted)}
        
        for t, tracked_objects in enumerate(agent_states):
            for agent in tracked_objects.get_tracked_objects_of_types(
                interested_objects_types
            ):
                if agent.track_token not in agent_ids_dict:
                    continue
                
                idx = agent_ids_dict[agent.track_token]
                polygon = agent.box.geometry
                self.fill_polygon(agent_temporal_mask[t], polygon, value=(1, 2, 0))
                
                if agent.tracked_object_type == TrackedObjectType.VEHICLE:
                    self.fill_polygon(vehicle_temporal_mask[t], polygon, value=(1, 0, 0))
                elif agent.tracked_object_type == TrackedObjectType.PEDESTRIAN:
                    self.fill_polygon(pedestrian_temporal_mask[t], polygon, value=(1, 0, 0))
                elif agent.tracked_object_type == TrackedObjectType.BICYCLE:
                    self.fill_polygon(cyclist_temporal_mask[t], polygon, value=(1, 0, 0))
                
                #JY
                agent_pixel_idx = np.where(agent_temporal_mask[t, :, :, 1] == 2)
                if t != present_idx:
                    pass
                else:
                    occupancy_agent_pixel_indx[0, agent_pixel_idx[0], agent_pixel_idx[1]] = idx * np.ones(agent_pixel_idx[0].shape)
                    occupancy_agent_pixel_indx_padding[0, agent_pixel_idx[0], agent_pixel_idx[1]] = True
                    
                agent_temporal_mask[t, agent_pixel_idx[0], agent_pixel_idx[1], 1] = 0
    
        # import matplotlib.pyplot as plt
        # fig1, axes1 = plt.subplots(1, 1, figsize=(10, 10))
        # for time in range(0, 101, 10):
        #     axes1.imshow(agent_temporal_mask[time, :, :, 0], cmap="binary")
        #     plt.tight_layout() 
        #     plt.savefig(f"/home/jyyun/workshop/spa_pluto_CVPR/tmp/agent_temporal_mask_{time}.png")
            
        # import matplotlib.pyplot as plt
        # fig1, axes1 = plt.subplots(1, 1, figsize=(10, 10))
        # for time in range(0, 101, 10):
        #     axes1.imshow(av_temporal_mask[time, :, :, 0], cmap="binary")
        #     plt.tight_layout() 
        #     plt.savefig(f"/home/jyyun/workshop/spa_pluto_CVPR/tmp/av_temporal_mask_{time}.png")

        return av_temporal_mask[:, :, :, 0], agent_temporal_mask[:, :, :, 0], vehicle_temporal_mask[:, :, :, 0], pedestrian_temporal_mask[:, :, :, 0], cyclist_temporal_mask[:, :, :, 0], \
            occupancy_av_pixel_indx, occupancy_av_pixel_indx_padding, occupancy_agent_pixel_indx, occupancy_agent_pixel_indx_padding
    
    def global_to_pixel(self, coord: np.ndarray):
        coord = np.matmul(coord - self.origin, self.rot_mat)
        coord = coord / self.resolution_hw + self.offset
        return coord

    def fill_polygon(self, mask, polygon, value=1):
        polygon = self.global_to_pixel(np.stack(polygon.exterior.coords.xy, axis=1))
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], value)

    def fill_convex_polygon(self, mask, polygon, value=1):
        polygon = self.global_to_pixel(np.stack(polygon.exterior.coords.xy, axis=1))
        cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), value)

    def fill_polyline(self, mask, polyline, value=1):
        polyline = self.global_to_pixel(polyline)
        cv2.polylines(
            mask,
            [np.round(polyline.reshape(-1, 1, 2)).astype(np.int32)],
            isClosed=False,
            color=value,
            thickness=1,
        )
