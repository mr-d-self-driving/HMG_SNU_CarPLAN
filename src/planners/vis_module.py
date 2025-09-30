from nuplan.common.actor_state.ego_state import EgoState
from typing import List, Optional, Type

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.route_utils import route_roadblock_correction
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling 
from nuplan.common.maps.maps_datatypes import SemanticMapLayer 
from nuplan.common.actor_state.state_representation import Point2D
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
from PIL import Image, ImageDraw, ImageFont 
import gc 
from itertools import chain 
import numpy as np
import torch
import os
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)

class main_vis_tool():
    def __init__(self):
        self._observation=None
        self._map_api=None
        self.out=None
        self.scenario=None
        self.planner_input=None
        vis_setting = {}
        vis_setting['About_vis'] ={}
        vis_setting['About_vis']["draw_range"]= 60 # 40-> fig1용, 60 -> fig2용
        vis_setting['About_vis']["fig_size"]=16
        vis_setting['About_vis']["vis_lane_activate"]=False
        vis_setting['About_vis']["vis_lane_centerline_activate"]=True
        vis_setting['About_vis']["vis_ego_history_activate"]=False
        vis_setting['About_vis']["vis_others_history_activate"]=False
        vis_setting['About_vis']["vis_current_occupancy_activate"]=True
        vis_setting['About_vis']["vis_future_occupancy_activate"]=False
        vis_setting['About_vis']["vis_occupied_object"]=True
        vis_setting['About_vis']["vis_current_traffic_activate"]=True
        vis_setting['About_vis']["vis_GT"]=True
        vis_setting['About_vis']["vis_centerline"]=False
        vis_setting['About_vis']["vis_ego_future"]=True
        vis_setting['About_vis']["vis_legend"]=False
        vis_setting['About_vis']["vis_inform_obj"]=False
        vis_setting['About_vis']["vis_attn_obj"]=False
        vis_setting['About_vis']["vis_plan_candidate"]=True
        vis_setting['About_vis']["goal_route"]=True
        vis_setting['About_vis']['vis_one_prediction']=False
        vis_setting['About_vis']['vis_all_prediction']=True
        vis_setting['About_vis']['vis_front_agent']=False
        vis_setting['About_vis']['split_front_goal']=True
        vis_setting['About_vis']['disp_agent']=False
        vis_setting['About_vis']['disp_map']=False
        vis_setting['About_vis']['RB_color'] = "#E9F1F7" #"#F5F9FC" ##CDF6CD
        vis_setting['About_vis']['RBC_color'] = "#D4DCDF"#"#D4DCDF" ##A7F0A7
        vis_setting['About_vis']['route_color'] = "#D7F2E7"#"#D7F2E7" #skyblue
        vis_setting['About_vis']['route_alpha'] = 0.8
        vis_setting['About_vis']['AV_color'] = "#993330"
        vis_setting['About_vis']['GT_traj_color'] = "#FEE079"  #"#686CF0"-> 오리지날,  #D6BBEB->보라색
        vis_setting['About_vis']['plan_color'] = "#D99694" #"#29A6E3"-> 오리지날,  #376092->초록색
        vis_setting['About_vis']['agent_color'] = 'black'
        vis_setting['About_vis']['pedes_color'] = 'black' # '#9800ED'
        vis_setting['About_vis']['pred_minimum_len'] = 4
        vis_setting['About_vis']['pred_color'] =  "#D99694"# "#6CBA94"
        vis_setting["chanllenge"]="closed_loop_nonreactive_agents"
        vis_setting['About_vis']['vis_title'] = False
        self.vis_setting = vis_setting
        
    def _vis(self, current_input, traj, scenario_token, iteration, save_path):
        vis_pack = {}
        vis_setting = self.vis_setting
        current_ego_state: EgoState = current_input.history.ego_states[-1]
        current_pose: StateSE2 = current_ego_state.center
        current_ego_global_pose: Point2D = Point2D(current_pose.x, current_pose.y)
        occupancy = self._observation

        LANE_LAYER = [SemanticMapLayer.LANE,
                        SemanticMapLayer.LANE_CONNECTOR,
                        SemanticMapLayer.INTERSECTION,
                        SemanticMapLayer.ROADBLOCK,
                        SemanticMapLayer.ROADBLOCK_CONNECTOR,
                        SemanticMapLayer.STOP_LINE,
                        SemanticMapLayer.CROSSWALK,
                        SemanticMapLayer.WALKWAYS,
                        SemanticMapLayer.CARPARK_AREA,
                        SemanticMapLayer.CROSSWALK]
        draw_range = vis_setting['About_vis']['draw_range']
        LANE_obj = self._map_api.get_proximal_map_objects(current_ego_global_pose, draw_range, LANE_LAYER)

        LANES = LANE_obj[list(LANE_obj.keys())[0]]
        LANE_CON = LANE_obj[list(LANE_obj.keys())[1]]
        ROADBLOCKS = LANE_obj[list(LANE_obj.keys())[3]]
        ROADBLOCK_CONS = LANE_obj[list(LANE_obj.keys())[4]]
        roadblock_poly_RB = [RB.polygon for RB in ROADBLOCKS]
        roadblock_poly_RBC = [RB.polygon for RB in ROADBLOCK_CONS]
        if vis_setting['About_vis']["goal_route"]:
            route_poly = [RB.polygon for RB in ROADBLOCKS if RB.id in list(self._route_roadblock_dict.keys())]
            route_poly = [RB.polygon for RB in ROADBLOCK_CONS if RB.id in list(self._route_roadblock_dict.keys())] + route_poly
            vis_pack['route'] = route_poly

        LANE_poly = [lane.polygon for lane in LANES]
        LANE_poly = [lane_con.polygon for lane_con in LANE_CON] + LANE_poly
        
        Lane_line = [lane.baseline_path.discrete_path for lane in LANES]
        Lane_line = [lane.baseline_path.discrete_path for lane in LANE_CON] + Lane_line
        Lane_line = list(chain(*Lane_line))
        Lane_line = np.array([[elem.x, elem.y] for elem in Lane_line])
        vis_pack['roadblock_poly_RB'] = roadblock_poly_RB
        vis_pack['roadblock_poly_RBC'] = roadblock_poly_RBC
        vis_pack['lane'] = LANE_poly


        agent_w = current_ego_state.agent.box.length
        agent_h = current_ego_state.agent.box.width #JY: width 써야되는거 아니야?
        # agent_h += 0.2 #JY: 왜 더하기 0.2?
        fig = plt.figure(figsize=(vis_setting['About_vis']['fig_size'],vis_setting['About_vis']['fig_size']))
        plot_fig = fig.add_subplot(1, 1, 1)
        plot_fig.axis(xmin=-draw_range, xmax=+draw_range)
        plot_fig.axis(ymin=-draw_range, ymax=+draw_range)
        
        for local_route_ in vis_pack['roadblock_poly_RB']:
            x_, y_ = local_route_.exterior.xy
            tmp_arr = np.concatenate((np.array([x_, y_]).T, np.zeros((len(x_), 1))), axis=1)
            tmp_arr = self._global_to_local(tmp_arr, current_ego_state)[:, :2]
            x_, y_ = tmp_arr[:,0], tmp_arr[:,1]
            plot_fig.fill(x_, y_, color=vis_setting['About_vis']['RB_color'], alpha=1, zorder=0)
        for local_route_ in vis_pack['roadblock_poly_RBC']:
            x_, y_ = local_route_.exterior.xy
            tmp_arr = np.concatenate((np.array([x_, y_]).T, np.zeros((len(x_), 1))), axis=1)
            tmp_arr = self._global_to_local(tmp_arr, current_ego_state)[:, :2]
            x_, y_ = tmp_arr[:,0], tmp_arr[:,1]
            plot_fig.fill(x_, y_, color=vis_setting['About_vis']['RBC_color'], alpha=1, zorder=0)
        if vis_setting['About_vis']["goal_route"]:
            for local_route_ in vis_pack['route']:
                if vis_setting['About_vis']['split_front_goal']:
                    x_, y_ = local_route_.exterior.xy
                    tmp_pos = torch.stack([torch.tensor(x_),torch.tensor(y_)]).permute(1, 0)
                    tmp_pos = torch.concat((tmp_pos, tmp_pos.new_zeros(tmp_pos.shape[0], 1)), dim=-1)
                    tmp_pos = np.array(tmp_pos)
                    tmp_pos = self._global_to_local(tmp_pos, current_ego_state)
                    tmp_valid_mask = tmp_pos[:, 0] > 0
                    if tmp_valid_mask.all()==False and tmp_valid_mask.any():
                        x__ = np.array(x_)[tmp_valid_mask]
                        y__ = np.array(y_)[tmp_valid_mask]
                        x_ = np.concatenate((x__, x__[:1]), axis=-1)
                        y_ = np.concatenate((y__, y__[:1]), axis=-1)
                        tmp_arr = np.concatenate((np.array([x_, y_]).T, np.zeros((len(x_), 1))), axis=1)
                        tmp_arr = self._global_to_local(tmp_arr, current_ego_state)[:, :2]
                        x_, y_ = tmp_arr[:,0], tmp_arr[:,1]
                        plot_fig.fill(x_, y_, color=vis_setting['About_vis']['route_color'], \
                                    alpha=vis_setting['About_vis']['route_alpha'], zorder=0)    
                    else:
                        tmp_arr = np.concatenate((np.array([x_, y_]).T, np.zeros((len(x_), 1))), axis=1)
                        tmp_arr = self._global_to_local(tmp_arr, current_ego_state)[:, :2]
                        x_, y_ = tmp_arr[:,0], tmp_arr[:,1]
                        plot_fig.fill(np.array(x_)[tmp_valid_mask], np.array(y_)[tmp_valid_mask], color=vis_setting['About_vis']['route_color'], \
                                    alpha=vis_setting['About_vis']['route_alpha'], zorder=0)    
                else:
                    x_, y_ = local_route_.exterior.xy
                    tmp_arr = np.concatenate((np.array([x_, y_]).T, np.zeros((len(x_), 1))), axis=1)
                    tmp_arr = self._global_to_local(tmp_arr, current_ego_state)[:, :2]
                    x_, y_ = tmp_arr[:,0], tmp_arr[:,1]
                    plot_fig.fill(x_, y_, color=vis_setting['About_vis']['route_color'], \
                                alpha=vis_setting['About_vis']['route_alpha'], zorder=0)    

        rect = patches.Rectangle(( -agent_w / 2, -agent_h / 2), agent_w, agent_h, linewidth=1,
                            edgecolor=vis_setting['About_vis']['AV_color'], facecolor=vis_setting['About_vis']['AV_color'], alpha = 1, zorder=20)
        ax = plt.gca()
        t = patches.transforms.Affine2D().rotate_around(0, 0, 0) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        if vis_setting['About_vis']['vis_front_agent']:
            vec_x_ = (agent_w/2 - 0.7)*np.cos(0)
            vec_y_ = (agent_w/2 - 0.7)*np.sin(0)
            vec_x = (agent_w/2)*np.cos(0)
            vec_y = (agent_w/2)*np.sin(0)
            plt.annotate(f"", xy=(+vec_x, +vec_y), \
                         xytext=(+vec_x_, +vec_y_), \
                            arrowprops=dict(arrowstyle="->", color="white", lw=2, mutation_scale=18), zorder=21)

        if vis_setting['About_vis']["vis_ego_future"]:
            traj = self._global_to_local(traj, current_ego_state)
            plt.plot(traj[:, 0], traj[:, 1], linewidth=4, color=vis_setting['About_vis']['plan_color'], zorder=10, label="ego_future", alpha=1)

        if vis_setting['About_vis']['vis_lane_activate']:
            for local_route_ in vis_pack['lane']:
                x_, y_ = local_route_.exterior.xy
                tmp_arr = np.concatenate((np.array([x_, y_]).T, np.zeros((len(x_), 1))), axis=1)
                tmp_arr = self._global_to_local(tmp_arr, current_ego_state)[:, :2]
                x_, y_ = tmp_arr[:,0], tmp_arr[:,1]
                plot_fig.fill(x_, y_, color='lightgreen', alpha=0.20, zorder=1)
    
        if vis_setting['About_vis']['vis_lane_centerline_activate']:
            Lane_line_ = np.concatenate((Lane_line, np.zeros((len(Lane_line), 1))), axis=1)
            Lane_line_ = self._global_to_local(Lane_line_, current_ego_state)[:, :2]
            plt.scatter(Lane_line_[::3, 0], Lane_line_[::3, 1], s=0.3, zorder=2, color="gray")

        if vis_setting['About_vis']['vis_ego_history_activate']:
            ego_hist_traj = np.array([[state.rear_axle.x, state.rear_axle.y] for state in current_input.history.ego_states])
            plt.plot(ego_hist_traj[:, 0], ego_hist_traj[:, 1], linewidth=4, color="black", zorder=3, alpha=0.4, label="ego_history")
            plt.scatter(ego_hist_traj[:, 0], ego_hist_traj[:, 1], s=3, color="black", zorder=3, alpha=0.2)

        if vis_setting['About_vis']['disp_agent']:
            current_occ = occupancy._occupancy_maps_object_dynamic[0]
            current_observe = current_input.history.observations[-1].tracked_objects.tracked_objects
            valid_track_token_vehicle = [elem.track_token for elem in current_observe if elem._tracked_object_type.name == 'VEHICLE']
            valid_track_token_pedestrian = [elem.track_token for elem in current_observe if elem._tracked_object_type.name == 'PEDESTRIAN']
            all_current_obs = current_input.history.observations[-1].tracked_objects.tracked_objects
            agent_pos = [[elem.center.x, elem.center.y, elem.center.heading] for \
                        elem in all_current_obs if elem._tracked_object_type.name=='VEHICLE']
            agent_pos = np.array(agent_pos)
            pedes_pos = [[elem.center.x, elem.center.y, elem.center.heading] for \
                        elem in all_current_obs if elem._tracked_object_type.name=='PEDESTRIAN']
            pedes_pos = np.array(pedes_pos)
            agent_pos = self._global_to_local(agent_pos, current_ego_state)[:, :2]
            pedes_pos = self._global_to_local(pedes_pos, current_ego_state)[:, :2]
            agent_pos_alphas = (1/np.linalg.norm(agent_pos, axis=1))/max(1/np.linalg.norm(agent_pos, axis=1))
            pedes_pos_alphas = (1/np.linalg.norm(pedes_pos, axis=1))/max(1/np.linalg.norm(pedes_pos, axis=1))
            agent_pos = np.concatenate((np.zeros_like(np.expand_dims(agent_pos, 1)), np.expand_dims(agent_pos, 1)), axis=1)
            pedes_pos = np.concatenate((np.zeros_like(np.expand_dims(pedes_pos, 1)), np.expand_dims(pedes_pos, 1)), axis=1)
            for agent_pos_alpha, distance__, token in zip(agent_pos_alphas, agent_pos, valid_track_token_vehicle):
                if token in current_occ.tokens:
                    plt.plot(distance__[:, 0], distance__[:, 1], color="red", alpha=agent_pos_alpha/2, zorder=3)
            for pedes_pos_alpha, distance__, token in zip(pedes_pos_alphas, pedes_pos, valid_track_token_pedestrian):
                if token in current_occ.tokens:
                    plt.plot(distance__[:, 0], distance__[:, 1], color="orange", alpha=pedes_pos_alpha/2, zorder=3)
        if vis_setting['About_vis']['disp_map']:
            Lane_line_ = np.concatenate((Lane_line[::100, :], np.zeros((Lane_line[::100, :].shape[0],1))), axis=1)
            Lane_line_ = self._global_to_local(Lane_line_, current_ego_state)[:, :2]
            Lane_pos_alphas = (1/np.linalg.norm(Lane_line_, axis=1))/max(1/np.linalg.norm(Lane_line_, axis=1))
            Lane_pos_ = np.concatenate((np.zeros_like(np.expand_dims(Lane_line_, 1)), np.expand_dims(Lane_line_, 1)), axis=1)
            for Lane_pos_alpha, distance__ in zip(Lane_pos_alphas, Lane_pos_):
                plt.plot(distance__[:, 0], distance__[:, 1], color="purple", alpha=Lane_pos_alpha/2, zorder=3)
                
        if vis_setting['About_vis']['vis_all_prediction']:
            mask_agent = (self.planner_input['agent']['category'][0, 1:] == 1).detach().cpu().numpy()
            mask_ = mask_agent # * mask_xm
            valid_index = np.where(mask_ == True)[0]
            if valid_index.shape[0] > 0:
                example_out = self.out['output_prediction'][0, valid_index, :, :3].detach().cpu().numpy()
                # example_out = self._local_to_global(example_out, current_ego_state)[..., :2]
                for index in range(example_out.shape[0]):
                    plt.plot(example_out[index, :, 0], example_out[index, :, 1], linewidth=4, color=vis_setting['About_vis']['pred_color'], zorder=3, label="pred", alpha=1)

        if vis_setting['About_vis']['vis_others_history_activate']:
            current_observe = current_input.history.observations[-1].tracked_objects.tracked_objects
            valid_track_token = [elem.track_token for elem in current_observe]
            object_history = {}
            for object_buffer in current_input.history.observations:
                tracked_data = object_buffer.tracked_objects.tracked_objects
                for elem in tracked_data:
                    if elem.track_token not in valid_track_token:
                        continue
                    if elem.track_token not in object_history:
                        object_history[elem.track_token] = [[elem.center.x, elem.center.y]]
                    else:
                        object_history[elem.track_token].append([elem.center.x, elem.center.y])
            for key, val in object_history.items():
                object_history[key] = np.array(val)
            for key, val in object_history.items():
                plt.plot(val[:, 0], val[:, 1], linewidth=4, color="black", zorder=3, alpha=0.1)

        if vis_setting['About_vis']['vis_current_occupancy_activate']:
            for future_index in range(1): #JY: 왜 future index가 0이지?
                current_occ = occupancy._occupancy_maps_object_dynamic[future_index]
                current_occ_poly =  current_occ._geometries
                current_observe = current_input.history.observations[-1].tracked_objects.tracked_objects
                valid_track_token_vehicle = [elem.track_token for elem in current_observe if elem._tracked_object_type.name != 'PEDESTRIAN']
                valid_track_token_pedestrian = [elem.track_token for elem in current_observe if elem._tracked_object_type.name == 'PEDESTRIAN']
                all_current_obs = current_input.history.observations[-1].tracked_objects.tracked_objects
                agent_pos = [[elem.center.x, elem.center.y, elem.center.heading] for \
                            elem in all_current_obs if elem._tracked_object_type.name=='VEHICLE']
                agent_pos = np.array(agent_pos)
                agent_size = [elem.box.half_length for \
                                elem in all_current_obs if elem._tracked_object_type.name=='VEHICLE']
                for local_polygon, token_id in zip(current_occ_poly, current_occ.tokens):
                    x_, y_ = local_polygon.exterior.xy
                    if token_id in valid_track_token_vehicle:
                        tmp_arr = np.concatenate((np.array([x_, y_]).T, np.zeros((len(x_), 1))), axis=1)
                        tmp_arr = self._global_to_local(tmp_arr, current_ego_state)[:, :2]
                        x_, y_ = tmp_arr[:,0], tmp_arr[:,1]
                        plot_fig.fill(x_, y_, color=vis_setting['About_vis']['agent_color'], alpha=1, zorder=3)
                        if vis_setting['About_vis']['vis_front_agent']:
                            index_ = np.linalg.norm((agent_pos[:, :2] - np.array([local_polygon.centroid.x, local_polygon.centroid.y])), axis=1).argmin()
                            pos_ = agent_pos[index_]
                            pos_ = self._global_to_local(pos_, current_ego_state)
                            agent_w = agent_size[index_]
                            vec_x_ = (agent_w - 0.7)*np.cos(pos_[2])
                            vec_y_ = (agent_w - 0.7)*np.sin(pos_[2])
                            vec_x = (agent_w)*np.cos(pos_[2])
                            vec_y = (agent_w)*np.sin(pos_[2])
                            plt.annotate("", xy=(pos_[0]+vec_x, pos_[1]+vec_y), \
                                        xytext=(pos_[0]+vec_x_, pos_[1]+vec_y_), \
                                            arrowprops=dict(arrowstyle="->", color="white", lw=2, mutation_scale=18), zorder=21)
                    if token_id in valid_track_token_pedestrian:
                        tmp_arr = np.concatenate((np.array([x_, y_]).T, np.zeros((len(x_), 1))), axis=1)
                        tmp_arr = self._global_to_local(tmp_arr, current_ego_state)[:, :2]
                        x_, y_ = tmp_arr[:,0], tmp_arr[:,1]
                        plot_fig.fill(x_, y_, color= vis_setting['About_vis']['pedes_color'], alpha=1, zorder=3)
                current_occ = occupancy._occupancy_maps_object_static[future_index]
                current_occ_poly =  current_occ._geometries
                for local_polygon in current_occ_poly:
                    x_, y_ = local_polygon.exterior.xy
                    tmp_arr = np.concatenate((np.array([x_, y_]).T, np.zeros((len(x_), 1))), axis=1)
                    tmp_arr = self._global_to_local(tmp_arr, current_ego_state)[:, :2]
                    x_, y_ = tmp_arr[:,0], tmp_arr[:,1]
                    plot_fig.fill(x_, y_, color='black', alpha=0.25, zorder=3)

        
        if vis_setting['About_vis']['vis_future_occupancy_activate']:
            for future_index in range(len(occupancy._occupancy_maps_object_dynamic))[::5]:
                current_occ = occupancy._occupancy_maps_object_dynamic[future_index]
                current_occ_poly =  current_occ._geometries
                for local_polygon in current_occ_poly:
                    x_, y_ = local_polygon.exterior.xy
                    plot_fig.fill(x_, y_, color='blue', alpha=0.05, zorder=2)
                current_occ = occupancy._occupancy_maps_object_static[future_index]
                current_occ_poly =  current_occ._geometries
                for local_polygon in current_occ_poly:
                    x_, y_ = local_polygon.exterior.xy
                    plot_fig.fill(x_, y_, color='blue', alpha=0.05, zorder=2)
                    
        if vis_setting['About_vis']['vis_occupied_object']:
            for future_index in range(1):
                current_occ = occupancy._occupancy_maps_object_col_dynamic[future_index]
                current_occ_poly =  current_occ._geometries
                for local_polygon in current_occ_poly:
                    x_, y_ = local_polygon.exterior.xy
                    tmp_arr = np.concatenate((np.array([x_, y_]).T, np.zeros((len(x_), 1))), axis=1)
                    tmp_arr = self._global_to_local(tmp_arr, current_ego_state)[:, :2]
                    x_, y_ = tmp_arr[:,0], tmp_arr[:,1]
                    plot_fig.fill(x_, y_, color='red', alpha=1.0, zorder=3)

        if vis_setting['About_vis']['vis_centerline']:
            centerline_for_vis = np.array([[elem.x, elem.y] for elem in self._centerline.discrete_path])
            # 120m까지만 자르기
            my_my_mask = np.where(np.concatenate((np.array([True]), np.cumsum(np.linalg.norm(np.diff(centerline_for_vis, axis=0), axis=1)) < 120)))
            centerline_for_vis = centerline_for_vis[my_my_mask]
            plt.plot(centerline_for_vis[:, 0], centerline_for_vis[:, 1], linewidth=2, color="blue", zorder=0, label="centerline", alpha=0.7)

        if vis_setting['About_vis']['vis_current_traffic_activate']:
            for future_index in range(1):
                current_occ = occupancy._occupancy_maps_traffic[future_index]
                current_occ_poly =  current_occ._geometries
                for local_polygon in current_occ_poly:
                    x_, y_ = local_polygon.exterior.xy
                    tmp_arr = np.concatenate((np.array([x_, y_]).T, np.zeros((len(x_), 1))), axis=1)
                    tmp_arr = self._global_to_local(tmp_arr, current_ego_state)[:, :2]
                    x_, y_ = tmp_arr[:,0], tmp_arr[:,1]
                    plot_fig.fill(x_, y_, color='red', alpha=0.4, zorder=3)
                    
        if vis_setting['About_vis']["vis_GT"]:
            if vis_setting["chanllenge"] == "open_loop_boxes":
                GT_traj = np.array([[i.center.x, i.center.y] for i in self.scenario.get_ego_future_trajectory(iteration = iteration, time_horizon=8)])
                plt.plot(GT_traj[:, 0], GT_traj[:, 1], linewidth=9, color="blue", zorder=2, alpha=0.2, label="GT trajectory")
                plt.scatter(GT_traj[:, 0], GT_traj[:, 1], s=3, color="blue", zorder=1, alpha=0.07)
            else:
                GT_traj = np.array([[i.center.x, i.center.y] for i in self.scenario.get_ego_future_trajectory(iteration = 0, time_horizon=15)])
                GT_traj = np.concatenate((GT_traj, np.zeros((len(GT_traj), 1))), axis=1)
                GT_traj = self._global_to_local(GT_traj, current_ego_state)[:, :2]
                plt.plot(GT_traj[:, 0], GT_traj[:, 1], linewidth=13, color=vis_setting['About_vis']['GT_traj_color'], zorder=2, alpha=1, label="GT trajectory")

        if vis_setting['About_vis']["vis_legend"]:
            plt.legend(loc="upper left", fontsize=10)

        if vis_setting['About_vis']["vis_title"]:
            plt.title(f"{scenario_token}_{iteration}", fontsize=16, fontweight='bold', color='black')

        img_path = f"{save_path}/{scenario_token}"
        os.makedirs(img_path, exist_ok=True)
        img_name = f"{img_path}/{iteration}.png"
        plt.axis('off')
        plt.savefig(img_name, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close("all")
        gc.collect()
        
    def _route_roadblock_correction(self, ego_state: EgoState) -> None:
        route_roadblock_ids = route_roadblock_correction(ego_state, self._map_api, self._route_roadblock_dict)
        self._load_route_dicts(route_roadblock_ids)
        
    def _load_route_dicts(self, route_roadblock_ids: List[str]) -> None:
        # remove repeated ids while remaining order in list
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

        self._route_roadblock_dict = {}
        self._route_lane_dict = {}

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )

            self._route_roadblock_dict[block.id] = block

            for lane in block.interior_edges:
                self._route_lane_dict[lane.id] = lane
    def _local_to_global(self, local_trajectory: np.ndarray, ego_state: EgoState):
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(local_trajectory[..., :2], rot_mat) + origin
        heading = local_trajectory[..., 2] + angle

        return np.concatenate([position, heading[..., None]], axis=-1)

    def _global_to_local(self, global_trajectory: np.ndarray, ego_state: EgoState):
        if isinstance(global_trajectory, InterpolatedTrajectory):
            states: List[EgoState] = global_trajectory.get_sampled_trajectory()
            global_trajectory = np.stack(
                [
                    np.array(
                        [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
                    )
                    for state in states
                ],
                axis=0,
            )

        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(global_trajectory[..., :2] - origin, rot_mat)
        heading = global_trajectory[..., 2] - angle

        return np.concatenate([position, heading[..., None]], axis=-1)
