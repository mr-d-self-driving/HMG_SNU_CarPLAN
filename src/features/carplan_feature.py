from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from torch.nn.utils.rnn import pad_sequence

from src.utils.utils import to_device, to_numpy, to_tensor


@dataclass
class CarPLANFeature(AbstractModelFeature):
    data: Dict[str, Any]  # anchor sample
    data_p: Dict[str, Any] = None  # positive sample
    data_n: Dict[str, Any] = None  # negative sample
    data_n_info: Dict[str, Any] = None  # negative sample info

    @classmethod
    def collate(cls, feature_list: List[CarPLANFeature]) -> CarPLANFeature:
        batch_data = {}

        pad_keys = ["agent", "map"]

        stack_keys = ["current_state", "origin", "angle"]
        str_keys = ["scenario_type"] #JY

        if "reference_line" in feature_list[0].data:
            pad_keys.append("reference_line")
        if "static_objects" in feature_list[0].data:
            pad_keys.append("static_objects")
        if "cost_maps" in feature_list[0].data:
            stack_keys.append("cost_maps")

        if "centerline" in feature_list[0].data:
            pad_keys.append("centerline")
            
        if "occ_agent_pixel_indx" in feature_list[0].data:
            stack_keys.append("occ_agent_pixel_indx")
        if "occ_agent_pixel_indx_padding" in feature_list[0].data:
            stack_keys.append("occ_agent_pixel_indx_padding")
            
            
        if feature_list[0].data_n is not None:
            for key in pad_keys:
                batch_data[key] = {
                    k: pad_sequence(
                        [f.data[key][k] for f in feature_list]
                        + [f.data_p[key][k] for f in feature_list]
                        + [f.data_n[key][k] for f in feature_list],
                        batch_first=True,
                    )
                    for k in feature_list[0].data[key].keys()
                }

            batch_data["data_n_valid_mask"] = torch.Tensor(
                [f.data_n_info["valid_mask"] for f in feature_list]
            ).bool()
            batch_data["data_n_type"] = torch.Tensor(
                [f.data_n_info["type"] for f in feature_list]
            ).long()

            for key in stack_keys:
                batch_data[key] = torch.stack(
                    [f.data[key] for f in feature_list]
                    + [f.data_p[key] for f in feature_list]
                    + [f.data_n[key] for f in feature_list],
                    dim=0,
                )

        elif feature_list[0].data_p is not None:
            for key in pad_keys:
                batch_data[key] = {
                    k: pad_sequence(
                        [f.data[key][k] for f in feature_list]
                        + [f.data_p[key][k] for f in feature_list],
                        batch_first=True,
                    )
                    for k in feature_list[0].data[key].keys()
                }

            for key in stack_keys:
                batch_data[key] = torch.stack(
                    [f.data[key] for f in feature_list]
                    + [f.data_p[key] for f in feature_list],
                    dim=0,
                )
        else:
            for key in pad_keys:
                batch_data[key] = {
                    k: pad_sequence(
                        [f.data[key][k] for f in feature_list], batch_first=True
                    )
                    for k in feature_list[0].data[key].keys()
                }

            for key in stack_keys:
                batch_data[key] = torch.stack(
                    [f.data[key] for f in feature_list], dim=0
                )
                
            for key in str_keys:
                batch_data[key] = [f.data[key] for f in feature_list]
        
        return CarPLANFeature(data=batch_data)

    def to_feature_tensor(self) -> CarPLANFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_tensor(v)

        if self.data_p is not None:
            new_data_p = {}
            for k, v in self.data_p.items():
                new_data_p[k] = to_tensor(v)
        else:
            new_data_p = None

        if self.data_n is not None:
            new_data_n = {}
            new_data_n_info = {}
            for k, v in self.data_n.items():
                new_data_n[k] = to_tensor(v)
            for k, v in self.data_n_info.items():
                new_data_n_info[k] = to_tensor(v)
        else:
            new_data_n = None
            new_data_n_info = None

        return CarPLANFeature(
            data=new_data,
            data_p=new_data_p,
            data_n=new_data_n,
            data_n_info=new_data_n_info,
        )

    def to_numpy(self) -> CarPLANFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_numpy(v)
        if self.data_p is not None:
            new_data_p = {}
            for k, v in self.data_p.items():
                new_data_p[k] = to_numpy(v)
        else:
            new_data_p = None
        if self.data_n is not None:
            new_data_n = {}
            for k, v in self.data_n.items():
                new_data_n[k] = to_numpy(v)
        else:
            new_data_n = None
        return CarPLANFeature(data=new_data, data_p=new_data_p, data_n=new_data_n)

    def to_device(self, device: torch.device) -> CarPLANFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_device(v, device)
        return CarPLANFeature(data=new_data)

    def serialize(self) -> Dict[str, Any]:
        return {"data": self.data}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> CarPLANFeature:
        return CarPLANFeature(data=data["data"])

    def unpack(self) -> List[AbstractModelFeature]:
        raise NotImplementedError

    @property
    def is_valid(self) -> bool:
        try: #JY
            if "reference_line" in self.data:
                return self.data["reference_line"]["valid_mask"].any()
            else:
                return self.data["map"]["point_position"].shape[0] > 0
        except:
            # return self.data["centerline"]["position"].shape[0] > 0
            return self.data["occ_agent_pixel_indx"].shape[1] > 0 

    @classmethod
    def normalize(
        self, data, first_time=False, radius=None, hist_steps=21
    ) -> CarPLANFeature:
        cur_state = data["current_state"]
        center_xy, center_angle = cur_state[:2].copy(), cur_state[2].copy()

        rotate_mat = np.array(
            [
                [np.cos(center_angle), -np.sin(center_angle)],
                [np.sin(center_angle), np.cos(center_angle)],
            ],
            dtype=np.float64,
        )

        data["current_state"][:3] = 0
        
        if "agent" in data: #JY
            data["agent"]["position"] = np.matmul(
                data["agent"]["position"] - center_xy, rotate_mat
            )
            data["agent"]["velocity"] = np.matmul(data["agent"]["velocity"], rotate_mat)
            data["agent"]["heading"] -= center_angle

        if "map" in data: #JY
            data["map"]["point_position"] = np.matmul(
                data["map"]["point_position"] - center_xy, rotate_mat
            )
            data["map"]["point_vector"] = np.matmul(data["map"]["point_vector"], rotate_mat)
            data["map"]["point_orientation"] -= center_angle

            data["map"]["polygon_center"][..., :2] = np.matmul(
                data["map"]["polygon_center"][..., :2] - center_xy, rotate_mat
            )
            data["map"]["polygon_center"][..., 2] -= center_angle
            data["map"]["polygon_position"] = np.matmul(
                data["map"]["polygon_position"] - center_xy, rotate_mat
            )
            data["map"]["polygon_orientation"] -= center_angle

        if "causal" in data:
            if len(data["causal"]["free_path_points"]) > 0:
                data["causal"]["free_path_points"][..., :2] = np.matmul(
                    data["causal"]["free_path_points"][..., :2] - center_xy, rotate_mat
                )
                data["causal"]["free_path_points"][..., 2] -= center_angle
        if "static_objects" in data:
            data["static_objects"]["position"] = np.matmul(
                data["static_objects"]["position"] - center_xy, rotate_mat
            )
            data["static_objects"]["heading"] -= center_angle
        if "route" in data:
            data["route"]["position"] = np.matmul(
                data["route"]["position"] - center_xy, rotate_mat
            )
        if "reference_line" in data:
            data["reference_line"]["position"] = np.matmul(
                data["reference_line"]["position"] - center_xy, rotate_mat
            )
            data["reference_line"]["vector"] = np.matmul(
                data["reference_line"]["vector"], rotate_mat
            )
            data["reference_line"]["orientation"] -= center_angle
        
        #JY
        if "centerline" in data:
            data["centerline"]["position"] = np.matmul(
                data["centerline"]["position"] - center_xy, rotate_mat
            )
            data["centerline"]["vector"] = np.matmul(
                data["centerline"]["vector"], rotate_mat
            )
            data["centerline"]["orientation"] -= center_angle
        
        if "agent" in data: #JY
            target_position = (
                data["agent"]["position"][:, hist_steps:]
                - data["agent"]["position"][:, hist_steps - 1][:, None]
            )
            target_heading = (
                data["agent"]["heading"][:, hist_steps:]
                - data["agent"]["heading"][:, hist_steps - 1][:, None]
            )
            target = np.concatenate([target_position, target_heading[..., None]], -1)
            target[~data["agent"]["valid_mask"][:, hist_steps:]] = 0
            data["agent"]["target"] = target

        if first_time:
            if "map" in data: #JY
                point_position = data["map"]["point_position"]
                x_max, x_min = radius, -radius
                y_max, y_min = radius, -radius
                valid_mask = (
                    (point_position[:, 0, :, 0] < x_max)
                    & (point_position[:, 0, :, 0] > x_min)
                    & (point_position[:, 0, :, 1] < y_max)
                    & (point_position[:, 0, :, 1] > y_min)
                ) #(M, 20)
                valid_polygon = valid_mask.any(-1) #(M,)
                data["map"]["valid_mask"] = valid_mask

                for k, v in data["map"].items():
                    data["map"][k] = v[valid_polygon]

            if "causal" in data:
                data["causal"]["ego_care_red_light_mask"] = data["causal"][
                    "ego_care_red_light_mask"
                ][valid_polygon]

            data["origin"] = center_xy
            data["angle"] = center_angle

        return CarPLANFeature(data=data)
