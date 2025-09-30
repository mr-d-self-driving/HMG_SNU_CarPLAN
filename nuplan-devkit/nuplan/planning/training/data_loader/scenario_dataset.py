import logging
from typing import List, Optional, Tuple

import torch.utils.data

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor

#JY
from src.features.carplan_feature import CarPLANFeature 
from copy import deepcopy
from nuplan.planning.training.data_augmentation.data_augmentation_util import (
    ParameterToScale,
    ScalingDirection,
    UniformNoise,
)
import numpy as np
import numpy.typing as npt
from src.utils.collision_checker import CollisionChecker
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

logger = logging.getLogger(__name__)


class ScenarioDataset(torch.utils.data.Dataset):
    """
    Dataset responsible for consuming scenarios and producing pairs of model inputs/outputs.
    """

    def __init__(
        self,
        scenarios: List[AbstractScenario],
        feature_preprocessor: FeaturePreprocessor,
        augmentors: Optional[List[AbstractAugmentor]] = None,
    ) -> None:
        """
        Initializes the scenario dataset.
        :param scenarios: List of scenarios to use as dataset examples.
        :param feature_preprocessor: Feature and targets builder that converts samples to model features.
        :param augmentors: Augmentor object for providing data augmentation to data samples.
        """
        super().__init__()

        if len(scenarios) == 0:
            logger.warning('The dataset has no samples')

        self._scenarios = scenarios
        self._feature_preprocessor = feature_preprocessor
        self._augmentors = augmentors

    def __getitem__(self, idx: int) -> Tuple[FeaturesType, TargetsType, ScenarioListType]:
        """
        Retrieves the dataset examples corresponding to the input index
        :param idx: input index
        :return: model features and targets
        """
        scenario = self._scenarios[idx]

        features, targets, _ = self._feature_preprocessor.compute_features(scenario)
        
        
        if hasattr(self._feature_preprocessor.feature_builders[0], 'av_state_noise'):
            if self._feature_preprocessor.feature_builders[0].av_state_noise:
                history_steps = self._feature_preprocessor.feature_builders[0].history_samples + 1
                
                data = features['feature'].data
                new_data = deepcopy(data)

                current_state = data["current_state"]
                _random_offset_generator = UniformNoise([0.0, -1.5, -0.35, -1, -0.5, -0.2, -0.2], [2.0, 1.5, 0.35, 1, 0.5, 0.2, 0.2])
                noise = _random_offset_generator.sample()

                num_tries, scale = 0, 1.0
                agents_position = data["agent"]["position"][1:, history_steps - 1]
                agents_shape = data["agent"]["shape"][1:, history_steps - 1]
                agents_heading = data["agent"]["heading"][1:, history_steps - 1]
                agents_shape = data["agent"]["shape"][1:, history_steps - 1]

                while num_tries < 5:
                    new_noise = noise * scale
                    new_state = current_state + new_noise
                    new_state[3] = max(0.0, new_state[3])

                    if self.safety_check(
                        ego_position=new_state[:2],
                        ego_heading=new_state[2],
                        agents_position=agents_position,
                        agents_heading=agents_heading,
                        agents_shape=agents_shape,
                    ):
                        break

                    num_tries += 1
                    scale *= 0.5

                new_data["current_state"] = new_state
                new_data["agent"]["position"][0, history_steps - 1] = new_state[:2]
                new_data["agent"]["heading"][0, history_steps - 1] = new_state[2]
                
                new_data = PlutoFeature.normalize(new_data).data
                features['feature'].data = new_data

        if self._augmentors is not None:
            for augmentor in self._augmentors:
                augmentor.validate(features, targets)
                features, targets = augmentor.augment(features, targets, scenario)
        try:
            features_ = {key: value.to_feature_tensor() for key, value in features.items() if key}
            # for key, value in features.items():
            #     try:
            #         key: value.to_feature_tensor()
            #     except Exception as e:
            #         print(key)
            
            targets_ = {key: value.to_feature_tensor() for key, value in targets.items()}
            scenarios_ = [scenario]
        except:
            print(scenario)

        return features_, targets_, scenarios_

    def __len__(self) -> int:
        """
        Returns the size of the dataset (number of samples)

        :return: size of dataset
        """
        return len(self._scenarios)

    def safety_check(
        self,
        ego_position: npt.NDArray[np.float32],
        ego_heading: npt.NDArray[np.float32],
        agents_position: npt.NDArray[np.float32],
        agents_heading: npt.NDArray[np.float32],
        agents_shape: npt.NDArray[np.float32],
    ) -> bool:
        _collision_checker = CollisionChecker()
        _rear_to_cog = get_pacifica_parameters().rear_axle_to_center
        
        if len(agents_position) == 0:
            return True

        ego_center = (
            ego_position
            + np.stack([np.cos(ego_heading), np.sin(ego_heading)], axis=-1)
            * _rear_to_cog
        )
        ego_state = torch.from_numpy(
            np.concatenate([ego_center, [ego_heading]], axis=-1)
        ).unsqueeze(0)
        objects_state = torch.from_numpy(
            np.concatenate([agents_position, agents_heading[..., None]], axis=-1)
        ).unsqueeze(0)

        collisions = _collision_checker.collision_check(
            ego_state=ego_state,
            objects=objects_state,
            objects_width=torch.from_numpy(agents_shape[:, 0]).unsqueeze(0),
            objects_length=torch.from_numpy(agents_shape[:, 1]).unsqueeze(0),
        )

        return not collisions.any()