import logging
import os
import pprint
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Union

import hydra
import pandas as pd
import pytorch_lightning as pl
from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.script.builders.simulation_builder import build_simulations
from nuplan.planning.script.builders.simulation_callback_builder import (
    build_callbacks_worker,
    build_simulation_callbacks,
)
from nuplan.planning.script.utils import (
    run_runners,
    set_default_path,
    set_up_common_builder,
)
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from omegaconf import DictConfig, OmegaConf

from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv("NUPLAN_HYDRA_CONFIG_PATH", "config/simulation")


def print_simulation_results(file=None):
    if file is not None:
        df = pd.read_parquet(file)
    else:
        root = Path(os.getcwd()) / "aggregator_metric"
        result = list(root.glob("*.parquet"))
        result = max(result, key=lambda item: item.stat().st_ctime)
        df = pd.read_parquet(result)
    final_score = df[df["scenario"] == "final_score"]
    final_score = final_score.to_dict(orient="records")[0]
    pprint.PrettyPrinter(indent=4).pprint(final_score)


def run_simulation(
    cfg: DictConfig,
    planners: Optional[Union[AbstractPlanner, List[AbstractPlanner]]] = None,
) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Helper function for main to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    :param planners: Pre-built planner(s) to run in simulation. Can either be a single planner or list of planners.
    """
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    profiler_name = "building_simulation"
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)

    # Build simulation callbacks
    callbacks_worker_pool = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(
        cfg=cfg, output_dir=common_builder.output_dir, worker=callbacks_worker_pool
    )

    # Remove planner from config to make sure run_simulation does not receive multiple planner specifications.
    if planners and "planner" in cfg.keys():
        logger.info("Using pre-instantiated planner. Ignoring planner in config")
        OmegaConf.set_struct(cfg, False)
        cfg.pop("planner")
        OmegaConf.set_struct(cfg, True)

    # Construct simulations
    if isinstance(planners, AbstractPlanner):
        planners = [planners]

    runners = build_simulations(
        cfg=cfg,
        callbacks=callbacks,
        worker=common_builder.worker,
        pre_built_planners=planners,
        callbacks_worker=callbacks_worker_pool,
    )

    if common_builder.profiler:
        # Stop simulation construction profiling
        common_builder.profiler.save_profiler(profiler_name)

    logger.info("Running simulation...")
    run_runners(
        runners=runners,
        common_builder=common_builder,
        cfg=cfg,
        profiler_name="running_simulation",
    )
    logger.info("Finished running simulation!")


def clean_up_s3_artifacts() -> None:
    """
    Cleanup lingering s3 artifacts that are written locally.
    This happens because some minor write-to-s3 functionality isn't yet implemented.
    """
    # Lingering artifacts get written locally to a 's3:' directory. Hydra changes
    # the working directory to a subdirectory of this, so we serach the working
    # path for it.
    working_path = os.getcwd()
    s3_dirname = "s3:"
    s3_ind = working_path.find(s3_dirname)
    if s3_ind != -1:
        local_s3_path = working_path[: working_path.find(s3_dirname) + len(s3_dirname)]
        rmtree(local_s3_path)


@hydra.main(config_path="./config", config_name="default_simulation")
def main(cfg: DictConfig) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Calls run_simulation to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    """
    assert (
        cfg.simulation_log_main_path is None
    ), "Simulation_log_main_path must not be set when running simulation."

    run_simulation(cfg=cfg)

    if is_s3_path(Path(cfg.output_dir)):
        clean_up_s3_artifacts()

    print_simulation_results()


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--version', type=str, default='carplan')
    parser.add_argument('--ckpt', type=str, required=False, default="3")
    parser.add_argument('--challenge', type=str, required=False, default="CLS_NR")
    parser.add_argument('--threads', type=int, required=False, default=20)
    parser.add_argument('--postprocessing', type=bool, required=False, default=False)
    parser.add_argument('--visualize', type=bool, required=False, default=False)
    parser.add_argument('--filter', type=str, required=False, default="test14_hard")
    
    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    CONFIG_PATH = './config'
    CONFIG_NAME = 'default_simulation'

    CHECK_POINT_PATH = f"./exp/training/{args.version}/checkpoints/epoch\={args.ckpt}.ckpt"
    
    if args.challenge == "OLS":
        CHALLENGE = 'open_loop_boxes'
    elif args.challenge == "CLS_NR":
        CHALLENGE = 'closed_loop_nonreactive_agents'
    else:
        CHALLENGE = 'closed_loop_reactive_agents'
    
    if args.threads == 0:
        worker_type = ['worker=sequential']
    else:
        worker_type = ['worker=ray_distributed', f'worker.threads_per_node={args.threads}', '+planner.pluto_planner.use_gpu=False', 'number_of_gpus_allocated_per_simulation=0']
    
    PLANNER = "carplan_planner"
        
    FILTER = args.filter #"test14_hard" #mini_demo_scenario #reduced_val14_benchmark #reduced_val14_benchmark_v2 #test14_hard
    RENDER = args.visualize
    SIM_OUT = f"./exp/{CHALLENGE}/{args.version}"
    VIDEO_SAVE_DIR = f"./exp/{CHALLENGE}/{args.version}"
        
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=CONFIG_PATH)
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    f'+simulation={CHALLENGE}',
    f'planner={PLANNER}',
    f'model={args.version}',
    f'model.is_simulation=True',
    'scenario_builder=nuplan_challenge', #nuplan_challenge
    f'scenario_filter={FILTER}',
    f'verbose=true',
    f'planner.pluto_planner.render={RENDER}',
    f'planner.pluto_planner.planner_ckpt={CHECK_POINT_PATH}',
    f'planner.pluto_planner.postprocessing={args.postprocessing}',
    f'+planner.pluto_planner.save_dir={VIDEO_SAVE_DIR}',
    f'experiment_uid={args.version}',
    f'output_dir={SIM_OUT}',
    *worker_type
    ])
    
    main(cfg)
