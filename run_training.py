import logging
from typing import Optional

import hydra
import numpy
import pytorch_lightning as pl
from nuplan.planning.script.builders.folder_builder import (
    build_training_experiment_folder,
)
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.experiments.caching import cache_data
from omegaconf import DictConfig

from src.custom_training.custom_training_builder import (
    TrainingEngine,
    build_training_engine,
    update_config_for_training,
)

import sys
sys.path.append('.')

from argparse import ArgumentParser

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = "./config"
CONFIG_NAME = "default_training"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """
    pl.seed_everything(cfg.seed, workers=True)

    # Configure logger
    build_logger(cfg)
    import os;os.path.abspath(os.curdir)
    # Override configs based on setup, and print config
    update_config_for_training(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    # Build worker
    worker = build_worker(cfg)
    
    if cfg.py_func == "train":
        # Build training engine
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # Run training
        logger.info("Starting training...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):
            engine.trainer.fit(
                model=engine.model,
                datamodule=engine.datamodule,
                ckpt_path=cfg.checkpoint,
            )
        return engine
    if cfg.py_func == "validate":
        # Build training engine
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # Run training
        logger.info("Starting training...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "validate"):
            engine.trainer.validate(
                model=engine.model,
                datamodule=engine.datamodule,
                ckpt_path=cfg.checkpoint,
            )
        return engine
    elif cfg.py_func == "test":
        # Build training engine
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # Test model
        logger.info("Starting testing...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "testing"):
            engine.trainer.test(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == "cache":
        # Precompute and cache all features
        logger.info("Starting caching...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "caching"):
            cache_data(cfg=cfg, worker=worker)
        return None
    else:
        raise NameError(f"Function {cfg.py_func} does not exist")


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--version', type=str, default='CarPLAN_1M')
    parser.add_argument('--batch_size', type=int, required=False, default=2)
    parser.add_argument('--epoch', type=int, required=False, default=41) #JY
    parser.add_argument('--ckpt', type=str, required=False, default="default")
    parser.add_argument('--cache', type=str, required=False, default="sanity_check")
    parser.add_argument('--CIL', type=bool, required=False, default=True)
    parser.add_argument('--py_func', type=str, required=False, default="train")
    parser.add_argument('--warmup_epochs', type=int, required=False, default=3) #JY
    parser.add_argument('--trainer', type=str, required=False, default="base") #JY
    parser.add_argument('--check_val_every_n_epoch', type=int, required=False, default=2) #JY
    parser.add_argument('--save_top_k', type=int, required=False, default=5) #JY
    parser.add_argument('--lr', type=str, required=False, default="1e-3") #JY
    
    
    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    CONFIG_NAME = 'default_training'
    RESULT_SAVE_DIR = './exp/training/' 

    CACHE_PATH= f'./cache/{args.cache}' 
        
    if args.CIL:
        add_CIL_config = ["+custom_trainer.use_contrast_loss=true", "data_augmentation=contrastive_scenario_generator"]
    else:
        add_CIL_config = ["+custom_trainer.use_contrast_loss=false"]
    
    if args.ckpt == "default":
        TRAIN_START_WITH_CHECK_POINT = False
        CHECK_POINT_PATH = f"./exp/training/{args.version}/checkpoints/epoch\={args.ckpt}.ckpt"
        checkpoint=[]
    else:
        TRAIN_START_WITH_CHECK_POINT = True
        CHECK_POINT_PATH = f"./exp/training/{args.version}/checkpoints/epoch\={args.ckpt}.ckpt"
        checkpoint=[f"checkpoint={CHECK_POINT_PATH}"]

    if args.trainer == "CarPLAN":
        trainer = "train_carplan"
        add_model_config = []
    else:
        trainer = 'train_pluto'
        add_model_config = ["+model.is_prev_cache=False"]
    
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=CONFIG_PATH)
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    'seed=0',
    f'py_func={args.py_func}',
    f'+training={trainer}',  #JY # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory
    f'model={args.version}',
    'worker=single_machine_thread_pool',
    'worker.max_workers=4',
    'scenario_builder=nuplan_mini',
    f'cache.cache_path={str(CACHE_PATH)}',
    'cache.use_cache_without_dataset=true',
    f'data_loader.params.batch_size={args.batch_size}',
    'data_loader.params.num_workers=1',
    f'group={str(RESULT_SAVE_DIR)}',
    f'job_name={args.version}',
    f'experiment_uid={args.version}',
    f'lightning.trainer.checkpoint.resume_training={TRAIN_START_WITH_CHECK_POINT}',
    f'lightning.trainer.checkpoint.save_top_k={args.save_top_k}',
    f'lightning.trainer.params.max_epochs={args.epoch}',
    f'lightning.trainer.params.check_val_every_n_epoch={args.check_val_every_n_epoch}',
    f'epochs={args.epoch}',
    f'warmup_epochs={args.warmup_epochs}',
    f'optimizer.lr={args.lr}',
    f'+checkpoint_path={CHECK_POINT_PATH}',
    *add_CIL_config,
    *add_model_config,
    f'lightning.trainer.params.strategy=ddp_find_unused_parameters_true',
    *checkpoint,
    ])
    
    main(cfg)

