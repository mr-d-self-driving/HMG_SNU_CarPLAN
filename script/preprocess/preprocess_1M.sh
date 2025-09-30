
export PYTHONPATH=$PYTHONPATH:$(pwd)

 python ./run_preprocessing.py \
    py_func=cache \
    +training=train_carplan \
    scenario_builder=nuplan \
    cache.cache_path=./cache/carplan_1M_cache \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40