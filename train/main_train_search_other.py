import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = "0.98"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys
sys.path.append(".")

from datetime import datetime
import functools
import jax
import jax.numpy as jnp
import json
import os

from waymax import config as _config
from train.trainer_search import Trainer
from configs.consts import N_TRAINING_INTER, N_TRAINING
from configs.conf_train_search_other import config
from utils.dataloader import tf_examples_dataset, inter_filter_funct, speed_filter_funct, preprocess_serialized_womd_data
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Ckeckpoint path
current_time = datetime.now()
date_string = current_time.strftime("%Y%m%d_%H%M%S")
log_folder = f"logs/{date_string}"
log_folder = f"logs/train_search_other"
os.makedirs(log_folder, exist_ok='True')

config['log_folder'] = log_folder

# Save training config
training_args = config

with open(os.path.join(log_folder, 'args.json'), 'w') as json_file:
    json.dump(training_args, json_file, indent=4)

# Data iter config
WOD_1_1_0_TRAINING = _config.DatasetConfig(
    path=config['training_path'],
    max_num_rg_points=config['max_num_rg_points'],
    shuffle_seed=config['shuffle_seed'],
    shuffle_buffer_size=config['shuffle_buffer_size'],
    data_format=_config.DataFormat.TFRECORD,
    batch_dims = (config['num_envs'],),
    max_num_objects=config['max_num_obj'],
    include_sdc_paths=config['include_sdc_paths'],
    repeat=None
)

filter_functions = {'inter_filter_fun': inter_filter_funct,
                    'speed_filter_fun': speed_filter_funct}

# Training dataset
train_dataset = tf_examples_dataset(
    path=WOD_1_1_0_TRAINING.path,
    data_format=WOD_1_1_0_TRAINING.data_format,
    preprocess_fn=functools.partial(preprocess_serialized_womd_data, config=WOD_1_1_0_TRAINING),
    shuffle_seed=WOD_1_1_0_TRAINING.shuffle_seed,
    shuffle_buffer_size=WOD_1_1_0_TRAINING.shuffle_buffer_size,
    repeat=WOD_1_1_0_TRAINING.repeat,
    batch_dims=WOD_1_1_0_TRAINING.batch_dims,
    num_shards=WOD_1_1_0_TRAINING.num_shards,
    deterministic=WOD_1_1_0_TRAINING.deterministic,
    drop_remainder=WOD_1_1_0_TRAINING.drop_remainder,
    tf_data_service_address=WOD_1_1_0_TRAINING.tf_data_service_address,
    batch_by_scenario=WOD_1_1_0_TRAINING.batch_by_scenario,
    filter_function=functools.partial(filter_functions[config['filter_fun_name']], **config['filter_fun_args']),
    num_files = config['num_files'],
    dataset_cache = config['dataset_cache']
)
print("Training dataset constructed")

if config['filter_fun_name']:
    if config['filter_fun_name'] == 'inter_filter_fun':
        config['num_training_data'] = N_TRAINING_INTER
    else:
        raise ValueError('Number of training data satisfying the filtering condition is unknown')
else:
    config['num_training_data'] = N_TRAINING

# Env config
assert config['env_type'] in ['planning', 'multi_agent']
if config['env_type'] == 'planning':
    env_config = _config.EnvironmentConfig(controlled_object=_config.ObjectType.SDC, max_num_objects=config['max_num_obj'])
else:
    env_config = _config.EnvironmentConfig(controlled_object=_config.ObjectType.VALID, max_num_objects=config['max_num_obj'])

# Training
print(jax.devices())
training = Trainer(config,
                    env_config,
                    train_dataset,
                    None, # val_dataset
                    )

training_dict = training.train()
