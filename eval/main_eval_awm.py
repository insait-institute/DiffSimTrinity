import argparse
import functools
import jax
import json
import os
import pickle
import sys
sys.path.append('.')
sys.path.append('..')

from utils.dataloader import tf_examples_dataset, preprocess_serialized_womd_data

from waymax import config as _config
from waymax import dataloader
from eval.evaluation_awm import Evaluator

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(jax.devices())

import tensorflow as tf
# tf.config.experimental.enable_op_determinism()
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# Command line arguments
parser = argparse.ArgumentParser(description="Agent evaluation")
parser.add_argument('--expe_id', '-expe', type=str, help='Id of the experiment')
parser.add_argument('--epochs', '-e', type=int, help='Number of training epochs (at which to evaluate)')
parser.add_argument('--IDM', '-IDM', type=int, help='Use IDM for simulated agents', default=0)
parser.add_argument('--num_modes', '-m', type=int, help='Number of modes to evaluate for', default=1)
parser.add_argument('--num_envs', '-envs', type=int, help='How many environments for batched evaluation', default=0)
parser.add_argument('--max_batches', '-mb', type=int, help='How many batches to evaluate', default=-1)
parser.add_argument('--use_mpc', '-mpc', type=int, help='Whether to use model-predictive control (MPC)', default=1)
parser.add_argument('--planning_horizon', '-ph', type=int, help='How long to plan each trajectory in MPC', default=1)
parser.add_argument('--num_imagined_rollouts', '-nr', type=int, help='How many latent simulations to do in MPC', default=1)
parser.add_argument('--use_planner_for_eval', '-p', type=int, help='Whether to use planner for action selection when evaluating', default=0)
parser.add_argument('--use_rewards', '-r', type=int, help='Whether to use rewards or norm of inverse state in MPC', default=1)
parser.add_argument('--top_k', '-k', type=int, help='How many actions to aggregate in MPC', default=3)



if __name__ == "__main__":
    args = parser.parse_args()

    # Training config
    load_folder = 'logs'
    expe_num = args.expe_id

    os.makedirs(f'animation/{expe_num}', exist_ok=True)

    with open(os.path.join(load_folder, expe_num, 'args.json'), 'r') as file:
        config = json.load(file)

    n_epochs = args.epochs

    print('Create datasets')

    # Env config
    config['IDM'] = bool(args.IDM)
    config['GIF'] = bool(args.GIF)
    if config['GIF']:
        config['num_envs_eval'] = 1
    
    config['num_modes'] = args.num_modes
    config['max_batches'] = args.max_batches
    config['use_mpc'] = bool(args.use_mpc)
    config['num_imagined_rollouts'] = args.num_imagined_rollouts
    config['planning_horizon'] = args.planning_horizon
    config['use_planner_for_eval'] = bool(args.use_planner_for_eval)
    config['use_rewards'] = bool(args.use_rewards)
    config['top_k'] = args.top_k

    config['num_epochs'] = 1
    if args.num_envs > 0:
        config['num_envs_eval'] = args.num_envs
    else:
        if config['use_mpc'] == True:
            # config['num_envs_eval'] = 78
            config['num_envs_eval'] = 30
        else:
            config['num_envs_eval'] = 50 # 100


    # Make sure that whenever we predict future camera tokens, we use the right feature extractor
    if config['feature_extractor'] != 'KeyExtractorWithIntermediates':
        config['feature_extractor'] = 'KeyExtractorWithIntermediates'

    # Data iter config
    WOD_1_1_0_VALIDATION = _config.DatasetConfig(
        path=config['validation_path'],
        max_num_rg_points=config['max_num_rg_points'],
        shuffle_seed=None,
        data_format=_config.DataFormat.TFRECORD,
        batch_dims = (config['num_envs_eval'],),
        max_num_objects=config['max_num_obj'],
        include_sdc_paths=config['include_sdc_paths'],
        repeat=1
    )
    
    val_dataset = dataloader.tf_examples_dataset(
        path=WOD_1_1_0_VALIDATION.path,
        data_format=WOD_1_1_0_VALIDATION.data_format,
        preprocess_fn=functools.partial(preprocess_serialized_womd_data, config=WOD_1_1_0_VALIDATION),
        shuffle_seed=WOD_1_1_0_VALIDATION.shuffle_seed,
        shuffle_buffer_size=WOD_1_1_0_VALIDATION.shuffle_buffer_size,
        repeat=WOD_1_1_0_VALIDATION.repeat,
        batch_dims=WOD_1_1_0_VALIDATION.batch_dims,
        num_shards=WOD_1_1_0_VALIDATION.num_shards,
        deterministic=WOD_1_1_0_VALIDATION.deterministic,
        drop_remainder=WOD_1_1_0_VALIDATION.drop_remainder,
        tf_data_service_address=WOD_1_1_0_VALIDATION.tf_data_service_address,
        batch_by_scenario=WOD_1_1_0_VALIDATION.batch_by_scenario,
    )

    env_config = _config.EnvironmentConfig(controlled_object=_config.ObjectType.SDC, 
			metrics=_config.MetricsConfig(metrics_to_run=('log_divergence', 'overlap', 'offroad')),
			  max_num_objects=config['max_num_obj'],)

    # Evaluation
    print('Load network parameters')

    with open(os.path.join(load_folder, expe_num, f'params_{n_epochs}.pkl'), 'rb') as file:
        params = pickle.load(file)
        
    list_id = None
    evaluator = Evaluator(config, env_config, val_dataset, params, list_id)
    evaluation_dict = evaluator.train()
    
    if args.sub_valid_model is None:
        with open(os.path.join(load_folder, expe_num, f"eval_metrics_{n_epochs}_IDM_{config['IDM']}.pkl"), "wb") as pkl_file:
            pickle.dump(evaluation_dict['metrics'], pkl_file)
        
    else:
        with open(os.path.join(load_folder, expe_num, f"eval_metrics_{n_epochs}_IDM_{config['IDM']}_sub_validation_{args.sub_valid_model}.pkl"), "wb") as pkl_file:
            pickle.dump(evaluation_dict['metrics'], pkl_file)
