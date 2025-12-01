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
from eval.evaluation_search import Evaluator


import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.98"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
print(jax.devices())

import tensorflow as tf
# tf.config.experimental.enable_op_determinism()
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


parser = argparse.ArgumentParser(description="Agent evaluation")
parser.add_argument('--expe_id', '-expe', type=str, help='Id of the experiment')
parser.add_argument('--epochs', '-e', type=int, help='Number of training epochs (at which to evaluate)')
parser.add_argument('--IDM', '-IDM', type=int, help='Use IDM for simulated agents', default=1)
parser.add_argument('--num_modes', '-m', type=int, help='Number of modes to evaluate for', default=4)
parser.add_argument('--num_envs', '-envs', type=int, help='How many environments for batched evaluation', default=10)
parser.add_argument('--max_batches', '-mb', type=int, help='How many batches to evaluate', default=20)
parser.add_argument('--multi_agent', '-ma', type=int, help='Whether to use multi-agent simulation', default=False)
parser.add_argument('--do_search', '-s', type=int, help='Whether to use do search', default=True)
parser.add_argument('--imagination_length', '-l', type=int, help='How long to imagine', default=10)
parser.add_argument('--num_actions_to_commit_to', '-a', type=int, help='How long to imagine', default=3)
parser.add_argument("--step_size", nargs="+", type=float, help="Step size of gradient update", default=[1e3, 1e-2]) # [1000.0, 0.01]
parser.add_argument('--ego_policy_weights', '-ep', type=str,  help='Pickle file for weights of ego-policy, relative to logs/', \
        default='train_search_ego/params_39.pkl')
parser.add_argument('--deterministic_actions', '-det', type=int,  help='Whether to use deterministic actions', default=0)
parser.add_argument('--rng_key', '-key', type=int,  help='The random seed', default=120)
parser.add_argument('--use_collisions_in_loss', '-cilf', type=int,  help='Whether to use collision and offroad events in loss function', default=0)
parser.add_argument('--tau', '-tau', type=float,  help='Temperature when sampling', default=1.0)


if __name__ == "__main__":
    args = parser.parse_args()

    # Training config
    load_folder = 'logs'
    expe_num = args.expe_id

    os.makedirs(f'animation/{expe_num}', exist_ok=True)

    with open(os.path.join(load_folder, expe_num, 'args.json'), 'r') as file:
        config = json.load(file)
    
    n_epochs = args.epochs

    if 'proxy_goal' in config['feature_extractor_kwargs']['keys']:
        print("Using proxy_goal: True", )

    print('Create datasets')

    # Env config
    config['IDM'] = bool(args.IDM)
    
    config['num_modes'] = args.num_modes
    config['max_batches'] = args.max_batches
    config['multi_agent'] = bool(args.multi_agent)
    config['do_search'] = bool(args.do_search)
    config['num_actions_to_commit_to'] = int(args.num_actions_to_commit_to)
    config['imagination_length'] = int(args.imagination_length)
    config['step_size'] = args.step_size
    config['deterministic_actions'] = bool(args.deterministic_actions)
    config['key'] = int(args.rng_key)
    config['use_collisions_in_loss'] = bool(args.use_collisions_in_loss)
    config['tau'] = float(args.tau)

    config['num_epochs'] = 1
    if args.num_envs > 0:
        config['num_envs_eval'] = args.num_envs

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
    
    # Ego config
    config['ego_policy_weights'] = os.path.join(load_folder, args.ego_policy_weights)
    ego_expe_num, ego_policy_params_file = args.ego_policy_weights.split("/")
    with open(os.path.join(load_folder, ego_expe_num, 'args.json')) as f:
        ego_config = json.load(f)
    config['ego_config'] = ego_config

    list_id = None
    evaluator = Evaluator(config, env_config, val_dataset, params, list_id)

    # with jax.disable_jit(): # DEBUG
    # 	evaluation_dict = evaluator.evaluate()

    evaluation_dict = evaluator.evaluate()
    
    with open(os.path.join(load_folder, expe_num, f"eval_metrics_{n_epochs}_IDM_{config['IDM']}.pkl"), "wb") as pkl_file:
        pickle.dump(evaluation_dict['metrics'], pkl_file)
    