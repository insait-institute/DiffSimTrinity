import dataclasses
from flax.training.train_state import TrainState
import functools
import jax
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)
import jax.numpy as jnp
from jax import random
import numpy as np
import pickle


import json
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import optax
from tqdm import tqdm

from typing import NamedTuple
from waymax import dynamics
from waymax import dataloader
from waymax import datatypes
from waymax import agents
from waymax.metrics import MetricResult
from waymax.config import CoordinateFrame
from waymax import env as _env
from waymax import config as _config

import sys
sys.path.append('./')

from configs.consts import N_TRAINING, N_VALIDATION, TRAJ_LENGTH, N_FILES
from models.feature_extractor import KeyExtractor
from models.state_processing import ExtractObs
from models.rnn_policy import ActorCriticForSearch, ScannedRNN
from utils.obs_mask import SpeedConicObsMask, SpeedGaussianNoise, SpeedUniformNoise, ZeroMask
from waymax.datatypes.observation import ObjectPose2D
from waymax.utils.geometry import transform_points

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
from utils.env_and_state_manipulation import CustomBicycleDynamics

#plot_preds(sim_states, save='rand_test_img5.png', b_ix=23, plot_gt=True, plot_odometry=True, plot_inverse_state=True, plot_realised_traj=False)
#What looks good? 13, 9, 23?, 26. 38, 

class Transition(NamedTuple):
    done: jnp.ndarray
    expert_action: jnp.array
    obs: jnp.ndarray

extractors = {
    'ExtractObs': ExtractObs
}
feature_extractors = {
    'KeyExtractor': KeyExtractor,
}
obs_masks = {
    'ZeroMask': ZeroMask,
    'SpeedGaussianNoise': SpeedGaussianNoise,
    'SpeedUniformNoise': SpeedUniformNoise,
    'SpeedConicObsMask': SpeedConicObsMask
}

class Evaluator(object):

    def __init__(self, config, env_config, val_dataset, params, list_id):
        self.config = config
        self.env_config = env_config

        # Device
        self.devices = jax.devices()
        print(f'Available devices: {self.devices}')

        # Params
        self.params = params
        self._post_process = functools.partial(
            dataloader.womd_factories.simulator_state_from_womd_dict,
            include_sdc_paths=config['include_sdc_paths'],)

        self.val_dataset = val_dataset        
        self.list_id = list_id

        # Random key
        self.key = random.PRNGKey(self.config['key'])

        # DEFINE ENV
        if self.config['dynamics'] == 'bicycle':
            # self.wrapped_dynamics_model = dynamics.InvertibleBicycleModel()
            self.wrapped_dynamics_model = CustomBicycleDynamics()
        elif self.config['dynamics'] == 'delta':
            self.wrapped_dynamics_model = dynamics.DeltaLocal()
        else:
            raise ValueError('Unknown dynamics')
    
        if self.config['env_type'] == 'planning':
            self.dynamics_model = _env.PlanningAgentDynamics(self.wrapped_dynamics_model)
        else:
            self.dynamics_model = self.wrapped_dynamics_model

        assert (not config['discrete'])

        if config['IDM']:
            sim_actors = [agents.IDMRoutePolicy(is_controlled_func=lambda state: 1 -  state.object_metadata.is_sdc)]
            sim_agent_params = [{}]
        else:
            sim_actors = ()
            sim_agent_params = ()

        env_config_ma = _config.EnvironmentConfig(controlled_object=_config.ObjectType.VALID, allow_new_objects_after_warmup=False,
                    metrics=_config.MetricsConfig(metrics_to_run=('log_divergence', 'overlap', 'offroad')),
                    max_num_objects=self.config['max_num_obj'],)
        
        self.env = _env.PlanningAgentEnvironment(dynamics_model=self.wrapped_dynamics_model, config=env_config, sim_agent_actors=sim_actors, sim_agent_params=sim_agent_params)
        self.env_ma = _env.MultiAgentEnvironment(dynamics_model=self.wrapped_dynamics_model, config=env_config_ma)

        # DEFINE EXTRACTOR AND FEATURE_EXTRACTOR
        self.multi_agent = config.get("multi_agent", False)
        if config['do_search'] and (not self.multi_agent):
            self.multi_agent = True
            config['multi_agent'] = True
        if config['do_search']:
            config['ego_config']['multi_agent'] = False

        self.extractor = extractors[self.config['extractor']](self.config)
        self.ego_config = self.config['ego_config']
        self.ego_extractor = extractors[self.ego_config['extractor']](self.ego_config)

        self.feature_extractor = feature_extractors[self.config['feature_extractor']]
        self.feature_extractor_kwargs = self.config['feature_extractor_kwargs']
        self.num_modes = self.config['num_modes']

        # DEFINE OBSERVABILITY MASK
        if 'obs_mask' not in self.config.keys():
            self.config['obs_mask'] = None

        if self.config['obs_mask']:
            self.obs_mask = obs_masks[self.config['obs_mask']](**self.config['obs_mask_kwargs'])
        else:
            self.obs_mask = None
        
        # Additional settings
        self.do_search = config.get("do_search", False)
        print("Do search", self.do_search)
        print("Multi-agent", self.multi_agent)
        print("ego multi agent:", self.ego_config.get('multi_agent', False))
        self.imagination_length = config.get('imagination_length', 10)
        print("imag. length", self.imagination_length)
        self.num_actions_to_commit_to = config.get('num_actions_to_commit_to', 3)
        print("Num actions to commit to", self.num_actions_to_commit_to)
        self.tau = config.get('tau', 1.0)
        print("Tau:", self.tau)
        self.step_size = jnp.array(config.get("step_size"))
        print("Grad step size", self.step_size)
        self.deterministic_actions = config.get('deterministic_actions', False)
        print("Deterministic actions:", self.deterministic_actions)
        print("Num modes", self.num_modes)
        self.use_collisions_in_loss = config['use_collisions_in_loss']
        print("use_collisions_in_loss:", self.use_collisions_in_loss)

        # Load the conditional model with waypoints
        self.ego_policy_weights = config.get('ego_policy_weights', None)
        if self.ego_policy_weights is None:
            assert False
        with open(self.ego_policy_weights, 'rb') as file:
            self.ego_params = pickle.load(file)
        print("Loaded ego policy params (conditional)")


    # SCHEDULER
    def linear_schedule(self, count):
        n_update_per_epoch = (N_TRAINING * self.config['num_files'] / N_FILES) // self.config["num_envs"]
        n_epoch = jnp.array([count // n_update_per_epoch])
        frac = jnp.where(n_epoch <= 20, 1, 1 / (2**(n_epoch - 20)))
        return self.config["lr"] * frac

    def evaluate(self,):
        # Initialize networks and components
        network = ActorCriticForSearch(self.dynamics_model.action_spec().shape[0],
                                self.dynamics_model.action_spec().minimum,
                                self.dynamics_model.action_spec().maximum,
                                feature_extractor_class=self.feature_extractor,
                                feature_extractor_kwargs=self.feature_extractor_kwargs)
        feature_extractor_shape = self.feature_extractor_kwargs['final_hidden_layers']
        init_x = self.extractor.init_x(self.config["num_envs_eval"])
        init_rnn_state_train = ScannedRNN.initialize_carry((self.config["num_envs_eval"], feature_extractor_shape))
        network_params = network.init(self.key, init_rnn_state_train, 
            (init_x[0], init_x[1], random.PRNGKey(self.config['key']+1)))
        num_params = sum([x.size for x in jax.tree_leaves(network_params)])
        print("Num parameters:", num_params)
        print("Num parameters in loaded weights:", sum([x.size for x in jax.tree_leaves(self.params)]))
        feature_extractor_shape = self.feature_extractor_kwargs['final_hidden_layers']

        # Initialize ego policy net
        ego_network = ActorCriticForSearch(self.dynamics_model.action_spec().shape[0],
                                self.dynamics_model.action_spec().minimum,
                                self.dynamics_model.action_spec().maximum,
                                feature_extractor_class=feature_extractors[self.ego_config['feature_extractor']],
                                feature_extractor_kwargs=self.ego_config['feature_extractor_kwargs'])
        ego_feature_extractor_shape = self.ego_config['feature_extractor_kwargs']['final_hidden_layers']
        init_x = self.ego_extractor.init_x(self.config["num_envs_eval"])
        ego_network_params = ego_network.init(self.key, init_rnn_state_train, 
            (init_x[0], init_x[1], random.PRNGKey(self.config['key']+1)))
        num_params = sum([x.size for x in jax.tree_leaves(ego_network_params)])
        print("Num parameters in ego policy:", num_params)
        print("Num parameters in ego loaded weights:", sum([x.size for x in jax.tree_leaves(self.ego_params)]))
    
        if self.config["lr_scheduler"]:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["max_grad_norm"]),
                optax.adam(learning_rate=self.linear_schedule, eps=1e-5),)
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["max_grad_norm"]),
                optax.adam(self.config["lr"], eps=1e-5),)

        train_state = TrainState.create(apply_fn=network.apply, params=self.params, tx=tx,)
        init_rnn_state_eval = ScannedRNN.initialize_carry((self.config["num_envs_eval"], feature_extractor_shape))
        jit_postprocess_fn = jax.jit(self._post_process)
        self.num_envs_eval = self.config["num_envs_eval"]
        
        # Transform 16units key into 1unit key
        def create_id(id):
            if len(id.shape) == 3:
                return [int(''.join(map(str, x.squeeze()))) for x in id]
            return int(''.join(map(str, id.squeeze())))

        def extend(x):
            if isinstance(x, jnp.ndarray):
                return x[jnp.newaxis, ...]
            else:
                return x
                
        if self.list_id is not None:
            self.list_id = [create_id(id) for id in self.list_id]

        def _log_step(cary, aux_inputs):
            """
            Multi-agent version of _log_step.
            """
            rng_extract = aux_inputs
            current_state, rng = cary
            done = current_state.is_done
            # Extract obs in SDC referential

            if self.do_search: # When doing search need to extract both
                obs = datatypes.observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'], 
                            coordinate_frame = CoordinateFrame.OBJECT)
                ego_obs = datatypes.sdc_observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'])
            else: # If we don't do search, depends
                if self.multi_agent:
                    obs = datatypes.observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'], 
                            coordinate_frame = CoordinateFrame.OBJECT)
                else:
                    obs = datatypes.sdc_observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'])

            # Mask
            rng, rng_obs = jax.random.split(rng)
            if self.obs_mask is not None:
                obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)

            # Extract the features from the observation
            if self.do_search:
                obsv = self.extractor(current_state, obs, rng_extract, waypoint_mode=0)
                ego_obsv = self.ego_extractor(current_state, ego_obs, rng_extract, waypoint_mode=0)
                current_state = datatypes.update_state_by_log(current_state, num_steps=1)
                transition = Transition(done, None, obsv)
                ego_transition = Transition(done, None, ego_obsv)
                return (current_state, rng), (transition, ego_transition)
            else:
                obsv = self.extractor(current_state, obs, rng_extract, waypoint_mode=0)
                transition = Transition(done, None, obsv)

                # Update the simulator with the log trajectory
                current_state = datatypes.update_state_by_log(current_state, num_steps=1)
                return (current_state, rng), (transition, None)

        def run_mode_reactive(rng, starting_state, starting_rnn_state, time_length):
            """
            Runs a single realization where all actions -- both of the ego-vehicle and the other agents -- are selected by the policy.
            """
            rng, rng_extract = jax.random.split(rng)
            carry_k, all_actions = jax.lax.scan(f=_eval_step, 
                init=(starting_state, starting_rnn_state, rng), 
                xs=rng_extract[None].repeat(time_length, axis=0),
                length=time_length)
            return all_actions, carry_k[0]

        def _eval_step(cary, aux_inputs):
            """Evaluates a single step of multi-agent predictions.
            Ego-vehicle actions chosen using predefined Normal distribution parameters.
            Other agent actions chosen by the policy.
            ego_action_params should be (B, 2, 2)"""
            rng_extract = aux_inputs
            current_state, rnn_state, rng = cary
            done = current_state.is_done

            # Extract obs for each agent
            obs = datatypes.observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'], 
                coordinate_frame = CoordinateFrame.OBJECT)
            ego_obs = datatypes.sdc_observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'])
            
            # Detach here
            obs = jax.lax.stop_gradient(obs)
            ego_obs = jax.lax.stop_gradient(ego_obs)
            current_state = jax.lax.stop_gradient(current_state)
            rnn_state = jax.lax.stop_gradient(rnn_state)
            
            # Mask
            rng, rng_obs = jax.random.split(rng)
            if self.obs_mask is not None:
                obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)

            # Extract the features from the observation
            obsv = self.extractor(current_state, obs, rng_extract, waypoint_mode=1)
            ego_obsv = self.ego_extractor(current_state, ego_obs, rng_extract, waypoint_mode=1)

            # Get the predicted distribution
            all_rnn_state, all_weights, all_actor_mean, all_actor_std = jax.vmap(
                lambda rnn_state, obsv, done: network.apply(train_state.params, rnn_state, (obsv, done), method='action_dist_from_obs_inference'), 
                in_axes=(1, {k:2 for k in obsv.keys()}, None), out_axes=(1, 2, 2, 2)
                )(rnn_state, jax.tree_map(extend, obsv), done[None])

            # Get prediction for ego
            _, is_sdc = jax.lax.top_k(current_state.object_metadata.is_sdc, k=1) # (B, 1)
            ego_rnn_state = jnp.take_along_axis(rnn_state, indices=is_sdc[..., None], axis=-2).squeeze(-2) # (B, C)
            ego_rnn_state, ego_weights, ego_actor_mean, ego_actor_std = ego_network.apply(
                self.ego_params, ego_rnn_state, (jax.tree_map(extend, ego_obsv), done[None]), 
                method='action_dist_from_obs_inference')

            # Sample action for all agents
            action_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=all_weights),
                components_distribution=tfd.Normal(loc=all_actor_mean, scale=all_actor_std),
                reparameterize=True)
            rng, rng_sample, rng_sample_ego = jax.random.split(rng, 3)
            if self.deterministic_actions:
                action_data = all_actor_mean[0].mean(-1)
            else:
                action_data = action_dist.sample(seed=rng_sample).squeeze(0)

            # Sample action for ego agent
            action_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=ego_weights),
                components_distribution=tfd.Normal(loc=ego_actor_mean, scale=ego_actor_std),
                reparameterize=True)
            if self.deterministic_actions:
                ego_action_data = ego_actor_mean[0].mean(-1)
            else:
                ego_action_data = action_dist.sample(seed=rng_sample_ego).squeeze(0) # (B, 2)
            action_data = jnp.put_along_axis(arr=action_data, 
                indices=is_sdc[..., None].repeat(2, -1), values=ego_action_data[..., None, :], axis=-2, inplace=False)
            action = datatypes.Action(data=action_data, valid=jnp.ones((all_weights.shape[1], self.config['max_num_obj'], 1), dtype='bool'))
            action = jax.lax.stop_gradient(action)
            
            # Update the RNN of all agents
            rnn_state = jnp.put_along_axis(arr=all_rnn_state, 
                indices=is_sdc[..., None].repeat(all_rnn_state.shape[-1], -1), values=ego_rnn_state[:, None], axis=-2, inplace=False)    
            
            # Patch bug in waymax (squeeze timestep dimension when using reset --> need squeezed timestep for update)
            current_timestep = current_state['timestep']
            current_state = self.env_ma.step(
                dataclasses.replace(current_state,timestep=current_timestep[0]), action)

            # Unsqueeze timestep dim
            current_state = dataclasses.replace(current_state, timestep=current_timestep + 1)
            return (current_state, rnn_state, rng), action_data  
    
        def run_mode_with_conditional_actions_and_nondiff_events(all_actions, starting_state, starting_rnn_state, time_length):
            """
            Runs a single realization where all actions -- both of the ego-vehicle and the other agents -- are selected by the policy.
            Additionally, run all the non-differentiable event predictors and get their logits.
            """
            carry_k, (metrics, last_state, overlap, offroad) = jax.lax.scan(f=_eval_step_conditional_on_actions_and_nondiff_events, 
                init=(starting_state, starting_rnn_state), 
                xs=all_actions,
                length=time_length)
            return last_state, (metrics, overlap, offroad)

        def _eval_step_conditional_on_actions_and_nondiff_events(cary, aux_inputs):
            """Evaluates a single step of multi-agent predictions.
            Ego-vehicle actions chosen using predefined Normal distribution parameters.
            Other agent actions chosen by the policy.
            ego_action_params should be (B, 2, 2)"""
            actions_data = aux_inputs
            current_state, ego_rnn_state = cary
            done = current_state.is_done

            # Extract obs for each agent
            ego_obs = datatypes.sdc_observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'])
            
            # Detach here
            ego_obs = jax.lax.stop_gradient(ego_obs)

            # Extract the features from the observation
            ego_obsv = self.ego_extractor(current_state, ego_obs, None, waypoint_mode=1)

            # Get the predicted distribution
            rnn_state, state_features, _ = ego_network.apply(self.ego_params, ego_rnn_state, (jax.tree_map(extend, ego_obsv), done[None]), method='rnn_state_from_obs')
            overlap, offroad = ego_network.apply(self.ego_params, state_features, method='offroad_overlap_from_state_features')
            
            # Prepare the predefined action
            action = datatypes.Action(data=actions_data, valid=jnp.ones((actions_data.shape[0], self.config['max_num_obj'], 1), dtype='bool'))
            
            # Patch bug in waymax (squeeze timestep dimension when using reset --> need squeezed timestep for update)
            current_timestep = current_state['timestep']
            current_state = self.env_ma.step(
                dataclasses.replace(current_state,timestep=current_timestep[0]), action)

            # Unsqueeze timestep dim
            current_state = dataclasses.replace(current_state, timestep=current_timestep + 1)
            metrics = self.env.metrics(current_state)
            return (current_state, rnn_state), (metrics, current_state, overlap, offroad)

        def run_mode_with_conditional_actions(all_actions, starting_state, time_length):
            _, (scenario_metrics, states) = jax.lax.scan(f=_eval_step_conditional_on_actions, 
                init=starting_state, 
                xs=all_actions,
                length=time_length)
            return states, scenario_metrics

        def _eval_step_conditional_on_actions(cary, aux_inputs):
            """Evaluates a single step of multi-agent predictions where actions for all agents are previously provided."""
            actions_data = aux_inputs # actions should be (B, N, 2)
            current_state = cary

            current_state = jax.lax.stop_gradient(current_state)

            action = datatypes.Action(data=actions_data, valid=jnp.ones((actions_data.shape[0], self.config['max_num_obj'], 1), dtype='bool'))

            # Patch bug in waymax (squeeze timestep dimension when using reset --> need squeezed timestep for update)
            current_timestep = current_state['timestep']
            current_state = self.env_ma.step(
                dataclasses.replace(current_state,timestep=current_timestep[0]), action)

            # Unsqueeze timestep dim
            current_state = dataclasses.replace(current_state, timestep=current_timestep + 1)
            metrics = self.env.metrics(current_state)
            return current_state, (metrics, current_state)
            
        def loss_fn(ego_action_params, all_actions, starting_state, time_length):
            """
            Compute the loss function over a few different trajectories.
            ego_action_params is (K, T, B, 1, 2)
            all_actions is (K, T, B, N, 2) and contains ego_action_params
            """
            # Place the ego_action_params into all_actions so that ego_action_params are used in the gradient
            _, is_sdc = jax.lax.top_k(starting_state.object_metadata.is_sdc, k=1)
            indices = is_sdc[None, None, ..., None].repeat(ego_action_params.shape[0], axis=0).repeat(
                ego_action_params.shape[1], axis=1).repeat(ego_action_params.shape[-1], axis=-1)
            all_actions = jnp.put_along_axis(all_actions, indices=indices, values=ego_action_params, axis=-2, inplace=False)

            # Realize multiple modes from the same parameters
            all_states, metrics = jax.vmap(
                lambda action_params, s, T: run_mode_with_conditional_actions(action_params, s, T), 
                in_axes=(0, None, None), 
                out_axes=0)(all_actions, starting_state, self.imagination_length)

            # Mask out all values beyond time_length and compute the loss
            T = metrics['log_divergence'].valid.shape[1]
            for k in metrics.keys():
                metrics[k].valid = metrics[k].valid * (jnp.arange(T) < time_length)[None, :, None]

            _, is_sdc = jax.lax.top_k(all_states.object_metadata.is_sdc, k=1) #(K, T, B, 1)
            ego_state_sim = jnp.take_along_axis(all_states.current_sim_trajectory.stack_fields(('x', 'y', 'speed', 'vel_x', 'vel_y')), is_sdc[..., None, None], axis=-3).squeeze((-2, -3))
            ego_state_log = jnp.take_along_axis(all_states.current_log_trajectory.stack_fields(('x', 'y', 'speed', 'vel_x', 'vel_y')), is_sdc[..., None, None], axis=-3).squeeze((-2, -3))
            detailed_loss = jnp.linalg.norm(ego_state_sim - ego_state_log, ord=2, axis=-1, keepdims=False) # (K, T, B)
            mean_loss = (detailed_loss * metrics['log_divergence'].valid).sum() / (metrics['log_divergence'].valid.sum())
            aggregated_loss = (detailed_loss * metrics['log_divergence'].valid).sum((-2))/(metrics['log_divergence'].valid.sum((-2)))
            return mean_loss, (aggregated_loss, ego_state_sim)

        def loss_fn_with_collisions(ego_action_params, all_actions, starting_state, starting_rnn_state, time_length):
            """
            Compute the loss function with realistic collisions and offroad penalties over a few different trajectories.
            ego_action_params is (K, T, B, 1, 2)
            all_actions is (K, T, B, N, 2) and contains ego_action_params
            """
            # Place the ego_action_params into all_actions so that ego_action_params are used in the gradient
            _, is_sdc = jax.lax.top_k(starting_state.object_metadata.is_sdc, k=1)
            indices = is_sdc[None, None, ..., None].repeat(ego_action_params.shape[0], axis=0).repeat(
                ego_action_params.shape[1], axis=1).repeat(ego_action_params.shape[-1], axis=-1)
            all_actions = jnp.put_along_axis(all_actions, indices=indices, values=ego_action_params, axis=-2, inplace=False)

            # Realize multiple modes from the same parameters
            all_states, (metrics, overlap, offroad) = jax.vmap(
                lambda a, s, rnn, T: run_mode_with_conditional_actions_and_nondiff_events(a, s, rnn, T), 
                in_axes=(0, None, None, None), out_axes=0)(all_actions, starting_state, starting_rnn_state, self.imagination_length)

            # Mask out all values beyond time_length and compute the loss
            T = metrics['log_divergence'].valid.shape[1]
            for k in metrics.keys():
                metrics[k].valid = metrics[k].valid * (jnp.arange(T) < time_length)[None, :, None]

            # LAST WAYPOINT LOSS
            _, is_sdc = jax.lax.top_k(all_states.object_metadata.is_sdc, k=1) #(K, T, B, 1)
            ego_state_sim = jnp.take_along_axis(all_states.current_sim_trajectory.stack_fields(('x', 'y', 'speed', 'vel_x', 'vel_y')), is_sdc[..., None, None], axis=-3).squeeze((-2, -3))
            
            # COLLISION/OFFROAD LOSS WITH PREDICTORS
            # Invalidate all steps after a collision occurs, this will concetrate the gradients to minimize this undesired event
            K, T, B = metrics['offroad'].value.shape
            bad_ov   = metrics['overlap'].value > 0
            bad_of   = metrics['offroad'].value > 0
            combined = bad_ov | bad_of # (K,T,B)
            any_any   = jnp.any(combined, axis=1) # (K,B)
            first_pos = jnp.argmax(combined, axis=1)
            first_pos = jnp.where(any_any, first_pos, T)
            t_grid    = jnp.arange(T)[None, :, None]
            after_first = t_grid > first_pos[:, None, :]
            metrics['overlap'].valid = jnp.where(after_first, False, metrics['overlap'].valid)
            metrics['offroad'].valid = jnp.where(after_first, False, metrics['offroad'].valid)
            metrics['log_divergence'].valid = jnp.where(after_first, False, metrics['log_divergence'].valid)
            overlap_terms = metrics['overlap'].value * metrics['overlap'].valid * overlap.squeeze(-1).squeeze(2) # (K, T, B)
            offroad_terms = metrics['offroad'].value * metrics['offroad'].valid * offroad.squeeze(-1).squeeze(2) # (K, T, B)
            mean_loss = overlap_terms.sum() / metrics['overlap'].valid.sum() + offroad_terms.sum() / metrics['offroad'].valid.sum()
            aggregated_loss = overlap_terms.sum((-2)) / metrics['overlap'].valid.sum((-2)) + offroad_terms.sum((-2)) / metrics['offroad'].valid.sum((-2))

            return mean_loss, (aggregated_loss, ego_state_sim)
        
        @jax.jit
        def select_action(starting_state, starting_rnn_state, rng, time_length):
            """
            We always run imagination simulations for self.imagination_length steps.
            However, for shorter sequences (toward the end of the trajectory) we mask out everything after time_length
            so it doesn't contribute to the loss.
            """
            # Simulate once to obtain all actions
            rngs_n_rollouts = jax.random.split(rng, self.num_modes + 1)
            rng = rngs_n_rollouts[0]
            all_actions, sim_states = jax.vmap(lambda rngs, s, rnn, T: run_mode_reactive(rngs, s, rnn, T), 
                in_axes=(0, None, None, None), out_axes=0)(rngs_n_rollouts[1:], starting_state, starting_rnn_state, self.imagination_length)
            # all_actions is (K, T, B, N, 2), sim_states is (K, B)

            # Extract ego-actions
            _, is_sdc = jax.lax.top_k(sim_states.object_metadata.is_sdc, k=1)
            ego_actions = jnp.take_along_axis(all_actions, indices=is_sdc[:, None, ..., None], axis=-2)
            # jax.debug.print("Nans in grads {x}", x=jnp.isnan(ego_actions).any())

            if self.use_collisions_in_loss:
                starting_ego_rnn = jnp.take_along_axis(starting_rnn_state, indices=is_sdc[0, ..., None], axis=-2).squeeze(-2)
                grads, (loss, imag_traj) = jax.grad(loss_fn_with_collisions, argnums=0, has_aux=True)(ego_actions, all_actions, starting_state, starting_ego_rnn, time_length)
            else:
                grads, (loss, imag_traj) = jax.grad(loss_fn, argnums=0, has_aux=True)(ego_actions, all_actions, starting_state, time_length)
            # grads are (K, T, B)

            # Check Nan Grads
            # jax.debug.print("Nans in grads {x}, max grad {y}", x=jnp.isnan(grads).any(), y=jnp.abs(grads).max())
            w = jax.nn.softmax(-loss/self.tau, axis=0) # (K, B), normalized across K
            updated_actions = ego_actions - self.step_size * grads # (K, T, B, N, 2)
            first_few_actions = updated_actions[:, :self.num_actions_to_commit_to, :, 0] # (K, T', B, 2)
            actions_to_exec = (first_few_actions * w[:, None, :, None]).sum(0) # (T', B, 2)
            return actions_to_exec, imag_traj[..., :2]

        @jax.jit
        def process_physical_action(physical_action, current_state, rnn_state, rng):
            action = datatypes.Action(data=physical_action, valid=jnp.ones((physical_action.shape[0], 1), dtype='bool'))
            rng, rng_extract = jax.random.split(rng)
            obs = datatypes.observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'], 
                coordinate_frame = CoordinateFrame.OBJECT)
            ego_obs = datatypes.sdc_observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'])

            obsv = self.extractor(current_state, obs, rng_extract, waypoint_mode=1)
            ego_obsv = self.ego_extractor(current_state, ego_obs, rng_extract, waypoint_mode=1)
            done = current_state.is_done
            
            all_rnn_state, _, _ = jax.vmap(
                lambda rnn_state, obsv, done: network.apply(train_state.params, rnn_state, (obsv, done), method='rnn_state_from_obs'), 
                in_axes=(1, {k:2 for k in obsv.keys()}, None), out_axes=(1, 2, 2)
                )(rnn_state, jax.tree_map(extend, obsv), done[None]) # (B, N, C)

            _, is_sdc = jax.lax.top_k(current_state.object_metadata.is_sdc, k=1) #(B, 1)
            ego_rnn_state = jnp.take_along_axis(arr=rnn_state, indices=is_sdc[..., None], axis=-2).squeeze(-2)
            ego_rnn_state, _, _ = ego_network.apply(
                self.ego_params, ego_rnn_state, (jax.tree_map(extend, ego_obsv), done[None]), method='rnn_state_from_obs')
            rnn_state = jnp.put_along_axis(all_rnn_state, # (B, N, C)
                indices=is_sdc[..., None].repeat(ego_rnn_state.shape[-1], axis=-1), # (B, 1) --> (B, 1, C)
                values=ego_rnn_state[..., None, :], # (B, 1, C)
                axis=-2, inplace=False)
            
            current_timestep = current_state.timestep
            current_state = self.env.step(
                dataclasses.replace(current_state, timestep=current_timestep[0]), action)
            current_state = dataclasses.replace(current_state, timestep=current_timestep + 1)
            metrics = self.env.metrics(current_state)
            return current_state, rnn_state, metrics, rng

        def _eval_scenario_search(train_state, scenario, rng):
   
            @jax.jit
            def handle_history(scenario, rng, init_rnn_state_eval, train_state):
                # Replay history
                rng, rng_extract, rng_log = jax.random.split(rng, 3)
                (current_state, rng), (log_traj_batch, ego_log_traj_batch) = jax.lax.scan(f = _log_step, 
                    init = (scenario, rng_log),
                    xs = rng_extract[None].repeat(self.env.config.init_steps - 1, axis=0),
                    length=self.env.config.init_steps - 1)
                
                # Evolve the hidden state on the log history
                rnn_state, _, _ = jax.vmap(
                    lambda rnn_state, obs, done: network.apply(train_state.params, rnn_state, (obs, done), method='rnn_state_from_obs'), 
                    in_axes=(None, {k:2 for k in log_traj_batch.obs.keys()}, None), out_axes=(1,2,2)
                    )(init_rnn_state_eval, log_traj_batch.obs, log_traj_batch.done)
                # At this point rnn_state is (B, N, C) where N is all agents

                # Evolve the hidden state of the ego agent - it has different features, so we need to do it differently
                ego_rnn_state, _, _ = ego_network.apply(self.ego_params, 
                    init_rnn_state_eval, (ego_log_traj_batch.obs, log_traj_batch.done), method='rnn_state_from_obs')
                
                # Place the ego RNN state at its location
                _, is_sdc = jax.lax.top_k(current_state.object_metadata.is_sdc, k=1) #(B, 1)
                rnn_state = jnp.put_along_axis(rnn_state, # (B, N, C)
                    indices=is_sdc[..., None].repeat(ego_rnn_state.shape[-1], axis=-1), # (B, 1) --> (B, 1, C)
                    values=ego_rnn_state[..., None, :], # (B, 1, C)
                    axis=-2, inplace=False)
                return current_state, rnn_state, rng
            
            current_state, rnn_state, rng = handle_history(scenario, rng, init_rnn_state_eval, train_state)
            rng, rng_action_init = jax.random.split(rng)
            batch_size = current_state.shape[0]
            max_steps = TRAJ_LENGTH - self.env.config.init_steps
            
            # Store the imagined trajectories
            imag_traj_shape = (self.num_modes, self.imagination_length, self.num_envs_eval, 2)
            imag_traj_buffer = jnp.zeros((max_steps,) + imag_traj_shape)

            def body_fun(carry, x):
                t, state, rnn, rng, ego_buf, imag_traj, imag_traj_buffer = carry
                rng, rng_a = jax.random.split(rng)
                refill = (t % self.num_actions_to_commit_to) == 0
                use_imag = jnp.minimum(max_steps - t, self.imagination_length)
                ego_buf, imag_traj = jax.lax.cond(refill, 
                    lambda _: select_action(state, rnn, rng_a, use_imag), 
                    lambda _: (ego_buf, imag_traj), operand=None)
            
                # Store imag_traj at the current timestep
                imag_traj_buffer = jax.lax.cond(refill, lambda _: imag_traj_buffer.at[t].set(imag_traj), lambda _: imag_traj_buffer, operand=None)

                k = (t % self.num_actions_to_commit_to) #.astype(int)
                action = ego_buf[k]
                state, rnn, metric, rng = process_physical_action(action, state, rnn, rng)
                return (t + 1, state, rnn, rng, ego_buf, imag_traj, imag_traj_buffer), metric
    
            ego_actions = jnp.zeros((self.num_actions_to_commit_to, batch_size, 2))
            imag_traj = imag_traj_buffer[0]
            carry = (0, current_state, rnn_state, rng, ego_actions, imag_traj, imag_traj_buffer)
            (_, sim_states, _, _, _, _, imag_traj_buffer), scenario_metrics = jax.lax.scan(f=body_fun, init=carry, xs=None, length=max_steps)
                
            scenario_metrics = jax.tree.map(lambda x: x[None, ..., None], scenario_metrics) # (1, T, B, N=1)
            scenario_metrics['best_mode'] = jnp.zeros(batch_size, dtype=int)
            scenario_metrics['imag_traj'] = imag_traj_buffer
            return scenario_metrics, sim_states

        def _eval_scenario_reactive(train_state, scenario, rng):

            @jax.jit
            def handle_history(scenario, rng, init_rnn_state_eval, train_state):
                # Replay history
                rng, rng_extract, rng_log = jax.random.split(rng, 3)
                (current_state, rng), (log_traj_batch, ego_log_traj_batch) = jax.lax.scan(f = _log_step, 
                    init = (scenario, rng_log),
                    xs = rng_extract[None].repeat(self.env.config.init_steps - 1, axis=0),
                    length=self.env.config.init_steps - 1)
                
                # Evolve the hidden state on the log history
                if self.multi_agent:
                    rnn_state, _, x_cat_history = jax.vmap(
                        lambda rnn_state, obs, done: network.apply(train_state.params, rnn_state, (obs, done), method='rnn_state_from_obs'), 
                        in_axes=(None, {k:2 for k in log_traj_batch.obs.keys()}, None), out_axes=(1,2,2)
                        )(init_rnn_state_eval, log_traj_batch.obs, log_traj_batch.done)
                    # At this point rnn_state is (B, N, C) where N is all agents
                else:
                    rnn_state, _, x_cat_history = network.apply(train_state.params, init_rnn_state_eval, 
                        (log_traj_batch.obs, log_traj_batch.done), method='rnn_state_from_obs')
                    # rnn_state is (B, C)
                return current_state, rnn_state, rng
            
            rng, rng_history = jax.random.split(rng)
            current_state, rnn_state, rng = handle_history(scenario, rng_history, init_rnn_state_eval, train_state)
            rng, rng_action_init = jax.random.split(rng)
            batch_size = current_state.shape[0]

            def extend(x):
                if isinstance(x, jnp.ndarray):
                    return x[jnp.newaxis, ...]
                else:
                    return x

            def _eval_step(cary, aux_inputs):
                """Evaluates a single step without planning."""
                rng_extract = aux_inputs
                current_state, rnn_state, rng = cary
                done = current_state.is_done

                # Extract obs in SDC referential
                if self.multi_agent:
                    obs = datatypes.observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'], 
                        coordinate_frame = CoordinateFrame.OBJECT)
                else:
                    obs = datatypes.sdc_observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'])

                # Mask
                rng, rng_obs = jax.random.split(rng)
                if self.obs_mask is not None:
                    obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)

                # Extract the features from the observation
                obsv = self.extractor(current_state, obs, rng_extract, waypoint_mode=1)

                # Get the predicted distribution
                if self.multi_agent:
                    rnn_state, weights, actor_mean, actor_std = jax.vmap(
                        lambda rnn_state, obsv, done: network.apply(train_state.params, rnn_state, (obsv, done), method='action_dist_from_obs_inference'), 
                        in_axes=(1, {k:2 for k in obsv.keys()}, None), out_axes=(1, 2, 2, 2)
                        )(rnn_state, jax.tree_map(extend, obsv), done[None])
                else:
                    rnn_state, weights, actor_mean, actor_std = network.apply(train_state.params, rnn_state, \
                        (jax.tree_map(extend, obsv), done[None]), method='action_dist_from_obs_inference')

                # Sample action for all agents
                action_dist = tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(logits=weights),
                    components_distribution=tfd.Normal(loc=actor_mean, scale=actor_std),
                    reparameterize=True)

                rng, rng_sample = jax.random.split(rng)
                if self.deterministic_actions:
                    action_data = actor_mean[0].mean(-1)
                else:
                    action_data = action_dist.sample(seed=rng_sample).squeeze(0)
                
                current_timestep = current_state['timestep']
                if self.multi_agent:
                    action = datatypes.Action(data=action_data, valid=jnp.ones((weights.shape[1], self.config['max_num_obj'], 1), dtype='bool'))
                    current_state = self.env_ma.step(dataclasses.replace(current_state, timestep=current_timestep[0]), action)
                    metric = self.env_ma.metrics(current_state)
                else:
                    action = datatypes.Action(data=action_data, valid=jnp.ones((weights.shape[1], 1), dtype='bool'))
                    current_state = self.env.step(dataclasses.replace(current_state, timestep=current_timestep[0]), action)
                    metric = self.env.metrics(current_state)

                # Unsqueeze timestep dim
                current_state = dataclasses.replace(current_state, timestep=current_timestep + 1)
                return (current_state, rnn_state, rng), metric

            def run_mode_reactive(rng):
                """
                Runs a single realization where all actions -- both of the ego-vehicle and the other agents -- are selected by the policy.
                """
                rng, rng_extract = jax.random.split(rng)
                carry_k, scenario_metrics = jax.lax.scan(f=_eval_step, 
                    init=(current_state, rnn_state, rng), 
                    xs=rng_extract[None].repeat(TRAJ_LENGTH - self.env.config.init_steps, axis=0),
                    length=TRAJ_LENGTH - self.env.config.init_steps)
                return scenario_metrics, carry_k[0]
            
            # Reactive multi-mode (multi-agent) sampling
            rngs_n_rollouts = jax.random.split(rng, self.num_modes + 1)
            rng = rngs_n_rollouts[0]
            metrics_per_mode, sim_states = jax.vmap(lambda rngs: run_mode_reactive(rngs), in_axes=0, out_axes=0)(rngs_n_rollouts[1:])
            # Avg metrics per mode over the agents, (K, T, B, N)

            # Mean over agents and timesteps, take min over modes K
            if self.multi_agent:
                best_mode = ((metrics_per_mode['log_divergence'].value * metrics_per_mode['log_divergence'].valid).sum((-1, -3)) \
                    / metrics_per_mode['log_divergence'].valid.sum((-1, -3))).argmin(0)
            else:
                best_mode = ((metrics_per_mode['log_divergence'].value * metrics_per_mode['log_divergence'].valid).sum(-2) \
                    / metrics_per_mode['log_divergence'].valid.sum(-2)).argmin(0)         
            metrics_per_mode['best_mode'] = best_mode
            return metrics_per_mode, sim_states

        def eval_scenario(train_state, scenario, rng):
            if self.do_search:
                return _eval_scenario_search(train_state, scenario, rng)
            else:
                return _eval_scenario_reactive(train_state, scenario, rng)
        
        jit_eval_scenario = jax.jit(eval_scenario)

        def compute_any_valid(value, valid):
            masked_values = jnp.where(valid, value, False)
            if len(value.shape) >= 4:
                any_valid = jnp.any(masked_values, axis=(0, 1, -1))
            else:
                any_valid = jnp.any(masked_values, axis=(0, 1))
            return any_valid

        def compute_nanmean_valid(value, valid):
            masked_values = jnp.where(valid, value, jnp.nan)
            if len(value.shape) >= 4:
                mean_valid = jnp.nanmean(masked_values, axis=(0, 1, -1))
            else:
                mean_valid = jnp.nanmean(masked_values, axis=(0, 1))
            return mean_valid

        def compute_nanmax_valid(value, valid):
            masked_values = jnp.where(valid, value, jnp.nan)
            if len(value.shape) >= 4:
                max_valid = jnp.nanmax(masked_values, axis=(0, 1, -1))
            else:
                max_valid = jnp.nanmax(masked_values, axis=(0, 1))
            return max_valid

        # EVALUATION LOOP
        def _evaluate_epoch(train_state, rng):

            all_metrics = {'log_divergence': [], 'max_log_divergence': [], 'overlap_rate': [], 'overlap': [], 'offroad_rate': [], 'offroad': [], 
                           'sdc_kinematic_infeasibility': [], 'infeasibility_rate':[]}

            t = 0
            for data in tqdm(self.val_dataset.as_numpy_iterator(), desc='Validation', 
                    total=N_VALIDATION // self.config['num_envs_eval'] + 1, position=0, leave=True, ascii=True):
                t += 1
                
                if self.config['max_batches'] > 0 and t > self.config['max_batches']:
                    break

                scenario = jit_postprocess_fn(data)
                init_timestep = scenario.timestep
                scenario = self.env.reset(scenario)
                scenario = dataclasses.replace(scenario, timestep=init_timestep)

                # Scenario does not contain the SDC
                no_sdc = not jnp.any(scenario.object_metadata.is_sdc)
                if no_sdc:
                    pass
                else:
                    rng, rng_eval = jax.random.split(rng)
                    scenario_metrics, sim_states = jit_eval_scenario(train_state, scenario, rng_eval)
                    best_mode = scenario_metrics['best_mode'] # (B,)
                    selected_scenario_metrics = {}
                    for metric_name in ['log_divergence', 'offroad', 'overlap']:
                        value = scenario_metrics[metric_name].value
                        valid = scenario_metrics[metric_name].valid

                        if self.multi_agent:
                            selected_values = jnp.take_along_axis(value, best_mode[None, None, :, None], axis=0)
                            selected_valids = jnp.take_along_axis(valid, best_mode[None, None, :, None], axis=0)
                        else:
                            selected_values = jnp.take_along_axis(value, best_mode[None, None], axis=0)
                            selected_valids = jnp.take_along_axis(valid, best_mode[None, None], axis=0)
                        metric_result = MetricResult(value=selected_values, valid=selected_valids)
                        selected_scenario_metrics[metric_name] = metric_result
                    scenario_metrics = selected_scenario_metrics

                    for key, metric_value in scenario_metrics.items():
                        if jnp.any(metric_value.valid):
                            value = metric_value.value  # Shape (K, T, B)
                            valid = metric_value.valid  # Shape (K, T, B)
                            all_metrics[key].extend(compute_nanmean_valid(value, valid).tolist())

                    key = 'max_log_divergence'
                    if jnp.any(scenario_metrics['log_divergence'].valid):
                        value = scenario_metrics['log_divergence'].value  # Shape (K, T, B)
                        valid = scenario_metrics['log_divergence'].valid  # Shape (K, T, B)
                        all_metrics[key].extend(compute_nanmax_valid(value, valid).tolist())
                        
                    key = 'overlap_rate'
                    if jnp.any(scenario_metrics['overlap'].valid):
                        value = scenario_metrics['overlap'].value  # Shape (K, T, B)
                        valid = scenario_metrics['overlap'].valid  # Shape (K, T, B)
                        all_metrics[key].extend(compute_any_valid(value, valid).tolist())

                    key = 'offroad_rate'
                    if jnp.any(scenario_metrics['offroad'].valid):
                        value = scenario_metrics['offroad'].value  # Shape (K, T, B)
                        valid = scenario_metrics['offroad'].valid  # Shape (K, T, B)
                        all_metrics[key].extend(compute_any_valid(value, valid).tolist())

                    key = 'infeasibility_rate'
                    if not key in scenario_metrics:
                        continue
                    if jnp.any(scenario_metrics['sdc_kinematic_infeasibility'].valid):
                        value = scenario_metrics['sdc_kinematic_infeasibility'].value  # Shape (K, T, B)
                        valid = scenario_metrics['sdc_kinematic_infeasibility'].valid  # Shape (K, T, B)
                        all_metrics[key].extend(compute_any_valid(value, valid).tolist())

            return train_state, all_metrics

        metrics = {}
        rng = self.key
        for epoch in range(self.config["num_epochs"]):
            metrics[epoch] = {}

            # Validation
            rng, rng_eval = jax.random.split(rng)
            _, val_metric = _evaluate_epoch(train_state, rng_eval)
            metrics[epoch]['validation'] = val_metric
            for key, value in val_metric.items():
                print("Num validation samples", len(value))

            val_message = f'Epoch | {epoch} | Val (full)| '
            for key, value in val_metric.items():
                val_message += f" {key} | {jnp.array(value).mean():.4f} | "
            print(val_message)
        return {"train_state": train_state, "metrics": metrics,}
