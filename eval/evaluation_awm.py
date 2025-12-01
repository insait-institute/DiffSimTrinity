import dataclasses
from flax.training.train_state import TrainState
import functools
import jax
import jax.numpy as jnp
from jax import random
from PIL import Image
import numpy as np

import json
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import optax
from tqdm import tqdm

from typing import NamedTuple
from waymax import dynamics
from waymax import dataloader
from waymax import datatypes
from waymax import env as _env
from waymax import agents
from waymax.metrics import MetricResult

import sys
sys.path.append('./')

from configs.consts import N_TRAINING, N_VALIDATION, TRAJ_LENGTH, N_FILES
from models.feature_extractor import KeyExtractor, KeyExtractorWithIntermediates
from models.state_processing import ExtractObs, ExtractObsForAWM
from models.rnn_policy import ActorCriticForAWM, ScannedRNN
from utils.obs_mask import SpeedConicObsMask, SpeedGaussianNoise, SpeedUniformNoise, ZeroMask
from waymax.datatypes.observation import ObjectPose2D
from utils.env_and_state_manipulation import transform_traj, subtract_from_sdc_xyyaw, subtract_from_sdc_yawvel, \
    create_state_with_sdc_xyyaw, create_state_with_sdc_yawvel, inverse_kinematics
from utils.env_and_state_manipulation import *


class Transition(NamedTuple):
    done: jnp.ndarray
    expert_action: jnp.array
    obs: jnp.ndarray

extractors = {
    'ExtractObs': ExtractObs,
    'ExtractObsForAWM': ExtractObsForAWM
}
feature_extractors = {
    'KeyExtractor': KeyExtractor,
    'KeyExtractorWithIntermediates': KeyExtractorWithIntermediates
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
        self.list_id = list_id # Subset validation scenario id

        # Random key
        self.key = random.PRNGKey(self.config['key'])

        # DEFINE ENV
        if self.config['dynamics'] == 'bicycle':
            self.wrapped_dynamics_model = dynamics.InvertibleBicycleModel()
            self.wrapped_dynamics_noclip = CustomBicycleDynamicsNoActionClip()
        elif self.config['dynamics'] == 'delta':
            self.wrapped_dynamics_model = dynamics.DeltaLocal()
        else:
            raise ValueError('Unknown dynamics')
    
        if self.config['env_type'] == 'planning':
            self.dynamics_model = _env.PlanningAgentDynamics(self.wrapped_dynamics_model)
            self.dynamics_model_noclip = _env.PlanningAgentDynamics(self.wrapped_dynamics_noclip)
        else:
            self.dynamics_model = self.wrapped_dynamics_model

        if config['discrete']:
            action_space_dim = self.dynamics_model.action_spec().shape
            self.dynamics_model = dynamics.discretizer.DiscreteActionSpaceWrapper(dynamics_model=self.dynamics_model,
                                                                                  bins=config['bins'] * jnp.array(action_space_dim))
        else:
            self.dynamics_model = self.dynamics_model

        if config['IDM']:
            sim_actors = [agents.IDMRoutePolicy(is_controlled_func=lambda state: 1 - state.object_metadata.is_sdc)]
            sim_agent_params = [{}]
        else:
            sim_actors = ()
            sim_agent_params = ()

        if self.config['env_type'] == 'planning':
            self.env = _env.PlanningAgentEnvironment(dynamics_model=self.wrapped_dynamics_model, config=env_config, 
                sim_agent_actors=sim_actors, sim_agent_params=sim_agent_params)
        else:
            self.env = _env.MultiAgentEnvironment(dynamics_model=self.wrapped_dynamics_model, config=env_config)

        # DEFINE EXTRACTOR AND FEATURE_EXTRACTOR
        self.extractor = extractors[self.config['extractor']](self.config)
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
        self.use_mpc = config['use_mpc']
        self.num_imagined_rollouts = config.get("num_imagined_rollouts", 1)
        self.planning_length = config.get("planning_horizon", 10)
        self.use_planner_for_eval = config.get('use_planner_for_eval', False)
        self.use_inv_opt_state_as_rewards = False
        self.use_rewards = config.get('use_rewards', True)
        self.top_k = config.get('top_k', None)
        if self.top_k is None:
            assert False
        print("Using rewards", self.use_rewards, 'top_k', self.top_k)

    # SCHEDULER
    def linear_schedule(self, count):
        n_update_per_epoch = (N_TRAINING * self.config['num_files'] / N_FILES) // self.config["num_envs"]
        n_epoch = jnp.array([count // n_update_per_epoch])
        frac = jnp.where(n_epoch <= 20, 1, 1 / (2**(n_epoch - 20)))
        return self.config["lr"] * frac

    def train(self,):
        # Initialize networks and components
        network = ActorCriticForAWM(self.dynamics_model.action_spec().shape[0],
                                self.dynamics_model.action_spec().minimum,
                                self.dynamics_model.action_spec().maximum,
                                feature_extractor_class=self.feature_extractor,
                                feature_extractor_kwargs=self.feature_extractor_kwargs)
        feature_extractor_shape = self.feature_extractor_kwargs['final_hidden_layers']
        init_x = self.extractor.init_x(self.config["num_envs_eval"])
        init_rnn_state_train = ScannedRNN.initialize_carry((self.config["num_envs_eval"], feature_extractor_shape))
        network_params = network.init(self.key, init_rnn_state_train, 
            (init_x[0], init_x[1], random.PRNGKey(self.config['key']+1)), jnp.zeros((1, self.config["num_envs_eval"], 2)))
        num_params = sum([x.size for x in jax.tree_leaves(network_params)])
        print("Num parameters:", num_params)
        feature_extractor_shape = self.feature_extractor_kwargs['final_hidden_layers']

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
        
        # Transform 16units key into 1unit key
        def create_id(id):
            if len(id.shape) == 3:
                return [int(''.join(map(str, x.squeeze()))) for x in id]
            return int(''.join(map(str, id.squeeze())))
        
        if self.list_id is not None:
            self.list_id = [create_id(id) for id in self.list_id]

        def _log_step(cary, aux_inputs):
            """
            Updates the simulator state according to the log trajectory. Returns Transitions.
            """
            rng_extract = aux_inputs
            current_state, rng = cary
            done = current_state.is_done
            # Extract obs in SDC referential
            obs = datatypes.sdc_observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'])

            # Mask
            rng, rng_obs = jax.random.split(rng)
            if self.obs_mask is not None:
                obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)

            # Extract the features from the observation
            obsv = self.extractor(current_state, obs, rng_extract)
            transition = Transition(done, None, obsv)

            # Update the simulator with the log trajectory
            current_state = datatypes.update_state_by_log(current_state, num_steps=1)
            return (current_state, rng), transition

        # Evaluate
        def _eval_scenario(train_state, current_state, rng):
            if not self.use_mpc: # If we're not planning, then we simulate the futures in a reactive manner
                                # In that case, it doesn't matter whether we are goal-conditioned or not.
                                # Goal-conditioning affects only the way in which imagined trajectories are evaluated.
                rng, rng_extract = jax.random.split(rng)
                rng, rng_log = jax.random.split(rng)
                # Compute the rnn_state on first self.env.config.init_steps from the log trajectory
                (current_state, rng), log_traj_batch = jax.lax.scan(f = _log_step, 
                    init = (current_state, rng_log),
                    xs = rng_extract[None].repeat(self.env.config.init_steps - 1, axis=0),
                    length=self.env.config.init_steps - 1)

                 # Evolve the hidden state on the log history
                batch_size = log_traj_batch.done.shape[1]
                expert_actions = jnp.zeros((10, batch_size, 2)) # This gets passed at test time but we don't use it
                rnn_state, _, _, _, _, _, hist_latent_state, pred_rewards, hist_state_features, individual_feats, x_cat_h = network.apply(
                    train_state.params, init_rnn_state_eval, (log_traj_batch.obs, log_traj_batch.done, rng_log), expert_actions)

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
                    obs = datatypes.sdc_observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'])

                    # Mask
                    rng, rng_obs = jax.random.split(rng)
                    if self.obs_mask is not None:
                        obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)

                    # Extract the features from the observation
                    obsv = self.extractor(current_state, obs, rng_extract)

                    # Sample action and update scenario
                    rnn_state, action_dist, weights, actor_mean, _, state_features, ind_feats, x_cat_h = network.apply(train_state.params, 
                        rnn_state, (jax.tree_map(extend, obsv), done[jnp.newaxis, ...]), method='action_dist_from_obs')

                    rng, rng_sample = jax.random.split(rng)
                    action_data = action_dist.sample(seed=rng_sample).squeeze(0)

                    # Prepare poses for transforming between global and local
                    _, is_sdc = jax.lax.top_k(current_state.object_metadata.is_sdc, 1)
                    sdc_xy = jnp.take_along_axis(current_state.current_sim_trajectory.xy, is_sdc[..., None, None], 1)
                    sdc_yaw = jnp.take_along_axis(current_state.current_sim_trajectory.yaw, is_sdc[..., None], 1)
                    st_pose = ObjectPose2D.from_center_and_yaw(xy=sdc_xy, yaw=sdc_yaw)
                    st_pose_matrix = st_pose.matrix[..., 0, :, :]
                    st_delta_yaw = st_pose.delta_yaw[..., 0, :]
                    st_pose_matrix_inv = jnp.linalg.inv(st_pose_matrix)

                    if self.use_planner_for_eval:
                        p = network.apply(train_state.params, state_features, action_data[None], x_cat_h, method='world_dynamics')[0]['planner_next_state'] # (1, 100, 3)
                        v = subtract_from_sdc_yawvel(jax.lax.stop_gradient(current_state), -p[0][:, None], st_pose_matrix, st_delta_yaw, st_pose_matrix_inv)
                        alt_next_state_plan = create_state_with_sdc_yawvel(jax.lax.stop_gradient(dataclasses.replace(datatypes.update_state_by_log(current_state, 1), timestep=current_state.timestep[0] + 1)), v)
                        action = inverse_kinematics(current_state, alt_next_state_plan, self.dynamics_model_noclip)
                    else:
                        action = datatypes.Action(data=action_data, valid=jnp.ones((weights.shape[1], 1), dtype='bool'))

                    # Patch bug in waymax (squeeze timestep dimension when using reset --> need squeezed timestep for update)
                    current_timestep = current_state['timestep']
                    # Squeeze timestep dim
                    current_state = dataclasses.replace(current_state, timestep=current_timestep[0])
                    current_state = self.env.step(current_state, action)

                    # Unsqueeze timestep dim
                    current_state = dataclasses.replace(current_state, timestep=current_timestep + 1)
                    metric = self.env.metrics(dataclasses.replace(current_state, timestep=current_state.timestep[0])) # Fix another bug here
                    return (current_state, rnn_state, rng), metric

                # We run K roll-outs and compute K sets of metrics, then we report those from the best one + ADE
                imgs = []
                metrics_per_mode = []
                sim_states = []
                for _ in range(self.num_modes):
                    rng, rng_extract = jax.random.split(rng)
                    rng, rng_extract = jax.random.split(rng_extract)
                    # with jax.disable_jit():
                    carry_k, scenario_metrics = jax.lax.scan(f=_eval_step, 
                        init=(current_state, rnn_state, rng), 
                        xs=rng_extract[None].repeat(TRAJ_LENGTH - self.env.config.init_steps, axis=0),
                        length=TRAJ_LENGTH - self.env.config.init_steps)
                    metrics_per_mode.append(scenario_metrics)
                    sim_states.append(carry_k[0])

                # We store all modes and the best one
                agg_metric = jnp.stack([i['log_divergence'].value for i in metrics_per_mode], axis=0) # [K, T, A=B=1]
                agg_metric_valid = jnp.stack([i['log_divergence'].valid for i in metrics_per_mode], axis=0) # [K, T, A=B=1]
                best_mode = ((agg_metric * agg_metric_valid).sum(1) / agg_metric_valid.sum(1)).argmin(0)
                metrics_per_mode.append(best_mode)
                scenario_metrics = metrics_per_mode
                return imgs, scenario_metrics, sim_states
            
            else: # If we are planning --> Do MPC
                def extend(x):
                    if isinstance(x, jnp.ndarray):
                        return x[jnp.newaxis, ...]
                    else:
                        return x

                def plan(current_state, rnn_state, rng, rng_extract):
                    """
                    Selects an action for the current state using MPC planning.
                    Takes s(t), c(t), h(t-1). Returns a(t) and rg(t)
                    t = current time step to produce action for
                    s(t) = simulator state
                    h(t-1) = RNN hidden state which has not processed the current state yet
                    a(t) = action
                    rg(t) = roadgraph features from the current state.
                    """                
                    def simulate(state_features, rnn_state, dones, x_cat_h, current_state, rng, action_component=-1):
                        """
                        Iterate the world model to produce imagined trajectories.
                        g(t), h(t) --> rewards and actions
                        g(t) = state_features
                        h(t) = rnn_state which has processed the current observation
                        rewards are r(t), r(t+1), r(t+2), ...
                        actions are a(t), a(t+1), a(t+2), ...
                        """
                        rewards = []
                        actions = []
                        wm_preds_all = []
                        for t in range(self.planning_length):
                            # Select action, h(t), g(t) --> a(t)
                            action_dist, weights, actor_mean, _ = network.apply(train_state.params, x_cat_h, method='action_dist_from_features')
                            rng, rng_sample = jax.random.split(rng)
                            
                            if action_component >= 0:
                                action_data = actor_mean[0, :, :, action_component]
                            else:
                                action_data = action_dist.sample(seed=rng_sample).squeeze(0)

                            B = action_data.shape[0]
                            # action_data = jnp.stack((jnp.zeros((B)), jnp.ones((B)) * 0.1), axis=-1) # Left turn
                            # action_data = jnp.stack((jnp.zeros((B)), jnp.ones((B)) * (-0.1)), axis=-1) # Right turn
                            # action_data = jnp.stack((jnp.ones((B)) * 6, jnp.zeros((B))), axis=-1) # Max acceleration
                            # action_data = jnp.stack((jnp.ones((B)) * (-6), jnp.zeros((B))), axis=-1) # Max break

                            # Prepare poses for transforming between global and local
                            _, is_sdc = jax.lax.top_k(current_state.object_metadata.is_sdc, 1)
                            sdc_xy = jnp.take_along_axis(current_state.current_sim_trajectory.xy, is_sdc[..., None, None], 1)
                            sdc_yaw = jnp.take_along_axis(current_state.current_sim_trajectory.yaw, is_sdc[..., None], 1)
                            st_pose = ObjectPose2D.from_center_and_yaw(xy=sdc_xy, yaw=sdc_yaw)
                            st_pose_matrix = st_pose.matrix[..., 0, :, :]
                            st_delta_yaw = st_pose.delta_yaw[..., 0, :]
                            st_pose_matrix_inv = jnp.linalg.inv(st_pose_matrix)

                            if self.use_planner_for_eval:
                                p = network.apply(train_state.params, state_features, action_data[None], x_cat_h, method='world_dynamics')[0]['planner_next_state'] # (1, 100, 3)
                                v = subtract_from_sdc_yawvel(jax.lax.stop_gradient(current_state), -p[0][:, None], st_pose_matrix, st_delta_yaw, st_pose_matrix_inv)
                                alt_next_state_plan = create_state_with_sdc_yawvel(jax.lax.stop_gradient(dataclasses.replace(datatypes.update_state_by_log(current_state, 1), timestep=current_state.timestep[0] + 1)), v)
                                action = inverse_kinematics(current_state, alt_next_state_plan, self.dynamics_model_noclip)
                                action_data = action.data
                            else:
                                action = datatypes.Action(data=action_data, valid=jnp.ones((weights.shape[1], 1), dtype='bool'))
  
                            # Predict world dynamics, g(t), a(t) --> g(t+1), r(t)
                            wm_preds, imagined_reward = network.apply(train_state.params, state_features, action_data[None], x_cat_h, method='world_dynamics')
                            if self.use_inv_opt_state_as_rewards:
                                imagined_reward = -jnp.linalg.norm(wm_preds['inv_opt_state'][..., :2], axis=-1)[..., None]
                            next_state_features = wm_preds['next_latent_state']

                            # Update RNN - we don't make dones recurrent because they should always be False, we're never done in the simulation
                            # g(t+1), h(t) --> h(t+1)
                            new_rnn_state, _ = network.apply(train_state.params, next_state_features, rnn_state, dones, method='update_rnn_only')

                            # Update x_cat_h
                            x_cat_h = jnp.concatenate((new_rnn_state[None], next_state_features), axis=-1)

                            # Store memories and update
                            actions.append(action_data)
                            rewards.append(imagined_reward)
                            wm_preds_all.append(wm_preds)
                            state_features = next_state_features # g(t) := g(t+1)
                            rnn_state = new_rnn_state # h(t) := h(t+1)
                        return rewards, actions, wm_preds_all, rng

                    # Obtain state features - here we observe s(t) and obtain the latent world state g(t), the RNN feats h(t), and the roadgraph features rg(t)
                    # g(t) is called state_features
                    # h(t) is called rnn_state
                    done = current_state.is_done
                    obs = datatypes.sdc_observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'])
                    rng, rng_obs = jax.random.split(rng)

                    if self.obs_mask is not None:
                        obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)
                    obsv = self.extractor(current_state, obs, rng_extract)
                    rnn_state, state_features, _, x_cat_h = network.apply(train_state.params, 
                        rnn_state, (jax.tree_map(extend, obsv), done[jnp.newaxis, ...]), method='rnn_state_from_obs')

                    # Roll-out simulations
                    simulations = [] # list[ (list[rewards], list[actions]), list[dicts]]
                    for j in range(self.num_imagined_rollouts):
                        rng, rng_sim = jax.random.split(rng)
                        r, a, wm_preds, rng = simulate(state_features, rnn_state, done[None], x_cat_h, 
                                current_state, rng_sim, action_component=-1)
                        for i in range(len(wm_preds)):
                            del wm_preds[i]['next_latent_state']
                        simulations.append((r, a, wm_preds))
                    
                    # Evaluate simulations
                    if self.use_rewards:
                        simulation_rewards = [imagined_traj[0] for imagined_traj in simulations]
                        stacked_rewards = jnp.stack([jnp.concatenate(simulation_rewards[i], 0) for i in range(len(simulation_rewards))]) # (Nsim, PlanHorizon, B, 1)
                    else:
                        # inv_opt_states = jnp.concatenate([imagined_traj[-1][0]['inv_opt_state'] for imagined_traj in simulations], axis=0)[:, None]
                        inv_opt_states = jnp.concatenate([s[-1][0]['inv_opt_state'] for s in simulations], axis=0)
                        inv_opt_state_norms = jnp.linalg.norm(inv_opt_states, ord=2, axis=-1)

                    # Take best actions
                    simulation_actions = [imagined_traj[1] for imagined_traj in simulations]
                    if self.use_planner_for_eval:
                        stacked_actions = jnp.stack([jnp.stack([j for j in simulation_actions[i]], 0) for i in range(len(simulation_actions))], 0)
                    else:
                        stacked_actions = jnp.stack([jnp.stack(simulation_actions[i], 0) for i in range(len(simulation_actions))], 0) # (Nsim, PlanHorizon, B, 2)

                    # Transpose from (Nsim, PlanHorizon, B, 2) to (B, Nsim, PlanHorizon, 2)
                    # Take only first action --> (B, Nsim, 2)
                    # Best actions are (B,) --> turn them into (B, 1, 1)
                    # best_actions = jnp.take_along_axis(stacked_actions.transpose(2, 0, 1, 3)[:, :, 0], best_traj_indices[:, None, None], axis=1)
                    best_actions = stacked_actions.transpose(2, 0, 1, 3)[:, :, 0].mean(1, keepdims=True)

                    if self.use_rewards:
                        scores = stacked_rewards.transpose(2, 0, 1, 3).squeeze(-1).sum(-1)
                        n_sim = scores.shape[1]
                        top_scores = jnp.argsort(scores, -1, descending=False)[:, -min(self.top_k, n_sim):]
                    else:
                        inv_opt_states = jnp.concatenate([s[-1][0]['inv_opt_state'] for s in simulations], axis=0)
                        scores = jnp.linalg.norm(inv_opt_states, ord=2, axis=-1).transpose(1, 0)
                        n_sim = scores.shape[1]
                        top_scores = jnp.argsort(scores, -1, descending=False)[:, :min(self.top_k, n_sim)]

                    batch_indices = jnp.arange(scores.shape[0])[:, None]
                    actions = stacked_actions.transpose(2, 0, 1, 3)[..., 0, :] # Only the first action from each trajectory
                    # selected_actions = actions[batch_indices, top_scores]
                    selected_actions = jnp.take_along_axis(actions, top_scores[..., None], axis=1)
                    best_actions = selected_actions.mean(-2, keepdims=True)
                    return best_actions.squeeze(1), simulations, rng

                rng, rng_extract = jax.random.split(rng)
                rng, rng_log = jax.random.split(rng)

                def handle_history(current_state, rng_log):
                    """
                    Handle the history.
                    We return s(t), c(t), h(t-1)
                    t = current timestep
                    s(t) = simulator state
                    h(t-1) = RNN hidden state after processing 0:t-1 included. It has not processed the current step.
                    """
                    # Obtain observations on the first self.env.config.init_steps from the log trajectory
                    # After we execute this, current_state will be s(t)
                    (current_state, rng), log_traj_batch = jax.lax.scan(f = _log_step, 
                        init = (current_state, rng_log),
                        xs = rng_extract[None].repeat(self.env.config.init_steps - 1, axis=0),
                        length=self.env.config.init_steps - 1)

                    # Evolve the RNN hidden state on the log history, obtaining h(t-1)
                    batch_size = log_traj_batch.done.shape[1]
                    expert_actions = jnp.zeros((10, batch_size, 2)) # This gets passed at test time but we don't use it
                    rnn_state, _, _, _, _, _, hist_latent_state, pred_rewards, hist_state_features, individual_feats, x_cat_h = network.apply(
                        train_state.params, init_rnn_state_eval, (log_traj_batch.obs, log_traj_batch.done, rng_log), expert_actions)
                           
                    return current_state, rnn_state, rng
            
                def execute_action_and_update(selected_action, current_state, rnn_state, rng):
                    """ 
                    s(t), c(t), h(t-1), a(t) ----> s(t+1), c(t+1), h(t) 
                    t = current timestep
                    s(t) = simulator state
                    h(t-1) = RNN hidden state from previous step, i.e. it has processed everything from the past, but has not processed s(t)
                    a(t) = the selected action
                    """
                    # Update the RNN state, this requires creating a new observation using the new simulator state
                    # h(t-1), s(t) --> h(t)
                    done = current_state.is_done
                    obs = datatypes.sdc_observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'])
                    rng, rng_obs = jax.random.split(rng)
                    if self.obs_mask is not None:
                        obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)
                    obsv = self.extractor(current_state, obs, rng_extract)
                    rnn_state, _, _, _ = network.apply(train_state.params, rnn_state, 
                        (jax.tree_map(extend, obsv), done[jnp.newaxis, ...]), method='rnn_state_from_obs')
 
                    # Execute selected action
                    # s(t), a(t) --> s(t+1)
                    batch_size = rnn_state.shape[0]
                    action = datatypes.Action(data=selected_action, valid=jnp.ones((batch_size, 1), dtype='bool'))
                    current_timestep = current_state['timestep']
                    current_state = dataclasses.replace(current_state, timestep=current_timestep[0])
                    current_state = self.env.step(current_state, action)
                    current_state = dataclasses.replace(current_state, timestep=current_timestep + 1)

                    # Evaluate the real-world transition
                    metric = self.env.metrics(dataclasses.replace(current_state, timestep=current_state.timestep[0])) # Fix another bug here
                    return metric, current_state, rnn_state
                
                current_state, rnn_state, rng = handle_history(current_state, rng_log)
                # At this point current_state s(t) is the one to be processed
                # rnn_state h(t-1) is the RNN state after processing all the 10 history timesteps. It has not processed the current state yet.
                # Thus, we have s(t), c(t), h(t-1)

                def plan_and_execute(carry, x):
                    current_state, rnn_state, rng = carry
                    rng_extract = x

                    # Do the planning, obtaining current action to execute a(t), and actual roadgraph features rg(t)
                    rng_old = rng
                    selected_action, simulations, rng = plan(current_state=current_state, rnn_state=rnn_state, rng=rng, rng_extract=rng_extract)
                
                    # Now execute a(t) and update the actual agent state, also obtaining metrics m(t)
                    # s(t), c(t), h(t-1), a(t) ---> s(t+1), c(t+1), h(t), m(t)
                    metric, current_state, rnn_state = execute_action_and_update(
                            selected_action, current_state, rnn_state, rng_old)
                    return (current_state, rnn_state, rng), (current_state, simulations, metric)

                
                rng, rng_extract = jax.random.split(rng)
                rng, rng_extract = jax.random.split(rng_extract)
                carry_k, outs = jax.lax.scan(f=plan_and_execute, 
                        init=(current_state, rnn_state, rng), 
                        xs=rng_extract[None].repeat(TRAJ_LENGTH - self.env.config.init_steps, axis=0),
                        length=TRAJ_LENGTH - self.env.config.init_steps)

                imgs = None
                batch_size = rnn_state.shape[0]
                states, simulations, metrics = outs
                scenario_metrics = [metrics, jnp.zeros(batch_size, dtype=int)]
                sim_states = {'sim_states': states, 'simulations': simulations}
                return imgs, scenario_metrics, sim_states
            
        jit_eval_scenario = jax.jit(_eval_scenario)

        def compute_any_valid(value, valid):
            masked_values = jnp.where(valid, value, False)
            any_valid = jnp.any(masked_values, axis=(0, 1))
            return any_valid

        def compute_nanmean_valid(value, valid):
            masked_values = jnp.where(valid, value, jnp.nan)
            mean_valid = jnp.nanmean(masked_values, axis=(0, 1))
            return mean_valid

        def compute_nanmax_valid(value, valid):
            masked_values = jnp.where(valid, value, jnp.nan)
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

                all_scenario_metrics = {}

                scenario = jit_postprocess_fn(data)
                timestep = scenario.timestep
                current_state = self.env.reset(scenario) #TODO Continue here
                current_state = dataclasses.replace(current_state, timestep=timestep) # Fix a bug
                # current_state = scenario
                
                # Scenario does not contain the SDC
                no_sdc = not jnp.any(scenario.object_metadata.is_sdc)

                if no_sdc:
                    pass
                else:
                    rng, rng_eval = jax.random.split(rng)
                    _, scenario_metrics, sim_states = jit_eval_scenario(train_state, current_state, rng_eval)
                    best_mode = scenario_metrics[-1] # (B=A,)
                    
                    # Batch subsampling based on best mode - gets a bit complicated here
                    metric_names = scenario_metrics[0].keys()
                    selected_scenario_metrics = {}
                    for metric_name in metric_names:
                        # Stack metric values and valids
                        all_metric_values = jnp.stack([i[metric_name].value for i in scenario_metrics[:-1]], axis=0)
                        all_metric_valids = jnp.stack([i[metric_name].valid for i in scenario_metrics[:-1]], axis=0)
                        
                        # Take only the best mode for each sample
                        selected_values = jnp.take_along_axis(all_metric_values, best_mode[None, None], axis=0)
                        selected_valids = jnp.take_along_axis(all_metric_valids, best_mode[None, None], axis=0)

                        # Create a metric result
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
