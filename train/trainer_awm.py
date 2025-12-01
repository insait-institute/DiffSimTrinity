import dataclasses
from flax.training.train_state import TrainState
import functools
import jax
import jax.numpy as jnp
from jax import random

import os
import optax
from optax import softmax_cross_entropy_with_integer_labels
import pickle
from tqdm import tqdm, trange
from typing import NamedTuple
import time

from waymax import agents
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax import env as _env

import sys
sys.path.append('./')

from configs.consts import N_TRAINING, N_VALIDATION, TRAJ_LENGTH, N_FILES
from models.feature_extractor import KeyExtractor, KeyExtractorWithIntermediates
from models.state_processing import ExtractObs, ExtractObsForAWM
from models.rnn_policy import ActorCriticForAWM, ScannedRNN
from utils.obs_mask import SpeedConicObsMask, SpeedGaussianNoise, SpeedUniformNoise, ZeroMask
from waymax.utils.geometry import transform_direction, transform_points
from waymax.datatypes.observation import ObjectPose2D
from waymax.dynamics.bicycle_model import compute_inverse
from utils.env_and_state_manipulation import *
# import wandb

# wandb.init(project='waymax_vision', name='debug1')


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


class Trainer(object):

    def __init__(self, config, env_config, train_dataset, val_dataset):
        self.config = config
        self.env_config = env_config

        # Device
        self.devices = jax.devices()
        print(f'Available devices: {self.devices}')

        # Minibatch
        self.n_minibatch = max([n for n in range(len(self.devices), 0, -1) if self.config['num_envs'] % n == 0])
        self.n_minibatch_eval = max([n for n in range(len(self.devices), 0, -1) if self.config['num_envs_eval'] % n == 0])
        self.mini_batch_size = self.config['num_envs'] // self.n_minibatch
        self.mini_batch_size_eval = self.config['num_envs_eval'] // self.n_minibatch_eval

        self._post_process = functools.partial(dataloader.womd_factories.simulator_state_from_womd_dict, include_sdc_paths=config['include_sdc_paths'],)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.key = random.PRNGKey(self.config['key'])

        if 'dynamics' not in self.config.keys():
            self.config['dynamics'] = 'bicycle'

        if self.config['dynamics'] == 'bicycle':
            self.wrapped_dynamics_model = CustomBicycleDynamics()
            self.wrapped_dynamics_noclip = CustomBicycleDynamicsNoActionClip()
            # self.wrapped_dynamics_model = dynamics.InvertibleBicycleModel()
        elif self.config['dynamics'] == 'delta':
            self.wrapped_dynamics_model = dynamics.DeltaLocal()
        else:
            raise ValueError('Unknown dynamics')

        if config['env_type'] == 'planning':
            self.dynamics_model = _env.PlanningAgentDynamics(self.wrapped_dynamics_model)
            self.dynamics_model_noclip = _env.PlanningAgentDynamics(self.wrapped_dynamics_noclip)
        else:
            self.dynamics_model = self.wrapped_dynamics_model

        if config['discrete']:
            action_space_dim = self.dynamics_model.action_spec().shape
            self.dynamics_model = dynamics.discretizer.DiscreteActionSpaceWrapper(dynamics_model=self.dynamics_model,
                                                                                  bins=config['bins'] * jnp.array(action_space_dim))

        assert(not config['discrete'])

        if config['env_type'] == 'planning':
            # self.env = PlanningAgentEnvironmentDiffMetrics(dynamics_model=self.wrapped_dynamics_model, config=env_config)
            self.env = PlanningAgentEnvironment(dynamics_model=self.wrapped_dynamics_model, config=env_config)
            self.env_noclip = _env.PlanningAgentEnvironment(dynamics_model=self.wrapped_dynamics_noclip, config=env_config)
        else:
            self.env = _env.MultiAgentEnvironment(dynamics_model=self.wrapped_dynamics_model, config=env_config)


        # Params and settings
        self.should_validate = False
        self.extractor = extractors[self.config['extractor']](self.config)
        self.feature_extractor = feature_extractors[self.config['feature_extractor']]
        self.feature_extractor_kwargs = self.config['feature_extractor_kwargs']
        self.use_planner_for_train = self.config['use_planner_for_train']
        print("Using planner for train:", self.use_planner_for_train)
        
        self.select_action_from_full_gmm = self.config['select_action_from_full_gmm']
        print("Select action from full GMM:", self.select_action_from_full_gmm)

        # Observability mask
        if 'obs_mask' not in self.config.keys():
            self.config['obs_mask'] = None

        if self.config['obs_mask']:
            self.obs_mask = obs_masks[self.config['obs_mask']](**self.config['obs_mask_kwargs'])
        else:
            self.obs_mask = None

    # SCHEDULERS
    def linear_schedule(self, count):
        n_update_per_epoch = (self.config['num_training_data'] * self.config['num_files'] / N_FILES) // self.config["num_envs"]
        n_epoch = jnp.array([count // n_update_per_epoch])
        frac = jnp.where(n_epoch <= 20, 1, 1 / (2**(n_epoch - 20)))
        return self.config["lr"] * frac

    def train(self,):
        # Initialize networks and components
        network = ActorCriticForAWM(self.dynamics_model.action_spec().shape[0],
                                self.dynamics_model.action_spec().minimum,
                                self.dynamics_model.action_spec().maximum,
                                feature_extractor_class=self.feature_extractor ,
                                feature_extractor_kwargs=self.feature_extractor_kwargs)
        feature_extractor_shape = self.feature_extractor_kwargs['final_hidden_layers']
        init_x = self.extractor.init_x(self.mini_batch_size)
        init_rnn_state_train = ScannedRNN.initialize_carry((self.mini_batch_size, feature_extractor_shape))
        init_rnn_state_eval = ScannedRNN.initialize_carry((self.mini_batch_size_eval, feature_extractor_shape))
        network_params = network.init(self.key, init_rnn_state_train, 
                (init_x[0], init_x[1], random.PRNGKey(self.config['key']+1)), jnp.zeros((1, self.mini_batch_size, 2)))
        num_params = sum([x.size for x in jax.tree_leaves(network_params)])
        print("Num parameters:", num_params)


        if self.config["lr_scheduler"]:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["max_grad_norm"]),
                optax.adam(learning_rate=self.linear_schedule, eps=1e-5),
            )
        elif self.config['lr_cosine']:

            n_update_per_epoch = (self.config['num_training_data'] * self.config['num_files'] / N_FILES) // self.config["num_envs"]
            transition_step = n_update_per_epoch * self.config['lr_transition_epoch']
            cosine_annealing = optax.cosine_onecycle_schedule(transition_step, self.config['lr_max'], div_factor=25, final_div_factor=100)

            tx = optax.chain(
                optax.clip_by_global_norm(self.config["max_grad_norm"]),
                optax.adam(learning_rate=cosine_annealing, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["max_grad_norm"]),
                optax.adam(self.config["lr"], eps=1e-5),
            )

        # # If we want to load previous weights, except in a few layers, initialized at random (here odometry layers)
        # with open('File with trained weights',  'rb') as file:
        #     prev_params = pickle.load(file)
        #     prev_params['params']['odometry_fc1']['kernel'] = network_params['params']['odometry_fc1']['kernel']
        #     prev_params['params']['odometry_fc1']['bias'] = network_params['params']['odometry_fc1']['bias']
        #     prev_params['params']['odometry_fc2']['kernel'] = network_params['params']['odometry_fc2']['kernel']
        #     prev_params['params']['odometry_fc2']['bias'] = network_params['params']['odometry_fc2']['bias']
        #     prev_params['params']['odometry_fc3']['kernel'] = network_params['params']['odometry_fc3']['kernel']
        #     prev_params['params']['odometry_fc3']['bias'] = network_params['params']['odometry_fc3']['bias']
        #     prev_params['params']['ln7']['bias'] = network_params['params']['ln7']['bias']
        #     prev_params['params']['ln7']['scale'] = network_params['params']['ln7']['scale']
        #     prev_params['params']['ln8']['bias'] = network_params['params']['ln8']['bias']
        #     prev_params['params']['ln8']['scale'] = network_params['params']['ln8']['scale']
        #     network_params = prev_params

        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx,)

        def gaussian_entropy(std):
            entropy = 0.5 * jnp.log( 2 * jnp.pi * jnp.e * std**2 + 1e-8)
            entropy = entropy.sum(axis=(-2), keepdims=True)
            return entropy

        def categorical_entropy(probabilities):
            entropy = -jnp.sum(probabilities * jnp.log(probabilities + 1e-8), axis=-1)
            return entropy

        # Jitted functions
        jit_postprocess_fn = jax.jit(self._post_process)

        # Update simulator with log
        def _log_step(cary, aux_inputs):
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
            # rng, rng_extract = jax.random.split(rng)
            obsv = self.extractor(current_state, obs, rng_extract)
            transition = Transition(done, None, obsv)
            expert_action = agents.expert.infer_expert_action(current_state, self.env.dynamics)

            # Update the simulator with the log trajectory
            current_state = datatypes.update_state_by_log(current_state, num_steps=1)
            return (current_state, rng), (transition, expert_action)

        # Divide scenario into minibatches
        def _minibatch(scenario):
            minibatched_scenario = jax.tree_map(lambda x : x.reshape(self.n_minibatch, self.mini_batch_size, *x.shape[1:]), scenario)
            return minibatched_scenario

        # Compute loss, grad on a single minibatch
        def _single_update(train_state, data, rng):
            """
            Idea is that we act out the history according to the log. Then we simulate the future by executing the agents' actions.
            Then we directly supervise the trajectory (by using the differentiability of the environment).
            """
            scenario = jit_postprocess_fn(data)
            current_state = self.env.reset(scenario)
            current_state = dataclasses.replace(current_state, timestep=scenario.timestep) # Fix a bug
            rng, rng_extract = jax.random.split(rng)
            
            def extend(x):
                if isinstance(x, jnp.ndarray):
                    return x[jnp.newaxis, ...]
                return x

            def _loss_fn(params, init_rnn_state, rng, current_state):
                # Compute the rnn_state on first self.env.config.init_steps from the log trajectory
                rng, rng_log = jax.random.split(rng)
                (current_state, rng), (log_traj_batch, expert_action) = jax.lax.scan(f = _log_step,
                    init = (current_state, rng_log),
                    xs = rng_extract[None].repeat(self.env.config.init_steps - 1, axis=0),
                    length = self.env.config.init_steps - 1)
            
                 # Evolve the hidden state on the log history
                rnn_state, _, _, _, _, _, hist_latent_state, _, hist_state_features, _, _ = network.apply(
                        params, init_rnn_state, (log_traj_batch.obs, log_traj_batch.done, rng_log), expert_action.data)

                def _select_action_and_execute(cary, aux_inputs):
                    rng_extract = aux_inputs
                    current_state, rnn_state, rng = cary
                    current_state = jax.lax.stop_gradient(current_state) # Most likely crucial
                    done = current_state.is_done

                    # Extract obs in SDC referential
                    obs = datatypes.sdc_observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'])
                    obs = jax.lax.stop_gradient(obs) # This is crucial, otherwise we get NaN gradients!

                    # Mask
                    rng, rng_obs = jax.random.split(rng)
                    if self.obs_mask is not None:
                        obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)

                    # Extract the features from the observation
                    # rng, rng_extract = jax.random.split(rng)
                    obsv = self.extractor(current_state, obs, rng_extract)
                    rnn_state, action_dist, weights, actor_mean, actor_std, state_features, ind_feats, x_cat_h = network.apply(params, \
                        rnn_state, (jax.tree_map(extend, obsv), done[jnp.newaxis, ...]), method='action_dist_from_obs')
                    
                    rng, rng_sample = jax.random.split(rng)
                    
                    if self.select_action_from_full_gmm:
                        batched_a = datatypes.Action(data=actor_mean.squeeze(0).transpose(2, 0, 1), valid=jnp.ones((6, weights.shape[1], 1), dtype='bool')) # (6, B=2)
                        dist = None
                        selected_components = None
                        action_data = action_dist.sample(seed=rng_sample).squeeze(0)
                    else:
                        _, is_sdc = jax.lax.top_k(current_state.object_metadata.is_sdc, 1)
                        batched_a = datatypes.Action(data=actor_mean.squeeze(0).transpose(2, 0, 1), valid=jnp.ones((6, weights.shape[1], 1), dtype='bool')) # (6, B=2)
                        new_states = jax.vmap(self.env.step, in_axes=(None, 0))(
                            dataclasses.replace(current_state, timestep=current_state['timestep'][0]), 
                            batched_a)
                        sim_xy = jnp.take_along_axis(new_states.current_sim_trajectory.xy, is_sdc[None, ..., None, None], axis=2) # (6, B=2, 1, 1, 2)
                        log_xy = jnp.take_along_axis(new_states.current_log_trajectory.xy, is_sdc[None, ..., None, None], axis=2) # (6, B=2, 1, 1, 2)
                        dist = jnp.linalg.norm(sim_xy - log_xy, ord=2, axis=-1).squeeze(-1).squeeze(-1)
                        selected_components = jax.lax.stop_gradient(dist.argmin(axis=0)) # (B,)
                        selected_mu = jnp.take_along_axis(actor_mean, selected_components[None, :, None, None], axis=-1).squeeze(0).squeeze(-1) # (T=1, B=2, 2, 1) --> (B=2, 2)
                        selected_std = jnp.take_along_axis(actor_std, selected_components[None, :, None, None], axis=-1).squeeze(0).squeeze(-1) # (T=1, B=2, 2, 1) --> (B=2, 2)
                        action_data = jax.random.normal(key=rng_sample, shape=(selected_std.shape[0], 2)) * selected_std + selected_mu

                    
                    # Prepare poses for transforming between global and local
                    _, is_sdc = jax.lax.top_k(current_state.object_metadata.is_sdc, 1)
                    sdc_xy = jnp.take_along_axis(current_state.current_sim_trajectory.xy, is_sdc[..., None, None], 1)
                    sdc_yaw = jnp.take_along_axis(current_state.current_sim_trajectory.yaw, is_sdc[..., None], 1)
                    st_pose = ObjectPose2D.from_center_and_yaw(xy=sdc_xy, yaw=sdc_yaw)
                    st_pose_matrix = st_pose.matrix[..., 0, :, :]
                    st_delta_yaw = st_pose.delta_yaw[..., 0, :]
                    st_pose_matrix_inv = jnp.linalg.inv(st_pose_matrix)
                    
                    # If we're going to use the planner, we sample the next state to visit, and then get the action that reaches it
                    if self.use_planner_for_train:
                        p = network.apply(params, state_features, jax.lax.stop_gradient(action_data[None]), x_cat_h, method='world_dynamics')[0]['planner_next_state'] # (1, 100, 3)
                        v = subtract_from_sdc_yawvel(jax.lax.stop_gradient(current_state), -p[0][:, None], st_pose_matrix, st_delta_yaw, st_pose_matrix_inv)
                        alt_next_state_plan = create_state_with_sdc_yawvel(jax.lax.stop_gradient(dataclasses.replace(datatypes.update_state_by_log(current_state, 1), timestep=current_state.timestep[0] + 1)), v)
                        action = inverse_kinematics(current_state, alt_next_state_plan, self.dynamics_model_noclip)
                        # If we're planning, there's only ever one action, coming from the planner
                        batched_a = datatypes.Action(data=action.data[None].repeat(6, 0), valid=jnp.ones((6, action.data.shape[0], 1), dtype='bool')) # (6, B=2)
                    else:
                        action = datatypes.Action(data=action_data, valid=jnp.ones((weights.shape[1], 1), dtype='bool'))

                    # Predict world dynamics
                    next_latent_states, rewards = network.apply(params, jax.lax.stop_gradient(state_features), \
                        jax.lax.stop_gradient(action.data[None]), jax.lax.stop_gradient(x_cat_h), method='world_dynamics')
                    # all_action_data = jnp.concatenate((batched_a.data, action_data[None]), axis=0) # (7, B, 2)
                    # next_latent_states, rewards = network.apply(params, 
                    #     jax.lax.stop_gradient(state_features[:, None].repeat(7, 1)),
                    #     jax.lax.stop_gradient(all_action_data[None]), 
                    #     jax.lax.stop_gradient(x_cat_h[:, None].repeat(7, 1)), method='world_dynamics') 

                    # Patch bug in waymax (squeeze timestep dimension when using reset --> need squeezed timestep for update)
                    current_timestep = current_state['timestep']
                    current_state = dataclasses.replace(current_state, timestep=current_timestep[0]) # Squeeze timestep dim

                    # Do the same for the deterministic transition
                    hypothetical_state_det = self.env.step(current_state, action) # s(t+1)
                    hypothetical_state_det = dataclasses.replace(hypothetical_state_det, timestep=current_timestep + 1)

                    # Next state dynamics prediction requires that we subtract the delta from the next state
                    stp1_minus_delta = subtract_from_sdc_xyyaw(jax.lax.stop_gradient(hypothetical_state_det), next_latent_states['odometry_next_state'].transpose(1, 0, 2),
                            jax.lax.stop_gradient(st_pose_matrix), jax.lax.stop_gradient(st_delta_yaw), jax.lax.stop_gradient(st_pose_matrix_inv))
                    s_next_state_pred = create_state_with_sdc_xyyaw(jax.lax.stop_gradient(current_state), stp1_minus_delta)
                    next_state_odometry = self.env.step(s_next_state_pred, jax.lax.stop_gradient(action))
                    # all_actions = datatypes.Action(data=all_action_data, valid=jnp.ones((7, weights.shape[1], 1), dtype='bool'))
                    # all_hypothetical_state_det = jax.vmap(self.env.step, in_axes=(None, 0))(current_state, all_actions)
                    # all_hypothetical_state_det = dataclasses.replace(all_hypothetical_state_det, timestep=current_timestep[None].repeat(7, 0) + 1)
                    # stp1_minus_delta = jax.vmap(subtract_from_sdc_xyyaw, in_axes=(0, 0, None, None, None))(
                    #     jax.lax.stop_gradient(all_hypothetical_state_det), 
                    #     next_latent_states['odometry_next_state'].transpose(1, 2, 0, 3), 
                    #     jax.lax.stop_gradient(st_pose_matrix), 
                    #     jax.lax.stop_gradient(st_delta_yaw), jax.lax.stop_gradient(st_pose_matrix_inv))
                    # s_next_state_pred = jax.vmap(create_state_with_sdc_xyyaw, in_axes=(None, 0))(jax.lax.stop_gradient(current_state), stp1_minus_delta)
                    # next_state_odometry = jax.vmap(self.env.step, in_axes=(0, 0))(s_next_state_pred, jax.lax.stop_gradient(all_actions))
                    # stp1_local = jax.vmap(project_sdc_to_local, 
                    #     in_axes=(0, None, None))(all_hypothetical_state_det, st_pose_matrix, st_delta_yaw) # TODO: Delete this later, only for odometry finetuning

                    # Inverse optimal state prediction requires that we add the delta to the current state - we instead subtract the negative
                    st_plus_delta = subtract_from_sdc_xy(jax.lax.stop_gradient(current_state), -next_latent_states['inv_opt_state'].transpose(1, 0, 2),
                             jax.lax.stop_gradient(st_pose_matrix), jax.lax.stop_gradient(st_delta_yaw), jax.lax.stop_gradient(st_pose_matrix_inv))
                    alt_s_current = create_state_with_sdc_xy(jax.lax.stop_gradient(current_state), st_plus_delta)
                    next_state_inv = self.env.step(alt_s_current, jax.lax.stop_gradient(action))
                    # st_plus_delta = jax.vmap(subtract_from_sdc_xy, in_axes=(None, 0, None, None, None))(
                    #     jax.lax.stop_gradient(current_state), -next_latent_states['inv_opt_state'].transpose(1, 2, 0, 3),
                    #     jax.lax.stop_gradient(st_pose_matrix), jax.lax.stop_gradient(st_delta_yaw), jax.lax.stop_gradient(st_pose_matrix_inv))
                    # alt_s_current = jax.vmap(create_state_with_sdc_xy, in_axes=(None, 0))(jax.lax.stop_gradient(current_state), st_plus_delta)
                    # next_state_inv = jax.vmap(self.env.step, in_axes=(0, 0))(alt_s_current, jax.lax.stop_gradient(all_actions))

                    # Optimal planning - addition implemented as subtraction with a negative
                    st_plus_delta_opt_plan = subtract_from_sdc_yawvel(
                        jax.lax.stop_gradient(current_state), -next_latent_states['planner_next_state'].transpose(1, 0, 2),
                        jax.lax.stop_gradient(st_pose_matrix), jax.lax.stop_gradient(st_delta_yaw), jax.lax.stop_gradient(st_pose_matrix_inv))
                    alt_next_state_plan = create_state_with_sdc_yawvel(
                        jax.lax.stop_gradient(dataclasses.replace(jax.lax.stop_gradient(hypothetical_state_det), timestep=current_timestep[0]+1)), 
                        st_plus_delta_opt_plan)
                    alt_action_plan = inverse_kinematics(jax.lax.stop_gradient(current_state), alt_next_state_plan, self.dynamics_model_noclip)
                    opt_plan_state = self.env_noclip.step(jax.lax.stop_gradient(current_state), alt_action_plan)
                    # st_plus_delta_opt_plan = jax.vmap(subtract_from_sdc_yawvel, in_axes=(None, 0, None, None, None))(
                    #     jax.lax.stop_gradient(current_state), -next_latent_states['planner_next_state'].transpose(1, 2, 0, 3),
                    #     jax.lax.stop_gradient(st_pose_matrix), jax.lax.stop_gradient(st_delta_yaw), jax.lax.stop_gradient(st_pose_matrix_inv))
                    # alt_next_state_plan = jax.vmap(create_state_with_sdc_yawvel, in_axes=(0, 0))(
                    #     jax.lax.stop_gradient(dataclasses.replace(all_hypothetical_state_det, timestep=all_hypothetical_state_det.timestep[:, 0])), 
                    #     st_plus_delta_opt_plan)
                    # alt_action_plan = jax.vmap(inverse_kinematics, in_axes=(None, 0, None))(jax.lax.stop_gradient(current_state), alt_next_state_plan, self.dynamics_model_noclip)
                    # opt_plan_state = jax.vmap(self.env_noclip.step, in_axes=(None, 0))(jax.lax.stop_gradient(current_state), alt_action_plan)

                    current_state = hypothetical_state_det
                    # current_state = datatypes.operations.dynamic_index(all_hypothetical_state_det, index=-1, axis=0, keepdims=False)
                    return (current_state, rnn_state, rng), (weights, actor_mean, actor_std, \
                         hypothetical_state_det, next_state_odometry, next_state_inv, opt_plan_state, \
                         next_latent_states, rewards, state_features, selected_components)

                # Set embeddings of future camera tokens to zeros.
                # T, B, Ncam, N_tokens, D = camera_embeddings.transpose((1, 0, 2, 3, 4)).shape

                # Roll-out until the rest of the episode
                rng, rng_step = jax.random.split(rng)
                (current_state, rnn_state, rng), actor_outputs = jax.lax.scan(f=_select_action_and_execute,
                    init=(current_state, rnn_state, rng_step),
                    xs=rng_extract[None].repeat(self.config["num_steps"], axis=0),
                    length=self.config["num_steps"])

                weights, actor_mean, actor_std, det_state_traj, next_state_odometry, \
                    next_state_inv, opt_plan_state, next_latent, pred_reward, state_features, selected_components = actor_outputs
                
                is_sdc = det_state_traj.object_metadata.is_sdc[..., None, None].repeat(5, -1) # We repeat this to the fullest shape possible
                sim_vals = det_state_traj.current_sim_trajectory.stack_fields(('x', 'y', 'yaw', 'vel_x', 'vel_y'))
                log_vals = det_state_traj.current_log_trajectory.stack_fields(('x', 'y', 'yaw', 'vel_x', 'vel_y'))
                sdc_traj_det = jnp.where(is_sdc, sim_vals, jnp.zeros_like(sim_vals)) # This contains 0s for non-sdc
                sdc_log_traj_det = jnp.where(is_sdc, log_vals, jnp.zeros_like(log_vals)) # This contains 0s for non-sdc
                valid = det_state_traj.current_log_trajectory.valid[..., None].repeat(5, -1) # Repeat to fullest shape possible
                valid = jnp.where(is_sdc, valid, jnp.zeros_like(valid)) # (T, B, N, 1, 5)
                traj_loss_det = (optax.huber_loss(sdc_traj_det, sdc_log_traj_det) * valid).sum() / valid.sum()
                if (not self.select_action_from_full_gmm) and (not self.use_planner_for_train):
                    # Update the weights so that the selected components have larger weight
                    traj_loss_det += softmax_cross_entropy_with_integer_labels(
                        weights.squeeze().reshape(-1, weights.squeeze().shape[-1]), 
                        selected_components.reshape(-1)).mean()

                # World model loss
                all_states = jnp.concat((hist_state_features, state_features[:, 0]), axis=0) # (T, B, C)
                all_pred_world_states = jnp.concat((hist_latent_state['next_latent_state'], next_latent['next_latent_state'][:, 0]), axis=0)
                world_model_loss = optax.huber_loss(all_pred_world_states[:-1], jax.lax.stop_gradient(all_states[1:])).mean()
                
                # Reward loss - negative distance to the corresponding SDC log trajectory
                _, sdc_indices = jax.lax.top_k(det_state_traj.object_metadata.is_sdc, k=1)
                sim_sdc = jnp.take_along_axis(sim_vals, sdc_indices [..., None, None], axis=2)
                log_sdc = jnp.take_along_axis(log_vals, sdc_indices [..., None, None], axis=2)
                dists = jnp.linalg.norm(sim_sdc[..., :2] - log_sdc[..., :2], ord=2, axis=-1).squeeze(-1) # (80, 2, 1)
                reward_loss = optax.huber_loss(pred_reward[:-1].squeeze(1), jax.lax.stop_gradient(-dists[1:])).mean()
                # reward_loss = optax.huber_loss(pred_reward[:, 0, :-1, :, 0], jax.lax.stop_gradient(-dist)).mean()
                total_loss = traj_loss_det + world_model_loss + reward_loss

                # Odometry loss
                # _, sdc_idx = jax.lax.top_k(det_state_traj.object_metadata.is_sdc, k=1)
                pred_odometry = next_state_odometry.current_sim_trajectory.stack_fields(('x', 'y', 'yaw', 'vel_x', 'vel_y'))
                target_odometry = det_state_traj.current_sim_trajectory.stack_fields(('x', 'y', 'yaw', 'vel_x', 'vel_y'))
                odometry_loss = (optax.huber_loss(pred_odometry, jax.lax.stop_gradient(target_odometry)) * valid).sum()/valid.sum()
                total_loss += odometry_loss

                # Planner loss
                pred_planner = opt_plan_state.current_sim_trajectory.stack_fields(('x', 'y', 'yaw'))
                target_planner = det_state_traj.current_log_trajectory.stack_fields(('x', 'y', 'yaw'))
                planner_loss = (optax.huber_loss(pred_planner, jax.lax.stop_gradient(target_planner)) * valid[..., :3]).sum()/valid[..., :3].sum()
                if self.use_planner_for_train:
                    pass
                else:
                    total_loss += planner_loss

                # Inverse state loss
                pred_state_inv = next_state_inv.current_sim_trajectory.stack_fields(('x', 'y'))
                target_state_inv = det_state_traj.current_log_trajectory.stack_fields(('x', 'y'))
                state_inv_loss = (optax.huber_loss(pred_state_inv, jax.lax.stop_gradient(target_state_inv)) * valid[..., :2]).sum()/valid[..., :2].sum()
                total_loss += state_inv_loss

                # Other quantities      
                probabilities = jnp.exp(jax.nn.log_softmax(weights))
                gmm_entropy = (probabilities * gaussian_entropy(actor_std)).sum(axis=-1).mean()
                cat_entropy = categorical_entropy(probabilities).mean()
                return total_loss, (gmm_entropy, cat_entropy, traj_loss_det, world_model_loss, reward_loss, odometry_loss, planner_loss, state_inv_loss, 
                    (weights, actor_mean, actor_std, det_state_traj))

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (loss, (gmm_entropy, cat_entropy, traj_loss_det, world_model_loss, reward_loss, \
                    odometry_loss, planner_loss, state_inv_loss, actor_preds)), grads = grad_fn(train_state.params, init_rnn_state_train, rng_extract, current_state)
            return loss, gmm_entropy, cat_entropy, traj_loss_det, world_model_loss, reward_loss, \
                    odometry_loss, planner_loss, state_inv_loss, actor_preds, grads

        pmap_single_update = jax.pmap(_single_update)

        def _global_update_with_aux(train_state, losses, gmm_entropies, cat_entropies, traj_loss_det, world_model_loss, 
                reward_loss, odometry_loss, planner_loss, state_inv_loss, grads):
            mean_grads = jax.tree_map(lambda x: x.mean(0), grads)
            mean_loss = jax.tree_map(lambda x: x.mean(0), losses)
            mean_gmm_entropy = jax.tree_map(lambda x: x.mean(0), gmm_entropies)
            mean_cat_entropy = jax.tree_map(lambda x: x.mean(0), cat_entropies)
            mean_traj_loss_det = jax.tree_map(lambda x: x.mean(0), traj_loss_det)
            mean_world_model_loss = jax.tree_map(lambda x: x.mean(0), world_model_loss)
            mean_reward_loss = jax.tree_map(lambda x: x.mean(0), reward_loss)
            mean_odometry_loss = jax.tree_map(lambda x: x.mean(0), odometry_loss)
            mean_planner_loss = jax.tree_map(lambda x: x.mean(0), planner_loss)
            mean_state_inv_loss = jax.tree_map(lambda x: x.mean(0), state_inv_loss)
            train_state = train_state.apply_gradients(grads=mean_grads)
            return train_state, mean_loss, mean_gmm_entropy, mean_cat_entropy, mean_traj_loss_det, mean_world_model_loss, mean_reward_loss, \
                mean_odometry_loss, mean_planner_loss, mean_state_inv_loss, mean_grads

        jit_global_update_with_aux = jax.jit(_global_update_with_aux)

        # Evaluate
        def _eval_scenario(train_state, scenario, rng, init_rnn_state):
            rng, rng_extract = jax.random.split(rng)
            
            # Compute the rnn_state on first self.env.config.init_steps from the log trajectory

            rng, rng_log = jax.random.split(rng)
            (current_state, rng), (log_traj_batch, expert_action) = jax.lax.scan(f = _log_step,
                init = (scenario, rng_log),
                xs = rng_extract[None].repeat(self.env.config.init_steps - 1, axis=0),
                length = self.env.config.init_steps - 1)
        
                # Evolve the hidden state on the log history
            rnn_state, _, _, _, _, _, _, _, _, _, _ = network.apply(
                train_state.params, 
                init_rnn_state.repeat(log_traj_batch.obs['roadgraph_map'].shape[1], 0), 
                (log_traj_batch.obs, log_traj_batch.done, rng_log), 
                expert_action.data)

            def extend(x):
                if isinstance(x, jnp.ndarray):
                    return x[jnp.newaxis, ...]
                else:
                    return x

            def _eval_step(cary, aux_inputs):
                """Evaluates a single step without planning and with no camera predictions."""
                rng_extract = aux_inputs
                current_state, rnn_state, rng = cary
                done = current_state.is_done

                # Extract obs in SDC referential
                obs = datatypes.sdc_observation_from_state(current_state, roadgraph_top_k=self.config['roadgraph_top_k'])
                dones = current_state.is_done

                # Mask
                rng, rng_obs = jax.random.split(rng)
                if self.obs_mask is not None:
                    obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)

                # Extract the features from the observation
                # rng, rng_extract = jax.random.split(rng)
                obsv = self.extractor(current_state, obs, rng_extract)

                # Sample action and update scenario
                rnn_state, action_dist, weights, _, _, state_features, ind_feats, x_cat_h = network.apply(train_state.params, 
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

                if self.use_planner_for_train:
                    p = network.apply(train_state.params, jax.lax.stop_gradient(state_features), jax.lax.stop_gradient(action_data[None]), x_cat_h, method='world_dynamics')[0]['planner_next_state'] # (1, 100, 3)
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
            
            rng, rng_step = jax.random.split(rng)

            # Prepare camera embeddings
            # T, B, Ncam, N_tokens, D = camera_embeddings.transpose((1, 0, 2, 3, 4)).shape

            carry_k, scenario_metrics = jax.lax.scan(f=_eval_step, 
                init=(current_state, rnn_state, rng), 
                xs=rng_extract[None].repeat(TRAJ_LENGTH - self.env.config.init_steps, axis=0),
                length=TRAJ_LENGTH - self.env.config.init_steps)
            
            return scenario_metrics

        jit_eval_scenario = jax.jit(_eval_scenario)

        # TRAIN LOOP
        def _update_epoch(train_state, rng):

            # UPDATE NETWORK
            def _update_scenario(train_state, data, rng):

                minibatched_data = _minibatch(data)
                
                rng_pmap = jax.random.split(rng, self.n_minibatch)
                expanded_train_state = jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), self.n_minibatch, axis=0), train_state)

                loss, gmm_entropy, cat_entropy, traj_loss_det, world_model_loss, reward_loss, odometry_loss, planner_loss,\
                    state_inv_loss, actor_preds, grads = pmap_single_update(
                    expanded_train_state, minibatched_data, rng_pmap)
                
                # weights, actor_mean, actor_std = actor_preds
                train_state_new, mean_loss, mean_gmm_entropy, mean_cat_entropy, \
                    mean_traj_det_loss, mean_world_model_loss, mean_reward_loss, mean_odometry_loss, mean_planner_loss,\
                        mean_state_inv_loss, grads = \
                        jit_global_update_with_aux(train_state, loss, gmm_entropy, cat_entropy, 
                        traj_loss_det, world_model_loss, reward_loss, odometry_loss, planner_loss, state_inv_loss, grads)

                return train_state_new, mean_loss, mean_gmm_entropy, mean_cat_entropy, mean_traj_det_loss, mean_world_model_loss, \
                    mean_reward_loss, mean_odometry_loss, mean_planner_loss, mean_state_inv_loss, grads

            metric = {'loss': [], 'gmm_entropy': [], 'cat_entropy': [], "wm_loss": [], "traj_loss": [], \
                      'reward_loss': [], 'odometry_loss':[], 'planner_loss':[], 'state_inv_loss':[]}
            losses = []
            gmm_entropies = []
            cat_entropies = []
            traj_losses = []
            wm_losses = []
            reward_losses = []
            odometry_losses = []
            planner_losses = []
            state_inv_losses = []

            tt = 0
            for data in tqdm(self.train_dataset.as_numpy_iterator(), desc='Training', total=N_TRAINING // self.config['num_envs']):
                tt += 1

                rng, rng_train = jax.random.split(rng)
                train_state, loss, gmm_entropy, cat_entropy, traj_loss_det, wm_loss, reward_loss, odometry_loss, \
                    planner_loss, state_inv_loss, grads = _update_scenario(train_state, data, rng_train)

                losses.append(loss)
                gmm_entropies.append(gmm_entropy)
                cat_entropies.append(cat_entropy)
                traj_losses.append(traj_loss_det)
                wm_losses.append(wm_loss)
                reward_losses.append(reward_loss)
                odometry_losses.append(odometry_loss)
                planner_losses.append(planner_loss)
                state_inv_losses.append(state_inv_loss)

                if tt > (self.config['num_training_data'] * self.config['num_files'] / N_FILES) // self.config['num_envs']:
                    break

            metric['loss'].append(jnp.array(losses).mean())
            metric['gmm_entropy'].append(jnp.array(gmm_entropies).mean())
            metric['cat_entropy'].append(jnp.array(cat_entropies).mean())
            metric['wm_loss'].append(jnp.array(wm_losses).mean())
            metric['traj_loss'].append(jnp.array(traj_losses).mean())
            metric['reward_loss'].append(jnp.array(reward_losses).mean())
            metric['odometry_loss'].append(jnp.array(odometry_losses).mean())
            metric['planner_loss'].append(jnp.array(planner_losses).mean())
            metric['state_inv_loss'].append(jnp.array(state_inv_losses).mean())
            return train_state, metric

        # EVALUATION LOOP
        def _evaluate_epoch(train_state, rng):

            all_metrics = {'log_divergence': [],
                           'max_log_divergence': [],
                           'overlap_rate': [],
                           'overlap': [],
                           'offroad_rate': [],
                           'offroad': []
                           }

            eval_count = 0
            for data in tqdm(self.val_dataset.as_numpy_iterator(), desc='Validation', total=N_VALIDATION // self.config['num_envs_eval'] + 1):
                eval_count += 1
                
                scenario = jit_postprocess_fn(data)
                init_timestep = scenario.timestep
                scenario = self.env.reset(scenario)
                scenario = dataclasses.replace(scenario, timestep=init_timestep)
                if not jnp.any(scenario.object_metadata.is_sdc):
                    # Scenario does not contain the SDC
                    pass
                else:
                    rng, rng_eval = jax.random.split(rng)
                    scenario_metrics = jit_eval_scenario(train_state, scenario, rng_eval, init_rnn_state_eval)

                    for key, value in scenario_metrics.items():
                        if jnp.any(value.valid):
                            all_metrics[key].append(value.value[value.valid].mean())

                    key = 'max_log_divergence'
                    value = scenario_metrics['log_divergence']
                    if jnp.any(value.valid):
                        all_metrics[key].append(value.value.max(axis=0).mean())

                    key = 'overlap_rate'
                    value = scenario_metrics['overlap']
                    if jnp.any(value.valid):
                        all_metrics[key].append(jnp.any(value.value, axis=0).mean())

                    key = 'offroad_rate'
                    value = scenario_metrics['offroad']
                    if jnp.any(value.valid):
                        all_metrics[key].append(jnp.any(value.value, axis=0).mean())

            return train_state, all_metrics

        # LOGS AND CHECKPOINTS
        metrics = {}
        rng = self.key
        should_validate = self.should_validate
        for epoch in range(self.config["num_epochs"]):
            metrics[epoch] = {}

            # Training
            rng, rng_train = jax.random.split(rng)
            train_state, train_metric = _update_epoch(train_state, rng_train)
            metrics[epoch]['train'] = train_metric

            train_message = f"Epoch | {epoch} | Train "
            train_message += f"| loss | {jnp.array(train_metric['loss']).mean():.4f} "
            train_message += f"| gmm_entropy | {jnp.array(train_metric['gmm_entropy']).mean():.4f}"
            train_message += f"| cat_entropy | {jnp.array(train_metric['cat_entropy']).mean():.4f}"
            train_message += f"| traj_loss | {jnp.array(train_metric['traj_loss']).mean():.4f}"
            train_message += f"| wm_loss | {jnp.array(train_metric['wm_loss']).mean():.4f}"
            train_message += f"| reward_loss | {jnp.array(train_metric['reward_loss']).mean():.4f}"
            train_message += f"| odometry_loss | {jnp.array(train_metric['odometry_loss']).mean():.4f}"
            train_message += f"| planner_loss | {jnp.array(train_metric['planner_loss']).mean():.4f}"
            train_message += f"| state_inv_loss | {jnp.array(train_metric['state_inv_loss']).mean():.4f}"
            print(train_message)

            # Validation
            if should_validate and ((epoch % self.config['freq_eval'] == 0) or (epoch == self.config['num_epochs'] - 1)):
                rng, rng_eval = jax.random.split(rng)
                _, val_metric = _evaluate_epoch(train_state, rng_eval)
                metrics[epoch]['validation'] = val_metric
                val_message = f'Epoch | {epoch} | Val | '
                for key, value in val_metric.items():
                    val_message += f" {key} | {jnp.array(value).mean():.4f} | "
                print(val_message)

            if (epoch % self.config['freq_save'] == 0) or (epoch == self.config['num_epochs'] - 1):
                past_log_metric = os.path.join(self.config['log_folder'], f'training_metrics_{epoch - self.config["freq_save"]}.pkl')
                past_log_params = os.path.join(self.config['log_folder'], f'params_{epoch - self.config["freq_save"]}.pkl')

                if os.path.exists(past_log_metric):
                    os.remove(past_log_metric)

                if os.path.exists(past_log_params):
                    os.remove(past_log_params)

                # Checkpoint
                with open(os.path.join(self.config['log_folder'], f'training_metrics_{epoch}.pkl'), "wb") as pkl_file:
                    pickle.dump(metrics, pkl_file)

                # Save model weights
                with open(os.path.join(self.config['log_folder'], f'params_{epoch}.pkl'), 'wb') as f:
                    pickle.dump(train_state.params, f)
        return {"train_state": train_state, "metrics": metrics}
