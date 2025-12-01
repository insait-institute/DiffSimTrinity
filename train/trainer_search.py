import dataclasses
from flax.training.train_state import TrainState
import functools
import jax
import jax.numpy as jnp
from jax import random

import os
import optax
from optax import softmax_cross_entropy_with_integer_labels, sigmoid_binary_cross_entropy
import pickle
from tqdm import tqdm, trange
from typing import NamedTuple
import time
import flax.linen as nn

from waymax import agents
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax import env as _env
from waymax import config as _config

import sys
sys.path.append('./')

from configs.consts import N_TRAINING, N_VALIDATION, TRAJ_LENGTH, N_FILES
from models.feature_extractor import KeyExtractor
from models.state_processing import ExtractObs
from models.rnn_policy import ActorCriticForSearch, ScannedRNN
from utils.obs_mask import SpeedConicObsMask, SpeedGaussianNoise, SpeedUniformNoise, ZeroMask
from waymax.env import PlanningAgentEnvironment
from waymax.datatypes.observation import ObjectPose2D
from waymax.utils.geometry import transform_points
# from utils.env_and_state_manipulation import *


from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


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
            # self.wrapped_dynamics_model = CustomBicycleDynamics()
            self.wrapped_dynamics_model = dynamics.InvertibleBicycleModel()
        elif self.config['dynamics'] == 'delta':
            self.wrapped_dynamics_model = dynamics.DeltaLocal()
        else:
            raise ValueError('Unknown dynamics')

        if config['env_type'] == 'planning':
            self.dynamics_model = _env.PlanningAgentDynamics(self.wrapped_dynamics_model)
        else:
            self.dynamics_model = self.wrapped_dynamics_model

        if config['discrete']:
            action_space_dim = self.dynamics_model.action_spec().shape
            self.dynamics_model = dynamics.discretizer.DiscreteActionSpaceWrapper(dynamics_model=self.dynamics_model,
                                                                                  bins=config['bins'] * jnp.array(action_space_dim))

        assert(not config['discrete'])

        # Build environment
        self.env = _env.PlanningAgentEnvironment(dynamics_model=self.wrapped_dynamics_model, config=env_config)

        # Observation extractor and feature extractor
        self.config['multi_agent'] = False
        self.should_validate = False
        self.extractor = extractors[self.config['extractor']](self.config)
        self.feature_extractor = feature_extractors[self.config['feature_extractor']]
        self.feature_extractor_kwargs = self.config['feature_extractor_kwargs']
        
        self.select_action_from_full_gmm = False
        self.critic_loss_coeff = 1.0
        self.overlap_loss_coeff = 1.0
        self.offroad_loss_coeff = 1.0
        print("Select action from full GMM:", self.select_action_from_full_gmm)
        print("critic loss coeff:", self.critic_loss_coeff)
        print("overlap loss coeff", self.overlap_loss_coeff)
        print("offroad loss coeff", self.offroad_loss_coeff)
        print("Multi agent:", self.config['multi_agent'])

        # DEFINE OBSERVABILITY MASK
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
        network = ActorCriticForSearch(self.dynamics_model.action_spec().shape[0],
                                self.dynamics_model.action_spec().minimum,
                                self.dynamics_model.action_spec().maximum,
                                feature_extractor_class=self.feature_extractor ,
                                feature_extractor_kwargs=self.feature_extractor_kwargs)
        feature_extractor_shape = self.feature_extractor_kwargs['final_hidden_layers']
        init_x = self.extractor.init_x(self.mini_batch_size)
        init_rnn_state_train = ScannedRNN.initialize_carry((self.mini_batch_size, feature_extractor_shape))
        init_rnn_state_eval = ScannedRNN.initialize_carry((self.mini_batch_size_eval, feature_extractor_shape))
        network_params = network.init(self.key, init_rnn_state_train, 
                (init_x[0], init_x[1], random.PRNGKey(self.config['key']+1)))
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
            obsv = self.extractor(current_state, obs, rng_extract, waypoint_mode=0, custom_waypoint=None)
            transition = Transition(done, None, obsv)

            # Update the simulator with the log trajectory
            current_state = datatypes.update_state_by_log(current_state, num_steps=1)
            return (current_state, rng), transition

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
                (current_state, rng), log_traj_batch = jax.lax.scan(f = _log_step,
                    init = (current_state, rng_log),
                    xs = rng_extract[None].repeat(self.env.config.init_steps - 1, axis=0),
                    length = self.env.config.init_steps - 1)
            
                 # Evolve the hidden state on the log history
                rnn_state, _, _, _, _, _, _, _, _, _ = network.apply(
                        params, init_rnn_state, (log_traj_batch.obs, log_traj_batch.done, rng_log))
                
                
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
                    
                    obsv = self.extractor(current_state, obs, rng_extract, waypoint_mode=1, custom_waypoint=None)
                    rnn_state, action_dist, weights, actor_mean, actor_std, state_features, x_cat_h = network.apply(params, \
                        rnn_state, (jax.tree_map(extend, obsv), done[jnp.newaxis, ...]), method='action_dist_from_obs')
                    pred_value = network.apply(params, x_cat_h, method='state_value_from_features')

                    overlap, offroad = network.apply(params, state_features, method='offroad_overlap_from_state_features')
                    
                    rng, rng_sample = jax.random.split(rng)
                    
                    if self.select_action_from_full_gmm:
                        action_data = action_dist.sample(seed=rng_sample).squeeze(0)
                        selected_components = None
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
                        action_data = jax.random.normal(key=rng_sample, shape=(selected_mu.shape[0], 2)) * selected_std + selected_mu
                    
                    action = datatypes.Action(data=action_data, valid=jnp.ones((weights.shape[1], 1), dtype='bool'))
                    # Patch bug in waymax (squeeze timestep dimension when using reset --> need squeezed timestep for update)
                    current_timestep = current_state['timestep']
                    current_state = dataclasses.replace(current_state, timestep=current_timestep[0]) # Squeeze timestep dim

                    # Do the same for the deterministic transition
                    hypothetical_state_det = self.env.step(current_state, action) # s(t+1)
                    hypothetical_state_det = dataclasses.replace(hypothetical_state_det, timestep=current_timestep + 1)

                    # Rewards - computed from the next state (whether it has collisions/offroad)
                    r = self.env.reward(hypothetical_state_det, action=None)
                    metrics = self.env.metrics(current_state)

                    current_state = hypothetical_state_det
                    return (current_state, rnn_state, rng), (weights, actor_mean, actor_std, \
                         hypothetical_state_det, selected_components, r, pred_value, overlap, offroad, metrics)

                # Roll-out until the rest of the episode
                rng, rng_step = jax.random.split(rng)
                (current_state, rnn_state, rng), actor_outputs = jax.lax.scan(f=_select_action_and_execute,
                    init=(current_state, rnn_state, rng_step),
                    xs=rng_extract[None].repeat(self.config["num_steps"], axis=0),
                    length=self.config["num_steps"])

                weights, actor_mean, actor_std, det_state_traj, selected_components, \
                    rewards, pred_state_vals, overlap, offroad, metrics = actor_outputs

                # DiffSim loss
                is_sdc = det_state_traj.object_metadata.is_sdc[..., None, None].repeat(5, -1) # We repeat this to the fullest shape possible
                sim_vals = det_state_traj.current_sim_trajectory.stack_fields(('x', 'y', 'yaw', 'vel_x', 'vel_y'))
                log_vals = det_state_traj.current_log_trajectory.stack_fields(('x', 'y', 'yaw', 'vel_x', 'vel_y'))
                sdc_traj_det = jnp.where(is_sdc, sim_vals, jnp.zeros_like(sim_vals)) # This contains 0s for non-sdc
                sdc_log_traj_det = jnp.where(is_sdc, log_vals, jnp.zeros_like(log_vals)) # This contains 0s for non-sdc
                valid = det_state_traj.current_log_trajectory.valid[..., None].repeat(5, -1) # Repeat to fullest shape possible
                valid = jnp.where(is_sdc, valid, jnp.zeros_like(valid))
                traj_loss_det = (optax.huber_loss(sdc_traj_det, sdc_log_traj_det) * valid).sum() / valid.sum()
                if not self.select_action_from_full_gmm:
                    # Update the weights so that the selected components have larger weight
                    traj_loss_det += softmax_cross_entropy_with_integer_labels(
                        weights.squeeze().reshape(-1, weights.squeeze().shape[-1]), 
                        selected_components.reshape(-1)).mean()
                                
                # Offroad/overlap loss
                overlap_metrics_value = jax.lax.stop_gradient(metrics['overlap'].value).flatten()
                overlap_metrics_valid = jax.lax.stop_gradient(metrics['overlap'].valid).flatten()
                offroad_metrics_value = jax.lax.stop_gradient(metrics['offroad'].value).flatten()
                offroad_metrics_valid = jax.lax.stop_gradient(metrics['offroad'].valid).flatten()

                overlap_loss = (
                    sigmoid_binary_cross_entropy(overlap.squeeze().flatten(), overlap_metrics_value) * overlap_metrics_valid).sum() / overlap_metrics_valid.sum()
                offroad_loss = (
                    sigmoid_binary_cross_entropy(offroad.squeeze().flatten(), offroad_metrics_value) * offroad_metrics_valid).sum() / offroad_metrics_valid.sum()
                
                # Critic loss
                critic_target = jnp.take_along_axis(log_vals - sim_vals, 
                    jax.lax.top_k(current_state.object_metadata.is_sdc, k=1)[1][None, :, :, None, None], 
                    -3).squeeze(-2).squeeze(-2)[..., :2]
                critic_loss = optax.huber_loss(pred_state_vals.squeeze(-3)[1:], jax.lax.stop_gradient(critic_target[:-1])).mean()

                # Total loss
                total_loss = traj_loss_det + self.critic_loss_coeff * critic_loss + \
                        self.overlap_loss_coeff * overlap_loss + self.offroad_loss_coeff * offroad_loss

                # Other quantities      
                probabilities = jnp.exp(jax.nn.log_softmax(weights))
                gmm_entropy = (probabilities * gaussian_entropy(actor_std)).sum(axis=-1).mean()
                cat_entropy = categorical_entropy(probabilities).mean()
                rewards = rewards.sum(0).mean(0)
                return total_loss, (gmm_entropy, cat_entropy, traj_loss_det, critic_loss, overlap_loss, offroad_loss, \
                        rewards, (weights, actor_mean, actor_std, det_state_traj))

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (loss, (gmm_entropy, cat_entropy, traj_loss_det, critic_loss, 
                overlap_loss, offroad_loss, rewards, actor_preds)), grads = grad_fn(train_state.params, init_rnn_state_train, rng_extract, current_state)
            return loss, gmm_entropy, cat_entropy, traj_loss_det, critic_loss, overlap_loss, offroad_loss, rewards, actor_preds, grads

        pmap_single_update = jax.pmap(_single_update)

        def _global_update_with_aux(train_state, losses, gmm_entropies, cat_entropies, traj_loss_det, cr_loss, \
                overlap_loss, offroad_loss, rewards, grads):
            mean_grads = jax.tree_map(lambda x: x.mean(0), grads)
            mean_loss = jax.tree_map(lambda x: x.mean(0), losses)
            mean_gmm_entropy = jax.tree_map(lambda x: x.mean(0), gmm_entropies)
            mean_cat_entropy = jax.tree_map(lambda x: x.mean(0), cat_entropies)
            mean_traj_loss_det = jax.tree_map(lambda x: x.mean(0), traj_loss_det)
            mean_cr_loss = jax.tree_map(lambda x: x.mean(0), cr_loss)
            mean_overlap_loss = jax.tree_map(lambda x: x.mean(0), overlap_loss)
            mean_offroad_loss = jax.tree_map(lambda x: x.mean(0), offroad_loss)
            mean_reward = jax.tree_map(lambda x: x.mean(0), rewards)
            train_state = train_state.apply_gradients(grads=mean_grads)
            return train_state, mean_loss, mean_gmm_entropy, mean_cat_entropy, mean_traj_loss_det, mean_cr_loss, \
                mean_overlap_loss, mean_offroad_loss, mean_reward, mean_grads

        jit_global_update_with_aux = jax.jit(_global_update_with_aux)

        # Evaluate
        def _eval_scenario(train_state, scenario, rng, init_rnn_state):
            rng, rng_extract = jax.random.split(rng)
            
            # Compute the rnn_state on first self.env.config.init_steps from the log trajectory

            rng, rng_log = jax.random.split(rng)
            (current_state, rng), log_traj_batch = jax.lax.scan(f = _log_step,
                init = (scenario, rng_log),
                xs = rng_extract[None].repeat(self.env.config.init_steps - 1, axis=0),
                length = self.env.config.init_steps - 1)
        
            # Evolve the hidden state on the log history
            rnn_state, _, _, _, _, _, _, _, _, _ = network.apply(
                    train_state.params, 
                    init_rnn_state.repeat(log_traj_batch.obs['roadgraph_map'].shape[1], 0),
                    (log_traj_batch.obs, log_traj_batch.done, rng_log))

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
                dones = current_state.is_done

                # Mask
                rng, rng_obs = jax.random.split(rng)
                if self.obs_mask is not None:
                    obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)

                # Extract the features from the observation
                # rng, rng_extract = jax.random.split(rng)
                obsv = self.extractor(current_state, obs, rng_extract)

                # Sample action and update scenario
                rnn_state, action_dist, weights, actor_mean, actor_std, state_features, x_cat_h = network.apply(train_state.params, \
                    rnn_state, (jax.tree_map(extend, obsv), done[jnp.newaxis, ...]), method='action_dist_from_obs')
                
                rng, rng_sample = jax.random.split(rng)
                action_data = action_dist.sample(seed=rng_sample).squeeze(0)
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

                # Store the camera embeddings
                minibatched_data = _minibatch(data)
                
                rng_pmap = jax.random.split(rng, self.n_minibatch)
                expanded_train_state = jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), self.n_minibatch, axis=0), train_state)

                loss, gmm_entropy, cat_entropy, traj_loss_det, critic_loss, \
                    overlap_loss, offroad_loss, rewards, actor_preds, grads = pmap_single_update(expanded_train_state, minibatched_data, rng_pmap)
                
                # weights, actor_mean, actor_std = actor_preds
                train_state_new, mean_loss, mean_gmm_entropy, mean_cat_entropy, mean_traj_det_loss, \
                    mean_critic_loss, mean_overlap_loss, mean_offroad_loss, mean_r, grads = jit_global_update_with_aux(train_state, loss, gmm_entropy, 
                    cat_entropy, traj_loss_det, critic_loss, overlap_loss, offroad_loss, rewards, grads)
                return train_state_new, mean_loss, mean_gmm_entropy, mean_cat_entropy, mean_traj_det_loss, \
                    mean_critic_loss, mean_overlap_loss, mean_offroad_loss, mean_r, grads

            metric = {'loss': [], 'gmm_entropy': [], 'cat_entropy': [], "traj_loss": [], "critic_loss":[],
                    'overlap_loss': [], 'offroad_loss': []}
            losses = []
            gmm_entropies = []
            cat_entropies = []
            traj_losses = []
            critic_losses = []
            overlap_losses = []
            offroad_losses = []

            
            tt = 0
            for data in tqdm(self.train_dataset.as_numpy_iterator(), desc='Training', total=N_TRAINING // self.config['num_envs']):
                tt += 1

                rng, rng_train = jax.random.split(rng)
                train_state, loss, gmm_entropy, cat_entropy, traj_loss_det, cr_loss, overlap_loss, offroad_loss, rewards, grads = _update_scenario(train_state, data, rng_train)

                losses.append(loss)
                gmm_entropies.append(gmm_entropy)
                cat_entropies.append(cat_entropy)
                traj_losses.append(traj_loss_det)
                critic_losses.append(cr_loss)
                overlap_losses.append(overlap_loss)
                offroad_losses.append(offroad_loss)

                if tt > (self.config['num_training_data'] * self.config['num_files'] / N_FILES) // self.config['num_envs']:
                    break

            metric['loss'].append(jnp.array(losses).mean())
            metric['gmm_entropy'].append(jnp.array(gmm_entropies).mean())
            metric['cat_entropy'].append(jnp.array(cat_entropies).mean())
            metric['traj_loss'].append(jnp.array(traj_losses).mean())
            metric['critic_loss'].append(jnp.array(critic_losses).mean())
            metric['overlap_loss'].append(jnp.array(overlap_losses).mean())
            metric['offroad_loss'].append(jnp.array(offroad_losses).mean())
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
            train_message += f"| critic_loss | {jnp.array(train_metric['critic_loss']).mean():.4f}"
            train_message += f"| overlap_loss | {jnp.array(train_metric['overlap_loss']).mean():.4f}"
            train_message += f"| offroad_loss | {jnp.array(train_metric['offroad_loss']).mean():.4f}"
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
