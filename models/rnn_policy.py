import jax
import jax.numpy as jnp
import numpy as np

from typing import Dict, Optional, Union
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal, he_normal
import functools

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry((*ins.shape[:-1], ins.shape[-1])),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=512)(carry, ins) # 128
        # new_rnn_state = nn.LayerNorm()(new_rnn_state)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(input_size):
        # Use a dummy key since the default state init fn is just zeros.
        batch_size, hidden_size = input_size
        return nn.GRUCell(features=512).initialize_carry(jax.random.key(0), (batch_size, hidden_size)) # 128


class ActorCriticForAWM(nn.Module):
    """Used when planning with world models but no future camera prediction."""
    action_dim: int
    action_minimum: jnp.ndarray
    action_maximum: jnp.ndarray
    feature_extractor_class: nn.Module
    feature_extractor_kwargs: Optional[Union[Dict, None]]
    num_components: int = 6
    hidden_units: int = 512  # 128

    def setup(self):
        # Feature extractor module (now initialized in setup)
        self.feature_extractor = self.feature_extractor_class(**self.feature_extractor_kwargs)

        # Layers to be reused
        self.rnn_layer = ScannedRNN()
        
        # Actor network layers
        self.dense_actor_1 = nn.Dense(self.hidden_units, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.dense_actor_2 = nn.Dense(self.hidden_units, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.weight_layer_1 = nn.Dense(self.num_components, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.weight_layer_2 = nn.Dense(self.num_components, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.actor_mean_1 = nn.Dense(self.action_dim * self.num_components, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.actor_mean_2 = nn.Dense(self.action_dim * self.num_components, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.actor_log_std_1 = nn.Dense(self.action_dim * self.num_components, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.actor_log_std_2 = nn.Dense(self.action_dim * self.num_components, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        
        # World models
        self.world_dense_1 = nn.Dense(512)
        self.world_dense_2 = nn.Dense(512)
        self.world_dense_3 = nn.Dense(512)
        self.world_dense_4 = nn.Dense(512)
        self.reward_fc1 = nn.Dense(128)
        self.reward_fc2 = nn.Dense(128)
        self.reward_fc3 = nn.Dense(128)
        self.reward_fc4 = nn.Dense(1)

        # Task-heads
        self.odometry_fc1 = nn.Dense(128)
        self.odometry_fc2 = nn.Dense(128)
        self.odometry_fc3 = nn.Dense(3) # (dx, dy, dyaw)
        self.planner_fc1 = nn.Dense(128)
        self.planner_fc2 = nn.Dense(128)
        self.planner_fc3 = nn.Dense(3) # (dyaw, dvx, dvy)
        self.inv_opt_state_fc1 = nn.Dense(128)
        self.inv_opt_state_fc2 = nn.Dense(128)
        self.inv_opt_state_fc3 = nn.Dense(2) # (dx, dy)
        self.ln7 = nn.LayerNorm()
        self.ln8 = nn.LayerNorm()
        self.ln9 = nn.LayerNorm()
        self.ln10 = nn.LayerNorm()
        self.ln11 = nn.LayerNorm()
        self.ln12 = nn.LayerNorm()

    def __call__(self, rnn_state, x, expert_action):
        """Uses full observation to predict everything - action distribution, sampled action, next sensor features."""
        obs, dones, rng_sample = x

        # State feature extractor
        state_features = self.feature_extractor(obs)
        if isinstance(state_features, tuple):
            state_features, individual_features = state_features
        else:
            individual_features = None
        
        # RNN
        new_rnn_state, x = self.update_rnn_only(state_features, rnn_state, dones)

        # Actor network
        x_cat = jnp.concatenate((x, state_features), axis=-1)  # [T, B, C]
        pi, weights, actor_mean, actor_std = self.action_dist_from_features(x_cat)

        # Select action
        action_data = pi.sample(seed=rng_sample)

        # Predict next world dynamics
        next_latent_state, pred_rewards = self.world_dynamics(
            jax.lax.stop_gradient(state_features), 
            jax.lax.stop_gradient(expert_action), 
            jax.lax.stop_gradient(x_cat))
        return new_rnn_state, pi, weights, actor_mean, actor_std, action_data, next_latent_state, pred_rewards, state_features, individual_features, x_cat

    def action_dist_from_features(self, x_cat):
        """
        Returns an action distribution from concatenated observed features and RNN hidden state.
        """
        x_actor = jax.nn.relu(self.dense_actor_1(x_cat))
        x_actor = self.dense_actor_2(x_actor)

        weights = jax.nn.relu(self.weight_layer_1(x_actor))
        weights = self.weight_layer_2(weights)

        weights = weights.reshape((*weights.shape[:-1], 1, self.num_components))  # (T, B, N, 1, Ncomp)

        actor_mean = jax.nn.relu(self.actor_mean_1(x_actor))
        actor_mean = self.actor_mean_2(actor_mean)
        actor_mean = actor_mean.reshape((*actor_mean.shape[:-1], self.action_dim, self.num_components))

        actor_log_std = jax.nn.relu(self.actor_log_std_1(x_actor))
        actor_log_std = jnp.clip(self.actor_log_std_2(actor_log_std), a_min=-5, a_max=2)
        actor_std = jnp.exp(actor_log_std)
        actor_std = actor_std.reshape((*actor_std.shape[:-1], self.action_dim, self.num_components))

        # Action distribution
        pi = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=weights),
            components_distribution=tfd.Normal(loc=actor_mean, scale=actor_std),
            reparameterize=True)
        return pi, weights, actor_mean, actor_std

    def update_rnn_only(self, state_features, rnn_state, dones):
        """
        Updates the RNN using obs(t) and h(t-1). Returns h(t)
        """
        rnn_in = (state_features, dones)
        new_rnn_state, x = self.rnn_layer(rnn_state, rnn_in)
        return new_rnn_state, x

    def action_dist_from_obs(self, rnn_state, x):
        """Uses full observation to predict everything up to the action distribution."""
        obs, dones = x

        # State feature extractor
        state_features = self.feature_extractor(obs)
        if isinstance(state_features, tuple):
            state_features, individual_features = state_features
        else:
            individual_features = None
        
        # RNN
        new_rnn_state, x = self.update_rnn_only(state_features, rnn_state, dones)

        # Actor network
        x_cat = jnp.concatenate((x, state_features), axis=-1)  # [T, B, C]
        pi, weights, actor_mean, actor_std = self.action_dist_from_features(x_cat)

        return new_rnn_state, pi, weights, actor_mean, actor_std, state_features, individual_features, x_cat

    def world_dynamics(self, current_latent_state, action_data, x_cat):
        """
        Various world modeling predictions.
        current_latent_state = world features at timestep t
        action_data = action at t
        x_cat = world features at t concatenated with RNN hidden state at t (processed t) 
        """
        concat_inputs = jnp.concatenate((current_latent_state, action_data), axis=-1)
        
        # Next latent state
        next_latent_state = jax.nn.relu(self.world_dense_1(concat_inputs))
        next_latent_state = jax.nn.relu(self.world_dense_2(next_latent_state))
        next_latent_state = jax.nn.relu(self.world_dense_3(next_latent_state))
        next_latent_state = self.world_dense_4(next_latent_state)

        # Reward
        pred_rewards = jax.nn.relu(self.reward_fc1(concat_inputs))
        pred_rewards = jax.nn.relu(self.reward_fc2(pred_rewards))
        pred_rewards = jax.nn.relu(self.reward_fc3(pred_rewards))
        pred_rewards = self.reward_fc4(pred_rewards)

        # Relative odometry
        odometry_next_state = jax.nn.relu(self.ln7(self.odometry_fc1(concat_inputs)))
        odometry_next_state = jax.nn.relu(self.ln8(self.odometry_fc2(odometry_next_state)))
        odometry_next_state = self.odometry_fc3(odometry_next_state)
        # odometry_next_state = jnp.concatenate((
        #     odometry_next_state[..., :2],
        #     jnp.arctan2(jnp.sin(odometry_next_state[..., [2]]), jnp.cos(odometry_next_state[..., [2]]))), axis=-1)
        
        # Optimal state from given action
        inv_opt_state = jax.nn.relu(self.ln9(self.inv_opt_state_fc1(concat_inputs)))
        inv_opt_state = jax.nn.relu(self.ln10(self.inv_opt_state_fc2(inv_opt_state)))
        inv_opt_state = self.inv_opt_state_fc3(inv_opt_state)
        # inv_opt_state = jnp.concatenate((
        #     inv_opt_state[..., :2],
        #     jnp.arctan2(jnp.sin(inv_opt_state[..., [2]]), jnp.cos(inv_opt_state[..., [2]]))), axis=-1)
        
        # Planner
        planner_next_state = jax.nn.relu(self.ln11(self.planner_fc1(x_cat)))
        planner_next_state = jax.nn.relu(self.ln12(self.planner_fc2(planner_next_state)))
        planner_next_state = self.planner_fc3(planner_next_state)
        # planner_next_state = jnp.concatenate((
        #     jnp.arctan2(jnp.sin(planner_next_state[..., [0]]), jnp.cos(planner_next_state[..., [0]])), 
        #     planner_next_state[..., 1:]), axis=-1)
        
        return {'next_latent_state': next_latent_state, 
                'odometry_next_state': odometry_next_state, 
                'planner_next_state': planner_next_state, 
                'inv_opt_state': inv_opt_state}, pred_rewards

    def rnn_state_from_obs(self, rnn_state, x):
        """Uses full observation to predict everything - action distribution, sampled action, next sensor features."""
        obs, dones = x

        # State feature extractor
        state_features = self.feature_extractor(obs)
        if isinstance(state_features, tuple):
            state_features, individual_features = state_features
        else:
            individual_features = None
        
        # RNN
        new_rnn_state, x = self.update_rnn_only(state_features, rnn_state, dones)

        # Actor network
        x_cat = jnp.concatenate((x, state_features), axis=-1)  # [T, B, C]
        return new_rnn_state, state_features, individual_features, x_cat
    

class ActorCriticForSearch(nn.Module):
    """Used when planning with world models but no future camera prediction."""
    action_dim: int
    action_minimum: jnp.ndarray
    action_maximum: jnp.ndarray
    feature_extractor_class: nn.Module
    feature_extractor_kwargs: Optional[Union[Dict, None]]
    num_components: int = 6
    hidden_units: int = 512  # 128

    def setup(self):
        # Feature extractor module (now initialized in setup)
        self.feature_extractor = self.feature_extractor_class(**self.feature_extractor_kwargs)

        # Layers to be reused
        self.rnn_layer = ScannedRNN()
        
        # Actor network layers
        self.dense_actor_1 = nn.Dense(self.hidden_units, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.dense_actor_2 = nn.Dense(self.hidden_units, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.weight_layer_1 = nn.Dense(self.num_components, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.weight_layer_2 = nn.Dense(self.num_components, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.actor_mean_1 = nn.Dense(self.action_dim * self.num_components, kernel_init=orthogonal(.1), bias_init=constant(0.0))
        self.actor_mean_2 = nn.Dense(self.action_dim * self.num_components, kernel_init=orthogonal(.1), bias_init=constant(0.0))
        self.actor_log_std_1 = nn.Dense(self.action_dim * self.num_components, kernel_init=orthogonal(.1), bias_init=constant(0.0))
        self.actor_log_std_2 = nn.Dense(self.action_dim * self.num_components, kernel_init=orthogonal(.1), bias_init=constant(0.0))
        self.critic_1 = nn.Dense(self.hidden_units, use_bias=True)
        self.critic_2 = nn.Dense(2, use_bias=True)

        # Classifiers
        self.classifier_overlap1 = nn.Dense(self.hidden_units, use_bias=True)
        self.classifier_overlap2 = nn.Dense(1, use_bias=True)
        self.classifier_offroad1 = nn.Dense(self.hidden_units, use_bias=True)
        self.classifier_offroad2 = nn.Dense(1, use_bias=True)

    def __call__(self, rnn_state, x):
        """Uses full observation to predict everything - action distribution, sampled action, next sensor features."""
        obs, dones, rng_sample = x

        # State feature extractor
        state_features = self.feature_extractor(obs)

        # Offroad and overlap
        offroad, overlap = self.offroad_overlap_from_state_features(state_features)
        
        # RNN
        new_rnn_state, x = self.update_rnn_only(state_features, rnn_state, dones)

        # Actor network
        x_cat = jnp.concatenate((x, state_features), axis=-1)  # [T, B, C]
        pi, weights, actor_mean, actor_std = self.action_dist_from_features(x_cat)

        # Select action
        action_data = pi.sample(seed=rng_sample)

        # Predict value
        pred_val = self.state_value_from_features(x_cat)

        return new_rnn_state, pi, weights, actor_mean, actor_std, action_data, x_cat, pred_val, offroad, overlap

    def action_dist_from_features(self, x_cat):
        """
        Returns an action distribution from concatenated observed features and RNN hidden state.
        """
        x_actor = jax.nn.relu(self.dense_actor_1(x_cat))
        x_actor = self.dense_actor_2(x_actor)

        weights = jax.nn.relu(self.weight_layer_1(x_actor))
        weights = self.weight_layer_2(weights)
        weights = weights.reshape((*weights.shape[:-1], 1, self.num_components))  # (T, B, N, 1, Ncomp)

        actor_mean = jax.nn.relu(self.actor_mean_1(x_actor))
        actor_mean = self.actor_mean_2(actor_mean)
        actor_mean = actor_mean.reshape((*actor_mean.shape[:-1], self.action_dim, self.num_components))

        actor_log_std = jax.nn.relu(self.actor_log_std_1(x_actor))
        actor_log_std = jnp.clip(self.actor_log_std_2(actor_log_std), a_min=-5, a_max=2)
        actor_std = jnp.exp(actor_log_std)
        actor_std = actor_std.reshape((*actor_std.shape[:-1], self.action_dim, self.num_components))

        # Action distribution
        pi = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=weights),
            components_distribution=tfd.Normal(loc=actor_mean, scale=actor_std),
            reparameterize=True)
        return pi, weights, actor_mean, actor_std

    def state_value_from_features(self, x_cat):
        """
        Returns an action distribution from concatenated observed features and RNN hidden state.
        """
        v = jax.nn.relu(self.critic_1(x_cat))
        return self.critic_2(v)

    def update_rnn_only(self, state_features, rnn_state, dones):
        """
        Updates the RNN using obs(t) and h(t-1). Returns h(t)
        """
        rnn_in = (state_features, dones)
        new_rnn_state, x = self.rnn_layer(rnn_state, rnn_in)
        return new_rnn_state, x

    def action_dist_from_obs(self, rnn_state, x):
        """Uses full observation to predict everything up to the action distribution."""
        obs, dones = x

        # State feature extractor
        state_features = self.feature_extractor(obs)
        
        # RNN
        new_rnn_state, x = self.update_rnn_only(state_features, rnn_state, dones)

        # Actor network
        x_cat = jnp.concatenate((x, state_features), axis=-1)  # [T, B, C]
        pi, weights, actor_mean, actor_std = self.action_dist_from_features(x_cat)

        return new_rnn_state, pi, weights, actor_mean, actor_std, state_features, x_cat

    def rnn_state_from_obs(self, rnn_state, x):
        """Uses full observation to predict everything - action distribution, sampled action, next sensor features."""
        obs, dones = x

        # State feature extractor
        state_features = self.feature_extractor(obs)
        
        # RNN
        new_rnn_state, x = self.update_rnn_only(state_features, rnn_state, dones)

        # Actor network
        x_cat = jnp.concatenate((x, state_features), axis=-1)  # [T, B, C]
        return new_rnn_state, state_features, x_cat

    def offroad_overlap_from_state_features(self, x):
        overlap = self.classifier_overlap2(jax.nn.relu(self.classifier_overlap1(x)))
        offroad = self.classifier_offroad2(jax.nn.relu(self.classifier_offroad1(x)))
        return overlap, offroad

    def action_dist_from_obs_inference(self, rnn_state, x):
        """Uses full observation to predict everything up to the action distribution."""
        obs, dones = x

        # State feature extractor
        state_features = self.feature_extractor(obs)
        
        # RNN
        new_rnn_state, x = self.update_rnn_only(state_features, rnn_state, dones)

        # Actor network
        x_cat = jnp.concatenate((x, state_features), axis=-1)  # [T, B, C]
        pi, weights, actor_mean, actor_std = self.action_dist_from_features(x_cat)

        return new_rnn_state, weights, actor_mean, actor_std