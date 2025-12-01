import jax
import jax.numpy as jnp
from waymax.utils.geometry import transform_points, transform_direction, corners_from_bboxes
from waymax import datatypes, dynamics
from waymax.dynamics.bicycle_model import _SPEED_LIMIT
from waymax.env.planning_agent_environment import PlanningAgentEnvironment
from waymax.env import typedefs as types
from waymax import config as _config
from waymax.metrics import abstract_metric
from waymax.metrics import comfort
from waymax.metrics import imitation
from waymax.metrics import roadgraph
from waymax.metrics import route
from waymax.metrics.roadgraph import is_offroad, compute_signed_distance_to_nearest_road_edge_point
from jax.scipy.special import logsumexp
import chex

def transform_traj(X, pose_matrix, delta_yaw):
    """
    X should be of shape (..., 5)
    pose should be of shape (...)
    """
    xy = X[..., :2]
    yaw = X[..., 2]
    vel_xy = X[..., 3:]
    local_xy = transform_points(pts=xy, pose_matrix=pose_matrix)
    local_vel_xy = transform_direction(pts_dir=vel_xy, pose_matrix=pose_matrix)
    local_yaw = yaw + delta_yaw
    out = jnp.concatenate((local_xy, local_yaw[..., None], local_vel_xy), axis=-1)
    return out

def subtract_from_sdc_xyyaw(state, pred, pose_matrix, delta_yaw, pose_matrix_inv):
    """
    Takes the SDV values, projects to local, applies a difference there, reprojects to global.
    """
    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1) # (B, 1)
    # Temporarily concat zero velocities
    zero_vels = jnp.zeros((*pred.shape[:-1], 2))
    pred = jnp.concatenate((pred, zero_vels), axis=-1)
    vals = state.current_sim_trajectory.stack_fields(('x', 'y', 'yaw', 'vel_x', 'vel_y')) # (B, Nobj, T, 5)
    sdc_vals = jnp.take_along_axis(vals, sdc_idx[..., None, None], axis=1).squeeze(1)
    sdc_vals = transform_traj(sdc_vals, pose_matrix, delta_yaw) # From global to local
    imagined_sdc_current_vals = sdc_vals - pred # Add the transformation there
    imagined_sdc_current_vals = transform_traj(imagined_sdc_current_vals, pose_matrix_inv, -delta_yaw) # Transform back to global
    return imagined_sdc_current_vals # (B, T=1, 5)

def subtract_from_sdc_xy(state, pred, pose_matrix, delta_yaw, pose_matrix_inv):
    """
    Takes the SDV values, projects to local, applies a difference there, reprojects to global.
    """
    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1) # (B, 1)
    # Temporarily concat zero velocities
    zero_yaw_vels = jnp.zeros((*pred.shape[:-1], 3))
    pred = jnp.concatenate((pred, zero_yaw_vels), axis=-1)
    vals = state.current_sim_trajectory.stack_fields(('x', 'y', 'yaw', 'vel_x', 'vel_y')) # (B, Nobj, T, 5)
    sdc_vals = jnp.take_along_axis(vals, sdc_idx[..., None, None], axis=1).squeeze(1)
    sdc_vals = transform_traj(sdc_vals, pose_matrix, delta_yaw) # From global to local
    imagined_sdc_current_vals = sdc_vals - pred # Add the transformation there
    imagined_sdc_current_vals = transform_traj(imagined_sdc_current_vals, pose_matrix_inv, -delta_yaw) # Transform back to global
    return imagined_sdc_current_vals # (B, T=1, 5)

def subtract_from_sdc_yawvel(state, pred, pose_matrix, delta_yaw, pose_matrix_inv):
    """
    Takes the SDV values, projects to local, applies a difference there, reprojects to global.
    """
    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1) # (B, 1)
    # Temporarily concat zero velocities
    zero_xy = jnp.zeros((*pred.shape[:-1], 2))
    pred = jnp.concatenate((zero_xy, pred), axis=-1)
    vals = state.current_sim_trajectory.stack_fields(('x', 'y', 'yaw', 'vel_x', 'vel_y')) # (B, Nobj, T, 5)
    sdc_vals = jnp.take_along_axis(vals, sdc_idx[..., None, None], axis=1).squeeze(1)
    sdc_vals = transform_traj(sdc_vals, pose_matrix, delta_yaw) # From global to local
    imagined_sdc_current_vals = sdc_vals - pred # Add the transformation there
    imagined_sdc_current_vals = transform_traj(imagined_sdc_current_vals, pose_matrix_inv, -delta_yaw) # Transform back to global
    return imagined_sdc_current_vals # (B, T=1, 5)

def create_state_with_sdc_xyyaw(state, vals):
    """
    Used for the next-state predictor
    """
    new_x = vals[..., [0]]
    new_y = vals[..., [1]]
    new_yaw = vals[..., [2]]
    new_vel_x = vals[..., [3]]
    new_vel_y = vals[..., [4]]
    
    # Update the array values for the SDC vehicle within the current sim trajectory
    updated_x = jnp.where(state.object_metadata.is_sdc[..., None], new_x, state.current_sim_trajectory.x)
    updated_y = jnp.where(state.object_metadata.is_sdc[..., None], new_y, state.current_sim_trajectory.y)
    updated_yaw = jnp.where(state.object_metadata.is_sdc[..., None], new_yaw, state.current_sim_trajectory.yaw)

    # Within the current sim trajectory change the values
    new_current_sim_traj = state.current_sim_trajectory.replace(x=updated_x, y=updated_y, yaw=updated_yaw)

    # Update the trajectory at the current timestep
    updated_sim_traj = datatypes.update_by_slice_in_dim(state.sim_trajectory, new_current_sim_traj, inputs_start_idx=state.timestep, updates_start_idx=0, slice_size=1, axis=-1)

    # Update the state
    updated_state = state.replace(sim_trajectory=updated_sim_traj)
    return updated_state

def create_state_with_sdc_xy(state, vals):
    """
    Used for the inverse optimal state prediction
    """
    new_x = vals[..., [0]]
    new_y = vals[..., [1]]
    new_yaw = vals[..., [2]]
    new_vel_x = vals[..., [3]]
    new_vel_y = vals[..., [4]]
    
    # Update the array values for the SDC vehicle within the current sim trajectory
    updated_x = jnp.where(state.object_metadata.is_sdc[..., None], new_x, state.current_sim_trajectory.x)
    updated_y = jnp.where(state.object_metadata.is_sdc[..., None], new_y, state.current_sim_trajectory.y)
    updated_yaw = jnp.where(state.object_metadata.is_sdc[..., None], new_yaw, state.current_sim_trajectory.yaw)

    # Within the current sim trajectory change the values
    new_current_sim_traj = state.current_sim_trajectory.replace(x=updated_x, y=updated_y)

    # Update the trajectory at the current timestep
    updated_sim_traj = datatypes.update_by_slice_in_dim(state.sim_trajectory, new_current_sim_traj, inputs_start_idx=state.timestep, updates_start_idx=0, slice_size=1, axis=-1)

    # Update the state
    updated_state = state.replace(sim_trajectory=updated_sim_traj)
    return updated_state

def create_state_with_sdc_yawvel(state, vals):
    """
    Used for the optimal planner
    """
    new_x = vals[..., [0]]
    new_y = vals[..., [1]]
    new_yaw = vals[..., [2]]
    new_vel_x = vals[..., [3]]
    new_vel_y = vals[..., [4]]
    
    # Update the array values for the SDC vehicle within the current sim trajectory
    updated_yaw = jnp.where(state.object_metadata.is_sdc[..., None], new_yaw, state.current_sim_trajectory.yaw)
    updated_vel_x = jnp.where(state.object_metadata.is_sdc[..., None], new_vel_x, state.current_sim_trajectory.vel_x)
    updated_vel_y = jnp.where(state.object_metadata.is_sdc[..., None], new_vel_y, state.current_sim_trajectory.vel_y)

    # Within the current sim trajectory change the values
    new_current_sim_traj = state.current_sim_trajectory.replace(yaw=updated_yaw, vel_x=updated_vel_x, vel_y=updated_vel_y)

    # Update the trajectory at the current timestep
    updated_sim_traj = datatypes.update_by_slice_in_dim(state.sim_trajectory, new_current_sim_traj, inputs_start_idx=state.timestep, updates_start_idx=0, slice_size=1, axis=-1)

    # Update the state
    updated_state = state.replace(sim_trajectory=updated_sim_traj)
    return updated_state

def inverse_kinematics(
    simulator_state1: datatypes.SimulatorState,
    simulator_state2: datatypes.SimulatorState,
    dynamics_model: dynamics.DynamicsModel,
) -> datatypes.Action:
    """Infers an action.

    Args:
        simulator_state: State of the simulator at the current timestep. Will use
        the `sim_trajectory` and `log_trajectory` fields to calculate an action.
        dynamics_model: Dynamics model whose `inverse` function will be used to
        infer the expert action given the logged states.

    Returns:
        Inferred action
    """
    prev_sim_traj = datatypes.dynamic_slice(  # pytype: disable=wrong-arg-types  # jax-ndarray
        simulator_state1.sim_trajectory, simulator_state1.timestep, 1, axis=-1
    )
    next_logged_traj = datatypes.dynamic_slice(  # pytype: disable=wrong-arg-types  # jax-ndarray
        simulator_state2.sim_trajectory, simulator_state2.timestep, 1, axis=-1
    )
    combined_traj = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y], axis=-1),
        prev_sim_traj,
        next_logged_traj,
    )
    return dynamics_model.inverse(
        combined_traj, metadata=simulator_state1.object_metadata, timestep=0
    )


class CustomBicycleDynamics(dynamics.InvertibleBicycleModel):
    @jax.named_scope('CustomBicycleDynamics.compute_update')
    def compute_update(
        self,
        action: datatypes.Action,
        trajectory: datatypes.Trajectory,
    ) -> datatypes.TrajectoryUpdate:
        """
        Everything as the normal InvertibleBicycleModel except that we don't clip the actions
        """
        x = trajectory.x
        y = trajectory.y
        vel_x = trajectory.vel_x
        vel_y = trajectory.vel_y
        yaw = trajectory.yaw
        speed  = jnp.sqrt(vel_x**2 + vel_y**2 + 1e-5)

        # Shape: (..., num_objects, 2)
        action_array = self._clip_values(action.data)
        action_array = action.data
        accel, steering = jnp.split(action_array, 2, axis=-1)
        if self._normalize_actions:
            accel = accel * self._max_accel
            steering = steering * self._max_steering
        t = self._dt

        new_x = x + vel_x * t + 0.5 * accel * jnp.cos(yaw) * t**2
        new_y = y + vel_y * t + 0.5 * accel * jnp.sin(yaw) * t**2
        delta_yaw = steering * (speed * t + 0.5 * accel * t**2)
        new_yaw = jnp.arctan2(jnp.sin(yaw + delta_yaw + 1e-6), jnp.cos(yaw + delta_yaw + 1e-6))

        new_vel = speed + accel * t
        new_vel_x = new_vel * jnp.cos(new_yaw)
        new_vel_y = new_vel * jnp.sin(new_yaw)
        return datatypes.TrajectoryUpdate(
            x=new_x,
            y=new_y,
            yaw=new_yaw,
            vel_x=new_vel_x,
            vel_y=new_vel_y,
            valid=trajectory.valid & action.valid,
        )


def wrap_yaws(x):
    return jnp.arctan2(jnp.sin(x), jnp.cos(x))

def compute_inverse_diffwrapping(
    traj: datatypes.Trajectory,
    timestep: jax.typing.ArrayLike,
    dt: float = 0.1,
    estimate_yaw_with_velocity: bool = True,
) -> datatypes.Action:
    """Runs inverse dynamics model to infer actions for specified timestep.

    Inverse dynamics:
        accel = (new_vel - vel) / dt
        steering = (new_yaw - yaw) / (speed * dt + 1/2 * accel * dt ** 2)

    Args:
        traj: A Trajectory used to infer actions of shape (..., num_objects,
        num_timesteps).
        timestep: Index of time for actions.
        dt: The time step length used in the simulator.
        estimate_yaw_with_velocity: Whether to use the yaw recorded in `traj` for
        estimating the inverse action or use the yaw estimated from velocities. It
        is recommended to set this to True, as using the estimated yaw is
        generally less noisy than using the yaw directly recorded in the
        trajectory.

    Returns:
        An Action that converts traj[timestep] to traj[timestep+1] of shape
        (..., num_objects, dim=2).
    """
    xy_yaw_vel = jnp.stack(
        [traj.x, traj.y, traj.yaw, traj.vel_x, traj.vel_y], axis=-1
    )
    xy_yaw_vel_slice = jax.lax.dynamic_slice_in_dim(
        xy_yaw_vel, start_index=timestep, slice_size=2, axis=-2
    )
    # Each has shape (..., num_timesteps = 2, 1).
    _, _, yaw, vel_x, vel_y = jnp.split(xy_yaw_vel_slice, 5, axis=-1)
    valids = jax.lax.dynamic_slice_in_dim(
        traj.valid, start_index=timestep, slice_size=2, axis=-1
    )
    valid = valids[..., 0:1] & valids[..., 1:2]
    # Calculate acceleration.
    speed = jnp.sqrt(vel_x[..., 0:2, :] ** 2 + vel_y[..., 0:2, :] ** 2)
    new_speed = speed[..., 1:2, :]
    accel = (new_speed - speed[..., 0:1, :]) / dt

    # Calculate steering curvature.
    new_yaw = wrap_yaws(yaw[..., 1:2, :])
    yaw = wrap_yaws(yaw[..., 0:1, :])
    if estimate_yaw_with_velocity:
        real_new_yaw = jnp.arctan2(vel_y[..., 1:2, :], vel_x[..., 1:2, :])
    else:
        real_new_yaw = new_yaw
    real_new_yaw = jnp.where(
        jnp.abs(new_speed) <= _SPEED_LIMIT, new_yaw, real_new_yaw
    )
    delta_yaw = wrap_yaws(real_new_yaw - yaw)
    steering = delta_yaw / (speed[..., 0:1, :] * dt + 0.5 * accel * dt**2)
    # Set steering to 0.0 if speed is 0 to avoid NaN error.
    # When speed is small, delta_yaw sometimes can also be small, so the
    # calculation of steering is affected by the data noise and can lead to
    # overestimation of steering, filtering small speed can help to prevent
    # overestimation of steering.
    steering = jnp.where(jnp.abs(speed[..., 0:1, :]) < _SPEED_LIMIT, 0, steering)
    steering = jnp.where(jnp.abs(speed[..., 1:2, :]) < _SPEED_LIMIT, 0, steering)
    raw_action_array = jnp.concatenate([accel, steering], axis=-1).squeeze(-2)
    action_array = jnp.where(valid, raw_action_array, 0.0)
    return datatypes.Action(data=action_array, valid=valid)


class CustomBicycleDynamicsNoActionClip(dynamics.InvertibleBicycleModel):
    @jax.named_scope('CustomBicycleDynamicsNoActionClip.compute_update')
    def compute_update(
        self,
        action: datatypes.Action,
        trajectory: datatypes.Trajectory,
    ) -> datatypes.TrajectoryUpdate:
        """
        Everything as the normal InvertibleBicycleModel except that we don't clip the actions
        """
        x = trajectory.x
        y = trajectory.y
        vel_x = trajectory.vel_x
        vel_y = trajectory.vel_y
        yaw = trajectory.yaw
        speed = jnp.sqrt(trajectory.vel_x**2 + trajectory.vel_y**2)
        speed = jnp.maximum(speed, 1e-7)

        # Shape: (..., num_objects, 2)
        # action_array = self._clip_values(action.data)
        action_array = action.data
        accel, steering = jnp.split(action_array, 2, axis=-1)
        if self._normalize_actions:
            accel = accel * self._max_accel
            steering = steering * self._max_steering
        t = self._dt

        new_x = x + vel_x * t + 0.5 * accel * jnp.cos(yaw) * t**2
        new_y = y + vel_y * t + 0.5 * accel * jnp.sin(yaw) * t**2
        delta_yaw = steering * (speed * t + 0.5 * accel * t**2)
        # new_yaw = geometry.wrap_yaws(yaw + delta_yaw)
        new_yaw = jnp.arctan2(jnp.sin(yaw + delta_yaw), jnp.cos(yaw + delta_yaw))
        new_vel = speed + accel * t
        new_vel_x = new_vel * jnp.cos(new_yaw)
        new_vel_y = new_vel * jnp.sin(new_yaw)
        return datatypes.TrajectoryUpdate(
            x=new_x,
            y=new_y,
            yaw=new_yaw,
            vel_x=new_vel_x,
            vel_y=new_vel_y,
            valid=trajectory.valid & action.valid,
        )

    def inverse(
        self,
        trajectory: datatypes.Trajectory,
        metadata: datatypes.ObjectMetadata,
        timestep: int,
    ) -> datatypes.Action:
        """Runs inverse dynamics model to infer actions for specified timestep.

        Inverse dynamics:
        accel = (new_vel - vel) / dt
        steering = (new_yaw - yaw) / (speed * dt + 1/2 * accel * dt ** 2)

        Args:
        trajectory: A Trajectory used to infer actions (..., num_objects,
            num_timesteps),
        metadata: Object metadata for the trajectory of shape (..., num_objects).
        timestep: Index of time for actions.

        Returns:
        An Action that converts traj[timestep] to traj[timestep+1] of shape
            (..., num_objects, dim=2).
        """
        actions = compute_inverse_diffwrapping(trajectory, timestep, self._dt)
        if self._normalize_actions:
            action_array = actions.data
            # accel/steering shape: (..., num_objects)
            accel = action_array[..., 0] / self._max_accel
            steering = action_array[..., 1] / self._max_steering
            # action_array shape: (..., num_objects, 2)
            action_array = jnp.stack([accel, steering], axis=-1)
            # action_array = self._clip_values(action_array)
        else:
            action_array = actions.data
            # action_array = self._clip_values(actions.data)
        return datatypes.Action(data=action_array, valid=actions.valid)
    

def project_sdc_to_local(state, pose_matrix, delta_yaw):
    """
    Takes the SDC values, projects from global to local.
    """
    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1) # (B, 1)
    vals = state.current_sim_trajectory.stack_fields(('x', 'y', 'yaw', 'vel_x', 'vel_y')) # (B, Nobj, T, 5)
    sdc_vals = jnp.take_along_axis(vals, sdc_idx[..., None, None], axis=1).squeeze(1)
    sdc_vals = transform_traj(sdc_vals, pose_matrix, delta_yaw) # From global to local
    return sdc_vals # (B, T=1, 5)