from abc import ABC, abstractmethod
import dataclasses
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Any

from utils.model_utils import linear_clip_scale
from configs.consts import MASK_UNVALID_VALUE

class ObsMask(ABC):

    @abstractmethod
    def mask_obs(self, *args: Any, **kwds: Any) -> Any:
        pass

    @abstractmethod
    def mask_fun(self, *args: Any, **kwds: Any) -> Any:
        pass

    @abstractmethod
    def plot_mask_fun(self, *args: Any, **kwds: Any) -> None:
        pass

@dataclass
class DistanceObsMask(ObsMask):
    radius: float

    def mask_obs(self, state, obs, rng):
        x = obs.trajectory.x
        y = obs.trajectory.y

        visible_obj = self.mask_fun(x, y)

        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                 valid=visible_obj)

        limited_obs = dataclasses.replace(obs,
                                          trajectory=trajectory_limited)
        return limited_obs

    def mask_fun(self, obj_x, obj_y, eps=1e-3):
        is_center = (-eps <= obj_x) & (obj_x <= eps) & \
            (-eps <= obj_y) & (obj_y <= eps)

        squared_distance = obj_x**2 + obj_y**2

        return (squared_distance <= self.radius**2) | is_center

    def plot_mask_fun(self, ax, center=(0, 0)) -> None:
        theta = jnp.linspace(0, 2 * jnp.pi, 100)

        x = center[0] + self.radius * jnp.cos(theta)
        y = center[1] + self.radius * jnp.sin(theta)

        ax.plot(x, y)

@dataclass
class ConicObsMask(ObsMask):
    radius: float
    angle: float

    def mask_obs(self, state, obs, rng):
        x = obs.trajectory.x
        y = obs.trajectory.y

        visible_obj = self.mask_fun(x, y)

        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                 valid=visible_obj)

        limited_obs = dataclasses.replace(obs,
                                          trajectory=trajectory_limited)
        return limited_obs

    def mask_fun(self, obj_x, obj_y, eps=1e-3):
        is_center = (-eps <= obj_x) & (obj_x <= eps) & \
            (-eps <= obj_y) & (obj_y <= eps)

        squared_distance = obj_x**2 + obj_y**2

        obj_angle = jnp.arctan2(obj_y, obj_x)
        angle_condition = (- self.angle[:, None, None] / 2 <= obj_angle) &\
            (obj_angle <= self.angle[:, None, None] / 2)

        radius_condition = squared_distance <= self.radius**2

        return (angle_condition & radius_condition)| is_center

    def plot_mask_fun(self, ax, center=(0, 0), color='b') -> None:
        theta = jnp.linspace(- self.angle/2, self.angle/2, 100)

        x = center[0] + self.radius * jnp.cos(theta)
        y = center[1] + self.radius * jnp.sin(theta)

        x1 = center[0] + self.radius * jnp.cos(- self.angle/2)
        y1 = center[1] + self.radius * jnp.sin(- self.angle/2)

        x2 = center[0] + self.radius * jnp.cos(self.angle/2)
        y2 = center[1] + self.radius * jnp.sin(self.angle/2)

        ax.plot([0, x1[0]], [0, y1[0]], c=color)
        ax.plot([0, x2[0]], [0, y2[0]], c=color)
        ax.plot(x, y, c=color)


@dataclass
class BlindSpotObsMask(ObsMask):
    radius: float
    angle: float

    def mask_obs(self, state, obs, rng):

        x = obs.trajectory.x
        y = obs.trajectory.y

        visible_obj = self.mask_fun(x, y)

        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                 valid=visible_obj)

        limited_obs = dataclasses.replace(obs,
                                          trajectory=trajectory_limited)
        return limited_obs

    def mask_fun(self, obj_x, obj_y, eps=1e-3):
        assert(self.angle <= jnp.pi / 2)
        is_center = (-eps <= obj_x) & (obj_x <= eps) & \
            (-eps <= obj_y) & (obj_y <= eps)

        visible_angle = jnp.pi - 2 * self.angle
        squared_distance = obj_x**2 + obj_y**2

        obj_angle = jnp.arctan2(obj_y, obj_x)
        angle_condition_front = (- jnp.pi / 2 <= obj_angle) & (obj_angle <= jnp.pi / 2)
        angle_condition_back = (- (visible_angle / 2 + 2 * self.angle) >= obj_angle) | (obj_angle >= (visible_angle / 2 + 2 * self.angle))
        radius_condition = squared_distance <= self.radius**2

        return ((angle_condition_front | angle_condition_back) & radius_condition) | is_center

    def plot_mask_fun(self, ax, center=(0, 0), color='b') -> None:

        # Front
        theta = jnp.linspace(- jnp.pi / 2, jnp.pi / 2, 100)

        x = center[0] + self.radius * jnp.cos(theta)
        y = center[1] + self.radius * jnp.sin(theta)

        x1 = center[0] + self.radius * jnp.cos(- jnp.pi / 2)
        y1 = center[1] + self.radius * jnp.sin(- jnp.pi / 2)

        x2 = center[0] + self.radius * jnp.cos(jnp.pi / 2)
        y2 = center[1] + self.radius * jnp.sin(jnp.pi / 2)

        ax.plot([center[0], x1], [center[1], y1], c=color)
        ax.plot([center[0], x2], [center[1], y2], c=color)
        ax.plot(x, y, c=color)

        # Back
        visible_angle = jnp.pi - 2 * self.angle
        theta = jnp.linspace(visible_angle / 2, - visible_angle / 2, 100)

        x = center[0] - self.radius * jnp.cos(theta)
        y = center[1] - self.radius * jnp.sin(theta)

        x1 = center[0] + self.radius * jnp.cos(- (visible_angle / 2 + 2 * self.angle))
        y1 = center[1] + self.radius * jnp.sin(- (visible_angle / 2 + 2 * self.angle))

        x2 = center[0] + self.radius * jnp.cos(visible_angle / 2 + 2 * self.angle)
        y2 = center[1] + self.radius * jnp.sin(visible_angle / 2 + 2 * self.angle)

        ax.plot([center[0], x1], [center[1], y1], c=color)
        ax.plot([center[0], x2], [center[1], y2], c=color)
        ax.plot(x, y, c=color)

@dataclass
class SpeedConicObsMask(ObsMask):
    radius: float
    angle_max: float
    angle_min: float
    v_max: float

    def mask_obs(self, state, obs, rng):
        _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
        sdc_v = jnp.take_along_axis(obs.trajectory.speed, sdc_idx[..., None, None], axis=-2)

        x = obs.trajectory.x
        y = obs.trajectory.y

        visible_obj = self.mask_fun(x, y, sdc_v.squeeze())

        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                 valid=visible_obj)
        limited_obs = dataclasses.replace(obs,
                                          trajectory=trajectory_limited)

        return limited_obs

    def mask_fun(self, obj_x, obj_y, sdc_v, eps=1e-3):

        angle = (sdc_v.clip(0, self.v_max) * (self.angle_min - self.angle_max) / self.v_max + self.angle_max)[..., None]

        return ConicObsMask(self.radius, angle).mask_fun(obj_x, obj_y, eps=eps)

    def plot_mask_fun(self, ax, sdc_v, center=(0, 0), color='b') -> None:
        angle = sdc_v.clip(0, self.v_max) * (self.angle_min - self.angle_max) / self.v_max + self.angle_max

        ConicObsMask(self.radius, angle).plot_mask_fun(ax, center=center, color=color)

@dataclass
class SpeedGaussianNoise(ObsMask):
    v_max: float
    sigma_max: float
    sigma_min: float

    def mask_obs(self, state, obs, rng):
        noisy_xy = self.mask_fun(state, obs, rng)

        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                 x=noisy_xy[..., 0],
                                                 y=noisy_xy[..., 1])
        obs_limited = dataclasses.replace(obs,
                                          trajectory=trajectory_limited)
        return obs_limited

    def mask_fun(self, state, obs, rng):

        xy = obs.trajectory.xy

        _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
        sdc_v = jnp.take_along_axis(obs.trajectory.speed, sdc_idx[..., None, None], axis=-2)

        xy = obs.trajectory.xy
        is_obj = 1 - state.object_metadata.is_sdc
        sigma = jnp.where(is_obj[:, None, ..., None, None],
                          linear_clip_scale(sdc_v, self.v_max, self.sigma_min, self.sigma_max)[..., None] * jnp.ones_like(xy),
                          jnp.zeros_like(xy))

        gaussian_noise = jax.random.normal(rng, xy.shape) * sigma

        noisy_xy = xy + gaussian_noise

        return noisy_xy

    def plot_mask_fun(self, ax, center=(0, 0), color='b') -> None:
        pass

@dataclass
class SpeedUniformNoise(ObsMask):
    v_max: float
    bound_max: float
    bound_min: float

    def mask_obs(self, state, obs, rng):
        noisy_xy = self.mask_fun(state, obs, rng)

        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                 x=noisy_xy[..., 0],
                                                 y=noisy_xy[..., 1])
        obs_limited = dataclasses.replace(obs,
                                          trajectory=trajectory_limited)
        return obs_limited

    def mask_fun(self, state, obs, rng):

        _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
        sdc_v = jnp.take_along_axis(obs.trajectory.speed, sdc_idx[..., None, None], axis=-2)

        xy = obs.trajectory.xy
        is_obj = 1 - state.object_metadata.is_sdc
        bound = jnp.where(is_obj[:, None, ..., None, None],
                          linear_clip_scale(sdc_v, self.v_max, self.bound_min, self.bound_max)[..., None] * jnp.ones_like(xy),
                          jnp.zeros_like(xy))

        uniform_noise = jax.random.uniform(rng,
                                          minval=-bound,
                                          maxval=bound,
                                          shape=xy.shape)

        noisy_xy = xy + uniform_noise

        return noisy_xy

    def plot_mask_fun(self, ax, center=(0, 0), color='b') -> None:
        pass


@dataclass
class ZeroMask(ObsMask):

    def mask_obs(self, state, obs, rng):
        zero_xy = self.mask_fun(state, obs, rng)

        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                 x=zero_xy[..., 0],
                                                 y=zero_xy[..., 1])
        obs_limited = dataclasses.replace(obs,
                                          trajectory=trajectory_limited)
        return obs_limited

    def mask_fun(self, state, obs, rng):

        xy = obs.trajectory.xy

        return jnp.zeros_like(xy)

    def plot_mask_fun(self, ax, center=(0, 0), color='b') -> None:
        pass