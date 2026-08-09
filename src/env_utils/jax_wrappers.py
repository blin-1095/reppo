from functools import partial
from typing import Any, Tuple, Union

import chex
import gymnax
import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from flax import struct
from gymnax.environments import environment, spaces
from gymnax.environments.environment import Environment
from gymnax.environments.spaces import Box
from ml_collections import ConfigDict
from mujoco_playground import MjxEnv, registry
from mujoco_playground._src.wrapper import wrap_for_brax_training, Wrapper

# Imports for the Dexmachina Jax Wrapper
import sys
import math
import torch
import numpy as np
import jax.numpy as jnp
from typing import Dict
from gymnax.environments.environment import EnvState
from torch.utils.dlpack import to_dlpack as pt_to_dlpack, from_dlpack as pt_from_dlpack
from jax.dlpack import to_dlpack as jax_to_dlpack, from_dlpack as jax_from_dlpack


class MjxGymnaxWrapper(Environment):
    def __init__(
        self,
        env_or_name: str | MjxEnv,
        episode_length: int = 1000,
        action_repeat: int = 1,
        reward_scale: float = 1.0,
        push_distractions: bool = False,
        config: dict = None,
        asymmetric_observation: bool = False,
    ):
        if isinstance(env_or_name, str):
            if config is None:
                config = registry.get_default_config(env_or_name)
                is_humanoid_task = env_or_name in [
                    "G1JoystickRoughTerrain",
                    "G1JoystickFlatTerrain",
                    "T1JoystickRoughTerrain",
                    "T1JoystickFlatTerrain",
                ]
                if is_humanoid_task:
                    config.push_config.enable = push_distractions
            else:
                config = ConfigDict(config)
            env = registry.load(env_or_name, config=config)
            if episode_length is not None:
                env = wrap_for_brax_training(
                    env, episode_length=episode_length, action_repeat=action_repeat
                )
            self.env = env
        else:
            self.env = env_or_name
        self.reward_scale = reward_scale
        if isinstance(self.env.observation_size, int):
            self.dict_obs = False
        else:
            self.dict_obs = True
        if asymmetric_observation:
            self.dict_obs_key = "privileged_state"
        else:
            self.dict_obs_key = "state"
        print(self.dict_obs_key)
        super().__init__()

    def action_space(self, params):
        return gymnax.environments.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_size,),
        )

    def observation_space(self, params):
        if self.dict_obs:
            return Box(
                low=-float("inf"),
                high=float("inf"),
                shape=self.env.observation_size["state"],
            ), Box(
                low=-float("inf"),
                high=float("inf"),
                shape=self.env.observation_size[self.dict_obs_key],
            )
        else:
            return Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.env.observation_size,),
            ), Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.env.observation_size,),
            )

    @property
    def default_params(self) -> gymnax.EnvParams:
        return gymnax.EnvParams()

    def reset(self, key):
        state = self.env.reset(key)
        # state.info["truncation"] = 0.0
        obs = state.obs if not self.dict_obs else state.obs["state"]
        critic_obs = state.obs if not self.dict_obs else state.obs[self.dict_obs_key]
        return obs, critic_obs, state

    def step(self, key, state, action):
        # action = jnp.nan_to_num(action, 0.0)
        state = self.env.step(state, action)
        obs = state.obs if not self.dict_obs else state.obs["state"]
        critic_obs = state.obs if not self.dict_obs else state.obs[self.dict_obs_key]
        return (
            obs,
            critic_obs,
            state,
            state.reward * self.reward_scale,
            state.done > 0.5,
            {},
        )


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray
    timestep: jnp.ndarray
    truncated: jnp.ndarray
    info: Any = None

    def unwrapped(self):
        return self.env_state

    def set_env_state(self, env_state):
        return self.replace(env_state=env_state)


class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment, num_envs: int):
        super().__init__(env)
        self.num_envs = num_envs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key) -> Tuple[chex.Array, environment.EnvState]:
        obs, critic_obs, env_state = self.env.reset(key)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros((self.num_envs,)),
            episode_lengths=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            returned_episode_returns=jnp.zeros((self.num_envs,)),
            returned_episode_lengths=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            timestep=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            truncated=jnp.ones((self.num_envs,), dtype=jnp.float32),
            info={
                "returned_episode": jnp.zeros((self.num_envs,), dtype=jnp.bool_),
                "returned_episode_returns": jnp.zeros((self.num_envs,)),
                "timestep": jnp.zeros((self.num_envs,), dtype=jnp.int32),
                "returned_episode_lengths": jnp.zeros(
                    (self.num_envs,), dtype=jnp.int32
                ),
            },
        )
        return obs, critic_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, critic_obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        info["returned_episode_returns"] = (
            state.returned_episode_returns * (1 - done) + new_episode_return * done
        )
        info["returned_episode_lengths"] = (
            state.returned_episode_lengths * (1 - done) + new_episode_length * done
        )
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
            truncated=env_state.info["truncation"],
            info=info,
        )
        return obs, critic_obs, state, reward, done, info


class BraxGymnaxWrapper:
    def __init__(
        self,
        env_name,
        backend="generalized",
        episode_length=1000,
        reward_scaling=1.0,
        terminate=True,
    ):
        env = envs.get_environment(
            env_name=env_name, backend=backend, terminate_when_unhealthy=terminate
        )
        env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
        env = AutoResetWrapper(env)
        self.env = env
        self.action_size = self.env.action_size
        self.observation_size = (self.env.observation_size,)
        self.default_params = ()
        self.reward_scaling = reward_scaling

    def reset(self, key):
        state = self.env.reset(key)
        return state.obs, state

    def step(self, key, state, action):
        next_state = self.env.step(state, action)
        return (
            next_state.obs,
            next_state.obs,
            next_state,
            next_state.reward * self.reward_scaling,
            next_state.done > 0.5,
            {},
        )

    def observation_space(self):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.env.observation_size,),
        ), spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.env.observation_size,),
        )

    def action_space(self):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_size,),
        )


class ClipAction(Wrapper):
    def __init__(self, env, low=-0.999, high=0.999):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self.env.step(key, state, action)


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    critic_mean: jnp.ndarray
    critic_var: jnp.ndarray
    count: float
    env_state: environment.EnvState
    truncated: float
    info: Any = None

    def unwrapped(self):
        return self.env_state.unwrapped()

    def set_env_state(self, env_state):
        return self.replace(env_state=self.env_state.set_env_state(env_state))


class NormalizeVec(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def _init_state(self, key):
        obs, critic_obs, env_state = self.env.reset(key)
        return NormalizeVecObsEnvState(
            mean=jnp.mean(obs, axis=0),
            var=jnp.var(obs, axis=0),
            critic_mean=jnp.mean(critic_obs, axis=0),
            critic_var=jnp.var(critic_obs, axis=0),
            count=obs.shape[0],
            env_state=env_state,
        )

    def _compute_stats(self, mean, var, count, obs):
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count

        return new_mean, new_var

    def reset(self, key, params=None):
        obs, critic_obs, env_state = self.env.reset(key)
        if params is not None:
            mean = params.mean
            var = params.var
            critic_mean = params.critic_mean
            critic_var = params.critic_var
            count = params.count
        else:
            mean = jnp.mean(obs, axis=0)
            var = jnp.var(obs, axis=0)
            critic_mean = jnp.mean(critic_obs, axis=0)
            critic_var = jnp.var(critic_obs, axis=0)
            count = obs.shape[0]
        state = NormalizeVecObsEnvState(
            mean=mean,
            var=var,
            critic_mean=critic_mean,
            critic_var=critic_var,
            count=count,
            env_state=env_state,
            truncated=env_state.truncated,
            info=env_state.info,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-2),
            (critic_obs - state.critic_mean) / jnp.sqrt(state.critic_var + 1e-2),
            state,
        )

    def step(self, key, state, action):
        obs, critic_obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action
        )

        new_mean, new_var = self._compute_stats(state.mean, state.var, state.count, obs)
        new_critic_mean, new_critic_var = self._compute_stats(
            state.critic_mean, state.critic_var, state.count, critic_obs
        )

        new_count = state.count + obs.shape[0]

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            critic_mean=new_critic_mean,
            critic_var=new_critic_var,
            count=new_count,
            env_state=env_state,
            truncated=env_state.truncated,
            info=env_state.info,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-2),
            (critic_obs - state.critic_mean) / jnp.sqrt(state.critic_var + 1e-2),
            state,
            reward,
            done,
            info,
        )


# --- Zero-Copy DLPack Memory Converters for Dexmachina Jax Wrapper ---
def torch_to_jax(pt_tensor: torch.Tensor) -> jax.Array:
    """Converts a PyTorch GPU Tensor to a JAX GPU Array without copying memory."""
    if not pt_tensor.is_contiguous():
        pt_tensor = pt_tensor.contiguous()
    return jax_from_dlpack(pt_to_dlpack(pt_tensor))


def jax_to_torch(jax_array: jax.Array) -> torch.Tensor:
    """Converts a JAX GPU Array to a PyTorch GPU Tensor without copying memory."""
    return pt_from_dlpack(jax_to_dlpack(jax_array))

@struct.dataclass
class DexmachinaState:
    time: jnp.ndarray
    truncated: jnp.ndarray  # Added for NormalizeVec wrapper
    info: Dict[str, jnp.ndarray]


class DexmachinaGymnaxWrapper(Environment):
    """Adapts DexMachina Genesis PyTorch environment for JAX/Gymnax pipelines."""

    def __init__(
        self,
        env,
        clip_obs: float = math.inf,
        clip_actions: float = math.inf,
        asymmetric_obs: bool = True,
    ):
        super().__init__()
        self.env = env
        self.num_envs = env.num_envs
        self.obs_dim = env.num_obs
        self.action_dim = env.num_actions
        self.state_dim = getattr(env, "num_states", self.obs_dim)

        self._clip_obs = clip_obs
        self._clip_actions = clip_actions
        self.asymmetric_obs = asymmetric_obs

        self.env_type = "dexmachina"
        self.max_episode_steps = env.max_episode_length

        # Force Genesis to compile its physics kernels on the main thread 
        # before JAX takes over and locks the GPU context.
        print("Warming up Genesis and PyTorch kernels...")
        self.env.reset()
        dummy_actions = torch.zeros((self.num_envs, self.action_dim), device='cuda:0')
        self.env.step(dummy_actions)
        torch.cuda.synchronize()
        print("Genesis warmup complete!")

    # ---------------------------------------------------------
    # HOST FUNCTIONS (Run standard PyTorch + DLPack outside JIT)
    # ---------------------------------------------------------
    def _host_reset(self):
        obs_dict, _ = self.env.reset()
        obs, critic_obs = self._process_obs(obs_dict)
        if obs is None:
            raise ValueError("DexMachina reset returned None for policy observations!")
        if critic_obs is None:
            critic_obs = obs
            
        torch.cuda.synchronize()
        return torch_to_jax(obs), torch_to_jax(critic_obs)

    def _host_step(self, actions_jax):
        actions_pt = jax_to_torch(actions_jax)
        if not math.isinf(self._clip_actions):
            actions_pt = torch.clamp(actions_pt, -self._clip_actions, self._clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions_pt)
        obs, critic_obs = self._process_obs(obs_dict)

        if obs is None:
            raise ValueError("DexMachina step returned None!")
        if critic_obs is None:
            critic_obs = obs

        if rew is None:
            rew = torch.zeros(self.num_envs, dtype=torch.float32, device='cuda:0')
        elif not isinstance(rew, torch.Tensor):
            rew = torch.tensor(rew, dtype=torch.float32, device='cuda:0')

        if terminated is None:
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device='cuda:0')
        elif not isinstance(terminated, torch.Tensor):
            terminated = torch.tensor(terminated, dtype=torch.bool, device='cuda:0')
        else:
            terminated = terminated.to(torch.bool)

        if truncated is None:
            truncated = torch.zeros(self.num_envs, dtype=torch.bool, device='cuda:0')
        elif not isinstance(truncated, torch.Tensor):
            truncated = torch.tensor(truncated, dtype=torch.bool, device='cuda:0')
        else:
            truncated = truncated.to(torch.bool)

        torch.cuda.synchronize()

        return (
            torch_to_jax(obs),
            torch_to_jax(critic_obs),
            torch_to_jax(rew),
            torch_to_jax(terminated | truncated),
            torch_to_jax(truncated.to(torch.float32))
        )

    # ---------------------------------------------------------
    # JAX FUNCTIONS (JIT-Safe Callbacks)
    # ---------------------------------------------------------
    def reset(self, key, params=None):
        """JIT-safe reset using io_callback."""
        # Define expected memory shapes for JAX compiler
        result_shapes = (
            jax.ShapeDtypeStruct((self.num_envs, self.obs_dim), jnp.float32),
            jax.ShapeDtypeStruct((self.num_envs, self.state_dim), jnp.float32),
        )

        obs_jax, critic_jax = jax.experimental.io_callback(
            self._host_reset,
            result_shapes
        )
        
        # Initialize truncation states
        dummy_state = DexmachinaState(
            time=jnp.zeros(self.num_envs, dtype=jnp.float32),
            truncated=jnp.zeros(self.num_envs, dtype=jnp.float32),
            info={
                "steps": jnp.zeros(self.num_envs, dtype=jnp.int32),
                "truncation": jnp.zeros(self.num_envs, dtype=jnp.float32)
            }
        )
        return obs_jax, critic_jax, dummy_state

    def step(self, key, state, action):
        """JIT-safe step using io_callback."""
        result_shapes = (
            jax.ShapeDtypeStruct((self.num_envs, self.obs_dim), jnp.float32),      # obs
            jax.ShapeDtypeStruct((self.num_envs, self.state_dim), jnp.float32),    # critic_obs
            jax.ShapeDtypeStruct((self.num_envs,), jnp.float32),                   # rew
            jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_),                     # done
            jax.ShapeDtypeStruct((self.num_envs,), jnp.float32),                   # truncated
        )

        obs_jax, critic_jax, rew_jax, dones_jax, truncated_jax = jax.experimental.io_callback(
            self._host_step,
            result_shapes,
            action
        )

        # Update the dummy state with step counts and truncation
        new_state = DexmachinaState(
            time=state.time + 1.0,
            truncated=truncated_jax,
            info={
                "steps": state.info["steps"] + 1,
                "truncation": truncated_jax
            }
        )

        # Return Gymnax format: obs, critic_obs, state, reward, done, info
        # Pass truncation back in info dict
        return obs_jax, critic_jax, new_state, rew_jax, dones_jax, {}
    
    def render(self):
        """
        Renders using DexMachina's native floating camera.
        Adapted for JAX/Gymnax.
        """
        # self.env is already the DexMachina BaseEnv, no need to peel wrappers!
        if hasattr(self.env, "_floating_camera") and self.env._floating_camera is not None:
            try:
                # 1. Render and grab just the RGB frame from the tuple
                out = self.env._floating_camera.render(segmentation=False)
                frame = out[0] if isinstance(out, tuple) else out

                # 2. Move to CPU and convert to NumPy
                if isinstance(frame, torch.Tensor):
                    frame = frame.detach().cpu().numpy()

                # 3. Drop the Alpha channel if it's RGBA (WandB requires RGB)
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]

                # Note on Batch Dimensions: 
                # FlashSAC requires (1, H, W, C). 
                # If reppo expects a standard image (H, W, C), comment the next two lines out!
                if frame.ndim == 3:
                    frame = frame[np.newaxis, ...]

                return frame
            
            except Exception as e:
                print(f"[DEBUG] Camera extraction failed: {e}")

        # Ultimate Fallback (Returns a black screen so the run doesn't crash)
        return np.zeros((1, 256, 256, 3), dtype=np.uint8)

    def _process_obs(self, obs_dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Process observations safely in PyTorch before converting to JAX."""
        # Extract policy observation
        if isinstance(obs_dict, dict) and "policy" in obs_dict:
            obs = obs_dict["policy"]
        else:
            obs = obs_dict  # Fallback if it's a raw tensor

        if not math.isinf(self._clip_obs):
            obs = torch.clamp(obs, -self._clip_obs, self._clip_obs)

        # Extract critic state safely
        critic_obs = obs
        if self.asymmetric_obs:
            if isinstance(obs_dict, dict):
                if "critic" in obs_dict:
                    critic_obs = obs_dict["critic"]
                elif "state" in obs_dict:
                    critic_obs = obs_dict["state"]
                elif "states" in obs_dict:
                    critic_obs = obs_dict["states"]

        return obs, critic_obs

    @property
    def default_params(self):
        return gymnax.EnvParams()
    
    @property
    def num_actions(self):
        return self.action_dim

    @property
    def num_obs(self):
        return self.obs_dim

    def action_space(self, params=None):
        return Box(low=-1.0, high=1.0, shape=(self.action_dim,))

    def observation_space(self, params=None):
        return Box(low=-float("inf"), high=float("inf"), shape=(self.obs_dim,)), \
               Box(low=-float("inf"), high=float("inf"), shape=(self.state_dim,))