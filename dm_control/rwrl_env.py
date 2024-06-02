from dmc2gym.wrappers import DMCWrapper, _spec_to_box, _flatten_obs
from dm_control.rl import control
import dm_env
import gym
import numpy as np
import realworldrl_suite.environments as rwrl
from realworldrl_suite.utils.wrappers import LoggingEnv
import torch


def make_env(env_cfg):
    """Helper function to create dm_control environment"""
    domain_name = env_cfg.name.split("_")[0]
    task_name = "_".join(env_cfg.name.split("_")[1:])
    task_kwargs = env_cfg.task_kwargs if "task_kwargs" in env_cfg else None
    if "noisy" in env_cfg and env_cfg.noisy.enable:
        env = RWRLEnvWithNoisyAction(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            noise_mean=env_cfg.noisy.mean,
            noise_scale=env_cfg.noisy.scale,
            temperature=env_cfg.noisy.temperature if "temperature" in env_cfg.noisy else None,
            n_trials=env_cfg.noisy.n_trials if "n_trials" in env_cfg.noisy else None,
            use_reset=env_cfg.noisy.reset,
        )
    else:
        env = RWRLEnv(domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs)
    env.seed(env_cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class RWRLEnv(DMCWrapper):
    def __init__(
        self,
        domain_name,
        task_name,
        seed=0,
        task_kwargs=None,
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        episode_length=1000,
        channels_first=True,
    ):
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first
        self._max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

        # create task
        task_kwargs = task_kwargs or {}
        self._env = rwrl.load(
            domain_name=domain_name, task_name=f"realworld_{task_name}", **task_kwargs
        )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values(), np.float64
            )

        self._state_space = _spec_to_box(self._env.observation_spec().values(), np.float64)

        self.current_state = None

        # set seed
        self.seed(seed=seed)


class RWRLEnvWithNoisyAction(RWRLEnv):
    def __init__(
        self,
        *args,
        noise_mean=0.0,
        noise_scale=0.01,
        temperature=0.1,
        n_trials=10,
        use_reset=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        action_spec = self._env.action_spec()
        self.agent = None
        self._minimum = action_spec.minimum
        self._maximum = action_spec.maximum
        self._noise_std = noise_scale * (action_spec.maximum - action_spec.minimum)
        self._noise_mean = noise_mean * (action_spec.maximum - action_spec.minimum)
        self.temperature = temperature
        self.n_trials = n_trials
        self.use_reset = use_reset
        self.step_cnt = 0

    def step(self, action):
        self.step_cnt += 1
        if self.step_cnt >= 10000 and self.use_reset:
            return self.step_with_reset(action)
        else:
            return self.step_with_noise_action(action)

    def step_with_reset(self, action):
        # Call next_step multiple times with action + random noise
        # Reset the env state multiple times
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)

        # Forward multiple times
        old_state = self._env.physics.get_state().copy()
        extra = {"internal_state": self._env.physics.get_state().copy()}
        cand_obs = []
        cand_acts = []
        for _ in range(self.n_trials):
            reward = 0
            # sample action noise
            noisy_action = action + self._env.task.random.normal(
                loc=self._noise_mean, scale=self._noise_std
            )
            np.clip(noisy_action, self._minimum, self._maximum, out=noisy_action)
            # step
            for _ in range(self._frame_skip):
                time_step = self.env_fake_step(noisy_action)
                reward += time_step.reward or 0
                done = time_step.last()
                if done:
                    break
            cand_obs.append(self._get_obs(time_step))
            cand_acts.append(noisy_action)
            self._env.physics.set_state(old_state)
            self._env.physics.forward()

        # Transit to next state
        idx = self.sample_cand_state(cand_obs, cand_acts)
        # step
        reward = 0
        for _ in range(self._frame_skip):
            time_step = self._env.step(cand_acts[idx])
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra["discount"] = time_step.discount
        return obs, reward, done, extra

    def step_with_noise_action(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        noisy_action = action + self._env.task.random.normal(
            loc=self._noise_mean, scale=self._noise_std
        )
        reward = 0
        extra = {"internal_state": self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(noisy_action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra["discount"] = time_step.discount
        return obs, reward, done, extra

    def env_fake_step(self, action):
        if self._env._reset_next_step:
            timestep = self._env.reset()
        else:
            self._env._task.before_step(action, self._env._physics)
            self._env._physics.step(self._env._n_sub_steps)
            self._env._task.after_step(self._env._physics)
            reward = self._env._task.get_reward(self._env._physics)
            observation = self._env._task.get_observation(self._env._physics)
            if self._env._flat_observation:
                observation = control.flatten_observation(observation)
            timestep = dm_env.TimeStep(dm_env.StepType.MID, reward, 1.0, observation)
        # Only flatten observation if we're not forwarding one from a reset(),
        # as it will already be flattened.
        if self._env._flat_observation_ and not timestep.first():
            timestep = dm_env.TimeStep(
                step_type=timestep.step_type,
                reward=timestep.reward,
                discount=timestep.discount,
                observation=control.flatten_observation(timestep.observation)["observations"],
            )
        return timestep

    def sample_cand_state(self, obs, action):
        assert self.agent is not None  # assign the agent after construction
        obs = torch.tensor(np.stack(obs), device=self.agent.device, dtype=torch.float32)
        action = torch.tensor(np.stack(action), device=self.agent.device, dtype=torch.float32)
        with torch.no_grad():
            Q1, Q2 = self.agent.critic(obs, action)
            Q = torch.minimum(Q1, Q2).squeeze()
            # Q = Q - np.mean(self.agent.Q_history)
            Q = Q - Q.min()
            idx = torch.multinomial((-self.temperature * Q).exp(), 1).item()
        return idx
