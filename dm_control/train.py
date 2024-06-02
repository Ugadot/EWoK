import os
import time

import numpy as np
import torch
import hydra

from logger import Logger
from replay_buffer import ReplayBuffer
from rwrl_env import make_env
import utils


class Workspace(object):
    def __init__(self, cfg):
        # Set up logger, seed, etc.
        self.work_dir = os.getcwd()
        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name,
        )
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        print(f"workspace: {self.work_dir}")

        # Create env and agent
        self.env = make_env(cfg.env.train)
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)
        if cfg.env.train.noisy.enable and cfg.env.train.noisy.reset:
            self.env.agent = self.agent
        print(f"Observation space: {self.env.observation_space}")
        print(f"Action space: {self.env.action_space}")

        # Create replay and step counter
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
        )
        self.step = 0

        self.perturbed_values = np.linspace(
            cfg.env.test.noisy.mean_min,
            cfg.env.test.noisy.mean_max,
            cfg.env.test.n_perturb_values,
        )

    def evaluate(self, env):
        average_episode_reward = 0
        for _ in range(self.cfg.num_eval_episodes):
            obs = env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
            average_episode_reward += episode_reward
        average_episode_reward /= self.cfg.num_eval_episodes
        return average_episode_reward

    def run_evaluate(self):
        start_time = time.time()
        for step, value in enumerate(self.perturbed_values):
            self.cfg.env.test.noisy.mean = value
            test_env = make_env(self.cfg.env.test)
            avg_rew = self.evaluate(test_env)
            print(f"Evaluation: {step + 1}-th value: {value}, rew: {avg_rew}")
            self.logger.log(f"eval/episode_reward ({step + 1}-th)", avg_rew, step)
        self.logger.log("eval/duration", time.time() - start_time, step)
        self.logger.dump(step)

    def run(self):
        obs = self.env.reset()
        self.agent.reset()
        episode, episode_reward, episode_step, done = 0, 0, 0, False
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                self.logger.log("train/episode_reward", episode_reward, self.step)
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                self.logger.log("train/episode", episode, self.step)

            if self.step > 0 and self.step % self.cfg.log_frequency == 0:
                self.logger.log("train/duration", time.time() - start_time, self.step)
                start_time = time.time()
                self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

            if self.step % 500000 == 0 and self.step > 0:
                test_start_time = time.time()
                self.run_evaluate()
                start_time = start_time + time.time() - test_start_time

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

        # Save final policy
        torch.save(self.agent, os.path.join(self.work_dir, "agent.pt"))

        # Evaluate at the end of training
        self.run_evaluate()


@hydra.main(config_path="config/train.yaml", strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
