from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from agent import Agent
import utils

import hydra


class TD3Agent(Agent):
    """TD3 algorithm."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_range,
        device,
        critic_cfg,
        actor_cfg,
        discount,
        actor_lr,
        actor_betas,
        actor_update_frequency,
        critic_lr,
        critic_betas,
        critic_tau,
        critic_target_update_frequency,
        target_policy_noise_std,
        target_policy_noise_clip,
        batch_size,
    ):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.target_policy_noise_std = target_policy_noise_std
        self.target_policy_noise_clip = target_policy_noise_clip
        self.batch_size = batch_size

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
        self.actor_target = hydra.utils.instantiate(actor_cfg).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=actor_betas
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=critic_betas
        )

        self.train()
        self.critic_target.train()
        self.Q_history = deque(maxlen=100)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        action = self.actor(obs)
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step):
        with torch.no_grad():
            noise = action.clone().data.normal_(0, self.target_policy_noise_std)
            noise.clamp_(-self.target_policy_noise_clip, self.target_policy_noise_clip)
            next_action = self.actor_target(next_obs)
            next_action = torch.clamp(next_action + noise, -1, 1)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q_min = torch.min(target_Q1, target_Q2)
            self.Q_history.append(target_Q_min.mean().item())
            logger.log("train_critic/maxQ", target_Q_min.max(), step)
            logger.log("train_critic/minQ", target_Q_min.min(), step)
            logger.log("train_critic/avgQ", target_Q_min.mean(), step)
            target_Q = reward + not_done * self.discount * target_Q_min
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        logger.log("train_critic/loss", critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, obs, logger, step):
        action = self.actor(obs)
        actor_Q, _ = self.critic(obs, action)
        actor_loss = -actor_Q.mean()
        logger.log("train_actor/loss", actor_loss, step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size
        )

        logger.log("train/batch_reward", reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
            utils.soft_update_params(self.actor, self.actor_target, self.critic_tau)
