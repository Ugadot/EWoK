from gymnasium.envs.registration import register


def register_envs(env_names):
    for game in env_names:
        if game == "Cartpole":
            register(
                id=f"Noisy_{game}-v1",
                entry_point="envs.classic_control.cartpole:CartPoleEnv",
                max_episode_steps=500,
                reward_threshold=475.0,
            )