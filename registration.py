from gym.envs.registration import register
# from gymnasium.envs.registration import register

def register_env():
    register(
                id=f"cliff_env",
                # entry_point="envs.cliff.tmp_cliff:CliffWalkingEnv",
                entry_point="envs.cliff.tmp_cliff:CliffWalkingEnv_2",
                max_episode_steps=500,
                # reward_threshold=475.0,
            )