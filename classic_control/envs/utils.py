import numpy as np


def get_update_env_params(params_dict):
    def update_env_func(env):
        new_params = {}
        for param_name, param_details in params_dict.items():
            is_param_discrete = param_details["is_discrete"]
            param_range = param_details["range"]
            param = np.random.uniform(low=param_range[0], high=param_range[-1])
            if is_param_discrete:
                param = np.round(param)
            new_params[param_name] = param
        env.update_func(new_params)
    return update_env_func
