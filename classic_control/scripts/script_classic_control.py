import os
import argparse

def main(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    if args.ppo:
        algos = ["EWoKPPO"]
    else:
        algos = ["Ours"]
    if args.with_vanilla:
        if args.ppo:
            algos.append("PPO")
        else:
            algos.append("Vanilla")
    if args.dmn_rand:
        if args.ppo:
            algos.append("PPO_Domain_Rand")
        else:
            algos.append("Domain_Rand")
    LOAD = "--load" if args.load else ""
    USE_WANDB = f"--use_wandb --wandb_entity {args.wandb_entity} --wandb_key {args.wandb_key}" if args.use_wandb else ""
    common_arguments = (f"{LOAD}  --test_episodes {args.test_episodes} "
                        f"--nominal_noises {' '.join([str(n) for n in args.nominal_noises])} "
                        f"--env_name {args.env_name} {USE_WANDB}")
    for seed in args.train_seeds:
        for algo in algos:
            if algo == "Vanilla":
                cmd_line = f"python ./run_classic_control.py " \
                           f"--algo {algo} --train_seed {seed} {common_arguments}"
                print(f"Executing cmd_line:\t{cmd_line}")
                os.system(cmd_line)
            elif algo == "Domain_Rand":
                dmn_rand_keys = " ".join(args.dmn_rand_keys)
                cmd_line = f"python ./run_classic_control.py --env_name {args.env_name} " \
                           f"--algo {algo} --train_seed {seed} {common_arguments}" \
                           f"--dmn_rand_name {args.dmn_rand_name} --dmn_rand_keys {dmn_rand_keys}"
                print(f"Executing cmd_line:\t{cmd_line}")
                os.system(cmd_line)
            elif algo == "Ours":
               for kappa in args.kappa:
                   for n_sample in args.n_sample:
                       cmd_line = f"python ./run_classic_control.py --env_name {args.env_name} " \
                                  f"--algo {algo} --train_seed {seed} {common_arguments} " \
                                   f"--n_sample {n_sample} --kappa {kappa} "
                       print(f"Executing cmd_line:\t{cmd_line}")
                       os.system(cmd_line)
            elif algo == "PPO":
                cmd_line = f"python ./run_classic_control_PPO.py " \
                           f"--algo {algo} --train_seed {seed} {common_arguments}"
                print(f"Executing cmd_line:\t{cmd_line}")
                os.system(cmd_line)
            elif algo == "EWoKPPO":
               for kappa in args.kappa:
                   for n_sample in args.n_sample:
                       cmd_line = f"python ./run_classic_control_PPO.py --env_name {args.env_name} " \
                                  f"--algo {algo} --train_seed {seed} {common_arguments} " \
                                  f"--n_sample {n_sample} --kappa {kappa} "
                       print(f"Executing cmd_line:\t{cmd_line}")
                       os.system(cmd_line)
            elif algo == "PPO_Domain_Rand":
                dmn_rand_keys = " ".join(args.dmn_rand_keys)
                cmd_line = f"python ./run_classic_control_PPO.py --env_name {args.env_name} " \
                           f"--algo {algo} --train_seed {seed} {common_arguments}" \
                           f"--dmn_rand_name {args.dmn_rand_name} --dmn_rand_keys {dmn_rand_keys}"
                print(f"Executing cmd_line:\t{cmd_line}")
                os.system(cmd_line)

def get_params():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('-env_name', '--env_name', type=str, choices=['Cartpole'])
    parser.add_argument('-nominal_noises', '--nominal_noises', type=float, default=[0, 0.01, 0, 0.01], nargs="*")
    parser.add_argument('-k', '--kappa', type=float, default=[], nargs='*')
    parser.add_argument('-n', '--n_sample', type=int, default=[],  nargs='*')
    parser.add_argument('-load', '--load', action='store_true', default=False)  # on/off flag
    parser.add_argument('-with_vanilla', '--with_vanilla', action='store_true', default=False)  # on/off flag
    parser.add_argument('-ppo', '--ppo', action='store_true', default=False, help='Use domain randomization')  # on/off flag
    parser.add_argument('-dmn_rand', '--dmn_rand', action='store_true', default=False, help='Use domain randomization')  # on/off flag
    parser.add_argument('-test_episodes', '--test_episodes', type=int, default=30)
    parser.add_argument('-gpu', '--gpu', type=int, default=None)
    parser.add_argument('-train_seeds', '--train_seeds', type=int, default=[1],  nargs='*')

    parser.add_argument('-dmn_rand_keys', '--dmn_rand_keys', type=str, nargs="*")
    parser.add_argument('-dmn_rand_name', '--dmn_rand_name', type=str, default="NoName")
    parser.add_argument('-use_wandb', '--use_wandb', action='store_true', default=False)
    parser.add_argument('-wandb_key', '--wandb_key', type=str)
    parser.add_argument('-wandb_entity', '--wandb_entity', type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_params()
    main(args)
