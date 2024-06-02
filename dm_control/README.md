# Code for experiments on DM control with SAC/TD3

## Install dependencies

```
conda env create -f conda_env.yml
source activate sac
```

## Training & evaluation

To train a baseline SAC agent on the `walker-walk` task, run:
```
CUDA_VISIBLE_DEVICES=0 python train.py env.train.name=walker_walk \
    experiment=temp \
    env.train.noisy.enable=True env.train.noisy.scale=0.2 seed=4
```

To train an SAC agent with domain randomization on the `walker-walk` task, run:
```
CUDA_VISIBLE_DEVICES=0 python train_dr.py env.train.name=walker_walk \
    experiment=temp \
    env.train.noisy.enable=True env.train.noisy.scale=0.2 seed=4
```

To train an SAC agent with our method on the `walker-walk` task, run:
```
CUDA_VISIBLE_DEVICES=0 python train.py env.train.name=walker_walk \
    experiment=temp \
    env.train.noisy.enable=True env.train.noisy.scale=0.2 seed=4 \
    env.train.noisy.reset=True  env.train.noisy.temperature=2
```

Add `agent=td3` to switch the base algorithm from SAC to TD3.
