# Using WandB with ETNA library

## Colab example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EBSqqBPaYgLWCRdpC5vMy9RiLBsCEd7I?usp=sharing)  

![](assets/etna-wandb.png)

[Sweep Dashboard](https://wandb.ai/martins0n/wandb-etna-sweep/sweeps/c7e0r8sq/overview?workspace=user-martins0n)

## Steps to start

- Define your pipeline and hyperparameters in `pipeline.yaml`, in example we will optimize number of iterations `iterations` and `learning-rate`

- Define WandB sweeps config `sweep.yaml` and push it to cloud:

```bash
WANDB_PROJECT=<project_name> WandB sweep sweep.yaml 
```

- You may change `dataloader` function and add additional parameters for WandB logger like tags for example in `run.py`

- Run WandB agent for hyperparameters optimization start:

```bash
wandb agent <user_name>/<project_name>/<sweep_id>
```
