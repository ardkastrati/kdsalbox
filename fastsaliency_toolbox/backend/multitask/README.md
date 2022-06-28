# Introduction

This is a research framework under development.
It aims to provide tools to investigate how we can make saliency models generalize to new tasks and new image domains.
In essence we are trying to find a model takes a few image-saliency pairs as input and then outputs a model that best fits the task "described" by those pairs.

The framework currently supports meta-learning via Hypernetworks but should be easy to extend to new approaches.

# Framework
The frameworks core is a fully configurable experiment pipeline that currently supports the following stages:
| Name              | Description                                                                                             | Data                 | Status    |
|-------------------|---------------------------------------------------------------------------------------------------------|----------------------|-----------|
| pretrain_weights  | Pretrains the Hypernetwork to output the weights of the models in fastsaliency_toolbox/models           | Weights              | [x]       |
| pretrain_one_task | Primes the Hypernetwork on one task (e.g. AIM)                                                          | Image/Saliency Pairs | [x]       |
| train_catchup     | Freezes all shared Hypernetwork parameters and lets the individual parameters adjust to the shared ones | Image/Saliency Pairs | partially |
| train             | Main training stage                                                                                     | Image/Saliency Pairs | [x]       |
| test              | Generates a variety of metrics for a model                                                              | Image/Saliency Pairs | [x]       |
| run               | For a list of images it produces the saliency maps and stores them                                      | Images               | [x]       |

Also after each stage that modifies the model there is an ExportStage that simply saves the model.
The framework is intended to be used with [Weights & Biases](https://wandb.ai/home)

### Terminology
- hnet: Hypernetwork
- mnet: Main network
- cwl: Custom Weights Layer
- mv2: MobilenetV2

## Running an experiment
1. Follow the instructions on the main readme of this repo to setup the environment ect.
2. Create custom config.json specifying all the details about the model architecture and trainings
3. Create custom gridearch.json specifying all the configurations you want to run
4. Navigate to fastsaliency_toolbox

Now you can run via
```console
    run gmain.py gridsearch --name="name of your experiment" --description="description of your experiment" --conf_file="path to your config.json" --param_grid_file="path to your gridsearch.json" --input_images="path to folder containing input images" --input_saliencies="path to folder containing subfolder for each task containing saliency maps" 
  ```
Optional arguments are:
 - --wdb: to report to wandb
 - --skip: comma separated list of pipeline stage names that should be skipped



# Hypernetworks
Hypernetworks in a nutshell: A hnet is a network that produces the weights of other networks as outputs.
To train a hnet we feed a task_id to the network and it will generate a list of weights. We then use this weights as our parameters in the forward pass of the mnet and can backpropagate through the weights to optimize the hnet. Note that we can also have parameters that are shared between tasks and are learned by the mnet directly. The mnet typically consists of a encoder and a decoder.

### Custom Weights Layer
A cwl automatically keeps track of the parameters that are learned by the hnet and those that are learned internally by the mnet. There is currently support for doing conv2d and bn2d with custom weights (as well as all the layers of the mv2 architecture).

### MobilenetV2
We suggest using mv2 as an encoder. There is a cwl implementation of mv2 that supports to be used in a partial manner too (cutoff). The most simple option is to not learn the mv2 via the hnet but to have it shared between tasks.

## train-api
Plug and play API that facilitates experimenting around with a lot of different training procedures, architectures etc.

- DataProvider: layer of abstraction that provide data for training
- Checkpointer: decides when to create checkpoints and can restore the best model at the end of training
- Actions: basically a callback that can be invoked at any point during training. We distinguish EpochActions and BatchActions (basically the same but with some extra parameters)
- ProgressTrackers: action dedicated to tracking and reporting some metrics during training
- ATrainer: interface that exposes all training data etc to the actions, checkpointer etc
- TrainingStep: does one step of training

### train-impl
- DataProviders
  - MultitaskBatchProvider: provides batches of (task_id, X, y) for multiple tasks (each batch corresponds to just one task though). Allows to have a fixed amount of consecutive batches of the same task.
  - BatchAndTaskProvider: provides batches of (task_id, X, y) but with a const task_id
  - BatchProvider: basically wraps a dataloader that provides batches of (X,y)
- Actions
  - LrDecay: Decreases lr for all param groups by a factor
  - FreezeEncoder: Freezes the **internal/shared** weights of the encoder for a given amount of epochs
  - FreezeHNETForCatchup (not fully supported): Freezes all hnet params that are shared between tasks for a given amount of epochs
  - LogEpochLosses: simply prints train and val loss at the end of an epoch
  - BatchLogger: simply prints the accumulated loss at the end of a batch
  - WeightWatcher (not fully supported): prints some information about gradients and parameters of the hnet
- Trainer: implements a basic trainer that supports checkpointing & restoring, multiple progress trackers, start actions (start of training), epoch start actions (start of each epoch), epoch end actions (end of each epoch), batch actions (end of each batch), end actions (end of training)
- TrainingSteps
  - MultitaskTrainStep: training step in a multitask setting (task_id, X, y)
  - WeightsTrainStep: training step in weight setting (task_id, weights)
  
### train-impl-wandb
- Checkpointer: stores hnet and mnet every fixed amount of epochs or if it is a new best performing model. Can upload the model to wandb too.
- RunProgressTrackerWandb: generates a set of saliency-maps from a set of images and uploads them to wandb.
- Actions
  - WatchWandb (StartAction): Watches a networks params and gradients and reports to wandb
  - ReportLiveMetricsWandb (EpochAction): Reports losses and the learning rate to wandb

