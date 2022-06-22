"""
trainer_utils
-------------

Contains utility methods for training with wandb.

"""

from typing import Callable, Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import wandb

from backend.multitask.hnet.hyper_model import HyperModel
from backend.image_processing import process
from backend.parameters import ParameterMap

def run_model_on_images_and_report_to_wandb(title : str, 
    tasks : List[str], model : HyperModel, run_dataloader : DataLoader, 
    postprocess_parameter_map : ParameterMap, device):
    """
    Runs a model for all images provided by the dataloader and for all tasks
    and uploads the resulting saliency maps to wandb

    Args:
        title (str): title of the table
        tasks (List[str]): names of all the tasks that should be run
        model (HyperModel): the hypermodel that should be used for running
        run_dataloader (DataLoader): dataloader that provides (images, _, _)
        postprocess_parameter_map (ParameterMap): will be applied to the generated saliency map
        device : which device to do the computation on 

    """

    model.to(device)

    with torch.no_grad():
        cols = ["Model"]
        cols.extend([os.path.basename(output_path[0]) for (_, _, output_path) in run_dataloader])
        data = []
        for task in tasks:
            row = [task]
            task_id = model.task_to_id(task)

            for (image, _, _) in run_dataloader:
                image = image.to(device)
                saliency_map = model.compute_saliency(image, task_id)
                post_processed_image = np.clip((process(saliency_map.cpu().detach().numpy()[0, 0], postprocess_parameter_map)*255).astype(np.uint8), 0, 255)
                img = wandb.Image(post_processed_image)
                row.append(img)

                if torch.cuda.is_available(): # avoid GPU out of mem
                    del image
                    del saliency_map
                    torch.cuda.empty_cache()
            
            data.append(row)

        table = wandb.Table(data=data, columns=cols)
        wandb.log({title: table})

def train_one_epoch_multitask(tasks : List[str], model : HyperModel,
    dataloaders : Dict[str,Dict[str,DataLoader]], 
    criterion : nn.Module, optimizer : torch.optim.Optimizer, 
    mode : str, device,
    batches_per_task_train : int, batches_per_task_val : int,
    consecutive_batches_per_task : int = 1,
    log_freq : int = 100):
    """
    Trains one epoch in a multitask setting (that is with a dataloader per task)

    Args:
        tasks (List[str]): the names of the tasks we are learning
        model (HyperModel): the model that should be used
        dataloaders (Dict[str, Dict[str, DataLoader]]) : a dictionary containing two dictionaries (one for train and one for val).
            Each of those dictionaries then contain a dataloader per task.
        criterion (nn.Module): the loss function used for training
        optimizer (torch.optim.Optimizer): the optimizer used for training
        mode (str): {train, val}
        device: which device to operate on
        batches_per_task_train (int): how many batches should be sampled during training for each task
        batches_per_task_val (int): how many batches should be sampled during validation for each task
        consecutive_batches_per_task (int): how many batches in a row should be sampled from the same task
        log_freq (int): how many batches should pass between logging the accumulated loss to the console

    """
    model.to(device)

    if mode == "train": model.train()
    elif mode == "val": model.eval()

    all_loss = []

    # defines which batch will be loaded from which task/model
    if mode == "train":
        limit = batches_per_task_train // consecutive_batches_per_task
        all_batches = np.concatenate([np.repeat(model.task_to_id(task), limit) for task in tasks])
        np.random.shuffle(all_batches)
        all_batches = np.repeat(all_batches, consecutive_batches_per_task)
    else:
        all_batches = np.concatenate([np.repeat(model.task_to_id(task), batches_per_task_val) for task in tasks])

    # for each model
    data_iters = [iter(d) for d in dataloaders[mode].values()] # Note: DataLoader shuffles when iterator is created
    for (i,task_id) in enumerate(all_batches):
        X,y = next(data_iters[task_id])

        optimizer.zero_grad()

        # put data on GPU (if cuda)
        X = X.to(device)
        y = y.to(device)

        pred = model(task_id.item(), X)
        loss = criterion(pred, y)

        # training
        if mode == "train":
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            all_loss.append(loss.item())

            # logging
            if i%log_freq == 0:
                print(f"Batch {i}/{len(all_batches)}: current accumulated loss {np.mean(all_loss)}", flush=True)
        
        # remove batch from gpu (if cuda)
        if torch.cuda.is_available():
            del pred
            del loss
            del X
            del y
            torch.cuda.empty_cache()
            
    return np.mean(all_loss)

def train_one_epoch_just_one_task(model : HyperModel,
    dataloaders : Dict[str,DataLoader], 
    criterion : nn.Module, optimizer : torch.optim.Optimizer, 
    mode : str, device, task_id : int,
    log_freq : int = 100):
    """
    Trains one epoch in a multitask setting but only for one task (that is with only one dataloader)

    Args:
        model (HyperModel): the model that should be used
        dataloaders (Dict[str, Dict[str, DataLoader]]) : a dictionary containing two dictionaries (one for train and one for val).
            Each of those dictionaries then contain a dataloader per task.
        criterion (nn.Module): the loss function used for training
        optimizer (torch.optim.Optimizer): the optimizer used for training
        mode (str): {train, val}
        device: which device to operate on
        task_id (int): the id of the the task that the dataloader corresponds to
        log_freq (int): how many batches should pass between logging the accumulated loss to the console

    """
    model.to(device)

    if mode == "train": model.train()
    elif mode == "val": model.eval()

    all_loss = []

    # for each model
    dataloader = dataloaders[mode]
    for i,(X,y) in enumerate(dataloader):

        optimizer.zero_grad()

        # put data on GPU (if cuda)
        X = X.to(device)
        y = y.to(device)

        pred = model(task_id, X)
        loss = criterion(pred, y)

        # training
        if mode == "train":
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            all_loss.append(loss.item())

            # logging
            if i%log_freq == 0:
                print(f"Batch {i}/{len(dataloader)}: current accumulated loss {np.mean(all_loss)}", flush=True)
        
        # remove batch from gpu (if cuda)
        if torch.cuda.is_available():
            del pred
            del loss
            del X
            del y
            torch.cuda.empty_cache()
            
    return np.mean(all_loss)

def pretty_print_epoch(epoch : int, mode : str, loss : float, lr : float):
    """
    Prints the stats of an epoch to the console.

    Args:
        epoch (int): the current epoch
        mode (str): {train, val}
        loss (float): the loss of the epoch
        lr (float): the current lr
    """
    print("--------------------------------------------->>>>>>")
    print(f"Epoch {epoch}: loss {mode} {loss}, lr {lr}", flush=True)
    print("--------------------------------------------->>>>>>")

def save(path : str, model : HyperModel, save_to_wandb : bool = True):
    """
    Saves the model to a specific path and optionally also syncs it to wandb

    Args:
        path (str): the path where the model should be saved to
        model (HyperModel): the model that should be saved
        save_to_wandb (bool): whether or not to save the model to wandb as well
    """
    model.save(path)
    if save_to_wandb:
        wandb.save(path, base_path=wandb.run.dir)

def train_val_one_epoch_and_report_to_wandb(model : HyperModel, train_one_epoch : Callable[[str], float], 
    epoch : int, lr : float, smallest_loss : float, all_epochs : List[Tuple[int,float,float]],
    auto_checkpoint_steps : int, save_checkpoints_to_wandb : bool,
    logging_dir : str, verbose : bool, export_path_best : str,
    wandb_report_prefix : str = "",
    on_checkpoint : Callable[[bool], None] = None):
    """
    Trains and evaluates the model & reports to wandb.
    Checkpointing: whenever the current validation loss is smaller than the previously smallest loss
        or the epoch is a multiple of auto_checkpoint_steps a checkpoint will be created. That includes
        saving the model and calling the on_checkpoint callback function.
    Saves the best model to "export_path_best". Can later be used to restore the best model.
    
    Args:
        model (HyperModel): the model to be used
        train_one_epoch (Callable[[str], float]): function that can be invoked with a given mode,
            runs one epoch and then returns the loss of that epoch
        epoch (int): the current epoch
        lr (float): the current lr
        smallest_loss (float): the current smallest loss
        all_epochs (List[Tuple[int,float,float]]): list of stats (epoch, loss_train, loss_val). Will be modified in-place!
        auto_checkpoint_steps (int): will create a checkpoint every x epochs
        save_checkpoints_to_wandb (bool): whether or not to sync checkpoints to wandb
        logging_dir (str): directory where logs will be written to
        verbose (bool): whether or not to log to console
        export_path_best (str): the path where the best model will be saved
        wandb_report_prefix (str): prefix used when reporting metrics
        on_checkpoint (Callable[[bool]] = None): callback that is called whenever a new checkpoint is created. 
            Takes as an argument whether or not the checkpoint is created because a new best model has been trained.
    Returns:
        The updated smallest_loss value

    """

    # train the networks
    loss_train = train_one_epoch("train")
    if verbose: pretty_print_epoch(epoch, "train", loss_train, lr)

    # validate the networks
    loss_val = train_one_epoch("val")
    if verbose: pretty_print_epoch(epoch, "val", loss_val, lr)

    # STATS / REPORTING
    all_epochs.append([epoch, loss_train, loss_val])

    # if better performance than all previous => save weights as checkpoint
    is_best_model = smallest_loss is None or loss_val < smallest_loss
    if epoch % auto_checkpoint_steps == 0 or is_best_model:
        checkpoint_dir = os.path.join(logging_dir, f"checkpoint_in_epoch_{epoch}/")
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = f"{checkpoint_dir}/{epoch}_{loss_val:f}.pth"

        save(path, model, save_checkpoints_to_wandb)
        
        # overwrite the best model
        if is_best_model:
            smallest_loss = loss_val
            save(export_path_best, model, save_checkpoints_to_wandb)
        
        if on_checkpoint:
            on_checkpoint(is_best_model)
    
    # save/overwrite results at the end of each epoch
    stats_file = os.path.join(os.path.relpath(logging_dir, wandb.run.dir), "all_results").replace("\\", "/")
    table = wandb.Table(data=all_epochs, columns=["Epoch", "Loss-Train", "Loss-Val"])
    wandb.log({
            f"{wandb_report_prefix} - epoch": epoch,
            f"{wandb_report_prefix} - loss train": loss_train,
            f"{wandb_report_prefix} - loss val": loss_val,
            f"{wandb_report_prefix} - learning rate": lr,
            stats_file:table
        })

    return smallest_loss
