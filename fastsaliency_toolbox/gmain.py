#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast-Saliency Toolbox: Pseudo-models for fast saliency research.
"""

import os
import json
import click
import wandb

from backend.multimodel.hyper_experiment import HyperExperiment
from backend.multimodel.hyper_model import HyperModel
from backend.multimodel.hyper_runner import HyperRunner
from backend.multimodel.hyper_tester import HyperTester
from backend.multimodel.hyper_trainer import HyperTrainer

@click.group()
def cli():
    pass

@cli.command()
def version():
    """Displays version information."""
    click.echo("Fast Saliency Toolbox: Saliency (Psuedo-Model) Implementation ")

def run_with_conf(conf, resume_id=None, ext_model=None):
    experiment_conf = conf["experiment"]
    train_conf = conf["train"]
    test_conf = conf["test"]
    run_conf = conf["run"]

    experiment_name = experiment_conf["name"]
    experiment_description = experiment_conf["description"]

    resuming = False
    if resume_id:
        resuming = True
        resume = "must"
    else:
        resume_id = None
        resume = None

    original_logging_dir = experiment_conf["logging_dir"]
    wandb.login()
    with wandb.init(project="kdsalbox-generalization", entity="ba-yanickz", name=experiment_name, config=conf, id=resume_id, resume=resume, notes=experiment_description, reinit=True):
        # build & update paths relative to wandb run dir
        experiment_conf["logging_dir"] = os.path.join(wandb.run.dir, experiment_conf["logging_dir"])
        logging_dir = experiment_conf["logging_dir"]
        experiment_dir = os.path.abspath(os.path.join(logging_dir, experiment_name))

        train_conf["logging_dir"] = os.path.join(experiment_dir, "train_logs")
        test_conf["logging_dir"] = os.path.join(experiment_dir, "test_logs")
        run_conf["logging_dir"] = os.path.join(experiment_dir, "run_logs")

        model_path = os.path.join(train_conf["logging_dir"], train_conf["export_path"], "best.pth")
        test_conf["model_path"] = ext_model if ext_model else model_path
        run_conf["model_path"] = ext_model if ext_model else model_path

        # DO NOT FURTHER ADJUST THE CONF FROM THIS POINT ON

        # save the current config file into the experiment folder
        if resuming and not ext_model:
            rel_model_path = os.path.relpath(model_path, wandb.run.dir).replace("\\", "/") # wandb expects linux path
            wandb.restore(rel_model_path)
        else:
            config_dump = os.path.join(experiment_dir, "used_config.json")
            print(config_dump)
            os.makedirs(os.path.dirname(config_dump), exist_ok=True)
            with open(config_dump, "w") as f:
                json.dump(conf, f, indent=4)
            wandb.save(config_dump, base_path=wandb.run.dir)

        try:
            from backend.multimodel.hnet_contextmod.model import hnet_mnet_from_config as hmfc_contextmod
            from backend.multimodel.hnet_full_chunked.model import hnet_mnet_from_config as hmfc_full_chunked
            hnet_mnet_from_config = None
            if conf["type"] == "contextmod":
                hnet_mnet_from_config = hmfc_contextmod
            elif conf["type"] == "full_chunked":
                hnet_mnet_from_config = hmfc_full_chunked

            print(f"Running {conf['type']} experiment")

            t = HyperExperiment(conf, 
                    lambda c : HyperTrainer(c, HyperModel(lambda : hnet_mnet_from_config(c), c["train"]["tasks"])), 
                    lambda c : HyperTester(c, HyperModel(lambda : hnet_mnet_from_config(c), c["test"]["tasks"])), 
                    lambda c : HyperRunner(c, HyperModel(lambda : hnet_mnet_from_config(c), c["run"]["tasks"]))
                )
            t.execute()

        except ValueError as e:
            print(str(e))
            exit(64)

        experiment_conf["logging_dir"] = original_logging_dir # reset the logging dir for the next run, the other paths are then built ontop of it

########################################
# Generalization - Experiment
@cli.command()
@click.option("--skip", help="Comma-separated list of experiment stages that should be skipped {train, test, run}")
@click.option("-n", "--name", help="The name of the experiment.")
@click.option("-c", "--conf_file", help="The path to the configuration file.")

@click.option("-l", "--logging_dir", help="Where should the logs be stored?")
@click.option("-i", "--input_images", help="The images used for experimenting. Should contain three folders inside: train, val and run.")
@click.option("-s", "--input_saliencies", help="Specify the directory to the saliency images. Should contain a separated folder for each model/task name.")

@click.option("--wdb", is_flag=True, help="Do you want to report to wandb?")
@click.option("--resume_id", help="The id of a run you want to continue")
@click.option("--ext_model", help="Specify the location of a model that should be used for test/run.")

@click.option("--description", help="A description of what makes this run special")
def experiment(skip, name, conf_file, logging_dir, input_images, input_saliencies, wdb, resume_id, ext_model, description):
    """Experiment with models on images in a directory."""

    # load config file
    with open(conf_file) as f:
        conf = json.load(f)

    experiment_conf = conf["experiment"]
    train_conf = conf["train"]
    test_conf = conf["test"]
    run_conf = conf["run"]

    # overwrite params given as args
    if skip:
        experiment_conf["skip"] = skip.split(",")
    if name:
        experiment_conf["name"] = name
    if description:
        experiment_conf["description"] = description
    if logging_dir:
        experiment_conf["logging_dir"] = logging_dir
    if input_images:
        train_conf["input_images_train"] = os.path.join(input_images, "train")
        train_conf["input_images_val"] = os.path.join(input_images, "val")
        train_conf["input_images_run"] = os.path.join(input_images, "run")
        test_conf["input_images_test"] = os.path.join(input_images, "val") # TODO: change to /test once testdata available
        run_conf["input_images_run"] = os.path.join(input_images, "run")
    if input_saliencies:
        train_conf["input_saliencies"] = input_saliencies
        test_conf["input_saliencies"] = input_saliencies
    
    os.environ["WANDB_MODE"] = "online" if wdb else "offline"

    run_with_conf(conf, resume_id, ext_model)

    ########################################
# Generalization - Gridsearch
@cli.command()
@click.option("--skip", help="Comma-separated list of experiment stages that should be skipped {train, test, run}")
@click.option("-n", "--name", help="The name of the experiment.")
@click.option("-c", "--conf_file", help="The path to the configuration file.")

@click.option("-l", "--logging_dir", help="Where should the logs be stored?")
@click.option("-i", "--input_images", help="The images used for experimenting. Should contain three folders inside: train, val and run.")
@click.option("-s", "--input_saliencies", help="Specify the directory to the saliency images. Should contain a separated folder for each model/task name.")

@click.option("--wdb", is_flag=True, help="Do you want to report to wandb?")

@click.option("--description", help="A description of what makes this run special")
def gridsearch(skip, name, conf_file, logging_dir, input_images, input_saliencies, wdb, description):
    # load config file
    with open(conf_file) as f:
        conf = json.load(f)

    experiment_conf = conf["experiment"]
    train_conf = conf["train"]
    test_conf = conf["test"]
    run_conf = conf["run"]

    # overwrite params given as args
    if skip:
        experiment_conf["skip"] = skip.split(",")
    if name:
        experiment_conf["name"] = name
    if description:
        experiment_conf["description"] = description
    if logging_dir:
        experiment_conf["logging_dir"] = logging_dir
    if input_images:
        train_conf["input_images_train"] = os.path.join(input_images, "train")
        train_conf["input_images_val"] = os.path.join(input_images, "val")
        train_conf["input_images_run"] = os.path.join(input_images, "run")
        test_conf["input_images_test"] = os.path.join(input_images, "val") # TODO: change to /test once testdata available
        run_conf["input_images_run"] = os.path.join(input_images, "run")
    if input_saliencies:
        train_conf["input_saliencies"] = input_saliencies
        test_conf["input_saliencies"] = input_saliencies
    
    os.environ["WANDB_MODE"] = "online" if wdb else "offline"

    # TODO:
    # 3t
    # hnet_embedding_size
    # hnet_hidden_layers (amount x size)

    base_name = experiment_conf["name"]
    base_description = experiment_conf["name"]
    hnets = [
        [128, 128, 128],
        [128, 128, 128, 128],
        [256, 256, 256],
    ]
    consecutive_batches_per_tasks = [4,16,64]
    for i,(hnet_hidden_layers,consecutive_batches_per_task) in enumerate([(h, b) for h in hnets for b in consecutive_batches_per_tasks]):
        print("#############################")
        print(f"NOW RUNNING {i}")
        print("#############################")

        conf["model"]["hnet_hidden_layers"] = hnet_hidden_layers
        train_conf["consecutive_batches_per_task"] = consecutive_batches_per_task
        experiment_conf["name"] = f"({i}) {base_name} - {hnet_hidden_layers} - {consecutive_batches_per_task}"
        experiment_conf["description"] = f"{base_description}\nhnet_hidden_layers={hnet_hidden_layers}\nconsecutive_batches_per_task={consecutive_batches_per_task}"
        run_with_conf(conf)

if __name__ == "__main__":
    cli()