#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast-Saliency Toolbox: Pseudo-models for fast saliency research.
"""

import os
import json
import click

@click.group()
def cli():
    pass

@cli.command()
def version():
    """Displays version information."""
    click.echo("Fast Saliency Toolbox: Saliency (Psuedo-Model) Implementation ")


########################################
# Generalization - Experiment
@cli.command()
@click.option("--skip", help="Comma-separated list of experiment stages that should be skipped {train, test, run}")
@click.option("-n", "--name", help="The name of the experiment.")
@click.option("-c", "--conf_file", help="The path to the configuration file.")

@click.option("-l", "--logging_dir", help="Where should the logs be stored?")
@click.option("-i", "--input_images", help="The images used for experimenting. Should contain three folders inside: train, val and run.")
@click.option("-s", "--input_saliencies", help="Specify the directory to the saliency images. Should contain a separated folder for each model/task name.")

@click.option("--wandb", is_flag=True, help="Do you want to report to wandb?")
def experiment(skip, name, conf_file, logging_dir, input_images, input_saliencies, wandb):
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
    if logging_dir:
        experiment_conf["logging_dir"] = logging_dir
    if input_images:
        train_conf["input_images_train"] = os.path.join(input_images, "train")
        train_conf["input_images_val"] = os.path.join(input_images, "val")
        test_conf["input_images_test"] = os.path.join(input_images, "val") # TODO: change to /test once testdata available
        run_conf["input_images_run"] = os.path.join(input_images, "run")
    if input_saliencies:
        train_conf["input_saliencies"] = input_saliencies
        test_conf["input_saliencies"] = input_saliencies
    
    os.environ["WANDB_MODE"] = "online" if wandb else "offline"

    # build & update paths
    logging_dir = experiment_conf["logging_dir"]
    experiment_name = experiment_conf["name"]
    experiment_dir = os.path.abspath(os.path.join(logging_dir, experiment_name))

    train_conf["logging_dir"] = os.path.join(experiment_dir, "train_logs")
    test_conf["logging_dir"] = os.path.join(experiment_dir, "test_logs")
    run_conf["logging_dir"] = os.path.join(experiment_dir, "run_logs")

    model_path = os.path.join(train_conf["logging_dir"], train_conf["export_path"], "best.pth")
    test_conf["model_path"] = model_path
    run_conf["model_path"] = model_path

    # DO NOT FURTHER ADJUST THE CONF FROM THIS POINT ON

    # save the current config file into the experiment folder
    config_dump = os.path.join(experiment_dir, "used_config.json")
    os.makedirs(os.path.dirname(config_dump), exist_ok=True)
    with open(config_dump, "w") as f:
        json.dump(conf, f, indent=4)

    try:
        from backend.multimodel.hnet_contextmod.pipeline import Experiment
        import wandb as wdb
        
        wdb.login()
        with wdb.init(project="kdsalbox-generalization", entity="ba-yanickz", config=conf):
            t = Experiment(conf)
            t.execute()

    except ValueError as e:
        print(str(e))
        exit(64)

if __name__ == "__main__":
    cli()