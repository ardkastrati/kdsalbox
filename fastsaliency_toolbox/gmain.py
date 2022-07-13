"""
gmain.py
--------

Provides the CLI to the generalization component of the fastsaliency_toolbox.
Note: This part is currently work in progress but the interested user is encouraged to play around with the provided framework.

Available commands are:
    - gridsearch: Runs a gridsearch over a set of configurations specified by a .json file 
        (see the command for a list of possible arguments)

"""

import os
import json
import click
import wandb

from backend.multitask.hnet.models.hyper_model import HyperModel
from backend.multitask.hnet.stages.runner import Runner
from backend.multitask.hnet.stages.tester import Tester
from backend.multitask.hnet.stages.trainer_main import MainTrainer
from backend.multitask.hnet.stages.pretrainer_weights import PreTrainerWeights
from backend.multitask.hnet.stages.pretrainer_one_task import PreTrainerOneTask
from backend.multitask.hnet.stages.trainer_catchup import TrainerCatchup
from backend.multitask.hnet.stages.trainer_generalization_task import TrainerGeneralizationTask
from backend.multitask.pipeline.pipeline import Pipeline
from backend.multitask.pipeline.stages import ExportStage
from backend.multitask.hnet.stages.load_model import ModelLoader
from backend.multitask.hnet.stages.table_1 import Table1

@click.group()
def cli():
    pass

@cli.command()
def version():
    """Displays version information."""
    click.echo("Fast Saliency Toolbox: Generalization Implementation")

def run_with_conf(conf, group=None):
    run_name = conf["name"]
    run_description = conf["description"]

    wandb_entity = conf["wandb_entity"]
    wandb_project = conf["wandb_project"]

    wandb.login()
    start_method = "spawn" if os.name == "nt" else "fork" # check if windows or linux
    run = wandb.init(project=wandb_project, entity=wandb_entity, 
        name=run_name, notes=run_description, group=group, reinit=True, 
        config=conf, settings=wandb.Settings(start_method=start_method))

    # build & update paths relative to wandb run dir
    run_dir = os.path.abspath(os.path.join(wandb.run.dir, run_name))

    # DO NOT FURTHER ADJUST THE CONF FROM THIS POINT ON

    # save the current config file
    config_dump = os.path.join(run_dir, "used_config.json")
    print(config_dump)
    os.makedirs(os.path.dirname(config_dump), exist_ok=True)
    with open(config_dump, "w") as f:
        json.dump(conf, f, indent=4)
    wandb.save(config_dump, base_path=wandb.run.dir)
        
    # construct and run the experiment pipeline
    try:
        from backend.multitask.hnet.models.mnet import MNET
        from backend.multitask.hnet.models.hnet import HNET

        verbose = conf["verbose"]
        print(f"Running {run_name}")

        stages = []
        if "load_model" in conf.keys():
            stages.append(ModelLoader(conf, "load_model", verbose=verbose))
        if "pretrain_weights" in conf.keys():
            stages.append(PreTrainerWeights(conf, "pretrain_weights", verbose=verbose))
            stages.append(ExportStage("export - pretrain_weights", path=f"{os.path.join(run_dir, 'pretrain_weights', 'best.pth')}", verbose=verbose))
        if "pretrain_one_task" in conf.keys():
            stages.append(PreTrainerOneTask(conf, "pretrain_one_task", verbose=verbose))
            stages.append(ExportStage("export - pretrain_one_task", path=f"{os.path.join(run_dir, 'pretrain_one_task', 'best.pth')}", verbose=verbose))
        if "train_catchup" in conf.keys():
            stages.append(TrainerCatchup(conf, "train_catchup", verbose=verbose))
            stages.append(ExportStage("export - train_catchup", path=f"{os.path.join(run_dir, 'train_catchup', 'best.pth')}", verbose=verbose))
        if "train" in conf.keys():
            stages.append(MainTrainer(conf, "train", verbose=verbose))
            stages.append(ExportStage("export - train", path=f"{os.path.join(run_dir, 'train', 'best.pth')}", verbose=verbose))
        if "train_generalization_task" in conf.keys():
            stages.append(TrainerGeneralizationTask(conf, "train_generalization_task", verbose=verbose))
        if "test" in conf.keys():
            stages.append(Tester(conf, "test", verbose=verbose))
        if "run" in conf.keys():
            stages.append(Runner(conf, "run", verbose=verbose))
        
        # generate tables for each category
        CAT = ['Action', 'Affective', 'Art', 'BlackWhite', 'Cartoon', 'Fractal', 'Indoor', 'Inverted', 'Jumbled', 'LineDrawing', 'LowResolution', 'Noisy', 'Object', 'OutdoorManMade', 'OutdoorNatural', 'Pattern', 'Random', 'Satelite', 'Sketch', 'Social']
        UMSI = ['ads', 'infographics', 'mobile_uis', 'movie_posters', 'webpages']
        
        base_path = "/itet-stor/yanickz/net_scratch/data/dataset"

        for cat in CAT:
            base_path_sal = os.path.join(base_path, "saliency/CAT")
            base_path_img = os.path.join(base_path, "DATASET/CAT")
            stages.append(Table1(conf, os.path.join(base_path_sal, cat), os.path.join(base_path_img, cat), f"table_cat_{cat}", verbose=verbose))
        
        for umsi in UMSI:
            base_path_sal = os.path.join(base_path, "saliency/UMSI")
            base_path_img = os.path.join(base_path, "DATASET/UMSI")
            stages.append(Table1(conf, os.path.join(base_path_sal, umsi), os.path.join(base_path_img, umsi), f"table_umsi_{cat}", verbose=verbose))

        base_path_sal = os.path.join(base_path, "SALICON")
        base_path_img = os.path.join(base_path, "SALICON/Images/val")
        stages.append(Table1(conf, base_path_sal, base_path_img, f"table_salicon", verbose=verbose))


        mnet_fn = lambda : MNET(conf["model"]["mnet"])
        hnet_fn = lambda mnet : HNET(mnet.get_cw_param_shapes(), conf["model"]["hnet"])

        pipeline = Pipeline(
            input = HyperModel(mnet_fn, hnet_fn, conf["all_tasks"], conf["model"]["use_eval_mode"]).build(),
            work_dir_path=run_dir,
            stages = stages
        )
        
        pipeline.execute(exclude=conf["skip"])

    except ValueError as e:
        print(str(e))
        exit(64)

    run.finish()

########################################
# Generalization - Gridsearch
@cli.command()
@click.option("--skip", help="Comma-separated list of experiment stages that should be skipped {train, test, run}")
@click.option("-n", "--name", help="The name of the experiment.")
@click.option("-c", "--conf_file", help="The path to the configuration file.")
@click.option("-p", "--param_grid_file", help="The path to the gridsearch params file.")

@click.option("-i", "--input_images", help="The images used for experimenting. Should contain three folders inside: train, val and run.")
@click.option("-s", "--input_saliencies", help="Specify the directory to the saliency images. Should contain a separated folder for each model/task name.")

@click.option("--wdb", is_flag=True, help="Do you want to report to wandb?")

@click.option("--description", help="A description of what makes this run special")
def gridsearch(skip, name, conf_file, param_grid_file, input_images, input_saliencies, wdb, description):
    def set_value_if_exists(dict : dict, key, value):
        if key in dict.keys():
            dict[key] = value

    # load param_grid
    with open(param_grid_file) as f:
        param_grid = json.load(f)
    
    # load config file
    with open(conf_file) as f:
        conf : dict = json.load(f)

    # overwrite params given as args
    if skip:
        conf["skip"] = skip.split(",")
    if name:
        conf["name"] = name
    if description:
        conf["description"] = description
    if input_images:
        # replace all occurrences of input_images with the updated path
        for stage_conf in [v for v in conf.values() if isinstance(v, dict)]:
            set_value_if_exists(stage_conf, "input_images_train", os.path.join(input_images, "train"))
            set_value_if_exists(stage_conf, "input_images_val", os.path.join(input_images, "val"))
            set_value_if_exists(stage_conf, "input_images_test", os.path.join(input_images, "val")) # TODO: change to /test once test saliency data available
            set_value_if_exists(stage_conf, "input_images_run", os.path.join(input_images, "run"))

    if input_saliencies:
        # replace all occurrences of input_saliencies with the updated path
        for stage_conf in [v for v in conf.values() if isinstance(v, dict)]:
            set_value_if_exists(stage_conf, "input_saliencies", input_saliencies)
    
    os.environ["WANDB_MODE"] = "online" if wdb else "offline"
    os.environ["WANDB_IGNORE_GLOBS"] = "*.ignore_wandb"

    base_name = conf["name"]

    def _gridsearch(conf, param_grid, indices=[]):
        param_grid = param_grid.copy()
        if len(param_grid) == 0:
            print("#############################")
            print(f"NOW RUNNING {indices}")
            print("#############################")

            run_with_conf(conf, group=base_name)
        else:
            path,values = param_grid.pop(0)
            paths = path.split("/")
            c = conf.copy()
            bn = c["name"]
            bd = c["description"] 
            d = c
            for p in paths[0:-1]:
                d = d[p]
            
            for i,v in enumerate(values):
                c["name"] = f"{bn} - {v}"
                c["description"] = f"{bd}\n{path} = {v}"
                d[paths[-1]] = v
                _gridsearch(c, param_grid, indices+[i])
        
    _gridsearch(conf, param_grid)

if __name__ == "__main__":
    cli()