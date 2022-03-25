#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast-Saliency Toolbox: Pseudo-models for fast saliency research.

Available Commands:
    - Train (trains the models using some original images and the corresponding saliency images) 
        [python main.py train <ARGUMENTS>]
    - Test (evaluates how good the models do)
        [python main.py test <ARGUMENTS>]
    - Run (generates images using the trained models)
        [python main.py run <ARGUMENTS>]
    - Experiment (runs Train, Test and Run in sequence)
        [python main.py experiment <ARGUMENTS>]

    - python main.py version

    Check out the commands to see all the supported arguments.

"""

import os
import click


@click.group()
def cli():
    pass

########################################
# Run
@cli.command()
@click.option('-m', '--model', help="The model you want to run.")
@click.option('-p', '--student_path', help="The path where to load the weights (If not chosen, uses the default pretrained models).")
@click.option('--histogram_matching', help="The type of histogram matching. Possible values: none, biased, equalization.")
@click.option('--scale', help="The type of scaling to be done. Possible values: log-density, min-max, none, normalized")
@click.option('--blur', help="The type of blurring to be done. Possible values: custom, proportional")
@click.option('--center_prior', help="The Gaussian center prior. Possible values none, proportional_add, proportional_mult")
@click.option(
    '-r',
    '--recursive',
    is_flag=True,
    help="Recursively process input directory.")
@click.option(
    '-o',
    '--overwrite',
    is_flag=True,
    help="Overwrite existing images in output directory.")
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help="Verbose mode and debugging messages.")
@click.argument('in_dir', type=click.Path(exists=True), required=False)
@click.argument('out_dir', type=click.Path(), required=False)
def run(model, student_path, histogram_matching, scale, blur, center_prior, recursive, overwrite, verbose, in_dir, out_dir):
    """Runs model on images in a directory."""
    from backend.config import Config
    c = Config('config.json')

    if model:
        c.run_parameter_map.set("model", model)
    if student_path:
        c.run_parameter_map.set("student_path", student_path)
    if histogram_matching:
        c.postprocessing_parameter_map.set("histogram_matching", histogram_matching)
    if scale:
        c.postprocessing_parameter_map.set("scale_output", scale)
    if blur:
        c.postprocessing_parameter_map.set("do_smoothing", blur)
    if center_prior:
        c.postprocessing_parameter_map.set("center_prior", center_prior)
    if recursive:
        c.run_parameter_map.set("recursive", recursive)
    if overwrite:
        c.run_parameter_map.set("overwrite", overwrite)
    if verbose:
        c.run_parameter_map.set("verbose", verbose)
    if in_dir:
        c.run_parameter_map.set("input_images", in_dir)
    if out_dir:
        c.run_parameter_map.set("output_dir", out_dir)
    
    c.run_parameter_map.set("output_dir", os.path.join(c.run_parameter_map.get_val('output_dir'), c.run_parameter_map.get_val("model")))
    real_in_dir = os.path.realpath(c.run_parameter_map.get_val('input_images'))
    real_out_dir = os.path.realpath(c.run_parameter_map.get_val('output_dir'))

    if not os.path.isdir(real_in_dir):
        click.echo("ERROR: '{}' is not a directory!".format(real_in_dir))
        return 64

    if not os.path.exists(real_out_dir):
        os.makedirs(real_out_dir)
            
    elif not os.path.isdir(real_out_dir):
        click.echo("ERROR: '{}' is not a directory!".format(real_out_dir))
        return 64

    try:
        from backend.pseudomodels import ModelManager
        m = ModelManager('models/', verbose=c.run_parameter_map.get_val("verbose"), pretrained=True)
        if c.run_parameter_map.get_val("student_path") != "default":
            m.update_model(c.run_parameter_map.get_val("model"), c.run_parameter_map.get_val("student_path"))
        print(m._model_map)

        c.run_parameter_map.pretty_print()
        c.postprocessing_parameter_map.pretty_print()
        
        from backend.runner import Runner
        t = Runner(m, c.run_parameter_map, c.postprocessing_parameter_map)
        t.execute()

    except ValueError as e:
        print(str(e))
        exit(64)

########################################
# Train
@cli.command()
@click.option('-m', '--model', help="The model you want to train.")
@click.option('--histogram_matching', help="The type of histogram matching. Possible values: none, biased, equalization. This is applied before training which basically means that the trained model learns to output directly histogram matched saliency images.")
@click.option('--scale', help="The type of scaling to be done. Possible values: log-density, min-max, none, normalized. This is applied before training which basically means that the trained model learns to output directly scaled saliency images.")
@click.option('--blur', help="The type of blurring to be done. Possible values: custom, proportional. This is applied before training which basically means that the trained model learns to output directly blurred saliency images.")
@click.option('--center_prior', help="The Gaussian center prior. Possible values none, proportional_add, proportional_mult. This is applied before training which basically means that the trained model learns to output directly Gaussian center biased saliency images.")

@click.option('-l', '--logging_dir', help="The logs where should be stored.")
@click.option('-i', '--input_images', help="The images used for training. Should contain two folders inside: train and val.")
@click.option('-s', '--input_saliencies', help="The saliency images used for training.")
@click.option('-e', '--export_path', help="The relative path where to export.")
@click.option(
    '-r',
    '--recursive',
    is_flag=True,
    help="Recursively process input directory.")
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help="Verbose mode and debugging messages.")
@click.option('-b', '--batch_size', help="The size of the batches used for training.")
@click.option('-f', '--freeze_encoder_steps', help="Specify for how many epochs to freeze the encoder.")
def train(model, histogram_matching, scale, blur, center_prior, logging_dir, input_images, input_saliencies, export_path, recursive, verbose, batch_size, freeze_encoder_steps):
    """Trains model on images in a directory."""

    from backend.config import Config
    c = Config('config.json')

    if model:
        c.train_parameter_map.set("model", model)
    if histogram_matching:
        c.preprocessing_parameter_map.set("histogram_matching", histogram_matching)
    if scale:
        c.preprocessing_parameter_map.set("scale_output", scale)
    if blur:
        c.preprocessing_parameter_map.set("do_smoothing", blur)
    if center_prior:
        c.preprocessing_parameter_map.set("center_prior", center_prior)
    if logging_dir:
        c.train_parameter_map.set("logging_dir", logging_dir)
    if input_images:
        c.train_parameter_map.set("input_images", input_images)
    if input_saliencies:
        c.train_parameter_map.set("input_saliencies", input_saliencies)
    if export_path:
        c.train_parameter_map.set("export_path", export_path)
    if recursive:
        c.train_parameter_map.set("recursive", recursive)
    if verbose:
        c.train_parameter_map.set("verbose", verbose)
    if batch_size:
        c.train_parameter_map.set("batch_size", batch_size)
    if freeze_encoder_steps:
        c.train_parameter_map.set("freeze_encoder_steps", freeze_encoder_steps)
    
    real_in_dir = os.path.realpath(c.train_parameter_map.get_val('input_images'))
    real_sa_dir = os.path.realpath(c.train_parameter_map.get_val('input_saliencies'))

    if not (os.path.isdir(real_in_dir) or os.path.isdir(real_sa_dir)):
        click.echo("ERROR: '{}' or '{}' is not a directory!".format(real_in_dir, real_sa_dir))
        return 64

    try:
        from backend.pseudomodels import ModelManager
        m = ModelManager('models/', verbose=c.train_parameter_map.get_val("verbose"), pretrained=False)
        print(m._model_map)

        c.train_parameter_map.pretty_print()
        from backend.trainer import Trainer
        t = Trainer(m, c.train_parameter_map, c.preprocessing_parameter_map)
        t.execute()

    except ValueError as e:
        print(str(e))
        exit(64)

########################################
# Test
@cli.command()
@click.option('-m', '--model', help="The model you want to train.")
@click.option('-p', '--student_path', help="The path where to load the weights (If not chosen, uses the default pretrained models).")

@click.option('--train_histogram_matching', help="The type of histogram matching. Possible values: none, biased, equalization. This is done to the ground truth to compare the models.")
@click.option('--train_scale', help="The type of scaling to be done. Possible values: log-density, min-max, none, normalized. This is done to the ground truth to compare the models.")
@click.option('--train_blur', help="The type of blurring to be done. Possible values: custom, proportional. This is done to the ground truth to compare the models.")
@click.option('--train_center_prior', help="The Gaussian center prior. Possible values none, proportional_add, proportional_mult. This is done to the ground truth to compare the models.")

@click.option('--post_histogram_matching', help="The type of histogram matching. Possible values: none, biased, equalization. This is done to the predicted saliency maps to compare the models (can be useful as inverse histogram matching procedure).")
@click.option('--post_scale', help="The type of scaling to be done. Possible values: log-density, min-max, none, normalized. This is done to the predicted saliency maps to compare the models. Should be used only if the model is trained without scaling.")
@click.option('--post_blur', help="The type of blurring to be done. Possible values: custom, proportional. This is done to the predicted saliency maps to compare the models. Should be used only if the model is trained without blurring.")
@click.option('--post_center_prior', help="The Gaussian center prior. Possible values none, proportional_add, proportional_mult. This is done to the predicted saliency maps to compare the models. Should be used only if the model is trained without gaussian center prior.")

@click.option('-l', '--logging_dir', help="The logs where should be stored.")
@click.option('-i', '--input_images', help="The images used for testing.")
@click.option('-s', '--input_saliencies', help="The saliency images used for testing.")
@click.option(
    '-r',
    '--recursive',
    is_flag=True,
    help="Recursively process input directory.")
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help="Verbose mode and debugging messages.")
@click.option(
    '-d',
    '--detailed',
    is_flag=True,
    help="More detailed testing (Such as per image statistics).")
@click.option('-b', '--batch_size', help="The size of the batches used for testing.")
def test(model, student_path, train_histogram_matching, train_scale, train_blur, train_center_prior, post_histogram_matching, post_scale, post_blur, post_center_prior, logging_dir, input_images, input_saliencies, recursive, verbose, detailed, batch_size):
    """Trains model on images in a directory."""

    from backend.config import Config
    c = Config('config.json')

    if model:
        c.test_parameter_map.set("model", model)
    if student_path:
        c.test_parameter_map.set("student_path", student_path)
    if train_histogram_matching:
        c.preprocessing_parameter_map.set("histogram_matching", train_histogram_matching)
    if train_scale:
        c.preprocessing_parameter_map.set("scale_output", train_scale)
    if train_blur:
        c.preprocessing_parameter_map.set("do_smoothing", train_blur)
    if train_center_prior:
        c.preprocessing_parameter_map.set("center_prior", train_center_prior)
    if post_histogram_matching:
        c.postprocessing_parameter_map.set("histogram_matching", post_histogram_matching)
    if post_scale:
        c.postprocessing_parameter_map.set("scale_output", post_scale)
    if post_blur:
        c.postprocessing_parameter_map.set("do_smoothing", post_blur)
    if post_center_prior:
        c.postprocessing_parameter_map.set("center_prior", post_center_prior)
    if logging_dir:
        c.test_parameter_map.set("logging_dir", logging_dir)
    if input_images:
        c.test_parameter_map.set("input_images", input_images)
    if input_saliencies:
        c.test_parameter_map.set("input_saliencies", input_saliencies)
    if recursive:
        c.test_parameter_map.set("recursive", recursive)
    if verbose:
        c.test_parameter_map.set("verbose", verbose)
    if batch_size:
        c.test_parameter_map.set("batch_size", batch_size)
    if detailed:
        c.test_parameter_map.set("per_image_statistics", detailed)
    
    real_in_dir = os.path.realpath(c.train_parameter_map.get_val('input_images'))
    real_sa_dir = os.path.realpath(c.train_parameter_map.get_val('input_saliencies'))

    if not (os.path.isdir(real_in_dir) or os.path.isdir(real_sa_dir)):
        click.echo("ERROR: '{}' or '{}' is not a directory!".format(real_in_dir, real_sa_dir))
        return 64

    try:
        from backend.pseudomodels import ModelManager
        m = ModelManager('models/', verbose=c.test_parameter_map.get_val("verbose"), pretrained=True)
        print(m._model_map)
        c.test_parameter_map.pretty_print()
        c.preprocessing_parameter_map.pretty_print()
        c.postprocessing_parameter_map.pretty_print()

        from backend.tester import Tester
        t = Tester(m, c.test_parameter_map, c.preprocessing_parameter_map, c.postprocessing_parameter_map)
        t.execute()

    except ValueError as e:
        print(str(e))
        exit(64)


########################################
# Experiment
@cli.command()
@click.option('-n', '--name', help="The name of the experiment.")
@click.option('-m', '--models', help="The models you want to do the experiment (with comma separated).")

@click.option('--train_histogram_matching', help="The type of histogram matching. Possible values: none, biased, equalization. This is done to the ground truth to compare the models.")
@click.option('--train_scale', help="The type of scaling to be done. Possible values: log-density, min-max, none, normalized. This is done to the ground truth to compare the models.")
@click.option('--train_blur', help="The type of blurring to be done. Possible values: custom, proportional. This is done to the ground truth to compare the models.")
@click.option('--train_center_prior', help="The Gaussian center prior. Possible values none, proportional_add, proportional_mult. This is done to the ground truth to compare the models.")

@click.option('--post_histogram_matching', help="The type of histogram matching. Possible values: none, biased, equalization. This is done to the predicted saliency maps to compare the models (can be useful as inverse histogram matching procedure).")
@click.option('--post_scale', help="The type of scaling to be done. Possible values: log-density, min-max, none, normalized. This is done to the predicted saliency maps to compare the models. Should be used only if the model is trained without scaling.")
@click.option('--post_blur', help="The type of blurring to be done. Possible values: custom, proportional. This is done to the predicted saliency maps to compare the models. Should be used only if the model is trained without blurring.")
@click.option('--post_center_prior', help="The Gaussian center prior. Possible values none, proportional_add, proportional_mult. This is done to the predicted saliency maps to compare the models. Should be used only if the model is trained without gaussian center prior.")

@click.option('-l', '--logging_dir', help="The logs where should be stored.")
@click.option('-i', '--input_images', help="The images used for experimenting. Should contain two folders inside: train and val.")
@click.option('-s', '--input_saliencies', help="Specify the directory to the saliency images. Should contain a separated folder for each model name.")
@click.option('-g', '--gpu', help="Specify the gpu where to run.")

@click.option(
    '-r',
    '--recursive',
    is_flag=True,
    help="Recursively process input directory.")
@click.option(
    '-o',
    '--overwrite',
    is_flag=True,
    help="Overwrite the files.")
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help="Verbose mode and debugging messages.")
def experiment(name, models, train_histogram_matching, train_scale, train_blur, train_center_prior, post_histogram_matching, post_scale, post_blur, post_center_prior, logging_dir, input_images, input_saliencies, recursive, overwrite, verbose, gpu=0):
    """Experiment with models on images in a directory."""

    from backend.config import Config
    c = Config('config.json')

    if name:
        c.experiment_parameter_map.set("experiment_name", name)
    if models:
        c.experiment_parameter_map.set("models", models)
    if train_histogram_matching:
        c.preprocessing_parameter_map.set("histogram_matching", train_histogram_matching)
    if train_scale:
        c.preprocessing_parameter_map.set("scale_output", train_scale)
    if train_blur:
        c.preprocessing_parameter_map.set("do_smoothing", train_blur)
    if train_center_prior:
        c.preprocessing_parameter_map.set("center_prior", train_center_prior)
    if post_histogram_matching:
        c.postprocessing_parameter_map.set("histogram_matching", post_histogram_matching)
    if post_scale:
        c.postprocessing_parameter_map.set("scale_output", post_scale)
    if post_blur:
        c.postprocessing_parameter_map.set("do_smoothing", post_blur)
    if post_center_prior:
        c.postprocessing_parameter_map.set("center_prior", post_center_prior)
    if logging_dir:
        c.experiment_parameter_map.set("logging_dir", logging_dir)
    if input_images:
        c.experiment_parameter_map.set("input_images", input_images)
    if input_saliencies:
        c.experiment_parameter_map.set("input_saliencies", input_saliencies)
    if recursive:
        c.experiment_parameter_map.set("recursive", recursive)
    if overwrite:
        c.experiment_parameter_map.set("overwrite", overwrite)
    if verbose:
        c.experiment_parameter_map.set("verbose", verbose)
    
    real_in_dir = os.path.realpath(c.train_parameter_map.get_val('input_images'))
    real_sa_dir = os.path.realpath(c.train_parameter_map.get_val('input_saliencies'))

    if not (os.path.isdir(real_in_dir) or os.path.isdir(real_sa_dir)):
        click.echo("ERROR: '{}' or '{}' is not a directory!".format(real_in_dir, real_sa_dir))
        return 64

    try:
        from backend.experiment import Experiment
        t = Experiment(c, gpu)
        t.execute()

    except ValueError as e:
        print(str(e))
        exit(64)

@cli.command()
def version():
    """Displays version information."""
    click.echo("Fast Saliency Toolbox: Saliency (Psuedo-Model) Implementation ")


########################################
# Generalization
@cli.command()
@click.option('--histogram_matching', help="The type of histogram matching. Possible values: none, biased, equalization. This is applied before training which basically means that the trained model learns to output directly histogram matched saliency images.")
@click.option('--scale', help="The type of scaling to be done. Possible values: log-density, min-max, none, normalized. This is applied before training which basically means that the trained model learns to output directly scaled saliency images.")
@click.option('--blur', help="The type of blurring to be done. Possible values: custom, proportional. This is applied before training which basically means that the trained model learns to output directly blurred saliency images.")
@click.option('--center_prior', help="The Gaussian center prior. Possible values none, proportional_add, proportional_mult. This is applied before training which basically means that the trained model learns to output directly Gaussian center biased saliency images.")

@click.option('-l', '--logging_dir', help="The logs where should be stored.")
@click.option('-e', '--export_path', help="The relative path where to export.")
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help="Verbose mode and debugging messages.")
@click.option('-b', '--batch_size', help="The size of the batches used for training.")
@click.option('-f', '--freeze_encoder_steps', help="Specify for how many epochs to freeze the encoder.")
def generalization(histogram_matching, scale, blur, center_prior, logging_dir, export_path, verbose, batch_size, freeze_encoder_steps):
    from backend.config import Config
    c = Config('config.json')

    # preprocessing
    if histogram_matching:
        c.preprocessing_parameter_map.set("histogram_matching", histogram_matching)
    if scale:
        c.preprocessing_parameter_map.set("scale_output", scale)
    if blur:
        c.preprocessing_parameter_map.set("do_smoothing", blur)
    if center_prior:
        c.preprocessing_parameter_map.set("center_prior", center_prior)
    
    # training
    if logging_dir:
        c.train_parameter_map.set("logging_dir", logging_dir)
    if export_path:
        c.train_parameter_map.set("export_path", export_path)
    if verbose:
        c.train_parameter_map.set("verbose", verbose)
    if batch_size:
        c.train_parameter_map.set("batch_size", batch_size)
    if freeze_encoder_steps:
        c.train_parameter_map.set("freeze_encoder_steps", freeze_encoder_steps)

    # TODO: paths as params
    # setup paths to data folders
    train_folders_base_path = "/media/yanick/Yanick Zengaffinen Ext-Festpl/DATA/Images"
    train_folders_paths = [
        (os.path.join(train_folders_base_path, "Images/train"), os.path.join(train_folders_base_path, "AIM")),
        (os.path.join(train_folders_base_path, "Images/train"), os.path.join(train_folders_base_path, "IKN")),
        (os.path.join(train_folders_base_path, "Images/train"), os.path.join(train_folders_base_path, "GBVS")),
    ]

    validation_folders_base_path = "/media/yanick/Yanick Zengaffinen Ext-Festpl/DATA/Images"
    validation_folders_paths = [
        (os.path.join(validation_folders_base_path, "Images/val"), os.path.join(validation_folders_base_path, "AIM")),
        (os.path.join(validation_folders_base_path, "Images/val"), os.path.join(validation_folders_base_path, "IKN")),
        (os.path.join(validation_folders_base_path, "Images/val"), os.path.join(validation_folders_base_path, "GBVS")),
    ]    

    # Do you want to report to wandb?
    REPORT_WANDB = True
    os.environ['WANDB_MODE'] = 'online' if REPORT_WANDB else 'offline'

    conf = dict(
        gpu = 0,
        preprocess_parameter_map = c.preprocessing_parameter_map,
        train_parameter_map = c.train_parameter_map,
        train_folders_paths = train_folders_paths,
        val_folders_paths =  validation_folders_paths
    )

    try:
        from backend.generalization.embed.trainer import Trainer
        t = Trainer(conf)
        t.execute()

    except ValueError as e:
        print(str(e))
        exit(64)

if __name__ == '__main__':
    cli()