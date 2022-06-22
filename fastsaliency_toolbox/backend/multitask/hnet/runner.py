"""
Runner
------

For each task in conf["run"]["tasks"], 
this runner will go over all images in conf["run"]["input_images_run"]
and compute & store the saliency map for each of them.

"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import wandb

from backend.parameters import ParameterMap
from backend.datasets import RunDataManager
from backend.image_processing import process
from backend.multitask.pipeline.pipeline import AStage
from backend.multitask.hnet.hyper_model import HyperModel

class Runner(AStage):
    def __init__(self, conf, name, verbose):
        super().__init__(name=name, verbose=verbose)
        self._model : HyperModel = None

        run_conf = conf[name]
        self._tasks = conf["tasks"]
        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"
        self._input_dir = run_conf["input_images_run"]
        self._overwrite = run_conf["overwrite"]

        # convert params
        self._postprocess_parameter_map = ParameterMap().set_from_dict(conf["postprocess"])

    # run for one task
    def _run_one_task(self, task):
        """ Runs the given model for one specific task """

        model = self._model
        model.to(self._device)
        model.eval()
        
        task_id = model.task_to_id(task)

        if self._verbose: print(f"Running the model on task {task}...")

        # load data
        output_dir = os.path.join(self._logging_dir, task)
        os.makedirs(output_dir, exist_ok=True)
        dataloader = DataLoader(RunDataManager(self._input_dir, output_dir, self._verbose, recursive=False), batch_size=1)

        # for each image in the specified folder
        for img_number, (image, input_path, output_path) in enumerate(dataloader):
            printable_input_path = os.path.relpath(input_path[0], self._input_dir) # [0] since we use dataloader which returns batch of size 1
            out_path = output_path[0]

            # If we aren't overwriting and the file exists, skip it.
            if not self._overwrite and os.path.isfile(out_path):
                if self._verbose: print(f"SKIP (already exists) image [{img_number + 1}/{len(dataloader)}]: {printable_input_path}")
                continue

            
            if self._verbose: print(f"Running image [{img_number + 1}/{len(dataloader)}]: {printable_input_path}")

            # compute the saliency map
            image = image.to(self._device)
            saliency_map = model.compute_saliency(image, task_id)
            post_processed_image = np.clip((process(saliency_map.cpu().detach().numpy()[0, 0], self._postprocess_parameter_map)*255).astype(np.uint8), 0, 255)

            # store the saliency map
            img_path = os.path.relpath(out_path, wandb.run.dir).replace("\\", "/").replace(".jpg", "")
            img = wandb.Image(post_processed_image)
            wandb.log({img_path:img})
            
            # remove batch from gpu
            if torch.cuda.is_available():
                del image
                del input_path
                del output_path
                torch.cuda.empty_cache()

        if self._verbose: print(f"Done with {task}!")

    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)
        assert input is not None and isinstance(input, HyperModel), "Runner expects a HyperModel to be passed as an input."
        assert work_dir_path is not None, "Working directory path cannot be None."

        self._model = input
        self._logging_dir = work_dir_path

        # prepare logging
        os.makedirs(self._logging_dir, exist_ok=True)

    def execute(self):
        super().execute()
        
        for task in self._tasks:
            self._run_one_task(task)

        return self._model

    def cleanup(self):
        super().cleanup()