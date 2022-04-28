import torch
from torch.utils.data import DataLoader
import numpy as np

import os
import wandb

from backend.parameters import ParameterMap
from backend.datasets import RunDataManager
from backend.image_processing import process
from backend.utils import print_pretty_header

class HyperRunner(object):
    def __init__(self, conf, hyper_model):
        self._hyper_model = hyper_model

        run_conf = conf["run"]
        postprocess_conf = conf["postprocess"]

        # params
        self._tasks = run_conf["tasks"]
        self._model_path = run_conf["model_path"]
        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"
        self._input_dir = run_conf["input_images_run"]
        self._logging_dir = run_conf["logging_dir"]
        self._verbose = run_conf["verbose"]
        self._recursive = run_conf["recursive"]
        self._overwrite = run_conf["overwrite"]

        # convert params
        self._postprocess_parameter_map = ParameterMap()
        self._postprocess_parameter_map.set_from_dict(postprocess_conf)

    # run for one task
    def run_one(self, model, task):
        task_id = model.task_to_id(task)

        if self._verbose: print(f"Running the model on task {task}...")

        # load data
        output_dir = os.path.join(self._logging_dir, task)
        os.makedirs(output_dir, exist_ok=True)
        dataloader = DataLoader(RunDataManager(self._input_dir, output_dir, self._verbose, self._recursive), batch_size=1)

        for img_number, (image, input_path, output_path) in enumerate(dataloader):
            printable_input_path = os.path.relpath(input_path[0], self._input_dir)
            out_path = output_path[0]

            # If we aren't overwriting and the file exists, skip it.
            if not self._overwrite and os.path.isfile(out_path):
                print(f"SKIP (already exists) image [{img_number + 1}/{len(dataloader)}]: {printable_input_path}")
                continue

            
            print(f"Running image [{img_number + 1}/{len(dataloader)}]: {printable_input_path}")

            image = image.to(self._device)
            saliency_map = self.compute_saliency(model, image, task_id)
            post_processed_image = np.clip((process(saliency_map.cpu().detach().numpy()[0, 0], self._postprocess_parameter_map)*255).astype(np.uint8), 0, 255)

            img_path = os.path.relpath(out_path, wandb.run.dir).replace("\\", "/").replace(".jpg", "")
            img = wandb.Image(post_processed_image)
            wandb.log({img_path:img})
            
            # Remove batch from gpu
            if torch.cuda.is_available():
                del image
                del input_path
                del output_path
                torch.cuda.empty_cache()

        print(f"Done with {task}!")

    def start_run(self):
        os.makedirs(self._logging_dir, exist_ok=True)

        model = self._hyper_model
        model.build()
        model.load(self._model_path, self._device)
        model.to(self._device)

        for task in self._tasks:
            self.run_one(model, task)

    def execute(self):
        if self._verbose: print_pretty_header("RUNNING " + self._model_path)
        if self._verbose: print("Runner started...")
        self.start_run()
        if self._verbose: print(f"Done with {self._model_path}!")
    
    # runs and returns the models on an image for a given task
    def compute_saliency(self, model, img, task_id):
        model.eval()
        sal = model(task_id, img)
        return sal