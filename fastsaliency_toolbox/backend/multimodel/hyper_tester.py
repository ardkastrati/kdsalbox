import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import wandb

from backend.utils import print_pretty_header
from backend.datasets import TestDataManager
from backend.metrics import NSS, CC, SIM
from backend.image_processing import process
from backend.parameters import ParameterMap


class HyperTester(object):
    def __init__(self, conf, hyper_model):
        self._hyper_model = hyper_model

        test_conf = conf["test"]
        preprocess_conf = conf["preprocess"]
        postprocess_conf = conf["postprocess"]

        # params
        batch_size = test_conf["batch_size"]
        tasks = test_conf["tasks"]
        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"
        self._recursive = test_conf["recursive"]
        self._per_image_statistics = test_conf["per_image_statistics"]
        self._model_path = test_conf["model_path"]
        self._logging_dir = test_conf["logging_dir"]
        self._verbose = test_conf["verbose"]

        # convert to pre-/postprocess params
        self._preprocess_parameter_map = ParameterMap()
        self._preprocess_parameter_map.set_from_dict(preprocess_conf)
        self._postprocess_parameter_map = ParameterMap()
        self._postprocess_parameter_map.set_from_dict(postprocess_conf)

        # data loading
        input_saliencies = test_conf["input_saliencies"]
        test_img_path = test_conf["input_images_test"]
        sal_folders = [os.path.join(input_saliencies, task) for task in tasks] # path to saliency folder for all models

        test_datasets = [TestDataManager(test_img_path, sal_path, self._verbose, self._preprocess_parameter_map) for sal_path in sal_folders]

        self._dataloaders = {task:DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4) for (task,ds) in zip(tasks,test_datasets)}

    def pretty_print(self, epoch, mode, loss, lr):
        print("--------------------------------------------->>>>>>")
        print(f"Epoch {epoch}: loss {mode} {loss}, lr {lr}", flush=True)
        print("--------------------------------------------->>>>>>")
    
    # runs the test for one task/model
    def test_one(self, model, task, dataloader):
        task_id = model.task_to_id(task)

        all_names, all_loss, all_NSS, all_CC, all_SIM = [], [], [], [], []
        my_loss = torch.nn.BCELoss()

        for i, (X, y, names) in enumerate(dataloader):
            X = X.to(self._device)
            y = y.to(self._device)

            pred = model(task_id, X)
            losses = my_loss(pred, y)
            detached_pred = pred.cpu().detach().numpy()
            detached_y = y.cpu().detach().numpy()

            # Doing the postprocessing steps needed for the metrics (We might want to do this also for Task Evaluation stuff?)
            y = process(y, self._postprocess_parameter_map)
            detached_pred = process(detached_pred, self._postprocess_parameter_map)
            NSSes = [NSS(map1, map2) for (map1, map2) in zip(detached_pred, detached_y)]
            CCes = [CC(map1, map2) for (map1, map2) in zip(detached_pred, detached_y)]
            SIMes = [SIM(map1, map2) for (map1, map2) in zip(detached_pred, detached_y)]

            with torch.no_grad():
                all_NSS.extend(NSSes)
                all_CC.extend(CCes)
                all_SIM.extend(SIMes)
                all_loss.append(losses.item())
                all_names.extend(names)

            if torch.cuda.is_available():
                # Remove batch from gpu
                del X
                del y
                torch.cuda.empty_cache()

            if i%100 == 0:
                print(f"Batch {i}: current accumulated loss {np.mean(all_loss)}", flush=True)

            # save per image stats
            if self._per_image_statistics:
                stats_file = os.path.join(os.path.relpath(self._logging_dir, wandb.run.dir), "image_statistics", task).replace("\\", "/")
                data = np.array([all_names, all_NSS, all_CC, all_SIM]).transpose().tolist()
                table = wandb.Table(data=data, columns=["Image", "NSS", "CC", "SIM"])
                wandb.run.log({stats_file:table})
            
        return np.mean(all_loss), np.nanmean(np.asarray(all_NSS)), np.nanmean(np.asarray(all_CC)), np.nanmean(np.asarray(all_SIM))

    def start_test(self):
        os.makedirs(self._logging_dir, exist_ok=True)

        model = self._hyper_model
        model.build()
        model.load(self._model_path, self._device)
        model.to(self._device)
        model.eval()

        # foreach task
        all_names, all_loss, all_NSS, all_CC, all_SIM = [], [], [], [], []
        for task,dataloader in self._dataloaders.items():
            loss, NSS, CC, SIM = self.test_one(model, task, dataloader)

            all_NSS.append(NSS)
            all_CC.append(CC)
            all_SIM.append(SIM)
            all_loss.append(loss)
            all_names.append(task)

        # total
        loss, NSS, CC, SIM = np.mean(all_loss), np.nanmean(np.asarray(all_NSS)), np.nanmean(np.asarray(all_CC)), np.nanmean(np.asarray(all_SIM))
        all_NSS.append(NSS)
        all_CC.append(CC)
        all_SIM.append(SIM)
        all_loss.append(loss)
        all_names.append("Average")

        # save test results
        stats_file = os.path.join(os.path.relpath(self._logging_dir, wandb.run.dir), "test_results").replace("\\", "/")
        data = np.array([all_names, all_loss, all_NSS, all_CC, all_SIM]).transpose().tolist()
        table = wandb.Table(data=data, columns=["Task", "Image", "NSS", "CC", "SIM"])
        wandb.run.log({stats_file:table})
        

    def execute(self):
        if self._verbose: print_pretty_header("TESTING " + self._model_path)
        if self._verbose: print("Tester started...")
        self.start_test()
        if self._verbose: print(f"Done with {self._model_path}!")

    def delete(self):
        del self._dataloaders