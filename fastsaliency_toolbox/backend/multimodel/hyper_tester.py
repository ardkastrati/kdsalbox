import os
import numpy as np
import torch
from torch.nn import KLDivLoss
from torch.utils.data import DataLoader

import wandb

from backend.utils import print_pretty_header
from backend.datasets import TestDataManager
from backend.metrics import NSS, CC, SIM, KL as KL_c
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
        imgs_per_task_test = test_conf["imgs_per_task_test"]
        tasks = test_conf["tasks"]

        self._device = f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu"
        self._recursive = test_conf["recursive"]
        self._per_image_statistics = test_conf["per_image_statistics"]
        self._batches_per_task_test = imgs_per_task_test // batch_size
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

        min_ds_len = min([len(ds) for ds in test_datasets])
        self._data_offset = 0 if min_ds_len - imgs_per_task_test == 0 else np.random.randint(0, (min_ds_len - imgs_per_task_test) // batch_size)
        self._dataloaders = {task:DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4) for (task,ds) in zip(tasks,test_datasets)}

        self.KL = KLDivLoss(reduce="batchmean")

        # sanity checks
        assert imgs_per_task_test <= min_ds_len

    def pretty_print(self, epoch, mode, loss, lr):
        print("--------------------------------------------->>>>>>")
        print(f"Epoch {epoch}: loss {mode} {loss}, lr {lr}", flush=True)
        print("--------------------------------------------->>>>>>")
    
    # runs the test for one task/model
    def test_one(self, model, task, dataloader):
        task_id = model.task_to_id(task)

        if self._verbose: print(f"Testing {task}...")

        all_names, all_loss, all_NSS, all_CC, all_SIM, all_KL, all_KL_c = [], [], [], [], [], [], []
        my_loss = torch.nn.BCELoss()

        data_iter = iter(dataloader)
        for i in range(self._data_offset): next(data_iter) # skip first few images

        for i in range(self._batches_per_task_test):
            if i > 0: break
            (X, y, names) = next(data_iter)
            X = X.to(self._device)
            y = y.to(self._device)

            pred = model(task_id, X)
            losses = my_loss(pred, y)
            detached_pred_t = pred.cpu().detach()
            detached_y_t = y.cpu().detach()
            detached_pred = detached_pred_t.numpy()
            detached_y = detached_y_t.numpy()

            # Doing the postprocessing steps needed for the metrics (We might want to do this also for Task Evaluation stuff?)
            y = process(y, self._postprocess_parameter_map)
            detached_pred = process(detached_pred, self._postprocess_parameter_map)
            NSSes = [NSS(map1, map2) for (map1, map2) in zip(detached_pred, detached_y)]
            CCes = [CC(map1, map2) for (map1, map2) in zip(detached_pred, detached_y)]
            SIMes = [SIM(map1, map2) for (map1, map2) in zip(detached_pred, detached_y)]
            KLes = [self.KL(map1, map2) for (map1, map2) in zip(detached_pred_t, detached_y_t)]
            KL_ces = [KL_c(map1, map2) for (map1, map2) in zip(detached_pred, detached_y)]

            with torch.no_grad():
                all_names.extend(names)
                all_loss.append(losses.item())
                all_NSS.extend(NSSes)
                all_CC.extend(CCes)
                all_SIM.extend(SIMes)
                all_KL.extend(KLes)
                all_KL_c.extend(KL_ces)

            if torch.cuda.is_available():
                # Remove batch from gpu
                del X
                del y
                torch.cuda.empty_cache()

            if i%100 == 0:
                print(f"Batch {i}/{self._batches_per_task_test}: current accumulated loss {np.mean(all_loss)}", flush=True)

        # save per image stats
        if self._per_image_statistics:
            stats_file = os.path.join(os.path.relpath(self._logging_dir, wandb.run.dir), "image_statistics", task).replace("\\", "/")
            data = np.array([all_names, all_NSS, all_CC, all_SIM, all_KL, all_KL_c]).transpose().tolist()
            table = wandb.Table(data=data, columns=["Image", "NSS", "CC", "SIM", "KL", "KL custom"])
            wandb.run.log({stats_file:table})
            
        return np.mean(all_loss), np.nanmean(np.asarray(all_NSS)), np.nanmean(np.asarray(all_CC)), np.nanmean(np.asarray(all_SIM)), np.nanmean(np.asarray(all_KL)), np.nanmean(np.asarray(all_KL_c))

    def start_test(self):
        os.makedirs(self._logging_dir, exist_ok=True)

        model = self._hyper_model
        model.build()
        model.load(self._model_path, self._device)
        model.to(self._device)
        model.eval()

        # foreach task
        all_names, all_loss, all_NSS, all_CC, all_SIM, all_KL, all_KL_c = [], [], [], [], [], [], []
        for task,dataloader in self._dataloaders.items():
            loss, NSS, CC, SIM, KL, KL_c = self.test_one(model, task, dataloader)

            all_names.append(task)
            all_loss.append(loss)
            all_NSS.append(NSS)
            all_CC.append(CC)
            all_SIM.append(SIM)
            all_KL.append(KL)
            all_KL_c.append(KL_c)

        # total
        loss, NSS, CC, SIM, KL, KL_c = np.mean(all_loss), np.nanmean(np.asarray(all_NSS)), np.nanmean(np.asarray(all_CC)), np.nanmean(np.asarray(all_SIM)), np.nanmean(np.asarray(all_KL)), np.nanmean(np.asarray(all_KL_c))
        all_names.append("Average")
        all_loss.append(loss)
        all_NSS.append(NSS)
        all_CC.append(CC)
        all_SIM.append(SIM)
        all_KL.append(KL)
        all_KL_c.append(KL_c)

        # save test results
        stats_file = os.path.join(os.path.relpath(self._logging_dir, wandb.run.dir), "test_results").replace("\\", "/")
        data = np.array([all_names, all_loss, all_NSS, all_CC, all_SIM, all_KL, all_KL_c]).transpose().tolist()
        table = wandb.Table(data=data, columns=["Task", "Loss", "NSS", "CC", "SIM", "KL", "KL custom"])
        wandb.run.log({stats_file:table})
        

    def execute(self):
        if self._verbose: print_pretty_header("TESTING " + self._model_path)
        if self._verbose: print("Tester started...")
        self.start_test()
        if self._verbose: print(f"Done with {self._model_path}!")

    def delete(self):
        del self._dataloaders