"""
PreTrainer
----------

Trains a hypernetwork to output some specific weights (e.g. to learn pretrained weights before the actual training begins)

"""

import os
from typing import Dict
import torch
from torch.utils.data import DataLoader

import backend.student as stud
from backend.multitask.hnet.datasets import WeightDataset
from backend.multitask.hnet.trainer import ATrainer
from backend.multitask.hnet.train_api.data import DataProvider
from backend.multitask.hnet.train_api.training import TrainStep
from backend.multitask.hnet.train_impl.data import BatchProvider
from backend.multitask.hnet.train_impl.training import WeightsTrainStep


class PreTrainerWeights(ATrainer):
    def __init__(self, conf, name, verbose):
        super().__init__(conf, name=name, verbose=verbose)

        train_conf = conf[self._name]
        self._target_model_weights = train_conf["target_model_weights"]

    def get_data_providers(self) -> Dict[str, DataProvider]:
        target_model_weights = self._target_model_weights
        model_paths = [(self._model.task_to_id(task), os.path.join(target_model_weights, task.upper(), "exported", f"{task.lower()}.pth")) for task in self._tasks] # paths to all trained models
        train_dataset = WeightDataset(model_paths, self.map_old_to_new_weights)
        train_dataloader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=False, num_workers=4) # TODO: custom validation dataset

        dataproviders = {
            "train": BatchProvider(train_dataloader),
            "val": BatchProvider(val_dataloader) 
        }

        return dataproviders

    # run the entire training
    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)

        # check if the extpected output format of the HNET matches the labels/weights loaded from the original models
        new_shapes = self._model.mnet.get_cw_param_shapes()
        old_stud = stud.Student()
        old_shapes = self.map_old_to_new_weights({n:p.size() for n,p in old_stud.named_parameters()}, old_stud, verbose=self._verbose) 
        assert len(new_shapes) == len(old_shapes), f"HNET output generates {len(new_shapes)} weight tensors whereas we only loaded {len(old_shapes)} into the label from the original models!"
        for new,old in zip(new_shapes, old_shapes):
            assert new == old, "Mismatch between HNET output format and loaded model weight format. Make sure the order of parameters is the same!"

    def get_stepper(self) -> TrainStep:
        return WeightsTrainStep()

    # evaluate how different the layers of the models are
    def _evaluate_model_differences(self, target_model_weights):
        model_p = [(task, os.path.join(target_model_weights, task.upper(), "exported", f"{task.lower()}.pth")) for task in self._tasks] # paths to all trained models
        model_weights = []
        for t,p in model_p:
            model = stud.Student()
            state_dict = torch.load(p, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict['student_model'])
            
            model_weights.append((t,{n:p.data.flatten() for n,p in model.named_parameters()}))
            

        for i in range(len(model_weights)):
            for j in range(i+1, len(model_weights)):
                t1,ws1 = model_weights[i]
                t2,ws2 = model_weights[j]

                # compare the two different models
                print(f"{t1} vs {t2}")

                total_dist = 0.0
                total_dist_enc = 0.0
                total_enc = 0
                total_dist_dec = 0.0
                total_dec = 0

                all_dist = []
                for n in ws1.keys():
                    w1 = ws1[n]
                    w2 = ws2[n]
                    dist = torch.sum(torch.abs(w1 - w2)) / len(w1)
                    total_dist += dist

                    if "encoder" in n:
                        total_dist_enc += dist
                        total_enc += 1
                    elif "decoder" in n:
                        total_dist_dec += dist
                        total_dec += 1
                    
                    all_dist.append((n, dist))
                

                print(f"Average {total_dist / len(ws1.keys())}")
                print(f"Average Encoder {total_dist_enc / total_enc}")
                print(f"Average Decoder {total_dist_dec / total_dec}")

                # all_dist = sorted(all_dist, key=(lambda t : t[1]), reverse=True)
                # for n,d in all_dist:
                #     if d < 0.5: continue
                #     print(f"\t{n}: {d:.3}")

    def map_old_to_new_weights(self, named_weights, model, verbose=False):
        decoder_selection = [
            "conv7_3.weight",   "conv7_3.bias",     "bn7_3.weight",     "bn7_3.bias",
            "conv8_1.weight",   "conv8_1.bias",     "bn8_1.weight",     "bn8_1.bias",
            "conv8_2.weight",   "conv8_2.bias",     "bn8_2.weight",     "bn8_2.bias", 
            "conv9_1.weight",   "conv9_1.bias",     "bn9_1.weight",     "bn9_1.bias",
            "conv9_2.weight",   "conv9_2.bias",     "bn9_2.weight",     "bn9_2.bias",
            "conv10_1.weight",  "conv10_1.bias",    "bn10_1.weight",    "bn10_1.bias",
            "conv10_2.weight",  "conv10_2.bias",    "bn10_2.weight",    "bn10_2.bias",
            "output.weight",    "output.bias"
        ]
        
        new_weights = []
        new_encoder_param_cnt = len(list(self._model.mnet.get_layer("encoder").parameters()))
        # select the first few params, assuming that 
        encoder_selection = [n for n,p in model.encoder.named_parameters()][new_encoder_param_cnt:] 

        if verbose: print("selecting encoder.")
        for s in encoder_selection:
            if verbose: print(f"\t{s}")
            new_weights.append(named_weights[f"encoder.{s}"])    

        if verbose: print("selecting decoder.")
        for s in decoder_selection:
            if verbose: print(f"\t{s}")
            new_weights.append(named_weights[f"decoder.{s}"])

        return new_weights
