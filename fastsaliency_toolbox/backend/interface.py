#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast-Saliency Toolbox: Pseudo-models for fast saliency research. This file offers a simple interface if one wishes to use the installed toolbox.
"""

import os
import torch
import numpy as np

from .parameters import ParameterMap
from .image_processing import process
from .metrics import NSS, CC, SIM

HERE_PATH = os.path.dirname(os.path.realpath(__file__))
PARENT_PATH = os.path.abspath(os.path.join(HERE_PATH, os.pardir))

class Interface:
    def __init__(self, pretrained_models_path : str = './models/', gpu : int = -1):
        self._gpu = 'cuda:' + str(gpu)
        from .pseudomodels import ModelManager
        from .config import Config
        print(HERE_PATH)
        print(PARENT_PATH)
        config_path = os.path.join(PARENT_PATH, 'config.json') # TO BE CHECKED
        self._c = Config(config_path)
        self._model_manager = ModelManager(pretrained_models_path, verbose=True, pretrained=True, gpu=self._gpu)
        selected_models = self._model_manager.get_matchings(self._c.experiment_parameter_map.get_val('models'))

        for selected_model in selected_models:
            if torch.cuda.is_available() and self._gpu != 'cuda:-1':
                print("Trying to move model " + selected_model.name + " to cuda!")
                self._model_manager.cuda(selected_model.name)
                self.memory_check("Position 1")
        print(self._model_manager._model_map)

    def memory_check(self, position = None):
        print(position)
        for i in range(8):
            print(torch.cuda.memory_reserved(i))
            print(torch.cuda.memory_allocated(i))
            print("")

    def postprocess(self, sal_map : torch.Tensor, postprocessing_parameter_map : ParameterMap) -> torch.Tensor:
        my_map = postprocessing_parameter_map.clone()
        postprocessed = process(sal_map, my_map)
        return np.interp(postprocessed, (postprocessed.min(), postprocessed.max()), (0, 1))

    def run(self, model_name, img, postprocessing_parameter_map=None):
        print("Computing saliency")
        img = np.transpose(img, (2, 0, 1)) / 255.0
        img = torch.FloatTensor(img)
        img = img.unsqueeze(0)
        print(img.shape[2])
        #Find model
        model = self._model_manager.get_matching(model_name)
        if torch.cuda.is_available() and self._gpu != 'cuda:-1':
            img = img.cuda(torch.device(self._gpu))

        saliency_map = model.compute_saliency(img)
        saliency_map = saliency_map.cpu().detach().numpy()[0, 0]
        if postprocessing_parameter_map is not None: saliency_map = self.postprocess(saliency_map, postprocessing_parameter_map)
        from skimage.transform import resize
        saliency_map = resize(saliency_map, (img.shape[2], img.shape[3]))
        return saliency_map

    def test(self, model, original_saliency, saliency):
        """Computes the score on the given image."""
        nss = NSS(original_saliency, saliency)
        cc = CC(original_saliency, saliency)
        sim = SIM(original_saliency, saliency)
        return [model, nss, cc, sim]

    def evaluate_task(self, model, original_saliency, annotation):
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        print(original_saliency.shape)
        original_saliency = np.interp(original_saliency, (original_saliency.min(), original_saliency.max()), (0, 1))
        print(annotation.shape)
        original_saliency = original_saliency.flatten()
        annotation = annotation.flatten()
        original_saliency = (original_saliency > 0.5).astype(np.float32)
        return [model, precision_score(original_saliency, annotation), recall_score(original_saliency, annotation), f1_score(original_saliency, annotation), accuracy_score(original_saliency, annotation)]

if __name__ == '__main__':
    print("TEST")
