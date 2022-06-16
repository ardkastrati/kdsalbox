"""
Config
------

Configures how the training, testing etc behave.

Configuration for:
    - Preprocessing (applied to saliency images when loaded by DataLoader)
    - Postprocessing (applied to computed saliency images)
    - Train (trains the models using some original images and the corresponding saliency images)
    - Test (evaluates how good the models do)
    - Run (generates images using the trained models)
    - Experiment (runs Train, Test and Run in sequence)

Look at fastsaliency_toolbox/config.json for more information.

"""

import json
import copy
import os

from .parameters import ParameterMap

HERE_PATH = os.path.dirname(os.path.realpath(__file__))
PARENT_PATH = os.path.abspath(os.path.join(HERE_PATH, os.pardir))

############################################################
# Config
############################################################
class Config(object):
    def __init__(self, config_path : str = None):
        if config_path is None:
            config_path = os.path.join(PARENT_PATH, 'config.json') # TO BE CHECKED

        with open(config_path, 'rb') as fp:
            self._config = json.load(fp)
        
        self.preprocessing_parameter_map = ParameterMap()
        self.postprocessing_parameter_map = ParameterMap()
        self.run_parameter_map = ParameterMap()
        self.train_parameter_map = ParameterMap()
        self.experiment_parameter_map = ParameterMap()
        self.test_parameter_map = ParameterMap()
        
        self.preprocessing_parameter_map.set_from_dict(self._config['preprocessing_parameters'])
        self.postprocessing_parameter_map.set_from_dict(self._config['postprocessing_parameters'])
        self.run_parameter_map.set_from_dict(self._config['run_parameters'])
        self.train_parameter_map.set_from_dict(self._config['train_parameters'])
        self.test_parameter_map.set_from_dict(self._config['test_parameters'])
        self.experiment_parameter_map.set_from_dict(self._config['experiment_parameters'])

    def clone(self):
        return copy.deepcopy(self)

    def pretty_print(self):
        print("Preprocessing parameters:  -----------------------------------------------")
        self.preprocessing_parameter_map.pretty_print()
        print("Postprocessing parameters:  ----------------------------------------------")
        self.postprocessing_parameter_map.pretty_print()
        print("Experiment parameters:  --------------------------------------------------")
        self.experiment_parameter_map.pretty_print()
        print("Run parameters:  ---------------------------------------------------------")
        self.run_parameter_map.pretty_print()
        print("Train parameters:  -------------------------------------------------------")
        self.train_parameter_map.pretty_print()
        print("Test parameters:  --------------------------------------------------------")
        self.test_parameter_map.pretty_print()

    def update(self, model=None, do_smoothing=None, smooth_size=None, smooth_std=None, smooth_prop=None, scale_output=None, scale_min=None, scale_max=None, 
               center_prior=None, center_prior_prop=None, center_prior_weight=None, center_prior_scale_first=None):
        
        if model is not None:
            self.run_parameter_map.set(name="model", value=model)

        if do_smoothing is not None:
            self.postprocessing_parameter_map.set(name="do_smoothing", value=do_smoothing)

        if smooth_size is not None:
            self.postprocessing_parameter_map.set(name="smooth_size", value=smooth_size)

        if smooth_std is not None:
            self.postprocessing_parameter_map.set(name="smooth_std", value=smooth_std)

        if smooth_prop is not None:
            self.postprocessing_parameter_map.set(name="smooth_prop", value=smooth_prop)

        if scale_output is not None:
            self.postprocessing_parameter_map.set(name="scale_output", value=scale_output)

        if scale_min is not None:
            self.postprocessing_parameter_map.set(name="scale_min", value=scale_min)

        if scale_max is not None:
            self.postprocessing_parameter_map.set(name="scale_max", value=scale_max)

        if center_prior is not None:
            self.postprocessing_parameter_map.set(name="center_prior", value=center_prior)

        if center_prior_prop is not None:
            self.postprocessing_parameter_map.set(name="center_prior_prop", value=center_prior_prop)

        if center_prior_weight is not None:
            self.postprocessing_parameter_map.set(name="center_prior_weight", value=center_prior_weight)

        if center_prior_weight is not None:
            self.postprocessing_parameter_map.set(name="center_prior_weight", value=center_prior_weight)

        if center_prior_scale_first is not None:
            self.postprocessing_parameter_map.set(name="center_prior_scale_first", value=center_prior_scale_first)


if __name__ == '__main__':
    c = Config('../config.json')
    c.pretty_print()
    #WORKS!