"""
Experiment
----------

Executes Trainer, Tester and Runner in sequence.

"""

import os
import torch
import time

from .runner import Runner
from .trainer import Trainer
from .tester import Tester

HERE_PATH = os.path.dirname(os.path.realpath(__file__))

class Experiment(object):
    def __init__(self, c, gpu):
        self._gpu = 'cuda:' + str(gpu)
        from .pseudomodels import ModelManager
        self._model_manager = ModelManager('models/', verbose=c.experiment_parameter_map.get_val("verbose"), pretrained=False, gpu=self._gpu)
        print(self._model_manager._model_map)
        self._experiment_config = c.clone()

        c.experiment_parameter_map.pretty_print()

        self._executions = []

        if self._experiment_config.experiment_parameter_map.get_val("experiment_name") == "NA":
            self._experiment_name = str(int(time.time()))
        else:
            self._experiment_name = self._experiment_config.experiment_parameter_map.get_val("experiment_name")

        if not self._experiment_config.experiment_parameter_map.exists_val("experiment_description"):
            self._experiment_description = "No description given."
        else:
            self._experiment_description = self._experiment_config.experiment_parameter_map.get_val("experiment_description")

        self._experiment_dir = os.path.abspath(os.path.join(self._experiment_config.experiment_parameter_map.get_val("logging_dir"), self._experiment_name))
        if not os.path.exists(self._experiment_dir):
            os.makedirs(self._experiment_dir)
        self.set_experiment()

    def set_experiment(self):
        my_experiment_map = self._experiment_config.experiment_parameter_map
        selected_models = self._model_manager.get_matchings(my_experiment_map.get_val('models'))

        for selected_model in selected_models:
            experiment_logging_dir = os.path.join(self._experiment_dir, selected_model.name)
            if not os.path.exists(experiment_logging_dir):
                os.makedirs(experiment_logging_dir)

            if torch.cuda.is_available(): 
                self._model_manager.cuda(selected_model.name)
                self.memory_check("Position 1")

            my_train_map = self._experiment_config.train_parameter_map.clone()
            my_train_map.set('model', selected_model.name)
            my_train_map.set('logging_dir', os.path.join(experiment_logging_dir, "train_logs"))
            my_train_map.set('input_images', my_experiment_map.get_val("input_images"))
            my_train_map.set('input_saliencies', os.path.join(my_experiment_map.get_val("input_saliencies"), selected_model.name))
            my_train_map.set('recursive', my_experiment_map.get_val('recursive'))
            my_train_map.set('verbose', my_experiment_map.get_val('verbose'))
            my_train_preprocess_map = self._experiment_config.preprocessing_parameter_map.clone()
            execution_train = Trainer(model_manager=self._model_manager, train_parameter_map=my_train_map, preprocess_parameter_map=my_train_preprocess_map, gpu=self._gpu)
            execution_train.execute()
            
            if torch.cuda.is_available():
                del execution_train
                torch.cuda.empty_cache()
                self.memory_check("Position 2")

            my_test_map = self._experiment_config.test_parameter_map.clone()
            my_test_map.set('model', selected_model.name)
            my_test_map.set('logging_dir', os.path.join(experiment_logging_dir, "validation_logs"))
            my_test_map.set('input_images', os.path.join(my_experiment_map.get_val("input_images"), "val"))
            my_test_map.set('input_saliencies', os.path.join(my_experiment_map.get_val("input_saliencies"), selected_model.name))
            my_test_map.set('recursive', my_experiment_map.get_val('recursive'))
            my_test_map.set('verbose', my_experiment_map.get_val('verbose'))
            my_test_preprocess_map = self._experiment_config.preprocessing_parameter_map.clone()
            my_test_postprocess_map = self._experiment_config.postprocessing_parameter_map.clone()
            execution_test = Tester(model_manager=self._model_manager, test_parameter_map=my_test_map, preprocessing_parameter_map=my_test_preprocess_map, postprocessing_parameter_map=my_test_postprocess_map, gpu=self._gpu)
            execution_test.execute()

            if torch.cuda.is_available(): 
                del execution_test
                torch.cuda.empty_cache()
                self.memory_check("Position 3")

            my_run_map = self._experiment_config.run_parameter_map.clone()
            my_run_map.set('model', selected_model.name)
            my_run_map.set('input_images', os.path.join(my_experiment_map.get_val("input_images"), "plot_test/"))
            my_run_map.set('output_dir', os.path.join(self._experiment_dir, selected_model.name, "plot_runs/"))
            my_run_map.set('recursive', my_experiment_map.get_val('recursive'))
            my_run_map.set('verbose', my_experiment_map.get_val('verbose'))
            my_run_postprocess_map = self._experiment_config.postprocessing_parameter_map.clone()

            execution_run = Runner(model_manager=self._model_manager, run_parameter_map=my_run_map, postprocessing_parameter_map=my_run_postprocess_map, gpu=self._gpu)
            execution_run.execute()

            if torch.cuda.is_available(): 
                execution_run.delete()
                torch.cuda.empty_cache()
                self.memory_check("Position 5")
                self._model_manager.delete(selected_model.name)
                torch.cuda.empty_cache()
                self.memory_check("Position 6")

    def execute(self):
        for execution in self._executions:
            print("EXECUTION IS STARTING")
            execution.execute()

    def memory_check(self, position=None):
        print(position)
        for i in range(8):
            print(torch.cuda.memory_reserved(i))
            print(torch.cuda.memory_allocated(i))
            print("")

    

if __name__ == '__main__':
    from pseudomodels import ModelManager
    m = ModelManager('../models/', verbose=True, pretrained=False)
    print(m._model_map)

    from config import Config
    c = Config('../config.json')
    c.train_parameter_map.pretty_print()
    c.test_parameter_map.pretty_print()
    c.run_parameter_map.pretty_print()

    t = Experiment(m, c)
    t.execute()


