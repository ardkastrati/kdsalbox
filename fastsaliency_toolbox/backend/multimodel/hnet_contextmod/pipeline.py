from backend.multimodel.hyper_experiment import HyperExperiment
from backend.multimodel.hyper_trainer import HyperTrainer
from backend.multimodel.hyper_tester import HyperTester
from backend.multimodel.hyper_runner import HyperRunner
from backend.multimodel.hyper_model import HyperModel
from backend.multimodel.hnet_contextmod.model import hnet_mnet_from_config

class Experiment(object):
    def __init__(self, conf):

        self._hyper_experiment = HyperExperiment(conf, 
            lambda c : HyperTrainer(c, HyperModel(lambda : hnet_mnet_from_config(c), c["train"]["tasks"])), 
            lambda c : HyperTester(c, HyperModel(lambda : hnet_mnet_from_config(c), c["test"]["tasks"])), 
            lambda c : HyperRunner(c, HyperModel(lambda : hnet_mnet_from_config(c), c["run"]["tasks"]))
        )

    def execute(self):
        self._hyper_experiment.execute()