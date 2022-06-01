class HyperExperiment(object):
    def __init__(self, conf, pretrainer_fn, trainer_fn, tester_fn, runner_fn):
        self._conf = conf
        self._pretrainer_fn = pretrainer_fn
        self._trainer_fn = trainer_fn
        self._tester_fn = tester_fn
        self._runner_fn = runner_fn

    def execute(self):
        conf = self._conf
        experiment_conf = conf["experiment"]

        if not "pretrain" in experiment_conf["skip"]:
            pretrain = self._pretrainer_fn(conf)
            pretrain.execute()
            del pretrain

        if not "train" in experiment_conf["skip"]:
            train = self._trainer_fn(conf)
            train.execute()
            del train

        if not "test" in experiment_conf["skip"]:
            test = self._tester_fn(conf)
            test.execute()
            del test

        if not "run" in experiment_conf["skip"]:
            run = self._runner_fn(conf)
            run.execute()
            del run