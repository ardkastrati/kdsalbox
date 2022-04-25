import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.utils import print_pretty_header
from backend.datasets import TestDataManager
from backend.metrics import NSS, CC, SIM
from backend.image_processing import process

from backend.generalization.embed.model import Student

from hypnettorch.hnets import HMLP


class Tester(object):
    def __init__(self, conf):
        self._conf = conf

        test_parameter_map = conf['test_parameter_map']
        self._logging_dir = test_parameter_map.get_val('logging_dir')
        self._recursive = test_parameter_map.get_val('recursive')
        self._verbose = test_parameter_map.get_val('verbose')
        self._batch_size = test_parameter_map.get_val('batch_size')
        self._per_image_statistics = test_parameter_map.get_val('per_image_statistics')

        self._preprocessing_parameter_map = conf["preprocessing_parameter_map"]
        self._postprocessing_parameter_map = conf["postprocessing_parameter_map"]

        self._model_path = conf["model_path"]
        self._task_cnt = conf["task_cnt"]
        # TODO: come up with a better way such that the model_id does not depend on the order of the paths
        test_folders_paths = self._conf["test_folders_paths"] # Make sure the order of models is the same as for the training!

        self._gpu = str(conf["gpu"])
        self._device = self._gpu if torch.cuda.is_available() else 'cpu'

        if self._verbose:
            print("Test setup:")

        test_datasets = [TestDataManager(img_path, sal_path, self._verbose, self._preprocessing_parameter_map) for (img_path,sal_path) in test_folders_paths]
        self._dataloaders = [DataLoader(ds, batch_size=self._batch_size, shuffle=False, num_workers=4) for ds in test_datasets]

    def pretty_print(self, epoch, mode, loss, lr):
        print('--------------------------------------------->>>>>>')
        print('Epoch {}: loss {} {}, lr {}'.format(epoch, mode, loss, lr))
        print('--------------------------------------------->>>>>>')
    
    def test_one(self, hnet, mnet, dataloaders):
        all_names, all_loss, all_NSS, all_CC, all_SIM = [], [], [], [], []
        my_loss = torch.nn.BCELoss()

        for (model_id, dataloader) in enumerate(dataloaders):
            for i, (X, y, names) in enumerate(dataloader):

                if torch.cuda.is_available():
                    X = X.cuda(torch.device(self._gpu))
                    y = y.cuda(torch.device(self._gpu))

                weights = hnet(cond_id=model_id)
                pred = mnet.forward(X, weights=weights)
                losses = my_loss(pred, y)
                detached_pred = pred.cpu().detach().numpy()
                detached_y = y.cpu().detach().numpy()

                # Doing the postprocessing steps needed for the metrics (We might want to do this also for Task Evaluation stuff?)
                y = process(y, self._postprocessing_parameter_map)
                detached_pred = process(detached_pred, self._postprocessing_parameter_map)
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
                    print('Batch {}: current accumulated loss {}'.format(i, np.mean(all_loss)))

                if self._per_image_statistics:
                    np.savetxt(os.path.join(self._logging_dir, 'image_statistics.csv'), np.array([all_names, all_NSS, all_CC, all_SIM]).T, fmt='%s', delimiter=',', header='IMAGE,NSS,CC,SIM', comments='')
            
        return np.mean(all_loss), np.nanmean(np.asarray(all_NSS)), np.nanmean(np.asarray(all_CC)), np.nanmean(np.asarray(all_SIM))


    def start_test(self):
        (hnet,mnet) = self.load_model(self._model_path)

        if not os.path.exists(self._logging_dir):
            os.makedirs(self._logging_dir)

        hnet.eval()
        mnet.eval()
        loss, NSS, CC, SIM = self.test_one(hnet, mnet, self._dataloaders)
        results = [[loss], [NSS], [CC], [SIM]]
        
        np.savetxt(os.path.join(self._logging_dir, 'test_results.csv'), np.array(results).T, fmt='%s', delimiter=',', header='LOSS,NSS,CC,SIM', comments='')

    def execute(self):
        if self._verbose: print_pretty_header("TESTING " + self._model_path)
        if self._verbose: print("Tester started...")
        self.start_test()
        if self._verbose: print("Done with {}!".format(self._model_path))

    def load_model(self, model_path):
        models = torch.load(model_path)

        mnet = Student()
        hnet = HMLP(
            mnet.external_param_shapes(), 
            layers=self._conf["hnet_hidden_layers"], # the sizes of the hidden layers (excluding the last layer that generates the weights)
            cond_in_size=self._conf["hnet_embedding_size"], # the size of the embeddings
            num_cond_embs=self._task_cnt # the number of embeddings we want to learn
        )

        mnet.load_state_dict(models["mnet_model"])
        hnet.load_state_dict(models["hnet_model"])

        mnet.to(self._device)
        hnet.to(self._device)

        return (hnet, mnet)

    def tester(self):
        del self._dataloaders