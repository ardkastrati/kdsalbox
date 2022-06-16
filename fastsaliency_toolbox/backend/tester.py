"""
Tester
------

Generates NSS, CC, SIM, loss metrics for all the original images in a folder.
If per_image_statistics is enabled it will additionally log the metrics per image.

"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import print_pretty_header
from .datasets import TestDataManager
from .metrics import NSS, CC, SIM
from .image_processing import process

class Tester(object):
    def __init__(self,
                 model_manager,
                 test_parameter_map, preprocessing_parameter_map, postprocessing_parameter_map, gpu=0):

        self._model = model_manager.get_matching(test_parameter_map.get_val('model'))
        self._logging_dir = test_parameter_map.get_val('logging_dir')
        self._input_images = test_parameter_map.get_val('input_images')
        self._input_saliencies = test_parameter_map.get_val('input_saliencies')
        self._recursive = test_parameter_map.get_val('recursive')
        self._verbose = test_parameter_map.get_val('verbose')
        self._batch_size = test_parameter_map.get_val('batch_size')
        self._per_image_statistics = test_parameter_map.get_val('per_image_statistics')

        self._preprocessing_parameter_map = preprocessing_parameter_map
        self._postprocessing_parameter_map = postprocessing_parameter_map

        self._gpu = str(gpu)

        if self._verbose:
            print("Test setup:")
        ds_test = TestDataManager(self._input_images, self._input_saliencies, self._verbose, self._preprocessing_parameter_map)
        self._dataloader = DataLoader(ds_test, batch_size=self._batch_size, shuffle=False, num_workers=4)

    def pretty_print(self, epoch, mode, loss, lr):
        print('--------------------------------------------->>>>>>')
        print('Epoch {}: loss {} {}, lr {}'.format(epoch, mode, loss, lr))
        print('--------------------------------------------->>>>>>')
    
    def test_one(self, model, dataloader):
        all_names, all_loss, all_NSS, all_CC, all_SIM = [], [], [], [], []
        my_loss = torch.nn.BCELoss()

        for i, (X, y, names) in enumerate(dataloader):

            if torch.cuda.is_available():
                X = X.cuda(torch.device(self._gpu))
                y = y.cuda(torch.device(self._gpu))

            pred = model.forward(X)
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
        student = self._model.get_student()

        if not os.path.exists(self._logging_dir):
            os.makedirs(self._logging_dir)

        student.eval()
        loss, NSS, CC, SIM = self.test_one(student, self._dataloader)
        results = [[loss], [NSS], [CC], [SIM]]
        
        np.savetxt(os.path.join(self._logging_dir, 'test_results.csv'), np.array(results).T, fmt='%s', delimiter=',', header='LOSS,NSS,CC,SIM', comments='')

    def execute(self):
        if self._verbose: print_pretty_header("TESTING " + self._model.name)
        if self._verbose: print("Tester started...")
        self.start_test()
        if self._verbose: print("Done with {}!".format(self._model.name))

    def tester(self):
        del self._dataloader

if __name__ == '__main__':
    from .pseudomodels import ModelManager
    m = ModelManager('models/', verbose=True, pretrained=True)
    print(m._model_map)

    from .config import Config
    c = Config('config.json')
    c.test_parameter_map.pretty_print()

    t = Tester(m, c.test_parameter_map)
    t.execute()
    #WORKSSS

