"""
Runner
------

Produces the saliency images for all original images in a folder for a specific model.

"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader


from .datasets import RunDataManager
from .pseudomodels import ModelManager
from .parameters import ParameterMap
from .image_processing import process
from .utils import print_pretty_header, save_image

class Runner(object):
    def __init__(self,
                 model_manager : ModelManager,
                 run_parameter_map : ParameterMap,
                 postprocessing_parameter_map : ParameterMap, 
                 gpu : int = 0):

        self._model = model_manager.get_matching(run_parameter_map.get_val('model'))
        self._input_dir = run_parameter_map.get_val('input_images')
        self._output_dir = run_parameter_map.get_val('output_dir')
        self._verbose = run_parameter_map.get_val('verbose')
        self._recursive = run_parameter_map.get_val('recursive')
        self._overwrite = run_parameter_map.get_val('overwrite')
        self._postprocessing_parameter_map = postprocessing_parameter_map
        self._gpu = str(gpu)
        if self._verbose:
            print("Run setup:")
        self._dataloader = DataLoader(RunDataManager(self._input_dir, self._output_dir, self._verbose, self._recursive), batch_size=1)

    def execute(self):
        print_pretty_header("RUNNING " + self._model.name)
        if self._verbose: print("Running the model...")
       

        for img_number, (image, input_path, output_path) in enumerate(self._dataloader):
            printable_input_path = os.path.relpath(input_path[0], self._input_dir)
            # If we aren't overwriting and the file exists, skip it.
            if not self._overwrite and os.path.isfile(output_path[0]):
                print(
                "SKIP (already exists) image [{}/{}]: {}".format(
                    img_number + 1, len(self._dataloader), printable_input_path))
            else:
                print(
                "Running image [{}/{}]: {}".format(img_number + 1, len(self._dataloader),
                                                   printable_input_path))

                if torch.cuda.is_available():
                    image = image.cuda(torch.device(self._gpu))
                
                saliency_map = self._model.compute_saliency(image)

                post_processed_image = np.clip((process(saliency_map.cpu().detach().numpy()[0, 0], self._postprocessing_parameter_map)*255).astype(np.uint8), 0, 255)
                print(post_processed_image.shape)
                save_image(output_path[0], post_processed_image)
                
                # Remove batch from gpu
                if torch.cuda.is_available():
                    del image
                    del input_path
                    del output_path
                    torch.cuda.empty_cache()
        print("Done with {}!".format(self._model.name))

    def delete(self):
        del self._dataloader


if __name__ == '__main__':
    from .pseudomodels import ModelManager
    m = ModelManager('models/', verbose=True, pretrained=True)
    print(m._model_map)

    from .config import Config
    c = Config('config.json')

    c.run_parameter_map.pretty_print()
    c.postprocessing_parameter_map.pretty_print()

    t = Runner(m, c.run_parameter_map, c.postprocessing_parameter_map)
    t.execute()
    #WORKSSS