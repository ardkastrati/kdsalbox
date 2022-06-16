import os
import textwrap
import numpy as np
from PIL import Image

from .parameters import ParameterMap


def create_dirs_if_none(path : str):
    parent_path = os.path.dirname(path)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)


def get_image_path_tuples(input_dir : str, output_dir : str, recursive : bool = False):
    if recursive:
        image_path_map = {}
        for dirpath, dirnames, filenames in os.walk(input_dir):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    input_path = os.path.join(dirpath, filename)
                    output_path = os.path.join(output_dir,
                                           os.path.relpath(dirpath, input_dir),
                                           filename)
                    image_path_map[input_path] = output_path
    else:
        image_path_map = {
            os.path.join(input_dir, f): os.path.join(output_dir, f)
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))
        }

    return sorted(image_path_map.items())


def print_pretty_header(header_text : str, width : int = 60):
    print("*" * width)
    print(" {} ".format(header_text).center(width, "*"))
    print("*" * width)


def pretty_print_parameters(parameter_list : ParameterMap):
    parameters = sorted(parameter_list, key=lambda x: x.name)
    if parameters:
        for param in parameters:
            print('')
            print(param.name)
            print('    Default:      {}'.format(param.value))
            if (param.description):
                print('    Description:  {}'.format('\n                  '.join(
                    textwrap.wrap(param.description, break_on_hyphens=False))))
            if (param.valid_values):
                print('    Valid Values: {}'.format('\n                   '.join(
                    textwrap.wrap(str(param.valid_values)))))
    else:
        print('    None.')


def save_image(path : str, image, create_parent : bool = True):
    if create_parent:
        create_dirs_if_none(path)

    result = Image.fromarray(image)
    result.save(path)


def read_image(path : str, dtype=np.float32):
    f = Image.open(path)
    img = np.asarray(f, dtype)
    if(len(img.shape) == 2):
        print("FOUND SOMETHING WEIRD")
        return None
    return img

def read_saliency(path : str, dtype=np.float32):
    f = Image.open(path)
    img = np.asarray(f, dtype)
    return img / 255.0

