# -*- coding: utf-8 -*-
import os
import json
import torch
import re

from .student import Student

############################################################
# FastSaliency Models
############################################################

class PseudoModel(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.original_model_name = kwargs.get('original_model_name')
        self.notes = kwargs.get('notes')
        self.student_path = kwargs.get('student_path')
        self.root_path = kwargs.get('root_path')
        self.version = kwargs.get('version')
        self.model_type = kwargs.get('model_type')
        self.verbose = kwargs.get('verbose')
        self.pretrained = kwargs.get('pretrained')
        self.gpu = kwargs.get('gpu')

        if self.verbose: print("Got name", self.name)
        if self.verbose: print("Got original model name", self.original_model_name)
        if self.verbose: print("Got notes", self.notes)
        if self.verbose: print("Got student path", self.student_path)
        if self.verbose: print("Got root path", self.root_path)
        if self.verbose: print("Got model type", self.model_type)
        if self.verbose: print("Got version", self.version)
        if self.verbose: print("Got pretrained", self.pretrained)

        if None in (self.name, self.original_model_name, self.student_path, self.model_type, self.version):
            raise ValueError("Invalid pseudomodel.json file contents.")

        self.my_student = Student()
        if self.pretrained:
            self.update_weights(os.path.join(self.root_path, self.student_path))

    def update_weights(self, path : str):
        print("Updating weights")
        if torch.cuda.is_available():
            checkpoint = torch.load(path, map_location=torch.device('cuda'))
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.my_student.load_state_dict(checkpoint['student_model'])

    def compute_saliency(self, img : torch.Tensor) -> torch.Tensor:
        self.my_student.eval()
        sal = self.my_student(img)
        return sal

    def cuda(self):
        if torch.cuda.is_available():
            print("Moving pseudomodel " + self.name + " in " + self.gpu)
            self.my_student.cuda(torch.device(self.gpu))

    def delete(self):
        if torch.cuda.is_available():
            print("Removing pseudomodel from cuda: " + self.name)
            del self.my_student
            torch.cuda.empty_cache()

    def get_student(self):
        return self.my_student


############################################################
# Model Manager
############################################################
class ModelManager(object):
    def __init__(self, models_path : str, verbose : bool, pretrained : bool = False, gpu : str = 'cuda:0'):
        self._models_path = models_path
        self._verbose = verbose
        self._pretrained = pretrained
        self._gpu = gpu

        self._model_map = self.find_and_load_models(self._models_path)

    def find_and_load_models(self, start_path : str):
        print("Entered Find and Load Models")
        def _yield_all_pseudomodel_jsons(start_path):
            for dirpath, dirnames, filenames in os.walk(start_path):
                for filename in filenames:
                    if filename == "pseudomodel.json":
                        yield os.path.join(dirpath, filename)

        models = {}
        for pseudomodel_json_path in _yield_all_pseudomodel_jsons(start_path):
            model = self.load_model(pseudomodel_json_path)
            if model is not None:
                models[model.name.lower()] = model
        if self._verbose: print(models)
        print("Finished with that")
        return models

    def load_model(self, pseudomodel_json_path : str):
        try:
            with open(pseudomodel_json_path) as fp:
                model_data = json.load(fp)
        except Exception as e:
            print("ERROR: Failed to read in {}".format(pseudomodel_json_path))
            print(e)
            exit(2)
        model_data['verbose'] = self._verbose
        model_data['root_path'] = self._models_path
        model_data['pretrained'] = self._pretrained
        model_data['gpu'] = self._gpu
        return PseudoModel(**model_data)

    def cuda(self, model_name : str):
        if model_name.lower() in self._model_map:
            self._model_map[model_name.lower()].cuda()
        else:
            msg = """Unknown model name '{}'.""".format(model_name)
            raise ValueError(msg)

    def delete(self, model_name : str):
        if model_name.lower() in self._model_map:
            self._model_map[model_name.lower()].delete()
            del self._model_map[model_name.lower()]
        else:
            msg = """Unknown model name '{}'.""".format(model_name)
            raise ValueError(msg)

    def update_model(self, model_name : str, student_path : str):
        if model_name.lower() in self._model_map:
            self._model_map[model_name.lower()].update_weights(student_path)
        else:
            msg = """Unknown model name '{}'.""".format(model_name)
            raise ValueError(msg)

    def get_matching(self, model_name : str):
        if model_name.lower() in self._model_map:
            return self._model_map[model_name.lower()]
        else:
            msg = """Unknown model name '{}'.""".format(model_name)
            raise ValueError(msg)

    def get_matchings(self, model_names : str):
        clean_names = [name for name in re.split("[ ,]", model_names) if name]
        result = set()
        for name in clean_names:
            if name.lower() in self._model_map:
                result.add(self._model_map[name.lower()])
            else:
                msg = """Unknown model name '{}'.""".format(name)
                raise ValueError(msg)

        result_list = list(result)
        result_list.sort(key=lambda x: x.name)
        return result_list

if __name__ == '__main__':
    m = ModelManager('../models/', verbose=True, pretrained=False)
    print(m._model_map)
    # WORKS
