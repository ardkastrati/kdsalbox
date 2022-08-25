"""
ParameterMap
------------

Represents a collection of named parameters, where each parameter has 
a value, description and a list of valid values.

"""

import copy
from typing import Any, List

class ParameterMap(object):
    def __init__(self):
        self._parameters = {}

    def set_from_dict(self, parameter_dict : dict):
        for name, properties in parameter_dict.items():
            if not isinstance(properties, dict):
                raise ValueError(
                    "Key '{}' has value '{}', expected dict.".format(
                        name, properties))
            self.set(
                name,
                properties['default'],
                description=properties.get('description'),
                valid_values=properties.get('valid_values'))
        return self

    def set(self, name : str, value : Any, description : str = None, valid_values : List[Any] = None):
        if name in self._parameters:
            self._parameters[name].update(
                value, description=description, valid_values=valid_values)
        else:
            self._parameters[name] = Parameter(
                name,
                value,
                description=description,
                valid_values=valid_values)

    def update(self, other_parameter_map : 'ParameterMap'):
        for name, parameter in other_parameter_map._parameters.items():
            self.set(name, parameter.value, parameter.description,
                     parameter.valid_values)

    def get_val(self, name : str):
        return self._parameters[name].value

    def exists_val(self, name : str):
        print([n.name for n in self._parameters.values()])
        return name in [n.name for n in self._parameters.values()]

    def get_pair_dict(self):
        pair_dict = {}

        for name, parameter in self._parameters.items():
            pair_dict[name] = parameter.value

        return pair_dict

    def get_parameters(self):
        return self._parameters.values()

    def clone(self):
        return copy.deepcopy(self)
    
    def pretty_print(self):
        for name in self._parameters:
            self._parameters[name].pretty_print()

    def __str__(self):
        res = ""
        for name in self._parameters:
            res += str(self._parameters[name]) + " \n "
        return res


class Parameter(object):
    def __init__(self, name : str, value : Any, description : str = None, valid_values : List[Any] = None):
        self.name = name
        self.value = value
        self.description = description
        self.valid_values = valid_values

    def update(self, value : Any, description :str = None, valid_values : List[Any] = None):
        if description is not None:
            self.description = description
        if valid_values is not None:
            self.valid_values = valid_values

        self.value = value

    def pretty_print(self):
        print(str(self))

    def __str__(self):
        return "Parameter: " + str(self.name) + ": " + str(self.value)
