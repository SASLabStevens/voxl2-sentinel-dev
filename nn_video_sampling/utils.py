# !/usr/bin/env python3

import yaml 

def load_yaml(path):
    with open(path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict
