import os

import yaml


def load_yaml(name):
    dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(dir, f"{name}.yaml"), "r") as file:
        config = yaml.safe_load(file)
    return config
