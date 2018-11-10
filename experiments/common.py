import os

EXPERIMENTS_DIR = os.path.dirname(os.path.realpath(__file__))
EXPERIMENTS_RESOURCES = os.path.join(EXPERIMENTS_DIR, 'resources')

if not os.path.exists(EXPERIMENTS_RESOURCES):
    os.makedirs(EXPERIMENTS_RESOURCES)


def get_resource_path(name):
    return os.path.join(EXPERIMENTS_RESOURCES, name)
