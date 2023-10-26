from datasets.CIFAR10 import CIFAR10
from datasets.CIFAR100 import CIFAR100
from datasets.TinyImageNet import TinyImageNet
from datasets.MNIST import MNIST
from datasets.PLACE365 import PLACE365

_Datasets = {}


def _add_dataset(dataset_fn):
    _Datasets[dataset_fn.__name__] = dataset_fn
    return dataset_fn


@_add_dataset
def cifar10(**kwargs):
    return CIFAR10(**kwargs)

@_add_dataset
def cifar100(**kwargs):
    return CIFAR100(**kwargs)

@_add_dataset
def mnist(**kwargs):
    return MNIST(**kwargs)


@_add_dataset
def tinyimagenet(**kwargs):
    return TinyImageNet(**kwargs)

@_add_dataset
def place365(**kwargs):
    return PLACE365(**kwargs)


def get_dataset(dataset_name, **kwargs):
    return _Datasets[dataset_name](**kwargs)
