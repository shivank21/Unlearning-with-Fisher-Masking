import torch


def model_init_fn(w):
    kaiming_normal(w)
    uniform(w)


# initializer for model
def binary(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)
        sigma = w.weight.data.std()
        w.weight.data = torch.sign(w.weight.data) * sigma


def kaiming_normal(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)


def kaiming_uniform(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(w.weight)


def orthogonal(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(w.weight)


# initializer for batch_norm
def uniform(w):
    if isinstance(w, torch.nn.BatchNorm2d):
        w.weight.data = torch.rand(w.weight.data.shape)
        w.bias.data = torch.zeros_like(w.bias.data)


def fixed(w):
    if isinstance(w, torch.nn.BatchNorm2d):
        w.weight.data = torch.ones_like(w.weight.data)
        w.bias.data = torch.zeros_like(w.bias.data)


def oneone(w):
    if isinstance(w, torch.nn.BatchNorm2d):
        w.weight.data = torch.ones_like(w.weight.data)
        w.bias.data = torch.ones_like(w.bias.data)


def positivenegative(w):
    if isinstance(w, torch.nn.BatchNorm2d):
        uniform(w)
        w.weight.data = w.weight.data * 2 - 1
        w.bias.data = torch.zeros_like(w.bias.data)

