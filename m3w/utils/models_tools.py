"""Tools for HARL."""
import copy
import math
import torch
import torch.nn as nn


def init_device(args):
    """Init device.
    Args:
        args: (dict) arguments
    Returns:
        device: (torch.device) device
    """
    if args["cuda"] and torch.cuda.is_available():
        if args.get("cuda_id", None) is not None:
            device = torch.device("cuda:" + str(args["cuda_id"]))
            print("choose to use gpu %d..." % args["cuda_id"])
        else:
            device = torch.device("cuda:0")
            print("choose to use gpu 0...")
        if args["cuda_deterministic"]:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
    torch.set_num_threads(args["torch_threads"])
    return device


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly
    Args:
        optimizer: (torch.optim) optimizer
        epoch: (int) current epoch
        total_num_epochs: (int) total number of epochs
        initial_lr: (float) initial learning rate
    """
    learning_rate = initial_lr - (initial_lr * ((epoch - 1) / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def init(module, weight_init, bias_init, gain=1):
    """Init module.
    Args:
        module: (torch.nn) module
        weight_init: (torch.nn) weight init
        bias_init: (torch.nn) bias init
        gain: (float) gain
    Returns:
        module: (torch.nn) module
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    """Clone module for N times."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_grad_norm(parameters):
    """Get gradient norm."""
    sum_grad = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        sum_grad += parameter.grad.norm() ** 2
    return math.sqrt(sum_grad)
