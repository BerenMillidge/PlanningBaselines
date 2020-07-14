import os
import time
from copy import deepcopy
import copy
import math
import json
import random
from datetime import datetime
import itertools
import pprint
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
matplotlib.rcParams["axes.linewidth"] = 1.1


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class Logger(object):
    def __init__(self, logdir, seed):
        self.logdir = logdir
        self.seed = seed
        self.path = f"log_{logdir}_{str(seed)}/"
        self.metrics_path = self.path + "metrics.json"
        self.print_path = self.path + "out.txt"
        os.makedirs(self.path, exist_ok=True)

        self._metrics = {}
        self._init_print()
        self._init_custom()

    def log_metric(self, metric, value):
        if metric not in self._metrics:
            self._metrics[metric] = []
        self._metrics[metric].append(value)

    def log(self, string):
        f = open(self.print_path, "a")
        f.write("\n")
        f.write(str(string))
        f.close()
        print(string)

    def save(self):
        self._save_json(self.metrics_path, self._metrics)
        self.log("Saved _metrics_")

    def save_metrics(self):
        self._save_json(self.metrics_path, self._metrics)
        self.log("Saved _metrics_")

    def log_args(self, args):
        for arg, value in sorted(vars(args).items()):
            self.log(f"{arg}: `{value}`")

    def log_stds(self, std):
        self._stds.append(std)

    def flush(self):
        _mean = np.mean(np.array(self._stds))
        _max = np.max(np.array(self._stds))
        _min = np.min(np.array(self._stds))
        self.log(f"Std mean {_mean} max {_max}, min {_min}")
        self._init_custom()

    def _init_custom(self):
        self._stds = []

    def _init_print(self):
        f = open(self.print_path, "w")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        f.write(current_time)
        f.close()

    def _save_json(self, path, obj):
        with open(path, "w") as file:
            json.dump(obj, file)