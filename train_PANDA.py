# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import random_split

from Model import *

import trainer
from opt import arg_parse
from utils import get_logger


if __name__ == '__main__':
    args = arg_parse()

    
