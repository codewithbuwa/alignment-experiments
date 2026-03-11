import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import math
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ==============================
# Config
# ==============================

BETA = 1.0
LR = 1e-3
STEPS = 2000
DATASET_SIZE = 10000
LAMBDA = 1.33
DEVICE = "cpu"

REF_MU = 5.0
REF_SIGMA = 2.0
TARGET = 7.0
ZONE = (5.5, 8.5)
N_COMPONENTS = 2



