import torch
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from modules.HDC_utils import EllipsoidModel
from modules.Basic_HD import DensityTrainer
import torch.backends.cudnn as cudnn
from dataset.kitti.parser import Parser

from sklearn.cluster import MiniBatchKMeans

class AugModel(EllipsoidModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

