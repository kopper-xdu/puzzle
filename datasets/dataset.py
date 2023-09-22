# from torch.utils.data import Dataset
# import os
# import numpy as np
# from PIL import Image
from .base_dataset import ImageFolderLMDB


class ImageNet(ImageFolderLMDB):
    def __init__(self, db_path, transform=None, target_transform=None):
        super().__init__(db_path, transform, target_transform)
