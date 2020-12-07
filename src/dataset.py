"""
Importing header files
"""

import torch

import os
import random
import warnings
import glob
import time

from os.path import join
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from tqdm.auto import tqdm
import numpy as np
import xarray as xr
import pyproj


from sklearn.model_selection import train_test_split

import utils
from config import *


class UnconditionalDataset(torch.utils.data.Dataset):
    """
    Abstract class for holding the observation dataset for unconditional GAN,
    and standardizing it.

    """
    def __init__(self, device='cpu', highres=True):
        """
        Parameters
        ----------
        device: str, default 'cpu'
            Train on cpu or cuda
        highres: bool, default True
            Use high resolution or low resolution images
        """

        self.device = device
        self.highres = highres

        self.load_observations()


    def load_observations(self):
        """
        Load observations
        """
        if self.highres:
            self.observed_images = np.ones((appended_obs_x_highres, appended_obs_y_highres))*0.1
        else:
            self.observed_images = np.ones((appended_obs_x_lowres, appended_obs_y_lowres))*0.1
        self.observed_images = utils.load_obs_from_cache(cluster_cache_dir, self.highres)
        self.observed_images = utils.standardize_images_clipping(self.observed_images)
        self.observed_images = np.expand_dims(self.observed_images, axis=1)

    def __len__(self):
        """
        Length of dataset
        """
        return len(self.observed_images)

    def __getitem__(self, index):
        """
        Returns elements of dataset
        """
        img = self.observed_images[index]
        img = torch.tensor(img, device=self.device)
        return img


class ConditionalDataset(torch.utils.data.Dataset):
    """
    Abstract class for holding observations and predictions
    for the condition GAN, and standardizing the images

    The conditional GAN conditions on the predictions and aims
    to generate the observed precipitation values.

    """
    def __init__(self, device='cpu', highres=True):
        """
        Parameters
        ----------
        device: str, default 'cpu'
            Train on cpu or cuda
        highres: bool,default True
            Use high resolution or low resolution images

        """
        self.device = device
        self.highres = highres

        #Load observations
        self.load_observations()

        #Load predictions
        self.load_predictions()

        assert (len(self.observed_images) == len(self.predicted_images)), "Number of observations and predictions should be the same."

    def load_observations(self):
        """
        Load observations
        """

        if self.highres:
            self.observed_images = np.ones((appended_obs_x_highres, appended_obs_y_highres))*0.1
        else:
            self.observed_images = np.ones((appended_obs_x_lowres, appended_obs_y_lowres))*0.1
        self.observed_images = utils.load_obs_from_cache(cluster_cache_dir, self.highres)
        self.observed_images = utils.standardize_images_clipping(self.observed_images)
        self.observed_images = np.expand_dims(self.observed_images, axis=1)

    def load_predictions(self):
        """
        Load predictions
        """
        self.predicted_images = np.ones((appended_pred_x, appended_pred_y))*0.1
        self.predicted_images = utils.load_predictions_from_cache(cluster_cache_dir)
        self.predicted_images = utils.standardize_images_clipping(self.predicted_images)
        self.predicted_images = np.expand_dims(self.predicted_images, axis=1)

    def __len__(self):
        """
        Length of dataset
        """
        return len(self.observed_images)

    def __getitem__(self, index):
        """
        Get elements of the dataset
        """
        predicted_image, observed_image = self.predicted_images[index], self.observed_images[index]
        predicted_image = torch.tensor(predicted_image, device=self.device)
        observed_image = torch.tensor(observed_image, device=self.device)
        return predicted_image, observed_image









