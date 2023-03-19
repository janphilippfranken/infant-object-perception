import torch
import tensorflow as tf
import numpy as np
import copy
import pickle

# grammar
from return_types import *
from data import *
from utils import *
from renderer import derender, render


# get error 
def get_residual(Masks, 
                 H, 
                 timesteps=10):

    residual = copy.deepcopy(Masks)
    for c in range(Masks.codes.shape[0]): # loop over codes 
        for time in range(timesteps - 1):
            pred = H[c]._evaluate(H[c], data=Bitmask(Masks.bitmasks[time, c]), methods=copy.deepcopy(H[c]._methods), t=time)[-1].bitmasks
            obs = Masks.bitmasks[time + 1, c]
            residual.bitmasks[time, c] = torch.abs(pred - obs)
    return residual

def get_residual_pixel(Masks, 
                       H, 
                       timesteps=10):


    residual = copy.deepcopy(Masks)

    for c in range(Masks.codes.shape[0]): # loop over codes 
        # assign unique IDs to 1s in initial tensor
        init_mask = Masks.bitmasks[:, c]
        init_ids = torch.zeros(init_mask.shape, dtype=torch.float)
        count = 1
        for t in range(init_mask.shape[0]):
            for i in range(init_mask.shape[1]):
                for j in range(init_mask.shape[2]):
                    if init_mask[t][i][j] == 1:
                        init_ids[t][i][j] = count
                        count += 1

        # assign unique IDs to 1s in predicted tensor
        predictions = torch.zeros_like(init_mask)
        prediction_ids = torch.zeros(predictions.shape, dtype=torch.float)
        count = 1
        for t in range(predictions.shape[0]):
            pred = H[c]._evaluate(H[c], data=Bitmask(init_mask[t]), methods=copy.deepcopy(H[c]._methods), t=t)[-1].bitmasks
            pred_ids = torch.zeros(pred.shape, dtype=torch.float)
            for i in range(predictions.shape[1]):
                for j in range(predictions.shape[2]):
                    if pred[i][j] == 1:
                        pred_ids[i][j] = count
                        count += 1
            predictions[t] = pred
            prediction_ids[t] = pred_ids

        # assign unique IDs to 1s in observed tensor
        obs = Masks.bitmasks[1:, c]
        obs_ids = torch.zeros(obs.shape, dtype=torch.float)
        count = 1
        for t in range(obs.shape[0]):
            for i in range(obs.shape[1]):
                for j in range(obs.shape[2]):
                    if obs[t][i][j] == 1:
                        obs_ids[t][i][j] = count
                        count += 1

        # create error tensor
        error_tensor = torch.zeros(init_mask.shape, dtype=torch.float)
        for t in range(init_mask.shape[0] - 1):
            for i in range(init_mask.shape[1]):
                for j in range(init_mask.shape[2]):
                    if init_mask[t][i][j] == 1:
                        pred_id = prediction_ids[t][i][j]
                        obs_id = obs_ids[t][i][j]
                        if pred_id != obs_id:
                            error_tensor[t][i][j] = 1.0

        residual.bitmasks[:, c] = error_tensor

    return residual

def get_new_mask(Masks, residual_pixel, H, timesteps=10):
    new_mask = copy.deepcopy(Masks)
    for c in range(Masks.codes.shape[0]):
        new_mask.bitmasks[:timesteps-2, c] = torch.logical_and(Masks.bitmasks[:timesteps-2, c], torch.logical_not(residual_pixel.bitmasks[:timesteps-2, c]))
        # impute final timesteps
        new_mask.bitmasks[timesteps-2, c] = H[c]._evaluate(H[c], data=Bitmask(new_mask.bitmasks[timesteps-3, c]), methods=copy.deepcopy(H[c]._methods), t=timesteps-3)[-1].bitmasks
        new_mask.bitmasks[timesteps-1, c] = H[c]._evaluate(H[c], data=Bitmask(new_mask.bitmasks[timesteps-2, c]), methods=copy.deepcopy(H[c]._methods), t=timesteps-2)[-1].bitmasks
    return new_mask

def get_new_bitmasks(Masks, new_mask):
    new_bitmasks = copy.deepcopy(Masks)
    for c in range(new_bitmasks.codes.shape[0]):
        new_bitmasks.bitmasks[:, c] = torch.logical_and(Masks.bitmasks[:, c], torch.logical_not(new_mask.bitmasks[:, c]))   
    new_bitmasks.bitmasks = torch.concat((new_bitmasks.bitmasks, new_mask.bitmasks), dim=1)
    new_bitmasks.codes = torch.cat((new_bitmasks.codes, torch.max(new_bitmasks.codes) + new_bitmasks.background + torch.tensor([1], dtype=torch.float)))
    return new_bitmasks