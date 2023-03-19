import torch
import copy


from return_types import *


def likelihood(Hypotheses, Masks: Bitmask, time_step_correction: int=0) -> Tensor:

    timesteps, n_perm, n_codes, h, w = Masks.union.shape

    imagined_masks = torch.zeros(timesteps, n_perm, n_codes, h, w)
    imagined_masks[0] = Masks.bitmasks[0]

    imaginary_errors = torch.zeros(n_perm, n_codes)
    observation_errors = torch.zeros(n_perm, n_codes)

    pixel_based_error = torch.zeros(timesteps, n_perm, n_codes, h, w)
    predictions = torch.zeros(timesteps, n_perm, n_codes, h, w)
    
    for t in range(timesteps - 1 - time_step_correction):
    
        imagined_masks[t + 1] = Masks.bitmasks[t + 1] # update imagined masks by future observations
        predicted = []
        
        for c, H in enumerate(Hypotheses):
            predicted.append(H._evaluate(H, data=Bitmask(imagined_masks[t, :, c]), methods=copy.deepcopy(H._methods), t=t)[-1].bitmasks)
            
        predicted = torch.stack(tuple(predicted)).permute(1, 0, 2, 3)
        predictions[t,:] = predicted

        observed = Masks.bitmasks[t + 1] # actual empirical observation
        imagined = torch.logical_and(predicted, Masks.union[t]).int() # imagined object based on prediction and potential occlusion / disappearance
        
        observation_error = torch.sum(abs(observed[None] - predicted), (2, 3)) # observation error vs imaginary error
        imaginary_error = torch.sum(abs(imagined - predicted), (2, 3))
    
        pixel_based_error[t,:] = abs(observed[None] - predicted)     
        imaginary_gain = (imaginary_error < observation_error).int().nonzero()
    
        for gain in imaginary_gain:
            imagined_masks[t + 1, gain[0], gain[1]] = torch.logical_or(imagined[gain[0], gain[1]], observed[gain[1]])

        observation_errors += -observation_error
        imaginary_errors += -imaginary_error

        # observation_errors += -out_penalizer * (reliability*observation_error) + -out_penalizer * (1-reliability) * torch.logical_not(observation_error) # fleet style of doing things
        # imaginary_errors += -out_penalizer * (reliability*imaginary_error) + -out_penalizer * (1-reliability) * torch.logical_not(imaginary_error)

    return imagined_masks, observation_errors, pixel_based_error, predictions


def imaginary_likelihood(Hypotheses, Masks: Bitmask) -> Tensor:
    
    timesteps, n_perm, n_codes, h, w = Masks.bitmasks.shape
    observation_errors = torch.log(torch.ones(n_perm, n_codes))
    
    for t in range(timesteps - 1):
        predicted = []
        for c, H in enumerate(Hypotheses):
            predicted.append(H._evaluate(H, data=Bitmask(Masks.bitmasks[t, :, c]), methods=copy.deepcopy(H._methods), t=t)[-1].bitmasks)
        
        predicted = torch.stack(tuple(predicted)).permute(1, 0, 2, 3)
        observed = Masks.bitmasks[t + 1] # actual empirical observation

        observation_error = torch.sum(abs(observed - predicted), (2, 3))

        observation_errors += -observation_error
    
    return observation_errors

