import torch
import numpy as np
import copy
import pickle
from multiprocessing import Pool
from os import getpid
from joblib import Parallel, delayed
import json
from tqdm import tqdm
from return_types import *
from data import *
from utils import *
from renderer import derender, render
from residual import get_residual, get_residual_pixel, get_new_mask, get_new_bitmasks
from mcmc import mcmc_sampler, imaginary_mcmc_sampler

def run_mcmc(world): # 
    print(f"process_id: {getpid()}")
    
    # SAMPLER 
    all_codebook_results = {}
    all_residuals = {}
    all_hypotheses = {}
    all_likelihoods = {}
    all_imaginary_bitmasks = {}

    
    n_repeat = 100
    n_cycles = 4
    mcmc_steps = 100

    for repeat in tqdm(range(n_repeat)):# define max number of wake-sleep cycles


        all_hypotheses[repeat] = []
        all_codebook_results[repeat] = []
        all_likelihoods[repeat] = []
        all_residuals[repeat] = []
        all_imaginary_bitmasks[repeat] = []

        # load codebooks
        timesteps = 10
        Codebooks = DataObject(root_dir='../codebooks/' + str(world) + '/codes_', timesteps=timesteps)
        Masks = derender(Codebooks)


        # keep track of previous likelihood
        joint_ll_prev = 0.0
        joint_ll_curr = 0.0

        # run 'wake-sleep' cycles
        for cycle in range(n_cycles):
            joint_ll_curr = 0.0
            print(f"Iteration {cycle+1}:")
            print(joint_ll_curr, joint_ll_prev)
            # WAKE PHASE 
            wake_res, wake_ll, wake_m_hat, wake_m_hat_tensor, wake_eps, wake_pred = mcmc_sampler(Masks=Masks,
                                                                                                p=0.4, 
                                                                                                mcmc_steps=mcmc_steps,
                                                                                                print_step=1, 
                                                                                                H_start=None)
            #  IMAGINARY PHASE
            max_joint_wake = torch.sum(torch.max(wake_ll[-1], dim=0).values)
            if world in ['3', '4', '8']: #  IMAGINARY PHASE FOR OCCLUSION
                imaginary_res, imaginary_ll = imaginary_mcmc_sampler(Masks=Masks, 
                                                                    predicted_masks_tensor=wake_m_hat_tensor, 
                                                                    p=0.4, 
                                                                    mcmc_steps=mcmc_steps,
                                                                    print_step=1)
                
                max_joint_imagined = torch.sum(torch.max(imaginary_ll[-1], dim=0).values)
                if world in ['8']: # ignore plank
                    max_joint_wake = torch.max(wake_ll[-1])
                    max_joint_imagined = torch.max(imaginary_ll[-1])
                if max_joint_wake < max_joint_imagined: # check if imaginary converged
                    max_idx = torch.max(imaginary_ll[-1], dim=0).indices[0] # winning permutation
                    Masks.bitmasks[:] = wake_m_hat_tensor[:, max_idx] # correct bitmasks
                    wake_res = imaginary_res # replace wake with imaginary results 
                    try:
                        all_likelihoods[repeat].append([int(t) for t in max_joint_imagined])
                    except:
                        all_likelihoods[repeat].append([int(max_joint_imagined)])
                    all_imaginary_bitmasks[repeat].append(copy.deepcopy(Masks)) 
                    all_hypotheses[repeat].append(len(wake_res[-1]))
                    # all_residuals[repeat].append(copy.deepcopy(residual))
                    print(f'Converged at repeat {repeat}, cycle {cycle}: imaginary', max_joint_imagined)
                    break
                
            # RESIDUAL PHASE
            for i, _ in enumerate(wake_res[-1]):
                joint_ll_curr += torch.max(wake_res[-1][i].likelihood) 
            residual = get_residual(Masks, wake_res[-1]) # compute residual using final sample from wake phase (i.e. surviving hypotheses)
            if cycle > 0: # if we are not in the first cycle
                # compare likelihood with previous results to check if converged 
                if torch.sum(joint_ll_curr) <= torch.sum(joint_ll_prev): # if current likelihood is less or equal than previous likelihood-converged 
                    print(f'Converged at repeat {repeat}, cycle {cycle}: residual', joint_ll_curr)
                    break

            joint_ll_prev = joint_ll_curr
            try:
                all_likelihoods[repeat].append([int(t) for t in joint_ll_curr])
            except:
                all_likelihoods[repeat].append([int(joint_ll_curr)])
            
            # Residual sampler 
            residual_res, residual_ll, residual_m_hat, residual_m_hat_tensor, residual_eps, residual_pred = mcmc_sampler(Masks=residual,
                                                                                                                        p=0.4,
                                                                                                                        mcmc_steps=mcmc_steps,
                                                                                                                        print_step=1,
                                                                                                                        H_start=None,
                                                                                                                        time_step_correction=1)
            # UPDATE REPRESENTSTION
            residual_pixel = get_residual_pixel(Masks, residual_res[-1])
            new_mask = get_new_mask(Masks, residual_pixel, residual_res[-1])
            new_bitmasks = get_new_bitmasks(Masks, new_mask)
            codebooks = render(new_bitmasks)
            Codebooks_prime = DataObject(codebooks=codebooks,timesteps=timesteps)
            Masks_prime = derender(Codebooks_prime)
            Masks = Masks_prime
            all_hypotheses[repeat].append(len(wake_res[-1]))


    with open(f"ebby_sim_res_paper/{world}_mcmc_{mcmc_steps}_ebby.pkl", "wb") as f:
        pickle.dump([all_hypotheses, all_likelihoods, all_imaginary_bitmasks], f)



if __name__ == "__main__":
    worlds = [str(i) for i in range(1,9)]
    n_jobs = 8
    Parallel(n_jobs=n_jobs)(delayed(run_mcmc)(world) for world in worlds)


