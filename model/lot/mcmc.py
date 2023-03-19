import torch
from tqdm import tqdm
import copy
from scipy.stats import geom

from return_types import *
from grammar import *
from hypothesis import *
from numpy.random import default_rng
from tree_regrowth import TreeRegrowth

from likelihood import likelihood, imaginary_likelihood

One = torch.tensor(1)

TR = TreeRegrowth()

rng = default_rng()


# hacky approach to get prior 
def _get_prior(h, 
               n_bitmask_functions: int=4, 
               n_number_functions: int=7, 
               n_set_function: int=2,
               n_numbers: int=5,
            ):
    methods = h._methods
    prior = 0.0
    for method in methods:
        method_name = method[0].__qualname__.split('.')[0] if isinstance(method, tuple) else method.__qualname__.split('.')[0]
        if method_name == "BitmaskFunction":
            prior += torch.log(One/n_bitmask_functions)
            if isinstance(method, tuple):
                n_node_methods = len(method[-1]._methods) - 1
                n_neg = sum(1 for m in method[-1]._methods[1:] if m[0].__name__ == "_neg")
                node_probs = [One/(n_number_functions * n_numbers)] * (n_node_methods - n_neg) + [One/n_number_functions] * n_neg
                prior += torch.sum(torch.log(torch.tensor(node_probs)))
        elif method_name == "SetFunction":
            prior += torch.log(One/n_set_function)
    return prior


def mcmc_sampler(Masks: Bitmask,
                 mcmc_steps: int=1000, 
                 p: float=0.4,
                 print_step: int=10,
		         world: int=0,
                 H_start=None,
                 time_step_correction: int=0,
                 ):
    """
    Generic mcmc sampler 

    """
    timesteps, n_perm, n_codes, h, w = Masks.union.shape
    
    # variables
    mcmc_res = []
    mcmc_likelihoods = []
    predicted_masks = []
    predicted_masks_tensor = torch.zeros(timesteps, n_perm, n_codes, h, w)
    pixel_based_errors = []
    predictions = []

    # ==========SEED===========
    Hypotheses = []
    for _ in range(Masks.codes.shape[0]): # create binary tree hypothesis for each code 
        H = HypothesisTree(1)
        n_expansions = 2 + geom.rvs(p=p, size=1)[0]
        nodes = 2 + rng.choice(1000, size=n_expansions - 2, replace=False)
        for n in nodes:
            H.add_node(n)
        H._traverse(H, data=Bitmask(Masks.bitmasks[0]))
        H.prior = _get_prior(H)
        Hypotheses.append(H)

    if H_start is not None:
        Hypotheses = copy.deepcopy(H_start)

    
    predicted_mask, likelihoods, pixel_based_error, prediction = likelihood(Hypotheses, Masks, time_step_correction=time_step_correction)

    for h_idx, H in enumerate(Hypotheses):
        H.likelihood = copy.deepcopy(likelihoods[:,h_idx])

    marginal_likelihoods = torch.max(likelihoods, dim=0)[0] # argmax over permutations 
    fits = [H.prior + ll for H, ll in zip(Hypotheses, marginal_likelihoods)]

    # ==========SAMPLE===========
    for step in range(mcmc_steps):
        Hypotheses_prime = []
        for h_idx in range(Masks.codes.shape[0]):
            H_prime = TR.regrowth(Hypotheses[h_idx])[0]
            H_prime.prior = _get_prior(H_prime)
            Hypotheses_prime.append(H_prime)
        
        check = False
        while check is False:
            try:
                predicted_mask_prime, likelihoods_prime, pixel_based_error_prime, prediction_prime = likelihood(Hypotheses_prime, Masks, time_step_correction=time_step_correction)
                check = True
            except:
                print('error in h types')
                Hypotheses_prime = []
                for h_idx in range(Masks.codes.shape[0]):
                    H_prime = TR.regrowth(Hypotheses[h_idx])[0]
                    H_prime.prior = _get_prior(H_prime)
                    Hypotheses_prime.append(H_prime)

        for h_prime_idx, H_prime in enumerate(Hypotheses_prime):
            H_prime.likelihood = copy.deepcopy(likelihoods_prime[:,h_prime_idx])

        marginal_likelihoods_prime = torch.max(likelihoods_prime, dim=0)[0]
        fits_prime = [H_prime.prior + ll for H_prime, ll in zip(Hypotheses_prime, marginal_likelihoods_prime)]
        acceptance_probs = torch.tensor([fit_prime - fit for fit_prime, fit in zip(fits_prime, fits)])
        
        # ==========UPDATE===========
        accepted = (acceptance_probs > torch.log(torch.rand(1)[0])).int() 
        
        for i, accept in enumerate(accepted):
            if accept == 1:
                Hypotheses[i] = Hypotheses_prime[i]
                fits[i] = fits_prime[i]
                marginal_likelihoods[i] = marginal_likelihoods_prime[i]
                predicted_mask[:,:,i] = copy.deepcopy(predicted_mask_prime[:,:,i])
                likelihoods[:,i] = likelihoods_prime[:,i]
                pixel_based_error[:,:,i] = copy.deepcopy(pixel_based_error_prime[:,:,i])
                prediction[:,:,i] = copy.deepcopy(prediction_prime[:,:,i])
    
        if (step + 1) % int(mcmc_steps/print_step) == 0:
            mcmc_res.append(copy.deepcopy(Hypotheses))
            predicted_masks.append(copy.deepcopy(predicted_mask))
            mcmc_likelihoods.append(copy.deepcopy(likelihoods))
            pixel_based_errors.append(copy.deepcopy(pixel_based_error))
            predictions.append(copy.deepcopy(prediction))


    predicted_masks_tensor[:] = predicted_masks[-1]
    
    return mcmc_res, mcmc_likelihoods, predicted_masks, predicted_masks_tensor, pixel_based_errors, predictions


def imaginary_mcmc_sampler(Masks: Bitmask, 
                           predicted_masks_tensor: Tensor, 
                           mcmc_steps: int=1000,
                           p: float=0.4,
                           print_step: int=10,
                           ):
    """
    Run sampler on predicted masks to see if it improves fit

    """
    mcmc_res = []
    imaginary_likelihoods = []

    # ==========SEED===========
    Hypotheses = []
    for _ in range(Masks.codes.shape[0]): # create binary tree hypothesis for each code 
        H = HypothesisTree(1)
        n_expansions = 2 + geom.rvs(p=p, size=1)[0]
        nodes = 2 + rng.choice(1000, size=n_expansions - 2, replace=False)
        for n in nodes:
            H.add_node(n)
        H._traverse(H, data=Bitmask(Masks.bitmasks[0]))
        H.prior = _get_prior(H)
        Hypotheses.append(H)

    likelihoods = imaginary_likelihood(Hypotheses, Bitmask(predicted_masks_tensor))

    marginal_likelihoods = torch.max(likelihoods, dim=0)[0]
    fits = [H.prior + ll for H, ll in zip(Hypotheses, marginal_likelihoods)]

    for h_idx, H in enumerate(Hypotheses):
        H.likelihood = copy.deepcopy(likelihoods[:,h_idx])

    # ==========SAMPLE===========
    for step in range(mcmc_steps):
        Hypotheses_prime = []
        for h_idx in range(Masks.codes.shape[0]):
            H_prime = TR.regrowth(Hypotheses[h_idx])[0]
            H_prime.prior = _get_prior(H_prime)
            Hypotheses_prime.append(H_prime)
    

        likelihoods_prime = imaginary_likelihood(Hypotheses_prime, Bitmask(predicted_masks_tensor))

        for h_prime_idx, H_prime in enumerate(Hypotheses_prime):
            H_prime.likelihood = copy.deepcopy(likelihoods_prime[:,h_prime_idx])


        marginal_likelihoods_prime = torch.max(likelihoods_prime, dim=0)[0]
        fits_prime = [H_prime.prior + ll for H_prime, ll in zip(Hypotheses_prime, marginal_likelihoods_prime)]
        acceptance_probs = torch.tensor([fit_prime - fit for fit_prime, fit in zip(fits_prime, fits)])
    
        # ==========UPDATE===========
        accepted = (acceptance_probs > torch.log(torch.rand(1)[0])).int() 

        for i, accept in enumerate(accepted):
            if accept == 1:
                Hypotheses[i] = Hypotheses_prime[i]
                fits[i] = fits_prime[i]
                marginal_likelihoods[i] = marginal_likelihoods_prime[i]
                likelihoods[:,i] = likelihoods_prime[:,i]    

        mcmc_res = [copy.deepcopy(Hypotheses)]
        imaginary_likelihoods = [copy.deepcopy(likelihoods)]
        
    return mcmc_res, imaginary_likelihoods