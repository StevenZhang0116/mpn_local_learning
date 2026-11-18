import numpy as np
import math

import torch
from torch import nn
from scipy.stats import lognorm
from typing import Dict

import helper

# Loss types a network can be configured with (net_params['loss_type']).
LOSS_TYPES = ('XE', 'MSE', 'MSE_grads', 'MSE_weighted', 'CS',
              'rl_direct_output', 'rl_output_deviations')


def rand_weight_init(n_inputs, n_outputs=None, init_type='gaussian', cell_types=None,
                     ei_balance_val=None, sparsity=None, weight_norm=None, self_couplings=True):
    """
    Returns a random weight initialization of the specified type, of size
    (n_inputs,) if n_ouputs is None, otherwise (n_outputs, n_inputs).

    Note: uses numpy throughout, converted to tensor after called

    ei_balance_val: balances strength of excitation and inhibition
    spasity: override sparsity 

    self_couplings: whether or not diagonal couplings can be nonzero

    """
    sparsity_p = 1.0 if sparsity is None else sparsity

    if n_outputs is not None: # 2d case
        weight_shape = (n_outputs, n_inputs,)
    else: # 1d case
        weight_shape = (n_inputs,)

    if init_type == 'xavier':
        if n_outputs is not None: # 2d case
            xavier_bound = np.sqrt(6/(n_inputs + n_outputs)) if weight_norm is None else weight_norm
        else: # 1d case
            xavier_bound = np.sqrt(6/(n_inputs)) if weight_norm is None else weight_norm
        rand_weights = np.random.uniform(low=-xavier_bound, high=xavier_bound, size=weight_shape)
    elif init_type in ('gaussian', 'sparse_gaussian', 'sparse_gaussian_ln_in'):
        norm_factor = 1/np.sqrt(n_inputs) if weight_norm is None else weight_norm
        rand_weights = norm_factor * np.random.normal(scale=1.0, size=weight_shape)
    elif init_type in ('mirror_gaussian',): # Two gaussians peaks centered at opposite means (fixed mean/scale ratio for now)
        norm_factor = 1/np.sqrt(n_inputs) if weight_norm is None else weight_norm
        rand_weights = norm_factor * (
            np.random.choice([-1, 1], size=weight_shape) * np.random.normal(loc=1.0, scale=0.5, size=weight_shape)
        )
    elif init_type in ('log_normal', 'sparse_log_normal'):
        norm_factor = 1/np.sqrt(n_inputs) if weight_norm is None else weight_norm # if init_type == 'log_normal' else n_outputs * sparsity_p
        rand_weights = norm_factor * np.random.lognormal(mean=0.0, sigma=1.0, size=weight_shape)
    elif init_type in ('ones', 'sparse_ones') or type(init_type) == float:
        if n_outputs is not None: # 2d case
            if n_outputs > 1:
                raise NotImplementedError('Normalization is weird for this, should think about it if we use this again')
            norm_factor = 1/np.sqrt(n_inputs) if weight_norm is None else weight_norm #if init_type == 'ones' else n_outputs * sparsity_p
        else: # 1d case
            norm_factor = 1/np.sqrt(n_inputs) if weight_norm is None else weight_norm #if init_type == 'ones' else n_inputs * n_outputs * sparsity_p
        if init_type in ('ones', 'sparse_ones'):
            rand_weights = norm_factor * np.ones(weight_shape)
        else:
            rand_weights = init_type * np.ones(weight_shape)
    elif init_type in ('rand_one_hot',):
        assert len(weight_shape) == 2
        shuffle_idxs = np.random.permutation(max(weight_shape))
        # Shuffle an identity matrix that is the larger of the two weight dimensions
        square_weights = np.eye(max(weight_shape))[shuffle_idxs, :]
        # Clip to appropriate size
        rand_weights = square_weights[:weight_shape[0], :weight_shape[1]]
    elif init_type in ('zeros',):
        rand_weights = np.zeros(weight_shape)
    else:
        raise NotImplementedError('Random weight init type {} not recognized!'.format(init_type))

    # Creates sparsification masks (masks that determine which elements are zero)
    if init_type in ('sparse_gaussian', 'sparse_log_normal', 'sparse_ones', 'sparse_gaussian_ln_in'):

        if init_type in ('sparse_gaussian', 'sparse_ones',): # Note this creates an in-degree distribution that is Gaussian distributed
            weight_mask = np.random.uniform(0, 1, size=rand_weights.shape) > sparsity_p # Which weights to zero
        elif init_type in ('sparse_log_normal', 'sparse_gaussian_ln_in',): 
            LOG_NORM_SHAPE = 0.25 # This could be adjusted to experimental data eventually

            # Adjusts scale relative to standard log normal to set desired mean
            standard_ln = lognorm(s=LOG_NORM_SHAPE)
            scaled_ln = lognorm(s=LOG_NORM_SHAPE, scale=sparsity_p / standard_ln.mean())

            # Draws twice as many as needed since will be truncating all > 1.0
            in_degrees = scaled_ln.rvs(size=2*n_outputs)
            in_degrees = in_degrees[in_degrees <=1.0] # Truncates to <1.0 in degrees

            if in_degrees.shape[0] < n_outputs:
                raise ValueError('Did not draw enough in-degrees that met threshold (only {}).'.format(
                    in_degrees.shape[0]
                ))

            weight_mask = np.zeros(weight_shape, dtype=bool)
            for out_idx in range(n_outputs): # Generates mask one output at a time
                in_degree_mask = np.zeros((n_inputs,), dtype=bool) # True for elements set to zero
                # Set all elements beyond index to be True, corresponding to no connection
                # (small in_degree[out_idx] corresponds to majority 1s)
                in_degree_mask[int(np.floor(in_degrees[out_idx] * n_inputs)):] = True
                np.random.shuffle(in_degree_mask)

                weight_mask[out_idx] = in_degree_mask

        if not self_couplings: # Sets all diagonal elements of mask so weights are set to zero
            assert n_inputs == n_outputs
            for neuron_idx in range(n_inputs):
                weight_mask[neuron_idx, neuron_idx] = True

        rand_weights[weight_mask] = 0.0

    if (cell_types is not None) and (n_outputs is not None): # Assigns cell types
        cell_types = cell_types.detach().cpu().numpy()
        assert cell_types.shape[0] == 1
        assert cell_types.shape[1] == n_inputs

        if ei_balance_val is not None:
             # Signed based on cell type, but also enhances strength appropriately
            cell_weights = np.where(cell_types < 0, ei_balance_val * cell_types, cell_types)
        else:
            cell_weights = cell_types # Just 1s or -1s to correct sign

        rand_weights = cell_weights * np.abs(rand_weights)

    # print(' Init type:', init_type)
    # print(' Cell types:', cell_types)
    # perc_pos = np.sum(rand_weights > 0) / np.prod(rand_weights.shape)
    # perc_neg = np.sum(rand_weights < 0) / np.prod(rand_weights.shape)
    # print(' Perc pos: {:.2f} perc neg: {:.2f}'.format(perc_pos, perc_neg))

    return rand_weights


def relu_fn(x):
    return torch.maximum(x, torch.zeros_like(x))
def relu_fn_np(x):
    return np.maximum(x, np.zeros_like(x))
def relu_fn_p(x):
    return torch.heaviside(x, torch.zeros_like(x)) # For x = 0, return 0 just like pytorch default

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))    
def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_p(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

def tanh_p(x):
    return (1/torch.cosh(x))**2 # Sech**2

def tanh_re(x):
    return torch.maximum(torch.tanh(x), torch.zeros_like(x))
def tanh_re_np(x):
    return np.maximum(np.tanh(x), np.zeros_like(x))    
def tanh_re_p(x): # Have to use where here instead of maximum because we want this to return 0 for x = 0, like ReLU (and sech(0) = 1)
    return torch.where(x > 1e-6, (1/torch.cosh(x))**2, 0.) # Sech**2

def tanh_re_super(x, alpha=2.0):
    return torch.maximum(torch.tanh(alpha*x), torch.zeros_like(x))
def tanh_re_super_np(x, alpha=2.0):
    return np.maximum(np.tanh(alpha*x), np.zeros_like(x))    
def tanh_re_super_p(x): # Have to use where here instead of maximum because we want this to return 0 for x = 0, like ReLU (and sech(0) = 1)
    raise NotImplementedError()
    return torch.where(x > 0., (1/torch.cosh(x))**2, 0.) # Sech**2

def linear_fn(x):
    return x     
def linear_fn_p(x):
    return torch.ones_like(x)

def cubed_re(x):
    return torch.maximum(x**3, torch.zeros_like(x))
def cubed_re_p(x):
    return torch.maximum(3*x**2, torch.zeros_like(x))

def tukey_fn(x):
    raw = 1/2 - 1/2 * torch.cos(np.pi * x) 
    raw[x < 0.] = 0.0
    raw[x > 1.] = 1.0
    return raw
def tukey_fn_np(x):
    raw = 1/2 - 1/2 * np.cos(np.pi * x) 
    raw[x < 0.] = 0.0
    raw[x > 1.] = 1.0
    return raw
def tukey_fn_p(x):
    raw = np.pi / 2 * torch.sin(np.pi * x)
    raw[x < 0.] = 0.0
    raw[x > 1.] = 0.0

def heaviside_p(x):
    raise NotImplementedError()

# 2025-11-17: add softplus function to match with LD's paper
def softplus_fn(x):
    return torch.nn.functional.softplus(x)

def softplus_fn_np(x):
    # stable: log(1 + exp(x))
    return np.log1p(np.exp(x))

def softplus_fn_p(x):
    # derivative of softplus is sigmoid(x)
    return torch.sigmoid(x)

def get_activation_function(act_fn):
    """ Returns pytorch version, numpy version, and pytorch derivative functions """
    if act_fn == 'ReLU':
        return relu_fn, relu_fn_np, relu_fn_p
    elif act_fn == 'sigmoid':
        return sigmoid, sigmoid_np, sigmoid_p
    elif act_fn == 'tanh_re':
        return tanh_re, tanh_re_np, tanh_re_p
    elif act_fn == 'tanh_re_super': # Supra linear version of Tanh
        return tanh_re_super, tanh_re_super_np, tanh_re_super_p
    elif act_fn == 'tanh':
        return torch.tanh, np.tanh, tanh_p
    elif act_fn == 'tukey':
        return tukey_fn, tukey_fn_np, tukey_fn_p
    elif act_fn == 'linear':
        return linear_fn, linear_fn, linear_fn_p
    elif act_fn == 'cubed_re':
        return cubed_re, None, cubed_re_p
    elif act_fn == 'heaviside':
        return lambda x : torch.heaviside(x, torch.tensor(0.5)), lambda x : np.heaviside(x, 0.5), heaviside_p
    elif act_fn == 'softplus':
        return softplus_fn, softplus_fn_np, softplus_fn_p
    else:
        raise ValueError('Activation function: {} not recoginized!'.format(act_fn))


def round_to_values_torch(array, round_vals):
    """ Round all elements of an array to closest value in round_vals, pytorch version """

    assert len(round_vals.shape) == 1

    round_vals_us = torch.clone(round_vals) # Copy of round_vals to be unsqueezed
    dims_unsqueeze = len(array.shape) # How many times to unsqueeze based on array shape
    for _ in range(dims_unsqueeze):
        round_vals_us = round_vals_us.unsqueeze(0)
    dists = torch.abs(
        array.unsqueeze(-1) - round_vals_us
    )

    return round_vals[torch.argmin(dists, axis=-1)]

def one_hot_argidx(x: torch.Tensor) -> torch.Tensor:
    """
    """
    if x.dim() != 3:
        raise ValueError(f'Input must be 3-D, got {x.shape}')

    # argmax gives the index of the sole “1” along D
    return torch.argmax(x, dim=-1)


def mean_by_group(A: torch.Tensor, B: torch.Tensor) -> Dict[int, float]:
    """
    Compute the mean of A for each integer label in B.

    Parameters
    ----------
    A : (N,) or (N,1) tensor
        Numeric values whose means you want.
    B : (N,) or (N,1) tensor of ints
        Integer labels; same length as A.

    Returns
    -------
    dict {label: mean_A_for_that_label}
    """
    # --- sanity checks ------------------------------------------------------
    if A.numel() != B.numel():
        raise ValueError(f"A and B must have the same number of elements "
                         f"(got {A.numel()} vs {B.numel()}).")
    # flatten to 1-D
    A = A.squeeze()
    B = B.squeeze().to(torch.int64)

    # --- fast vectorised solution using torch.bincount ----------------------
    # works if labels are non-negative ints (they don’t have to be contiguous)
    max_label = int(B.max())
    if B.min() < 0:
        raise ValueError("B must contain non-negative integers for bincount-based method.")

    sums   = torch.bincount(B, weights=A, minlength=max_label + 1)
    counts = torch.bincount(B, minlength=max_label + 1)

    # Avoid division by zero: mask out labels that never occur
    valid = counts > 0
    means = torch.zeros_like(sums, dtype=A.dtype)
    means[valid] = sums[valid] / counts[valid]

    # Build dict only for labels present in B
    return {int(label): float(means[label]) for label in torch.unique(B)}

##############################################################################################
##############################################################################################
##############################################################################################

class BaseNetworkFunctions(nn.Module):
    """
    Functions that may be used in networks and subnetwork layers. 

    Separate from BaseNetwork so that certain subnetwork layers can use these functions without having
    to fully init a new network each time.
    """

    def __init__(self):

        super().__init__()


    def parameter_or_buffer(self, name, tensor=None, verbose=False):
        """
        Helper function to register certain variables as parameters or buffers in networks.
        Also has capability of updating prameters to buffers or vice versa, in which case 
        a tensor value does not need to be passed

        If "name" is in the "self.params" tuple, then registers as parameter, otherwise buffer.
        """

        # If the attribute already exists, copy it to tensor then delete it
        update_name = True

        if hasattr(self, name): # Automatrically update if it doesnt exist, otherwise some checks

            # Find out if this is a parameter or buffer (probably a better way to do this)
            param = False
            buffer = False
            for buff_name, _ in self.named_buffers():
                if buff_name == name:
                    buffer = True
            for param_name, _ in self.named_parameters():
                if param_name == name:
                    param = True

            assert not (buffer and param)

            # If already the correct type
            if (buffer and (name not in self.params)) or (param and (name in self.params)):
                update_name = False
            else:
                if verbose: print('Updating atribute: ', name)
                tensor = getattr(self, name).detach().clone()
                delattr(self, name)

        if update_name:
            if name in self.params:
                self.register_parameter(name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)


class BaseNetwork(BaseNetworkFunctions):

    def __init__(self, net_params, verbose=False):

        super().__init__()
    
        self.loss_type = net_params.get('loss_type', 'XE')
        self.acc_measure = net_params.get('acc_measure', 'angle')

        # The loss itself is supplied by the gradient routines (mpn / rnn / gru:
        # masked MSE by default, or the task's loss_and_grad); loss_type only
        # selects how compute_acc scores outputs and whether prefs are stored.
        if self.loss_type not in LOSS_TYPES:
            raise ValueError('Loss type {} not recognized.'.format(self.loss_type))

        # Prefs can be used to compute approximate accuracy in loss setups where accuracy is not well-defined
        if self.loss_type in ('MSE', 'MSE_grads', 'MSE_weighted', 'CS',):
            if net_params.get('prefs', None) is not None:
                self.register_buffer('prefs', torch.tensor(net_params.get('prefs')))
            else:
                self.prefs = None

        self.verbose = verbose
        self.param_clamping = False # Will be overridden if needed.

        # Time difference between discrete steps, used to set biologically-realistic parameters
        self.dt = net_params.get('dt', 40) # units of ms

    def compute_acc(self, output, labels, output_mask, input_, round_type='prefs', mode="angle", isvalid=False):
        """
        output shape: (batches, seq_len, output_size)
        labels shape: (batches, seq_len)
        output_mask shape: (batches, seq_len, output_size)
        """		
        # batches * seq_len * output_size
        output_mask_alter = torch.full(output_mask.shape, float('nan'), device=output_mask.device) 

        if isvalid:
            # select the input part that is task related
            # under assumption for which the fixation off signal is provided (task_params)
            # 2026-02-04: modified to adapt when fixation off signal is not provided
            fixon = input_[:,:,0]
            fixoff_candidate = input_[:,:,1]
            s = fixon + fixoff_candidate
            fixoff_add = torch.allclose(
                s, torch.ones_like(s), rtol=1e-5, atol=1e-6
            )
            
            task_mask_truc = input_[:,:,6-abs(1-fixoff_add):] 

            task_mask = one_hot_argidx(task_mask_truc)
            
            output_mask_alter_taskalign = torch.full(output_mask_alter.shape, 
                                                     float('nan'), 
                                                     device=output_mask_alter.device)
            
            assert task_mask.shape[0] == output_mask_alter_taskalign.shape[0]
            
            for batch_iter in range(task_mask.shape[0]):
                # initial timestamp (definitely no paddling)
                task_label = task_mask[:,0][batch_iter]
                # broadcast the task information according to the output mask label
                output_mask_alter_taskalign[batch_iter,:,:] = task_label
                
        for batch_iter in range(output_mask.shape[0]):
            # identify chunks of zeros 
            # 1st: first 100ms; 2nd: ideally fix_offs and check_ons; 3rd: end of trail (paddling)
            ll = helper.find_zero_chunks(output_mask[batch_iter, :, :].clone().cpu().detach().numpy())
        
            if len(ll) < 3: # for delay task...
                ll.append([output_mask.shape[1], output_mask.shape[1]])
            # only focusing on the alignment during response (go) period; always placed at the final
            response_start, response_end = ll[-2][1] + 1, ll[-1][0]

            # ignore cut_proportion at the beginning of the response period
            # 2025-11-16: is 1/4 reasonable enough? 
            cut_proportion = 1/4
            response_duration = response_end - response_start
            response_start = response_start + int(response_duration * cut_proportion)

            output_part = output_mask[batch_iter, response_start:response_end, :]
            # originally the mask is not binary, but scaled based on different period to let the cost function work properly
            # that makes sense since we do not want to over-emphasize on periods e.g. fixon and delay
            # when calculating MSE loss
            output_part = (output_part > 0).int()

            output_mask_alter[batch_iter, response_start:response_end, :] = output_part

        assert self.loss_type in ('XE', 'MSE',)

        if self.loss_type in ('XE',):
            # Modify output by the mask (note just uses first output idx to mask)   
            masked_output = output[output_mask[:, :, 0]] # [B, T, out] -> [B*T_mask, out]
            masked_labels = labels[output_mask[:, :, 0]].squeeze(-1) # [B, T, 1] -> [B*T_mask, 1] -> [B*T_mask]
            return (torch.argmax(masked_output, dim=-1) == masked_labels).float().mean() 

        elif self.loss_type in ('MSE',):
            if output_mask.dtype == torch.bool:
                # Modify output by the mask (note just uses first output idx to mask)   
                masked_output = output[output_mask[:, :, 0]] # [B, T, 1] -> [B*T_mask, 1]
                masked_labels = labels[output_mask[:, :, 0]] # [B, T, 1] -> [B*T_mask, 1]
            elif output_mask.dtype == torch.float32:# Mask by some cost function instead
                masked_output = (output_mask_alter * output).reshape((-1, output.shape[-1])) # [B, T, out] -> [B*T, out]
                masked_labels = (output_mask_alter * labels).reshape((-1, output.shape[-1])) # [B, T, label_size] -> [B*T, label_size]
                if isvalid:
                    masked_tasks = (output_mask_alter * output_mask_alter_taskalign).reshape((-1, output.shape[-1]))
                    
            else:
                raise ValueError('Mask type {} not recognized.'.format(output_mask.dtype))

            if round_type in ('prefs',) and self.prefs is not None: # Round both the labels and the ouputs to self.prefs
                # compare the alignment by angle (relative position between two stimulus)
                if mode == "angle": 
                    # July 6th: this value is chosen based on testing dmsgo/dmcgo
                    mag_eps = 5e-2
                    
                    mo = masked_output[~torch.all(torch.isnan(masked_output), dim=1)]
                    ml = masked_labels[~torch.all(torch.isnan(masked_labels), dim=1)]
                    
                    if isvalid:
                        mtask = masked_tasks[~torch.all(torch.isnan(masked_tasks), dim=1)][:,0]
                        # print(torch.unique(mtask))

                    # round small magnitude
                    sin_vals = mo[:, 1]
                    cos_vals = mo[:, 2]
                    
                    tiny_sin = torch.abs(sin_vals) < mag_eps
                    tiny_cos = torch.abs(cos_vals) < mag_eps

                    # rounding to 0 only happens during calculating the accuracy
                    # so the training process will not be impacted
                    mo[:, 1] = torch.where(tiny_sin, torch.zeros_like(sin_vals), sin_vals)
                    mo[:, 2] = torch.where(tiny_cos, torch.zeros_like(cos_vals), cos_vals)

                    # 
                    mo_angle = torch.remainder(torch.atan2(mo[:, 1], mo[:, 2]), 2 * math.pi).reshape(-1, 1)
                    ml_angle = torch.remainder(torch.atan2(ml[:, 1], ml[:, 2]), 2 * math.pi).reshape(-1, 1)
                
                    append_pref = torch.tensor([2 * math.pi], device=self.prefs.device)
                    temp_prefs = torch.cat((self.prefs, append_pref), dim=0)

                    masked_output_round = round_to_values_torch(mo_angle, temp_prefs) 
                    masked_labels_round = round_to_values_torch(ml_angle, temp_prefs)
    
                    def clean(x: torch.Tensor, tol: float = 1e-6):
                        """Replace entries equal (within tol) to 2π with 0, in-place."""
                        two_pi = torch.tensor(2 * math.pi, device=x.device, dtype=x.dtype)
                        x.masked_fill_(torch.isclose(x, two_pi, atol=tol), 0.)
                        return x   
    
                    masked_output_round = clean(masked_output_round, tol=1e-3)
                    masked_labels_round = clean(masked_labels_round, tol=1e-3)
 
                    result_acc_ = (masked_output_round == masked_labels_round).float()
                    result_acc = result_acc_.mean() 

                # compare the alignment directly by stimulus 
                # this is technically a "unnecessarily stricter" version of the angle comparison
                # and is expected to be sensitive to noise 
                # July 1st: this might be more stable for tasks with overwhelmed 0 response, like dmsgo & dmcgo
                elif mode == "stimulus": 
                    mo = masked_output[~torch.all(torch.isnan(masked_output), dim=1)]
                    ml = masked_labels[~torch.all(torch.isnan(masked_labels), dim=1)]

                    if isvalid:
                        mtask = masked_tasks[~torch.all(torch.isnan(masked_tasks), dim=1)][:,0]

                    # sanity check
                    # make sure the angle between response during the truncated response period has a perfect 
                    # match with the pool 
                    append_pref = torch.tensor([2 * math.pi], device=self.prefs.device)
                    temp_prefs = torch.cat((self.prefs, append_pref), dim=0)
                    
                    ml_angle = torch.remainder(torch.atan2(ml[:, 1], ml[:, 2]), 2 * math.pi).reshape(-1, 1)

                    ml_angle = ml_angle.to(temp_prefs.dtype)

                    tol = 1e-6
                    matches = torch.isclose(                   
                        ml_angle.unsqueeze(-1),                 
                        temp_prefs,                              
                        atol=tol, rtol=0.0
                    )
                    
                    assert matches.any(dim=-1).all(), "Some angles are not within tol of any preference value"
                    
                    # this function is needed since self.prefs is rounded to decimals
                    def unique_with_tolerance(tensor, tolerance=1e-5):
                        # Scale by tolerance, round, then divide back to avoid floating-point precision issues
                        scaled = torch.round(tensor / tolerance) * tolerance
                        return torch.unique(scaled)

                    sines  = unique_with_tolerance(torch.sin(self.prefs)) 
                    cosines = unique_with_tolerance(torch.cos(self.prefs))
                    # round the output response to the possible decomposed stimulus 
                    # though 8 angles, only 5 possible stimuli for sin & cos (+-1, 0, +-\sqrt{2}/2)
                    output_stimulus1, output_stimulus2 = round_to_values_torch(mo[:, 1], sines), round_to_values_torch(mo[:, 2], cosines)
                    labels_stimulus1, labels_stimulus2 = round_to_values_torch(ml[:, 1], sines), round_to_values_torch(ml[:, 2], cosines) 
 
                    match1 = (output_stimulus1 == labels_stimulus1).int()
                    match2 = (output_stimulus2 == labels_stimulus2).int()
                    # only correct if both stimulus are matched with the label, therefore equivalent to the angle comparison
                    result_acc_ = ((match1 == 1) & (match2 == 1)).float()
  
                    result_acc = result_acc_.mean()
 
                else: 
                    raise ValueError('Mode {} not recognized.'.format(mode))

            elif round_type in ('int',):
                masked_output_round = torch.round(masked_output)
                masked_labels_round = masked_labels
            else:
                raise ValueError(f'Round type {round_type} not recognized or invalid.')

            acc_per_task = None 
            if isvalid:
                acc_per_task = mean_by_group(result_acc_, mtask)
                
            return result_acc, acc_per_task


    @torch.no_grad()
    def param_clamp(self):

        for p_name, p in self.named_parameters():
        
            if hasattr(self, p_name + '_bounds'): # If bounds exist, enforce them
                bounds = getattr(self, p_name + '_bounds')
                p.data.clamp_(bounds[0], bounds[1])
