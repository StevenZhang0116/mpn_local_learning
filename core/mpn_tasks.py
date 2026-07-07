"""Collections of tasks."""

from __future__ import division
import six
import numpy as np

import torch
import net_helpers
import time
import random
import sys

def get_rules(ruleset):
    return rules_dict[ruleset]

# def get_num_ring(ruleset):
#     '''get number of stimulus rings'''
#     return 3 if ruleset=='oicdmc' else 2

def get_num_rule(ruleset):
    '''get number of rules'''
    return len(rules_dict[ruleset])

def get_rule_index(rule, config):
    '''get the input index for the given rule'''
    return rule_index_map[config['ruleset']][rule]+config['rule_start']

def get_prefs(config):
    """ Perferred locations given the config, used in trial and also used for computing accuracy"""
    return np.arange(0, 2*np.pi, 2*np.pi / config['n_eachring']) # preferences dependent on n_eachring

def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))

def get_clostest_pref_idx(locs, prefs):
    """ Get the preference idx that the distance is closest to,
    Args:
        locs.shape = (n_batch)
        prefs.shape = (n_prefs)

    """
    return np.argmin(get_dist(locs[:, np.newaxis] - prefs[np.newaxis, :]), axis=-1) # Smallest distance

def pos_circle_4d(a):
    """
    Makes Laura's mapping positive definite
    """
    sin_a = np.sin(a)
    cos_a = np.cos(a)

    return np.array((cos_a * (cos_a > 0), -cos_a * (cos_a < 0),
                     sin_a * (sin_a > 0), -sin_a * (sin_a < 0)))

def batch_choice(options, batch_size, rng, dt=None):
    """ Batched version of rng.choice for variability across batches """
    if dt is None:
        return rng.choice(options, size=(batch_size,))
    else:
        return (rng.choice(options, size=(batch_size,))/dt).astype(np.int32)

class Trial(object):
    """Class representing a batch of trials."""

    def __init__(self, config, tdim, batch_size):
        """A batch of trials.

        Args:
            config: dictionary of configurations
            tdim: int or numpy array of batch size, number of time steps
            batch_size: int, batch size
        """
        self.float_type = 'float32' # This should be the default
        self.config = config
        self.dt = self.config['dt']
        self.in_out_mode = self.config['in_out_mode']
        self.fixate_off = self.config['fixate_off'] # Additional signal when fixation turns off
        self.n_fixate = 2 if self.fixate_off else 1

        if self.in_out_mode in ('high_dim',):
            self.n_eachring = self.config['n_eachring'] # This is the dimension of each ring input
        elif self.in_out_mode in ('low_dim', 'low_dim_pos'):
            self.n_eachring = 2 if self.in_out_mode == 'low_dim' else 4
        self.pref = get_prefs(self.config) # preferences dependent on config['n_eachring']
        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']

        self.batch_size = batch_size
        self.tdim = tdim # Individual batched t_dims
        self.tdim_max = np.max(self.tdim)
        self.x = np.zeros((self.tdim_max, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((self.tdim_max, batch_size, self.n_output), dtype=self.float_type)
        # if self.config['loss_type'] == 'lsq':
        #     self.y[:,:,:] = 0.05
        # y_loc is the stimulus location of the output, -1 for fixation, (0,2 pi) for response
        self.y_loc = -np.ones((self.tdim_max, batch_size), dtype=self.float_type)

        self._sigma_x = config['sigma_x']*np.sqrt(2/config['alpha'])

        # Saves metadata of task for easier analysis/labelling
        self.get_meta = self.config['get_meta']
        if self.get_meta:
            self.meta = {}
        else:
            self.meta = None

    def expand(self, var):
        """Expand an int/float to list."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1, mods=None):
        """ Add an input or stimulus output.

        Args:
            loc_type: str (fix_in, stim, fix_out, out), type of information to be added
            locs: array of list of float (batch_size,), locations to be added, only for loc_type=stim or out
            ons: int or list, index of onset time of type of information
            offs: int or list, index of offset time of type of information
            strengths: float or list, strength of input or target output
            mods: int or list, modalities of input or target output
        """

        # Automatically handles conversion of ints to lists
        ons = self.expand(ons)
        offs = self.expand(offs)
        strengths = self.expand(strengths)
        mods = self.expand(mods)

        for i in range(self.batch_size):
            if loc_type == 'fix_in':
                self.x[ons[i]: offs[i], i, 0] = 1
                if self.fixate_off: # Signal when fixation is off
                    self.x[offs[i]:, i, 1] = 1
            elif loc_type == 'stim':
                # Assuming that mods[i] starts from 1
                self.x[ons[i]: offs[i], i, self.n_fixate+(mods[i]-1)*self.n_eachring:self.n_fixate+mods[i]*self.n_eachring] \
                    += self.add_x_loc(locs[i])*strengths[i]
            elif loc_type == 'fix_out':
                # Just set this to 1.0, unlike Robert's code we don't use an output activation
                self.y[ons[i]: offs[i], i, 0] = 1.0
            elif loc_type == 'out':
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]: offs[i], i, 1:] += self.add_y_loc(locs[i])*strengths[i]
                else:
                    y_tmp = self.add_y_loc(locs[i])
                    y_tmp /= np.sum(y_tmp)
                    self.y[ons[i]: offs[i], i, 1:] += y_tmp
                self.y_loc[ons[i]: offs[i], i] = locs[i]
            else:
                raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        """Add input noise."""
        self.x += self.config['rng'].randn(*self.x.shape)*self._sigma_x

    def add_c_mask(self, pre_offs, post_ons, post_offs=None):
        """Add a cost mask.

        Usually there are two periods, pre and post response
        Scale the mask weight for the post period so in total it's as important
        as the pre period
        """

        pre_on   = int(100/self.dt) # never check the first 100ms
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)
        if post_offs is not None:
            post_offs = self.expand(post_offs)

        if self.config['loss_type'] == 'lsq':
            c_mask = np.zeros((self.tdim_max, self.batch_size, self.n_output), dtype=self.float_type)

            for i in range(self.batch_size):
                if post_offs is None: # When post response periods just go to the end of the task
                    c_mask[post_ons[i]:, i, :] = 5.
                else:
                    c_mask[post_ons[i]:post_offs[i], i, :] = 5.
                # Pre-response periods usually have different lengths across tasks
                # To keep cost comparable across tasks
                # Scale the cost mask of the pre-response period by a factor
                c_mask[pre_on:pre_offs[i], i, :] = 1.

            # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
            # c_mask[:, :, :self.n_fixate] *= 2. # Fixation is important
            # c_mask[:, :, :self.n_fixate] *= 0.5 # Fixation is trivial
            c_mask[:,:,0] *= 0.5

            # Reshaping is done in code, so not needed here.
            # self.c_mask = c_mask.reshape((self.tdim_max*self.batch_size, self.n_output))

            # Since this directly modifies the labels/output, to be equivalent after MSE need a sqrt
            self.c_mask = np.sqrt(c_mask)
        else:
            c_mask = np.zeros((self.tdim_max, self.batch_size), dtype=self.float_type)
            for i in range(self.batch_size):
                # Post response periods usually have the same length across tasks
                # Having it larger than 1 encourages the network to achieve higher performance
                c_mask[post_ons[i]:, i] = 5.
                # Pre-response periods usually have different lengths across tasks
                # To keep cost comparable across tasks
                # Scale the cost mask of the pre-response period by a factor
                c_mask[pre_on:pre_offs[i], i] = 1.

            self.c_mask = c_mask.reshape((self.tdim_max*self.batch_size,))
            self.c_mask /= self.c_mask.mean()

    def add_rule(self, rule, on=None, off=None, strength=1.):
        """ Add rule input. """
        if isinstance(rule, int):
            self.x[on:off, :, self.config['rule_start']+rule] = strength
        else:
            # ind_rule = get_rule_index(rule, self.config)
            ind_rule = self.config['rules'].index(rule)
            self.x[on:off, :, self.config['rule_start']+ind_rule] = strength

    def add_x_loc(self, x_loc):
        """Input activity given location."""
        dist = get_dist(x_loc-self.pref)  # periodic boundary
        if self.in_out_mode in ('high_dim',):
            dist /= np.pi/8
            return 0.8*np.exp(-dist**2/2)
        elif self.in_out_mode in ('low_dim', 'low_dim_pos',):
            closest_pref = self.pref[np.argmin(dist)] # Smallest distance
            if self.in_out_mode in ('low_dim',):
                return np.array((np.sin(closest_pref), np.cos(closest_pref)))
            elif self.in_out_mode in ('low_dim_pos',):
                return pos_circle_4d(closest_pref)

    def add_y_loc(self, y_loc):
        """Target response given location."""
        dist = get_dist(y_loc-self.pref)  # periodic boundary
        if self.in_out_mode in ('high_dim',):
            if self.config['loss_type'] == 'lsq':
                dist /= np.pi/8
                y = 0.8*np.exp(-dist**2/2)
            else:
                # One-hot output
                y = np.zeros_like(dist)
                ind = np.argmin(dist)
                y[ind] = 1.
            return y
        elif self.in_out_mode in ('low_dim', 'low_dim_pos',):
            closest_pref_idx = np.argmin(dist) # Smallest distance
            if self.config['loss_type'] == 'lsq':
                closest_pref = self.pref[closest_pref_idx]
                if self.in_out_mode in ('low_dim',):
                    y = np.array((np.sin(closest_pref), np.cos(closest_pref)))
                elif self.in_out_mode in ('low_dim_pos',):
                    y = pos_circle_4d(closest_pref)
            else:
                # One-hot output
                y = np.zeros_like(dist)
                y[closest_pref_idx] = 1.
            return y

def test_init(config, mode, **kwargs):
    '''
    Test initialization of model. mode is not actually used
    Fixation is on then off.
    '''
    dt = config['dt']
    tdim = int(10000/dt)
    fix_offs  = [int(800/dt)]
    batch_size = 1

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)

    return trial


def delaygo_(config, mode, anti_response, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    '''
    Fixate whenever fixation point is shown,
    saccade to the location of the previously shown stimulus
    whenever the fixation point is off
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The stimulus is shown between (stim_on, stim_off)

    The output should be fixation location for (0, fix_off)
    and the stimulus location for (fix_off, T)

    :param mode: the mode of generating. Options: 'random', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    
    if mode == 'random': # Randomly generate parameters, but uniform times/modalities across batch
        batch_size = kwargs['batch_size']
        # A list of locations of stimulus and on/off time
        stim_locs = rng.rand(batch_size)*2*np.pi
        if long_fixation == "normal":
            stim_ons = int(rng.uniform(300,700)/dt)
        elif long_fixation == "long":
            stim_ons = int(rng.choice([10000])/dt)
        if long_all == True: # overwrite
            stim_ons = int(700/dt)
        
        if long_stimulus == "normal":
            stim_offs = stim_ons + int(rng.uniform(200,1600)/dt)
        elif long_stimulus == "long":
            stim_offs = stim_ons + int(rng.choice([10000])/dt)
        if long_all == True: # overwrite
            stim_offs = stim_ons + int(1600/dt)
        
        if long_delay == "normal":
            fix_offs = stim_offs + int(rng.uniform(200, 1600)/dt)
        elif long_delay == "long":
            fix_offs = stim_offs + int(rng.choice([10000])/dt)
        if long_all == True: # overwrite
            fix_offs = stim_offs + int(1600/dt)

        if long_response == "normal": 
            tdim     = fix_offs + int(rng.uniform(300, 700)/dt)
        elif long_response == "long": 
            tdim     = fix_offs + int(rng.choice([10000])/dt)
        if long_all == True: # overwrite
            tdim     = fix_offs + int(700/dt)
            
        stim_mod  = rng.choice([1,2])
        
    elif mode == 'random_batch': # Randomly generate parameters, times/modalities different across batch
        batch_size = kwargs['batch_size']

        # A list of locations of stimuluss and on/off time
        stim_locs = rng.rand(batch_size)*2*np.pi
        if long_fixation == "normal":
            stim_ons = (rng.uniform(300,700, size=(batch_size,))/dt).astype(np.int32)
        elif long_fixation == "long":
            stim_ons = (rng.choice([10000], size=(batch_size,))/dt).astype(np.int32)
        if long_stimulus == "normal":
            stim_offs = stim_ons + (rng.uniform(200,1600, size=(batch_size,))/dt).astype(np.int32)
        elif long_stimulus == "long":
            stim_offs = stim_ons + (rng.choice([10000], size=(batch_size,))/dt).astype(np.int32)
        
        if long_delay == "normal": 
            fix_offs = stim_offs + (rng.uniform(200,1600, size=(batch_size,))/dt).astype(np.int32)
        elif long_delay == "long":
            fix_offs = stim_offs + batch_choice([10000], batch_size, rng, dt)

        if long_response == "normal": 
            tdim     = fix_offs + (rng.uniform(300,700, size=(batch_size,))/dt).astype(np.int32)
        elif long_response == "long": 
            tdim     = fix_offs + batch_choice([10000], batch_size, rng, dt)
            
        stim_mod  = batch_choice([1, 2], batch_size, rng)
        
    elif mode == 'test':
        tdim = int(2500/dt)
        n_stim_loc, n_stim_mod = batch_shape = 20, 2
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod = np.unravel_index(range(batch_size),batch_shape)

        fix_offs  = int(2000/dt)
        stim_locs  = 2*np.pi*ind_stim_loc/n_stim_loc
        stim_ons   = int(500/dt)
        stim_mod   = ind_stim_mod + 1
        stim_offs  = int(1000/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim_locs = p['stim_locs']
        # Time of stimuluss on/off
        stim_ons    = int(p['stim_ons']/dt)
        stim_offs   = int(p['stim_offs']/dt)
        delay_time = int(p['delay_time']/dt)
        fix_offs   = stim_offs + delay_time
        tdim       = int(400/dt) + fix_offs
        stim_mod    = 1

        batch_size = len(stim_locs)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100/dt)

    # Response locations
    stim_locs = np.array(stim_locs)

    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs+np.pi)%(2*np.pi)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs, offs=tdim)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons, post_offs=tdim)

    trial.epochs = {'fix1'     : (None, stim_ons),
                   'stim1'     : (stim_ons, stim_offs),
                   'delay1'   : (stim_offs, fix_offs),
                   'go1'      : (fix_offs, tdim)}

    if trial.get_meta:
        trial.meta = {
            'stim1': get_clostest_pref_idx(stim_locs, trial.pref),
            'resp1': get_clostest_pref_idx(response_locs, trial.pref),
        }

    return trial


def delaygo(config, mode, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    return delaygo_(config, mode, False, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation,long_all, **kwargs)

def delayanti(config, mode, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    return delaygo_(config, mode, True, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs)
    
def contextdm_genstim(batch_size, rng, stim_coh_range=None):
    stim_mean = rng.uniform(0.8, 1.2, (batch_size,))
    if stim_coh_range is None:
        stim_coh_range = np.array([0.16, 0.32, 0.64])*1.0
    stim_coh  = rng.choice(stim_coh_range, (batch_size,))
    stim_sign = rng.choice([+1, -1], (batch_size,))
    stim1_strengths = stim_mean + stim_coh*stim_sign
    stim2_strengths = stim_mean - stim_coh*stim_sign
    return stim1_strengths, stim2_strengths

def _contextdm(config, mode, attend_mod, separate_input, label_strength, long_delay, long_stimulus, long_fixation, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    Two stimuluss are shown in each ring,
    Saccade to the one with higher intensity for the attended ring
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The two stimuluss is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger stimulus

    In this task, if the model's strategy is to ignore context, and integrate both,
    then the maximum performance is 75%. So we need to make the highest correct performance
    much higher than that.

    :param mode: the mode of generating. Options: 'random', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''

    # Certain rules are incompatible with this mode because can't have multiple stimuli on a ringe at once
    # assert self.in_out_mode not in ('low_dim',)

    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of stimuluss, same locations for both modalities
        stim_dist = rng.uniform(0.5*np.pi, 1.5*np.pi,(batch_size,))*rng.choice([-1,1],(batch_size,))
        stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim2_locs = (stim1_locs+stim_dist)%(2*np.pi)

        stim_coh_range = np.array([0.01, 0.02, 0.04, 0.08])
        if ('easy_task' in config) and config['easy_task']:
            # stim_coh_range = np.array([0.1, 0.2, 0.4, 0.8])
            stim_coh_range *= 10

        if (attend_mod == 1) or (attend_mod == 2):
            stim1_mod1_strengths, stim2_mod1_strengths = contextdm_genstim(batch_size, rng, stim_coh_range)
            stim1_mod2_strengths, stim2_mod2_strengths = contextdm_genstim(batch_size, rng, stim_coh_range)
            if attend_mod == 1:
                stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
            else:
                stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
        else:
            stim1_strengths, stim2_strengths = contextdm_genstim(batch_size, rng, stim_coh_range)

            stim1_mod12_diff = stim1_strengths * \
                               np.random.uniform(0.2, 0.8, (batch_size,)) * \
                               np.random.choice([+1, -1], (batch_size,))
            stim1_mod1_strengths = stim1_strengths + stim1_mod12_diff/2
            stim1_mod2_strengths = stim1_strengths - stim1_mod12_diff/2

            stim2_mod12_diff = stim2_strengths * \
                               np.random.uniform(0.2, 0.8, (batch_size,)) * \
                               np.random.choice([+1, -1], (batch_size,))
            stim2_mod1_strengths = stim2_strengths + stim2_mod12_diff/2
            stim2_mod2_strengths = stim2_strengths - stim2_mod12_diff/2

        # Time of stimuluss on/off
        stim_on = int(rng.uniform(100,400)/dt)
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)
        stim_dur = int(rng.choice([400, 800, 1600])/dt)
        stim_offs = stim_ons+stim_dur

        # delay_dur = rng.choice((np.array([200, 400, 800])/dt).astype(int)) # Current setting
        delay_dur = 0
        fix_offs  = stim_offs + delay_dur

        # each batch consists of sequences of equal length
        tdim = stim_on+stim_dur+delay_dur+int(500/dt)
    elif mode == 'random_batch': # Randomly generate parameters, times/modalities different across batch
        batch_size = kwargs['batch_size']

        # Generate per-trial stimulus locations
        stim_dist = rng.uniform(0.5*np.pi, 1.5*np.pi, size=(batch_size,)) * rng.choice([-1, 1], size=(batch_size,))
        stim1_locs = rng.uniform(0, 2*np.pi, size=(batch_size,))
        stim2_locs = (stim1_locs + stim_dist) % (2*np.pi)

        stim_coh_range = np.array([0.01, 0.02, 0.04, 0.08])
        if ('easy_task' in config) and config['easy_task']:
            stim_coh_range *= 10

        # Generate stimulus strengths per modality
        if (attend_mod == 1) or (attend_mod == 2):
            stim1_mod1_strengths, stim2_mod1_strengths = contextdm_genstim(batch_size, rng, stim_coh_range)
            stim1_mod2_strengths, stim2_mod2_strengths = contextdm_genstim(batch_size, rng, stim_coh_range)
            if attend_mod == 1:
                stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
            else:
                stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
        else:
            stim1_strengths, stim2_strengths = contextdm_genstim(batch_size, rng, stim_coh_range)
            stim1_mod12_diff = stim1_strengths * rng.uniform(0.2, 0.8, size=(batch_size,)) * rng.choice([+1, -1], size=(batch_size,))
            stim1_mod1_strengths = stim1_strengths + stim1_mod12_diff / 2
            stim1_mod2_strengths = stim1_strengths - stim1_mod12_diff / 2

            stim2_mod12_diff = stim2_strengths * rng.uniform(0.2, 0.8, size=(batch_size,)) * rng.choice([+1, -1], size=(batch_size,))
            stim2_mod1_strengths = stim2_strengths + stim2_mod12_diff / 2
            stim2_mod2_strengths = stim2_strengths - stim2_mod12_diff / 2

        # Generate per-trial timing parameters
        stim_on = (rng.uniform(100, 400, size=(batch_size,)) / dt).astype(np.int32)
        stim_ons = stim_on  # each trial has its own stimulus onset time
        stim_dur = (rng.choice([400, 800, 1600], size=(batch_size,)) / dt).astype(np.int32)
        stim_offs = stim_ons + stim_dur
        delay_dur = np.zeros(batch_size, dtype=np.int32)  # here delay is 0 per trial
        fix_offs = stim_offs + delay_dur
        # Set overall trial length to be the maximum across trials so that every trial is padded equally.
        tdim = int(500/dt) + np.max(fix_offs)

    elif mode == 'test':
        tdim = int(2000/dt)
        n_stim_loc, n_stim_mod1_strength, n_stim_mod2_strength = batch_shape = 20, 5, 5
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod1_strength, ind_stim_mod2_strength = np.unravel_index(range(batch_size),batch_shape)
        fix_offs  = int(1500/dt)

        stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
        stim2_locs = (stim1_locs+np.pi)%(2*np.pi)
        stim1_mod1_strengths = 0.4*ind_stim_mod1_strength/n_stim_mod1_strength+0.8
        stim2_mod1_strengths = 2 - stim1_mod1_strengths
        stim1_mod2_strengths = 0.4*ind_stim_mod2_strength/n_stim_mod2_strength+0.8
        stim2_mod2_strengths = 2 - stim1_mod2_strengths
        stim_ons  = int(500/dt)
        stim_offs = int(1500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        stim1_mod1_strengths = p['stim1_mod1_strengths']
        stim2_mod1_strengths = p['stim2_mod1_strengths']
        stim1_mod2_strengths = p['stim1_mod2_strengths']
        stim2_mod2_strengths = p['stim2_mod2_strengths']
        stim_time = int(p['stim_time']/dt)
        batch_size = len(stim1_locs)

        # Time of stimuluss on/off
        stim_ons = int(500/dt)
        stim_offs = stim_ons + stim_time
        fix_offs = stim_offs
        tdim = int(500/dt) + fix_offs

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    if attend_mod == 1:
        stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
    elif attend_mod == 2:
        stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
    elif attend_mod == 'both':
        stim1_strengths = stim1_mod1_strengths + stim1_mod2_strengths
        stim2_strengths = stim2_mod1_strengths + stim2_mod2_strengths

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim_ons, offs=stim_offs, strengths=stim1_mod1_strengths, mods=1)
    trial.add('stim', stim2_locs, ons=stim_ons, offs=stim_offs, strengths=stim2_mod1_strengths, mods=1)
    trial.add('stim', stim1_locs, ons=stim_ons, offs=stim_offs, strengths=stim1_mod2_strengths, mods=2)
    trial.add('stim', stim2_locs, ons=stim_ons, offs=stim_offs, strengths=stim2_mod2_strengths, mods=2)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if (stim1_strengths[i]>stim2_strengths[i])
                else stim2_locs[i] for i in range(batch_size)]
    trial.add('out', stim_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1'     : (None, stim_ons),
                   'stim1'    : (stim_ons, stim_offs),
                   # 'delay1'   : (stim_offs, fix_offs),
                   'go1'      : (fix_offs, None)}

    if trial.get_meta:
        trial.meta = {
            'stim1': get_clostest_pref_idx(stim1_locs, trial.pref),
            'stim1_strs': stim1_strengths,
            'stim2': get_clostest_pref_idx(stim2_locs, trial.pref),
            'stim2_strs': stim2_strengths, 
            'resp1': get_clostest_pref_idx(np.array(stim_locs), trial.pref)
        }

    return trial

def contextdm1(config, mode, separate_input, label_strength, long_delay, long_stimulus, long_fixation, **kwargs):
    return _contextdm(config, mode, 1, separate_input, label_strength, long_delay, long_stimulus, long_fixation, **kwargs)

def contextdm2(config, mode, separate_input, label_strength, long_delay, long_stimulus, long_fixation, **kwargs):
    return _contextdm(config, mode, 2, separate_input, label_strength, long_delay, long_stimulus, long_fixation, **kwargs)

def multidm(config, mode, separate_input, label_strength, long_delay, long_stimulus, long_fixation, **kwargs):
    return _contextdm(config, mode, 'both', separate_input, label_strength, long_delay, long_stimulus, long_fixation, **kwargs)

def reactgo_(config, mode, anti_response, long_all, **kwargs):
    '''
    Fixate when fixation point is shown,
    A stimulus will be shown, and the output should saccade to the stimulus location
    Generate one batch of trials

    The fixation is shown between (0, T)
    The stimulus is shown between (fix_off,T)

    The output should be fixation location for (0, fix_off)
    Otherwise should be the stimulus location

    :param mode: the mode of generating. Options: 'random', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']

    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length
        # A list of locations of fixation points and fixation off time
        if not long_all: 
            stim_ons = int(rng.uniform(500,2500)/dt)
            tdim = int(rng.uniform(300,1700)/dt) + stim_ons
        else:
            stim_ons = int(2500/dt)
            tdim = int(1700/dt) + stim_ons

        # A list of locations of stimuluss (they are always on)
        stim_locs = rng.uniform(0, 2*np.pi, (batch_size,))

        stim_mod  = rng.choice([1,2])
    elif mode == 'random_batch': # Randomly generate parameters, times/modalities different across batch
        batch_size = kwargs['batch_size']

        stim_ons = (rng.uniform(500,2500, size=(batch_size,))/dt).astype(np.int32)
        tdim = (rng.uniform(300,1700, size=(batch_size,))/dt).astype(np.int32) + stim_ons

        # A list of locations of stimuluss (they are always on)
        stim_locs = rng.uniform(0, 2*np.pi, (batch_size,))

        stim_mod  = batch_choice([1, 2], batch_size, rng)
    elif mode == 'test':
        tdim = int(2500/dt)
        n_stim_loc, n_stim_mod = batch_shape = 20, 2
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod = np.unravel_index(range(batch_size),batch_shape)

        stim_ons  = int(2000/dt)
        stim_locs  = 2*np.pi*ind_stim_loc/n_stim_loc
        stim_mod   = ind_stim_mod + 1

    elif mode == 'psychometric':
        p = kwargs['params']
        stim_locs = p['stim_locs']
        batch_size = len(stim_locs)

        # Time of stimuluss on/off
        stim_ons = int(1000/dt)
        tdim = int(400/dt) + stim_ons
        stim_mod   = 1

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons  = stim_ons + int(100/dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs+np.pi)%(2*np.pi)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=tdim)
    trial.add('stim', stim_locs, ons=stim_ons, offs=tdim, mods=stim_mod)
    trial.add('fix_out', offs=stim_ons)
    trial.add('out', response_locs, ons=stim_ons, offs=tdim)
    trial.add_c_mask(pre_offs=stim_ons, post_ons=check_ons, post_offs=tdim)

    trial.epochs = {'fix1'     : (None, stim_ons),
                   'go1'      : (stim_ons, tdim)}

    if trial.get_meta:
        trial.meta = {
            'stim1': get_clostest_pref_idx(stim_locs, trial.pref),
            'resp1': get_clostest_pref_idx(response_locs, trial.pref),
        }

    return trial

def reactgo(config, mode, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    return reactgo_(config, mode, False, long_all, **kwargs)


def reactanti(config, mode, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    return reactgo_(config, mode, True, long_all, **kwargs)


def fdgo_(config, mode, anti_response, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    '''
    Go with inhibitory control. Important difference with Go task is that
    the stimulus is presented from the beginning.

    Fixate whenever fixation point is shown,
    A stimulus will be shown from the beginning
    And output should saccade to the stimulus location
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The stimulus is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise should be the stimulus location

    :param mode: the mode of generating. Options: 'random', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length
        # A list of locations of fixation points and fixation off time

        # A list of locations of stimulus (they are always on)
        stim_locs = rng.rand(batch_size)*2*np.pi
        stim_mod  = rng.choice([1,2])
        
        if not long_all:
            stim_ons  = int(rng.uniform(300,700)/dt)
            fix_offs  = stim_ons + int(rng.uniform(200,1500)/dt)
        else:
            stim_ons  = int(700/dt)
            fix_offs  = stim_ons + int(1500/dt)
        
        if long_response == "normal": 
            tdim = int(rng.uniform(300,700)/dt) + fix_offs
        elif long_response == "long": 
            tdim = int(rng.choice([5000])/dt) + fix_offs
        if long_all == True: # overwrite
            tdim = int(700/dt) + fix_offs
        
    elif mode == 'random_batch': # Randomly generate parameters, times/modalities different across batch
        batch_size = kwargs['batch_size']

        # A list of locations of stimulus (they are always on)
        stim_locs = rng.rand(batch_size)*2*np.pi
        stim_mod  = batch_choice([1, 2], batch_size, rng)
        stim_ons  = (rng.uniform(300,700, size=(batch_size,))/dt).astype(np.int32)

        # fix_offs  = stim_ons + (rng.uniform(500,1500, size=(batch_size,))/dt).astype(np.int32)
        fix_offs  = stim_ons + (rng.uniform(200,1500, size=(batch_size,))/dt).astype(np.int32)
        
        # tdim      = int(500/dt) + fix_offs
        if long_response == "normal": 
            tdim = (rng.uniform(300,700, size=(batch_size,))/dt).astype(np.int32) + fix_offs
        elif long_response == "long": 
            tdim = batch_choice([5000], batch_size, rng, dt) + fix_offs
            
    elif mode == 'test':
        tdim = int(2000/dt)
        n_stim_loc, n_stim_mod = batch_shape = 20, 2
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod = np.unravel_index(range(batch_size),batch_shape)

        stim_ons   = int(500/dt)
        fix_offs   = int(1500/dt)
        stim_locs  = 2*np.pi*ind_stim_loc/n_stim_loc
        stim_mod   = ind_stim_mod + 1

    elif mode == 'psychometric':
        p = kwargs['params']
        stim_locs = p['stim_locs']
        stim_time = int(p['stim_time']/dt)
        batch_size = len(stim_locs)

        # Time of stimuluss on/off
        stim_ons   = int(300/dt)
        fix_offs  = stim_ons + stim_time
        tdim      = int(400/dt) + fix_offs
        stim_mod   = 1

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs+np.pi)%(2*np.pi)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=tdim, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs, offs=tdim)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons, post_offs=tdim)

    trial.epochs = {'fix1'     : (None, stim_ons),
                   'stim1'     : (stim_ons, fix_offs),
                   'go1'      : (fix_offs, tdim)}

    if trial.get_meta:
        trial.meta = {
            'stim1': get_clostest_pref_idx(stim_locs, trial.pref),
            'resp1': get_clostest_pref_idx(response_locs, trial.pref),
        }

    return trial

def fdgo(config, mode, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    return fdgo_(config, mode, False, long_response, long_stimulus, long_fixation, long_all, **kwargs)


def fdanti(config, mode, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    return fdgo_(config, mode, True, long_response, long_stimulus, long_fixation, long_all, **kwargs)

def delaydm_(config, mode, stim_mod, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    ''' 
    Fixate whenever fixation point is shown.
    Two stimuluss are shown at different time, with different intensities

    The fixation is shown between (0, fix_off)
    The two stimuluss is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger stimulus

    :param mode: the mode of generating. Options: 'random', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of stimuluss (they are always on)
        stim_dist = rng.uniform(0.5*np.pi, 1.5*np.pi,(batch_size,))*rng.choice([-1,1],(batch_size,))
        stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim2_locs = (stim1_locs+stim_dist)%(2*np.pi)

        stims_mean = rng.uniform(0.8,1.2,(batch_size,))
        # stims_diff = rng.choice([0.32,0.64,1.28],(batch_size,))

        stim_coh_range = np.array([0.08,0.16,0.32])
        if ('easy_task' in config) and config['easy_task']:
            # stim_coh_range = np.array([0.16,0.32,0.64])
            stim_coh_range *= 2

        stims_coh  = rng.choice(stim_coh_range,(batch_size,))
        stims_sign = rng.choice([1,-1], (batch_size,))

        stim1_strengths = stims_mean + stims_coh*stims_sign
        stim2_strengths = stims_mean - stims_coh*stims_sign
        
        # Time of stimuluss on/off (keep your existing sampling for these)
        if not long_all:
            stim1_ons  = int(rng.uniform(200,600)/dt)
            stim1_offs = stim1_ons + int(rng.uniform(200,1600)/dt)
        else:
            stim1_ons  = int(600/dt)
            stim1_offs = stim1_ons + int(1600/dt)

        # NEW: delay lengths (apply to BOTH delay1 and delay2)
        if long_delay == "normal":
            delay1_len = int(rng.uniform(200,1600)/dt)   # between stim1_offs and stim2_ons
            delay2_len = int(rng.uniform(100,300)/dt)    # between stim2_offs and fix_offs
        elif long_delay == "long":
            delay1_len = int(5000/dt)
            delay2_len = int(5000/dt)
        else:
            raise ValueError(f"Unknown long_delay: {long_delay}")
        
        if long_all == True: # overwrite
            delay1_len = int(1600/dt)
            delay2_len = int(300/dt)

        stim2_ons  = stim1_offs + delay1_len
        if not long_all:
            stim2_offs = stim2_ons + int(rng.uniform(200,1600)/dt)
        else:
            stim2_offs = stim2_ons + int(1600/dt)
        fix_offs   = stim2_offs + delay2_len

        if not long_all:
            tdim = fix_offs + int(rng.uniform(300,700)/dt)
        else:
            tdim = fix_offs + int(700/dt)

    elif mode == 'random_batch': # Randomly generate parameters, times/modalities different across batch
        batch_size = kwargs['batch_size']

        # A list of locations of stimuluss (they are always on)
        stim_dist = rng.uniform(0.5*np.pi, 1.5*np.pi, size=(batch_size,))*rng.choice([-1,1],(batch_size,))
        stim1_locs = rng.uniform(0, 2*np.pi, size=(batch_size,))
        stim2_locs = (stim1_locs+stim_dist)%(2*np.pi)

        stims_mean = rng.uniform(0.8,1.2, size=(batch_size,))
        # stims_diff = rng.choice([0.32,0.64,1.28],(batch_size,))

        stim_coh_range = np.array([0.08,0.16,0.32])
        if ('easy_task' in config) and config['easy_task']:
            # stim_coh_range = np.array([0.16,0.32,0.64])
            stim_coh_range *= 2

        stims_coh  = rng.choice(stim_coh_range, size=(batch_size,))
        stims_sign = rng.choice([1,-1], size=(batch_size,))

        stim1_strengths = stims_mean + stims_coh*stims_sign
        stim2_strengths = stims_mean - stims_coh*stims_sign
        
        stim1_ons  = (rng.uniform(200,600,  size=(batch_size,))/dt).astype(np.int32)
        stim1_offs = stim1_ons + (rng.uniform(200,1600, size=(batch_size,))/dt).astype(np.int32)

        # NEW: delay lengths (apply to BOTH delay1 and delay2)
        if long_delay == "normal":
            delay1_len = (rng.uniform(200,1600, size=(batch_size,))/dt).astype(np.int32)
            delay2_len = (rng.uniform(100,300,  size=(batch_size,))/dt).astype(np.int32)
        elif long_delay == "long":
            delay1_len = np.full((batch_size,), int(5000/dt), dtype=np.int32)
            delay2_len = np.full((batch_size,), int(5000/dt), dtype=np.int32)
        else:
            raise ValueError(f"Unknown long_delay: {long_delay}")

        stim2_ons  = stim1_offs + delay1_len
        stim2_offs = stim2_ons + (rng.uniform(200,1600, size=(batch_size,))/dt).astype(np.int32)
        fix_offs   = stim2_offs + delay2_len

        tdim = fix_offs + (rng.uniform(300,700, size=(batch_size,))/dt).astype(np.int32)

    elif mode == 'test':
        tdim = int(3000/dt)
        n_stim_loc, n_stim1_strength = batch_shape = 20, 5
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim1_strength = np.unravel_index(range(batch_size),batch_shape)

        fix_offs  = int(2700/dt)
        stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
        stim2_locs = (stim1_locs+np.pi)%(2*np.pi)
        stim1_strengths = 1.0*ind_stim1_strength/n_stim1_strength+0.5
        stim2_strengths = 2 - stim1_strengths
        stim1_ons = int(500/dt)
        stim1_offs = int(1000/dt)
        stim2_ons = int(2000/dt)
        stim2_offs = int(2500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs       = p['stim1_locs']
        stim2_locs       = p['stim2_locs']
        stim1_strengths  = p['stim1_strengths']
        stim2_strengths  = p['stim2_strengths']
        stim1_ons        = int(p['stim1_ons']/dt)
        stim1_offs       = int(p['stim1_offs']/dt)
        stim2_ons        = int(p['stim2_ons']/dt)
        stim2_offs       = int(p['stim2_offs']/dt)
        batch_size = len(stim1_locs)

        fix_offs = int(200/dt) + stim2_offs
        tdim = int(300/dt) + fix_offs

    else:
        raise ValueError('Unknown mode: ' + str(mode))
    
    if label_strength:
        chosen_strengths = np.where(stim1_strengths > stim2_strengths,
                            stim1_strengths,
                            stim2_strengths)

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_strengths, mods=stim_mod)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_strengths, mods=stim_mod if not separate_input else 3 - stim_mod)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if (stim1_strengths[i]>stim2_strengths[i])
                else stim2_locs[i] for i in range(batch_size)]

    if label_strength:
        trial.add('out', stim_locs, ons=fix_offs, offs=tdim, strengths=chosen_strengths)
    else:
        trial.add('out', stim_locs, ons=fix_offs, offs=tdim)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons, post_offs=tdim)

    trial.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'delay1'   : (stim1_offs, stim2_ons),
                   'stim2'     : (stim2_ons, stim2_offs),
                   'delay2'   : (stim2_offs, fix_offs),
                   'go1'      : (fix_offs, tdim)}

    if trial.get_meta:
        trial.meta = {
            'stim1': get_clostest_pref_idx(stim1_locs, trial.pref),
            'stim1_strs': stim1_strengths,
            'stim2': get_clostest_pref_idx(stim2_locs, trial.pref),
            'stim2_strs': stim2_strengths,
            'resp1': get_clostest_pref_idx(np.array(stim_locs), trial.pref),
        }

    return trial


def delaydm1(config, mode, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    return delaydm_(config, mode, 1, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs)


def delaydm2(config, mode, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    return delaydm_(config, mode, 2, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs)


def contextdelaydm_(config, mode, attend_mod, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    Two stimuluss are shown in each ring,
    Saccade to the one with higher intensity for the attended ring
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The two stimuluss is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger stimulus

    In this task, if the model's strategy is to ignore context, and integrate both,
    then the maximum performance is 75%. So we need to make the highest correct performance
    much higher than that.

    :param mode: the mode of generating. Options: 'random', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of stimuluss, same locations for both modalities
        stim_dist = rng.uniform(0.5*np.pi,1.5*np.pi,(batch_size,))*rng.choice([-1,1],(batch_size,))
        stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim2_locs = (stim1_locs+stim_dist)%(2*np.pi)

        stim_coh_range = np.array([0.08,0.16,0.32])
        if ('easy_task' in config) and config['easy_task']:
            # stim_coh_range = np.array([0.16, 0.32, 0.64])
            stim_coh_range *= 2

        if (attend_mod == 1) or (attend_mod == 2):
            stim1_mod1_strengths, stim2_mod1_strengths = \
                contextdm_genstim(batch_size, rng, stim_coh_range)
            stim1_mod2_strengths, stim2_mod2_strengths = \
                contextdm_genstim(batch_size, rng, stim_coh_range)
            if attend_mod == 1:
                stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
            else:
                stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
        else:
            stim1_strengths, stim2_strengths = \
                contextdm_genstim(batch_size, rng, stim_coh_range)

            stim1_mod12_diff = stim1_strengths * \
                               np.random.uniform(0.2, 0.8, (batch_size,)) * \
                               np.random.choice([+1, -1], (batch_size,))
            stim1_mod1_strengths = stim1_strengths + stim1_mod12_diff/2
            stim1_mod2_strengths = stim1_strengths - stim1_mod12_diff/2

            stim2_mod12_diff = stim2_strengths * \
                               np.random.uniform(0.2, 0.8, (batch_size,)) * \
                               np.random.choice([+1, -1], (batch_size,))
            stim2_mod1_strengths = stim2_strengths + stim2_mod12_diff/2
            stim2_mod2_strengths = stim2_strengths - stim2_mod12_diff/2

        if not long_all:
            stim1_ons  = int(rng.uniform(200,600)/dt)
            stim1_offs = stim1_ons + int(rng.uniform(200,1600)/dt)
            stim2_ons  = stim1_offs + int(rng.uniform(200,1600)/dt)
            stim2_offs = stim2_ons + int(rng.uniform(200,1600)/dt)
            fix_offs  = stim2_offs + int(rng.uniform(100,300)/dt)
            tdim = fix_offs + int(rng.uniform(300,700)/dt) # longest trial
        else:
            stim1_ons  = int(600/dt)
            stim1_offs = stim1_ons + int(1600/dt)
            stim2_ons  = stim1_offs + int(1600/dt)
            stim2_offs = stim2_ons + int(1600/dt)
            fix_offs  = stim2_offs + int(300/dt)
            tdim = fix_offs + int(700/dt) # longest trial
        
    elif mode == 'random_batch': # Randomly generate parameters, times/modalities different across batch
        batch_size = kwargs['batch_size']

        # Generate stimulus locations (per trial)
        stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi, size=(batch_size,)) * rng.choice([-1, 1], size=(batch_size,))
        stim1_locs = rng.uniform(0, 2 * np.pi, size=(batch_size,))
        stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

        stim_coh_range = np.array([0.08, 0.16, 0.32])
        if ('easy_task' in config) and config['easy_task']:
            stim_coh_range *= 2

        if (attend_mod == 1) or (attend_mod == 2):
            stim1_mod1_strengths, stim2_mod1_strengths = contextdm_genstim(batch_size, rng, stim_coh_range)
            stim1_mod2_strengths, stim2_mod2_strengths = contextdm_genstim(batch_size, rng, stim_coh_range)
            if attend_mod == 1:
                stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
            else:
                stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
        else:
            stim1_strengths, stim2_strengths = contextdm_genstim(batch_size, rng, stim_coh_range)

            stim1_mod12_diff = stim1_strengths * rng.uniform(0.2, 0.8, size=(batch_size,)) * rng.choice([+1, -1], size=(batch_size,))
            stim1_mod1_strengths = stim1_strengths + stim1_mod12_diff / 2
            stim1_mod2_strengths = stim1_strengths - stim1_mod12_diff / 2

            stim2_mod12_diff = stim2_strengths * rng.uniform(0.2, 0.8, size=(batch_size,)) * rng.choice([+1, -1], size=(batch_size,))
            stim2_mod1_strengths = stim2_strengths + stim2_mod12_diff / 2
            stim2_mod2_strengths = stim2_strengths - stim2_mod12_diff / 2

        stim1_ons  = (rng.uniform(200,600, size=(batch_size,))/dt).astype(np.int32)
        stim1_offs = stim1_ons + (rng.uniform(200,1600, size=(batch_size,))/dt).astype(np.int32)
        stim2_ons  = stim1_offs + (rng.uniform(200,1600, size=(batch_size,))/dt).astype(np.int32)
        stim2_offs = stim2_ons + (rng.uniform(200,1600, size=(batch_size,))/dt).astype(np.int32)
        fix_offs  = stim2_offs + (rng.uniform(100,300, size=(batch_size,))/dt).astype(np.int32)
        tdim = fix_offs + (rng.uniform(300,700, size=(batch_size,))/dt).astype(np.int32) # longest trial

    elif mode == 'test':
        n_stim_loc, n_stim_mod1_strength, n_stim_mod2_strength = batch_shape = 20, 5, 5
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod1_strength, ind_stim_mod2_strength = np.unravel_index(range(batch_size),batch_shape)

        stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
        stim2_locs = (stim1_locs+np.pi)%(2*np.pi)
        stim1_mod1_strengths = 0.4*ind_stim_mod1_strength/n_stim_mod1_strength+0.8
        stim2_mod1_strengths = 2 - stim1_mod1_strengths
        stim1_mod2_strengths = 0.4*ind_stim_mod2_strength/n_stim_mod2_strength+0.8
        stim2_mod2_strengths = 2 - stim1_mod2_strengths

        stim1_ons = int(500/dt)
        stim1_offs = int(1000/dt)
        stim2_ons = int(2000/dt)
        stim2_offs = int(2500/dt)
        fix_offs  = int(3000/dt)
        tdim = int(3500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        stim1_mod1_strengths = p['stim1_mod1_strengths']
        stim2_mod1_strengths = p['stim2_mod1_strengths']
        stim1_mod2_strengths = p['stim1_mod2_strengths']
        stim2_mod2_strengths = p['stim2_mod2_strengths']
        # stim1_ons        = int(500/dt)
        # stim1_offs       = int(1000/dt)
        # stim2_ons        = int(p['stim_time']/dt) + stim1_offs
        # stim2_offs       = int(500/dt) + stim2_ons
        stim1_ons        = int(300/dt)
        stim1_offs       = int(600/dt)
        stim2_ons        = int(p['stim_time']/dt) + stim1_offs
        stim2_offs       = int(300/dt) + stim2_ons
        batch_size = len(stim1_locs)

        # Time of stimuluss on/off
        fix_offs = int(200/dt) + stim2_offs
        tdim = int(300/dt) + fix_offs

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    if attend_mod == 1:
        stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
    elif attend_mod == 2:
        stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
    elif attend_mod == 'both':
        stim1_strengths = stim1_mod1_strengths + stim1_mod2_strengths
        stim2_strengths = stim2_mod1_strengths + stim2_mod2_strengths

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_mod1_strengths, mods=1)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_mod1_strengths, mods=1)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_mod2_strengths, mods=2)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_mod2_strengths, mods=2)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if (stim1_strengths[i]>stim2_strengths[i])
                else stim2_locs[i] for i in range(batch_size)]
    trial.add('out', stim_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'delay1'   : (stim1_offs, stim2_ons),
                   'stim2'     : (stim2_ons, stim2_offs),
                   'delay2'   : (stim2_offs, fix_offs),
                   'go1'      : (fix_offs, None)}

    if trial.get_meta:
        trial.meta = {
            'stim1': get_clostest_pref_idx(stim1_locs, trial.pref),
            'stim1_mod1_strs': stim1_mod1_strengths,
            'stim1_mod2_strs': stim1_mod2_strengths,
            'stim2': get_clostest_pref_idx(stim2_locs, trial.pref),
            'stim2_mod1_strs': stim2_mod1_strengths,
            'stim2_mod2_strs': stim2_mod2_strengths,
            'resp1': get_clostest_pref_idx(np.array(stim_locs), trial.pref),
        }

    return trial


def contextdelaydm1(config, mode, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    return contextdelaydm_(config, mode, 1, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs)

def contextdelaydm2(config, mode, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    return contextdelaydm_(config, mode, 2, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs)

def multidelaydm(config, mode, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    return contextdelaydm_(config, mode, 'both', separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs)

def dms_(config, mode, matchnogo, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    '''
    Delay-match-to-sample

    Two stimuli are shown, separated in time, either at the same location or not
    Fixate before the second stimulus is shown

    If matchnogo is one, then:
    If the two stimuli are the same, then keep fixation.
    If the two stimuli are different, then saccade to the location of the stimulus

    If matchnogo is zero, then:
    If the two stimuli are different, then keep fixation.
    If the two stimuli are the same, then saccade to the location of the stimulus

    The first stimulus is shown between (stim1_on, stim1_off)
    The second stimulus is shown between (stim2_on, T)

    The output should be fixation location for (0, stim2_on)
    If two stimuli the different location, then for (stim2_on, T) go to stim2_loc
    Otherwise keep fixation

    :param mode: the mode of generating. Options: 'random', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        
        stim1_mod  = rng.choice([1,2])
        stim2_mod  = rng.choice([1,2])
        # A list of locations of stimuluss
        # Since stim1 is always shown first, it's important that we completely randomize their relative positions
        matchs    = rng.choice([0,1],(batch_size,)) # match or not?
        # stim_dist range between 1/18*pi and (2-1/18*pi), corresponding to 10 degree to 350 degree
        stim_dist  = rng.uniform(np.pi/9,np.pi*17./9.,(batch_size,))*rng.choice([-1,1],(batch_size,)) # 20-340
        stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim2_locs = (stim1_locs+stim_dist*(1-matchs))%(2*np.pi)

        if not long_all:
            stim1_ons  = int(rng.uniform(200,600)/dt)
            stim1_offs = stim1_ons + int(rng.uniform(200,1600)/dt)
            stim2_ons  = stim1_offs + int(rng.uniform(200,1600)/dt)
            tdim = stim2_ons + int(rng.uniform(300,700)/dt) # longest trial
        else:
            stim1_ons  = int(600/dt)
            stim1_offs = stim1_ons + int(1600/dt)
            stim2_ons  = stim1_offs + int(1600/dt)
            tdim = stim2_ons + int(700/dt) # longest trial
        
    elif mode == 'random_batch': # Randomly generate parameters, times/modalities different across batch
        batch_size = kwargs['batch_size']

        stim1_mod  = batch_choice([1,2], batch_size, rng)
        stim2_mod  = batch_choice([1,2], batch_size, rng)
        # A list of locations of stimuluss
        # Since stim1 is always shown first, it's important that we completely randomize their relative positions
        matchs    = rng.choice([0,1], size=(batch_size,)) # match or not?
        # stim_dist range between 1/18*pi and (2-1/18*pi), corresponding to 10 degree to 350 degree
        stim_dist  = rng.uniform(np.pi/9,np.pi*17./9., size=(batch_size,))*rng.choice([-1,1], size=(batch_size,)) # 20-340
        stim1_locs = rng.uniform(0, 2*np.pi, size=(batch_size,))
        stim2_locs = (stim1_locs+stim_dist*(1-matchs))%(2*np.pi)

        # # Time of stimuluss on/off
        # stim1_ons  = batch_choice([200, 400, 600], batch_size, rng, dt)
        # stim1_offs = stim1_ons + batch_choice([200, 400, 600], batch_size, rng, dt)
        # stim2_ons  = stim1_offs + batch_choice([200, 400, 800, 1600], batch_size, rng, dt)
        # tdim       = stim2_ons + int(500/dt)

        stim1_ons  = (rng.uniform(200,600, size=(batch_size,))/dt).astype(np.int32)
        stim1_offs = stim1_ons + (rng.uniform(200,1600, size=(batch_size,))/dt).astype(np.int32)
        stim2_ons  = stim1_offs + (rng.uniform(200,1600, size=(batch_size,))/dt).astype(np.int32)
        tdim = stim2_ons + (rng.uniform(300,700, size=(batch_size,))/dt).astype(np.int32) # longest trial
        
    elif mode == 'test':
        # Set this test so the model always respond
        n_stim_loc, n_mod1, n_mod2 = batch_shape = 20, 2, 2
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_mod1, ind_mod2 = np.unravel_index(range(batch_size),batch_shape)

        stim1_mod = ind_mod1 + 1
        stim2_mod = ind_mod2 + 1

        stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
        matchs = (1 - matchnogo)*np.ones(batch_size) # make sure the response is Go
        stim2_locs = (stim1_locs+np.pi*(1-matchs))%(2*np.pi)

        stim1_ons  = int(500/dt)
        stim1_offs = stim1_ons + int(500/dt)
        stim2_ons  = stim1_offs + int(1200/dt)
        tdim = stim2_ons + int(500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        matchs = get_dist(stim1_locs-stim2_locs)<np.pi/36. # 5 degree
        batch_size = len(stim1_locs)

        tdim = int(2500/dt)
        stim1_ons  = int(500/dt)
        stim1_offs = int(800/dt)
        stim2_ons  = int(2000/dt)
        stim1_mod = 1
        stim2_mod = 1

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons = stim2_ons + int(100/dt)

    trial = Trial(config, tdim, batch_size)

    trial.add('fix_in', offs=tdim)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=stim1_mod)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=tdim, mods=stim2_mod if not separate_input else 3 - stim1_mod)

    if hasattr(stim2_ons, '__iter__'):
        fix_out_offs = list(stim2_ons)
    else:
        fix_out_offs = [stim2_ons]*batch_size
    out_offs = [None]*batch_size

    for i in range(batch_size):
        if matchs[i] == matchnogo: # If match
            fix_out_offs[i] = None # Keep fixation
            out_offs[i] = 0 # And don't go to stimulus location

    trial.add('fix_out', offs=fix_out_offs)
    trial.add('out', stim2_locs, ons=stim2_ons, offs=out_offs)

    trial.add_c_mask(pre_offs=stim2_ons, post_ons=check_ons, post_offs=tdim)

    trial.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'delay1'   : (stim1_offs, stim2_ons),
                   'go1'      : (stim2_ons, tdim)}

    if trial.get_meta:
        trial.meta = {
            'stim1': get_clostest_pref_idx(stim1_locs, trial.pref),
            'stim1_mod': stim1_mod,
            'stim2': get_clostest_pref_idx(stim2_locs, trial.pref),
            'stim2_mod': stim2_mod if not separate_input else 3 - stim1_mod,
            'matches': matchs,
        }

    return trial


def dmsgo(config, mode, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    return dms_(config, mode, 0, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs)


def dmsnogo(config, mode, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    return dms_(config, mode, 1, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs)


def dmc_(config, mode, matchnogo, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    '''
    Delay-match-to-category

    Two stimuli are shown, separated in time, either at the locations of the same category or not
    Fixate before the second stimulus is shown

    If matchnogo is one, then:
    If the two stimuli are the same, then keep fixation.
    If the two stimuli are different, then saccade to the location of the stimulus

    If matchnogo is zero, then:
    If the two stimuli are different, then keep fixation.
    If the two stimuli are the same, then saccade to the location of the stimulus

    The first stimulus is shown between (stim1_on, stim1_off)
    The second stimulus is shown between (stim2_on, T)

    :param mode: the mode of generating. Options: 'random', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length

        # Use only mod 1 for input
        stim1_mod  = rng.choice([1,2])
        stim2_mod  = rng.choice([1,2])

        stim1_locs = rng.choice(np.array([0.05, 0.30, 0.55, 0.80, 1.05, 1.30, 1.55, 1.80])*np.pi,size=(batch_size,)) # avoid ambiguity
        stim2_locs = rng.choice(np.array([0.05, 0.30, 0.55, 0.80, 1.05, 1.30, 1.55, 1.80])*np.pi,size=(batch_size,))

        if not long_all:
            stim1_ons  = int(rng.uniform(200, 600) / dt)
            stim1_offs = stim1_ons + int(rng.uniform(200, 1600) / dt)

            # delay between stim1_offs and stim2_ons
            if long_delay == "normal":
                delay_len = int(rng.uniform(200, 1600) / dt)
            elif long_delay == "long":
                delay_len = int(10000 / dt)
            else:
                raise ValueError(f"Unknown long_delay: {long_delay}")

            stim2_ons = stim1_offs + delay_len
            tdim = stim2_ons + int(rng.uniform(300, 700) / dt)

        else:
            # longest selection within the normal range for all periods
            stim1_ons  = int(600 / dt)
            stim1_offs = stim1_ons + int(1600 / dt)
            stim2_ons  = stim1_offs + int(1600 / dt)
            tdim       = stim2_ons + int(700 / dt)

    elif mode == 'random_batch': # Randomly generate parameters, times/modalities different across batch
        batch_size = kwargs['batch_size']

        # Generate modality per trial for stimulus 1 and 2
        stim1_mod = rng.choice([1, 2], size=(batch_size,))
        stim2_mod = rng.choice([1, 2], size=(batch_size,))

        # Choose stimulus locations independently for each trial
        # possible_locs = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]) * np.pi
        possible_locs = np.array([0.05, 0.30, 0.55, 0.80, 1.05, 1.30, 1.55, 1.80]) * np.pi # avoid ambiguity
        stim1_locs = rng.choice(possible_locs, size=(batch_size,))
        stim2_locs = rng.choice(possible_locs, size=(batch_size,))
        
        # # Generate timing parameters as per-trial arrays
        # stim1_ons  = (rng.choice([200, 400, 600], size=(batch_size,)) / dt).astype(np.int32)
        # stim1_offs = stim1_ons + (rng.choice([200, 400, 600], size=(batch_size,)) / dt).astype(np.int32)
        # stim2_ons  = stim1_offs + (rng.choice([200, 400, 800, 1600], size=(batch_size,)) / dt).astype(np.int32)
        # # For the second stimulus, we generate a duration for each trial...
        # tdim = np.max(stim2_ons + (rng.choice([200, 400, 600], size=(batch_size,)) / dt).astype(np.int32))

        stim1_ons  = (rng.uniform(200,600, size=(batch_size,))/dt).astype(np.int32)
        stim1_offs = stim1_ons + (rng.uniform(200,1600, size=(batch_size,))/dt).astype(np.int32)
        
        # long_delay controls the delay between stim1_offs and stim2_ons (per trial)
        if long_delay == "normal":
            stim2_ons  = stim1_offs + (rng.uniform(200,1600, size=(batch_size,))/dt).astype(np.int32)
        elif long_delay == "long":
            stim2_ons  = stim1_offs + (rng.choice([10000], size=(batch_size,))/dt).astype(np.int32)
        else:
            raise ValueError(f"Unknown long_delay: {long_delay}")
        
        tdim = stim2_ons + (rng.uniform(300,700, size=(batch_size,))/dt).astype(np.int32) # longest trial

    elif mode == 'test':
        # Set this test so the model always respond
        n_stim_loc, n_mod1, n_mod2 = batch_shape = 20, 2, 2
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_mod1, ind_mod2 = np.unravel_index(range(batch_size),batch_shape)

        stim1_mod = ind_mod1 + 1
        stim2_mod = ind_mod2 + 1

        n_stim_loc2 = n_stim_loc/2
        stim1_locs_ = np.concatenate(((0.1+0.8*np.arange(n_stim_loc2)/n_stim_loc2),
                                    (1.1+0.8*np.arange(n_stim_loc2)/n_stim_loc2)))*np.pi
        stim1_locs = np.array([stim1_locs_[i] for i in ind_stim_loc])
        matchs = (1 - matchnogo)*np.ones(batch_size) # make sure the response is Go
        stim2_locs = (stim1_locs+np.pi*(1-matchs))%(2*np.pi)

        stim1_ons  = int(500/dt)
        stim1_offs = stim1_ons + int(500/dt)
        stim2_ons  = stim1_offs + int(1200/dt)
        tdim = stim2_ons + int(500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        batch_size = len(stim1_locs)

        tdim = int(2500/dt)
        stim1_ons  = int(500/dt)
        stim1_offs = int(800/dt)
        stim2_ons  = int(2000/dt)
        stim1_mod = 1
        stim2_mod = 1

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons = stim2_ons + int(100/dt)

    stim1_cats = stim1_locs<np.pi # Category of stimulus 1
    stim2_cats = stim2_locs<np.pi # Category of stimulus 2
    matchs    = stim1_cats==stim2_cats

    trial = Trial(config, tdim, batch_size)

    trial.add('fix_in', offs=tdim)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=stim1_mod)
    trial.add('stim', stim2_locs, ons=stim2_ons, mods=stim2_mod if not separate_input else 3 - stim1_mod)

    if hasattr(stim2_ons, '__iter__'):
        fix_out_offs = list(stim2_ons)
    else:
        fix_out_offs = [stim2_ons]*batch_size
    out_offs = [None]*batch_size

    for i in range(batch_size):
        if matchs[i] == matchnogo: # If match
            fix_out_offs[i] = None # Keep fixation
            out_offs[i] = 0 # And don't go to stimulus location

    trial.add('fix_out', offs=fix_out_offs)
    trial.add('out', stim2_locs, ons=stim2_ons, offs=out_offs)
    
    trial.add_c_mask(pre_offs=stim2_ons, post_ons=check_ons)

    trial.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'delay1'   : (stim1_offs, stim2_ons),
                   'go1'      : (stim2_ons, None)}

    if trial.get_meta:
        trial.meta = {
            'stim1': get_clostest_pref_idx(stim1_locs, trial.pref),
            'stim1_mod': stim1_mod,
            'stim2': get_clostest_pref_idx(stim2_locs, trial.pref),
            'stim2_mod': stim2_mod if not separate_input else 3 - stim1_mod,
            'matches': matchs
        }

    return trial

def dmcgo(config, mode, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    return dmc_(config, mode, 0, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs)

def dmcnogo(config, mode, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs):
    return dmc_(config, mode, 1, separate_input, label_strength, long_delay, long_response, long_stimulus, long_fixation, long_all, **kwargs)

rule_mapping = {'testinit': test_init,
                'fdgo': fdgo, # Var_batch
                'reactgo': reactgo, # Var_batch
                'delaygo': delaygo, # Var_batch
                'fdanti': fdanti, # Var_batch
                'reactanti': reactanti, # Var_batch
                'delayanti': delayanti, # Var_batch
                # 'dm1': dm1,
                # 'dm2': dm2,
                # 'contextdm1': contextdm1,
                # 'contextdm2': contextdm2,
                'multidm': multidm,
                'delaydm1': delaydm1, # Var_batch
                'delaydm2': delaydm2, # Var_batch
                'contextdelaydm1': contextdelaydm1,
                'contextdelaydm2': contextdelaydm2,
                'multidelaydm': multidelaydm,
                'dmsgo': dmsgo, # Var_batch
                'dmsnogo': dmsnogo, # Var_batch
                'dmcgo': dmcgo,
                'dmcnogo': dmcnogo,
                # 'oic': oic,
                # 'dmc': delaymatchcategory_original
                }

rule_name    = {'reactgo': 'RT Go',
                'delaygo': 'Dly Go',
                'fdgo': 'Go',
                'dm1': 'DM 1',
                'dm2': 'DM 2',
                'contextdm1': 'Ctx DM 1',
                'contextdm2': 'Ctx DM 2',
                'multidm': 'MultSen DM',
                'delaydm1': 'Dly DM 1',
                'delaydm2': 'Dly DM 2',
                'contextdelaydm1': 'Ctx Dly DM 1',
                'contextdelaydm2': 'Ctx Dly DM 2',
                'multidelaydm': 'MultSen Dly DM',
                'reactanti': 'RT Anti',
                'delayanti': 'Dly Anti',
                'fdanti': 'Anti',
                'dmsgo': 'DMS',
                'dmsnogo': 'DNMS',
                'dmcgo': 'DMC',
                'dmcnogo': 'DNMC',
                'oic': '1IC',
                'dmc': 'DMC'
                }


def generate_trials(rule, hp, mode, noise_on=True, 
                    separate_input=False, label_strength=False, 
                    long_delay="normal", long_response="normal", 
                    long_stimulus="normal", long_fixation="normal", 
                    long_all=False, **kwargs):
    """Generate one batch of data.

    Args:
        rule: str, the rule for this batch
        hp: dictionary of hyperparameters
        mode: str, the mode of generating. Options: random, test, psychometric
        noise_on: bool, whether input noise is given

    Return:
        trial: Trial class instance, containing input and target output
    """
    config = hp
    trial = rule_mapping[rule](config, 
                               mode, 
                               separate_input, 
                               label_strength, 
                               long_delay, 
                               long_response, 
                               long_stimulus, 
                               long_fixation, 
                               long_all=long_all, 
                               **kwargs)

    # Add rule input to every task
    if 'rule_on' in kwargs:
        rule_on = kwargs['rule_on']
    else: # default behavior
        rule_on = None
    if 'rule_off' in kwargs:
        rule_off = kwargs['rule_off']
    else: # default behavior
        rule_off = None

    # overwrite current rule for input
    if 'replace_rule' in kwargs:
        rule = kwargs['replace_rule']

    if rule == 'testinit':
        # Add no rule
        return trial

    if isinstance(rule, six.string_types):
        # rule is not iterable
        # Expand to list
        if 'rule_strength' in kwargs:
            rule_strength = [kwargs['rule_strength']]
        else:
            rule_strength = [1.]
        rule = [rule]

    else:
        if 'rule_strength' in kwargs:
            rule_strength = kwargs['rule_strength']
        else:
            rule_strength = [1.] * len(rule)

    for r, s in zip(rule, rule_strength):
        trial.add_rule(r, on=rule_on, off=rule_off, strength=s)

    if noise_on:
        trial.add_x_noise()

    return trial

seed = random.randint(1,10000)
# seed = 1000
np.random.RandomState(seed)

def convert_and_init_multitask_params(params):
    """
    Convert parameters used in our setup into equivalent parameters in the
    multitask setup and vice versa.

    """
    task_params, train_params, net_params = params

    # task_params['rules'] = get_rules(task_params['ruleset'])
    num_ring = 2 if 'oic' not in task_params['rules'] else 3
    n_rule = len(task_params['rules']) 
    task_params['n_rules'] = n_rule

    fixate_off = task_params.get('fixate_off', False) # Signal for when fixation turns off (go signal)
    task_info = task_params.get('task_info', True) 
    n_fixate = 2 if fixate_off else 1
    n_rule_revise = n_rule if not task_info else 0

    if task_params['in_out_mode'] in ('high_dim',):
        n_input = n_fixate + num_ring * task_params['n_eachring'] + n_rule # fix, ring inputs, rule inputs
        n_output = 1 + task_params['n_eachring'] # fix, output ring
    elif task_params['in_out_mode'] in ('low_dim',):
        n_input = n_fixate + num_ring * 2 + n_rule # fix, ring inputs, rule inputs
        n_output = 1 + 2 # fix, output ring
    elif task_params['in_out_mode'] in ('low_dim_pos',):
        n_input = n_fixate + num_ring * 4 + n_rule # fix, ring inputs, rule inputs
        n_output = 1 + 4 # fix, output ring

    if task_params['randomize_inputs']: # If passing through random matrix, use determined input size
        input_size = task_params['n_input']
    else: # Sets input size to task input size
        input_size = n_input
        task_params['n_input'] = input_size

    # Mar 16th: overwrite input & output
    if 'n_neurons' in net_params:
        net_params['n_neurons'][0] = input_size - n_rule_revise # ***
        net_params['n_neurons'][-1] = n_output
    else:
        net_params['n_input'] = input_size - n_rule_revise # ***
        net_params['n_output'] = n_output
    task_params['n_output'] = n_output

    net_params['tau'] = 200 # Time constant (ms), default is 100 ms, only used to set noise scale for task
    net_params['dt'] = task_params['dt'] # Default is 20 ms
    
    # net_params['alpha'] = net_params['dt'] / net_params['tau'] # Default is 0.2
    # 2025-10-30: overwrite alpha to be 0.8, to match with LD's paper
    net_params['alpha'] = 1 - net_params['dt'] / net_params['tau'] 
    
    hp = {
        'seed': seed,
        'rng': np.random.RandomState(seed),
        # 'ruleset': task_params['ruleset'],
        'in_out_mode': task_params['in_out_mode'],
        'batch_size_train': train_params['n_batches'], # number of batches for training
        'loss_type': 'lsq' if net_params['loss_type'] == 'MSE' else 'xe', # Type of loss functions
        'tau': net_params['tau'], # Time constant (ms)
        'dt': net_params['dt'],
        'alpha': net_params['alpha'], # discretization time step/time constant, used to set noise scale
        'sigma_x': task_params['sigma_x'], # input noise
        'n_eachring': task_params['n_eachring'],
        # number of rings
        'num_ring': num_ring,
        'rules': task_params['rules'],
        # number of rules
        'n_rule': n_rule,
        # first input index for rule units
        'rule_start': n_input - n_rule,
        # number of input units
        'n_input': n_input,
        # number of output units
        'n_output': n_output,
        'fixate_off': fixate_off,
        'task_info': task_info,
        'get_meta': True, # Collect metadata for easier labelling
    }

    task_params['loss_type'] = net_params['loss_type']
    task_params['hp'] = hp

    # Initiates some other parameters that are useful for training
    if 'rules_probs' not in task_params:
        task_params['rules_probs'] = 1 / n_rule * np.ones((n_rule,))

    if task_params['randomize_inputs']:
        # If the conversion is not already generated, create it
        if 'randomize_matrix' not in task_params:
            task_params['randomize_matrix'] = net_helpers.rand_weight_init(task_params['n_input'], n_input - n_rule_revise)

    params = task_params, train_params, net_params

    return params

def normalize_to_one(x, axis=None, eps=1e-12):
    """
    Normalize an array so its entries sum to 1.
    """
    x = np.asarray(x, dtype=float)

    # Detect axes whose total is exactly zero
    s = x.sum(axis=axis, keepdims=True)
    zero_sum_mask = (s == 0)

    if np.any(zero_sum_mask):
        # Add a tiny uniform mass along the zero-sum axes
        # so that their sum becomes ``eps * n`` and no entry is nan/inf.
        if axis is None:
            x = x + eps
        else:
            # Broadcast `eps` to the same shape as `x` on the target axes
            add = np.where(zero_sum_mask, eps, 0.0)
            x = x + add

        # Recompute the sums after the adjustment
        s = x.sum(axis=axis, keepdims=True)

    # Safe division: where s==0 can no longer happen, but keep guard
    return np.divide(x, s, where=s != 0, out=np.full_like(x, np.nan))

def insert_zeros_after_channel(x, K, after=5):
    """
    Insert K zero-valued channels into a 3D NumPy array immediately after a given channel index.

    Parameters
    ----------
    x : np.ndarray
        Input 3D array of shape (H, W, C), e.g., an image or feature map.
    K : int
        Number of zero channels to insert.
    after : int, optional
        Index (0-based) after which the zero channels should be inserted.
        Must satisfy 0 <= after < C. Default is 5.

    Returns
    -------
    np.ndarray
        A new array of shape (H, W, C + K) where K zero channels are inserted
        immediately after the `after`-th channel of `x`.
    """
    if x.ndim != 3:
        raise ValueError("Input must be 3‑D (H, W, C).")
    if not (0 <= after < x.shape[2]):
        raise ValueError("'after' index out of range.")

    first  = x[..., :after + 1]         
    second = x[..., after + 1 :]        

    zeros  = np.zeros((*x.shape[:2], K), dtype=x.dtype)

    return np.concatenate([first, zeros, second], axis=2)

def generate_trials_wrap(task_params, 
                         n_batches, 
                         device='cuda', 
                         verbose=False, 
                         rules=None,
                         mode_input="random_batch",
                         pretraining_shift=0,
                         pretraining_shift_pre=0,
                         long_all=False,
                         align_periods=False):
    """
    Wrapper to generate the raw datasets, including the inputs, labels, and masks.

    This is used for both training, validation, and testing functionality, which
    is controlled by the "rules" input.
    - If the "rules" input is none, generates
      data for a single random rule (usually used during training).
    - If the "rules" input is a single rule, generates data for that particular
      rule.
    - Finally, if the "rules" input is a list of many
      rules, generates data for all rules simultaneously and puts into a single
      batch (usually used during validation).

    Args:
        n_batches: Size of batch dimension, for multiple rules each rule has
            a batch of this size.
        rule: See above
    """
    if rules is None: # Draw a rule randomly, create tuple with single rule in it
        task_params['rules_probs'] = normalize_to_one(task_params['rules_probs'])
        if verbose: 
            print(f"task_params['rules_probs']: {task_params['rules_probs']}")
        rule_idx = np.random.choice(task_params['n_rules'], p=task_params['rules_probs'])
        rules = (task_params['rules'][rule_idx],)
        rule_idxs = (rule_idx,) # Corresponding rule_idx in task_params['rules']
        if verbose:
            print('Rule: {}'.format(rules[0]))
    elif type(rules) == str: # Single rule passed in as string
        assert rule in task_params['rules']
        rules = (rules,)
        rule_idxs = (task_params['rules'].index(rule),)
    elif type(rules) == tuple or type(rules) == list: # Many rules together
        rule_idxs = []
        for rule in rules:
            assert rule in task_params['rules']
            rule_idxs.append(task_params['rules'].index(rule),)
    else:
        raise ValueError

    n_rules = len(rules)

    inputs = []
    labels = []
    masks = []
    seq_lens = []
    rule_batch_idxs = [] # Used to tell what rule idx each batch_idx corresponds to
    trials = []
    max_seq_len = 0
    n_batches_all = 0

    def allocate_integer_to_divisions(integer, divisor):
        quotient = integer // divisor
        remainder = integer % divisor
        
        result = [quotient] * divisor        
        if remainder > 0:
            indices = np.random.choice(divisor, remainder, replace=False)
            for idx in indices:
                result[idx] += 1
        
        return result
    
    # 2026-03-29: we add an additional mode "long_all" as an additional mode for post-training analysis,
    # which generates deta trail with longest (but still in reasonable range) delay, response, stimulus, and fixation
    # currently only implemented in the "random" mode for which different task period is aligned across trials 
    # in one specific task 
    if long_all:
        assert mode_input == "random"

    # align_periods: reset the shared RNG to the same state before each rule so
    # that every rule in this call draws IDENTICAL epoch timing (stim_ons,
    # stim_offs, fix_offs, tdim). In 'random' mode these are per-call scalars, so
    # without this each rule would land on a different timeline and the two tasks'
    # periods (delay etc.) would not be matched. Requires 'random' mode (the only
    # mode with uniform-across-batch scalar timing).
    if align_periods:
        assert mode_input == "random", "align_periods requires mode_input='random'"
        _align_rng = task_params['hp']['rng']
        _align_rng_state = _align_rng.get_state()

    for rule, rule_idx in zip(rules, rule_idxs):
        if align_periods:
            _align_rng.set_state(_align_rng_state)
        trial = generate_trials(rule,
                                task_params['hp'], 
                                mode_input, 
                                batch_size=n_batches, 
                                separate_input=task_params['modality_diff'], 
                                label_strength=task_params['label_strength'], 
                                long_delay=task_params['long_delay'],
                                long_response=task_params['long_response'], 
                                long_stimulus=task_params['long_stimulus'], 
                                long_fixation=task_params['long_fixation'],
                                long_all=long_all)

        trial.x = trial.x[:,:,:-1] if not task_params["task_info"] else trial.x # ***
        
        # Transposes batch and sequence dimensions
        inputs.append(np.transpose(trial.x, (1, 0, 2)))
        labels.append(np.transpose(trial.y, (1, 0, 2)))

        seq_len = np.max(trial.tdim) # Within batch maximum sequence length
        if seq_len > max_seq_len:
            max_seq_len = seq_len
        if verbose:
            print('Rule {} seq_len {}, max_seq_len {}'.format(rule, seq_len, max_seq_len))
        n_batches_all += trial.x.shape[1]

        if task_params['mask_type'] in (None,):
            masks.append(np.ones_like(labels[-1]))
        elif task_params['mask_type'] in ('cost',):
            masks.append(np.transpose(trial.c_mask, (1, 0, 2)))
        else:
            raise ValueError('Mask type {} not recognized.'.format(task_params['mask_type']))

        trials.append(trial)
        if not hasattr(trial.tdim, '__iter__'):
            seq_lens.append(trial.tdim * np.ones((n_batches,), dtype=np.int32))
        else:
            seq_lens.append(trial.tdim)
        rule_batch_idxs.append(rule_idx * np.ones((n_batches,), dtype=np.int32))

    # Concatenates different rules together into one giant batch if necessary
    if n_rules == 1: # Not necessary, just take 0th element
        inputs_all = inputs[0]
        labels_all = labels[0]
        masks_all = masks[0]
        seq_lens_all = seq_lens[0]
        rule_batch_idxs_all = rule_batch_idxs[0]
    else:
        inputs_all = np.zeros((n_batches_all, max_seq_len, task_params['hp']['n_input']))
        labels_all = np.zeros((n_batches_all, max_seq_len, task_params['hp']['n_output']))
        masks_all = np.zeros((n_batches_all, max_seq_len, task_params['hp']['n_output'])) # Masks are zero by default

        batch_idx = 0
        for rule_idx, rule in enumerate(rules):
            n_batch_rule = inputs[rule_idx].shape[0]
            seq_len_rule = inputs[rule_idx].shape[1]
            inputs_all[batch_idx:batch_idx+n_batch_rule, :seq_len_rule, :] = inputs[rule_idx]
            labels_all[batch_idx:batch_idx+n_batch_rule, :seq_len_rule, :] = labels[rule_idx]
            masks_all[batch_idx:batch_idx+n_batch_rule, :seq_len_rule, :] = masks[rule_idx]

            batch_idx += n_batch_rule

        seq_lens_all = np.concatenate(seq_lens, axis=0)
        rule_batch_idxs_all = np.concatenate(rule_batch_idxs, axis=0)

    if task_params['randomize_inputs']: # Multiplies all inputs by fixed random matrix (needed for MPNs sometimes)
        inputs_all = np.matmul(inputs_all, task_params['randomize_matrix'])

    # print(pretraining_shift_pre, pretraining_shift, inputs_all.shape)
    # 2025-07-18: pretraining purprose, paddling the input based on the number of pretrained tasks 
    # mask and output should not be affected
    inputs_all = insert_zeros_after_channel(inputs_all, K=pretraining_shift_pre, after=inputs_all.shape[-1]-1)
    inputs_all = insert_zeros_after_channel(inputs_all, K=pretraining_shift, after=5)
    # print(inputs_all.shape)
    
    # 2025-10-30: at most one is True, the values will only be nonzero during the pretraining analysis
    # in which one value is used in pre-training and one value is used for post-training
    assert not (pretraining_shift_pre > 0 and pretraining_shift > 0), "Both values are > 0, but only one or none is allowed."

    if verbose:
        print(f"inputs_all paddled: {inputs_all.shape}")

    # Converts the data to be passed to network to torch form
    inputs_all = torch.from_numpy(inputs_all).type(torch.float).to(device) # inputs.shape
    if task_params['loss_type'] in ('XE'):
        labels_all = torch.from_numpy(labels_all).type(torch.long).to(device) # labels.shape
    elif task_params['loss_type'] in ('MSE'):
        labels_all = torch.from_numpy(labels_all).type(torch.float).to(device) # labels.shape
    else:
        raise ValueError('Loss type not recognized.')
    if task_params['mask_type'] in (None,):
        masks_all = torch.tensor(masks_all, dtype=torch.bool, device=device)
    elif task_params['mask_type'] in ('cost',):
        masks_all = torch.tensor(masks_all, dtype=torch.float, device=device)

    if verbose:
        print(f"inputs_all: {inputs_all.shape}")

    return (inputs_all, labels_all, masks_all), (seq_lens_all, trials, rule_batch_idxs_all,)
