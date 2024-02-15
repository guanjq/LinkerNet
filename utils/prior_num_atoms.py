"""Utils for sampling size of a linker."""

import numpy as np
import pickle
from collections import Counter


def setup_configs(meta_path='utils/prior_num_atoms.pkl', mode='frag_center_distance'):
    with open(meta_path, 'rb') as f:
        prior_meta = pickle.load(f)
    all_dist = prior_meta[mode]
    all_n_atoms = prior_meta['num_linker_atoms']
    bin_min, bin_max = np.floor(all_dist.min()), np.ceil(all_dist.max())
    BINS = np.arange(bin_min, bin_max, 1.)
    CONFIGS = {'bounds': BINS, 'distributions': []}

    for min_d, max_d in zip(BINS[:-1], BINS[1:]):
        valid_idx = (min_d < all_dist) & (all_dist < max_d)
        c = Counter(all_n_atoms[valid_idx])
        num_atoms_list, prob_list = list(c.keys()), np.array(list(c.values())) / np.sum(list(c.values()))
        CONFIGS['distributions'].append((num_atoms_list, prob_list))
    return CONFIGS


def _get_bin_idx(distance, config_dict):
    bounds = config_dict['bounds']
    for i in range(len(bounds) - 1):
        if distance < bounds[i + 1]:
            return i
    return len(bounds) - 2


def sample_atom_num(distance, config_dict):
    bin_idx = _get_bin_idx(distance, config_dict)
    num_atom_list, prob_list = config_dict['distributions'][bin_idx]
    return np.random.choice(num_atom_list, p=prob_list)
