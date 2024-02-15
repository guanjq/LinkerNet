import torch
import torch_scatter
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

FOLLOW_BATCH = ()


class FragLinkerData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # @staticmethod
    # def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
    #     instance = FragLinkerData(**kwargs)
    #
    #     if protein_dict is not None:
    #         for key, item in protein_dict.items():
    #             instance['protein_' + key] = item
    #
    #     if ligand_dict is not None:
    #         for key, item in ligand_dict.items():
    #             instance['ligand_' + key] = item
    #
    #     instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1])
    #                                               if instance.ligand_bond_index[0, k].item() == i]
    #                                    for i in instance.ligand_bond_index[0]}
    #     return instance

    # def __inc__(self, key, value, *args, **kwargs):
    #     if key == 'ligand_bond_index':
    #         return self['ligand_element'].size(0)
    #     # elif key == 'ligand_context_bond_index':
    #     #     return self['ligand_context_element'].size(0)
    #     else:
    #         return super().__inc__(key, value)


def batch_from_data_list(data_list):
    return Batch.from_data_list(data_list, follow_batch=FOLLOW_BATCH)


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output
