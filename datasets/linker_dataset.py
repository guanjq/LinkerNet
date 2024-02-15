import os
import pickle
import random

import joblib
import lmdb
import numpy as np
import torch
from networkx.algorithms import isomorphism
from rdkit import Chem
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm

from utils import frag_utils
from utils.data import process_from_mol
from utils.geometry import find_axes, global_to_local, rotation_matrix_from_vectors
from .linker_data import FragLinkerData, torchify_dict
from utils.transforms import dataset_info


def get_linker_dataset(cfg, transform_map, **kwargs):
    name = cfg.name
    root = cfg.path
    index_name = cfg.index_name
    if name == 'zinc':
        dataset = LinkerDataset(root, dset_name=name, version=cfg.version, index_name=index_name, **kwargs)
    elif name == 'protac':
        dataset = LinkerDataset(root, dset_name=name, version=cfg.version, index_name=index_name,
                                split_mode=cfg.split_mode, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if dataset.split is not None:
        subsets = {k: Subset(dataset, indices=v) for k, v in dataset.split.items()}
        t_subsets = {}
        for k, v in subsets.items():
            if transform_map is not None:
                t_subsets[k] = MapDataset(v, transform_map[k])
            else:
                t_subsets[k] = MapDataset(v, None)
        return dataset, t_subsets
    else:
        if transform_map is not None:
            dataset = MapDataset(dataset, transform_map)
        return dataset


class MapDataset(Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        if self.map is not None:
            return self.map(self.dataset[index])
        else:
            return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class LinkerDataset(Dataset):

    def __init__(self, raw_path, dset_name='zinc',
                 transform=None, version='v1', split_mode=None, index_name='index.pkl', reset=False):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.dset_name = dset_name
        self.index_path = os.path.join(self.raw_path, index_name)
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_{version}.lmdb')
        self.split_mode = split_mode
        if self.split_mode is None:
            self.split_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_{version}_split.pt')
        else:
            self.split_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_{version}_{self.split_mode}_split.pt')

        self.transform = transform
        # self.mode = mode
        self.db = None
        self.keys = None
        self.split = None
        if not os.path.exists(self.processed_path) or reset:
            print(f'{self.processed_path} do not exist, begin processing data')
            self._process()
        print('Load dataset from %s' % self.processed_path)

        if not os.path.exists(self.split_path):
            print(f'{self.split_path} do not exist, begin splitting data')
            self._split_dataset()
        print('Load split file from %s' % self.split_path)
        self.split = torch.load(self.split_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    @staticmethod
    def process_data(index):
        rdmol = index['mol']
        ligand_dict = process_from_mol(rdmol)

        # align centers of fragments to the x-axis
        frags_t = []
        for ind_key in ['atom_indices_f1', 'atom_indices_f2']:
            f_pos = ligand_dict['pos'][index[ind_key]]
            frags_t.append(f_pos.mean(-2))
        rot = rotation_matrix_from_vectors(
            torch.from_numpy(frags_t[1] - frags_t[0]), torch.tensor([1., 0., 0.])).numpy()
        tr = -rot @ ((frags_t[0] + frags_t[1]) / 2)
        ligand_dict['pos'] = (rot @ ligand_dict['pos'].T).T + tr

        # compute rotation and translation
        frags_axes = []
        frags_t = []
        for ind_key in ['atom_indices_f1', 'atom_indices_f2']:
            f_pos = ligand_dict['pos'][index[ind_key]]
            f_charge = np.array(
                [rdmol.GetAtomWithIdx(i).GetAtomicNum() for i in index[ind_key]], dtype=np.float32)
            axes = find_axes(f_pos, f_charge)
            frags_axes.append(axes)
            frags_t.append(f_pos.mean(-2))

        ligand_dict['frags_R'] = np.stack(frags_axes)
        ligand_dict['frags_t'] = np.stack(frags_t)
        # assert np.allclose(ligand_dict['frags_t'][:, 1:], 0.), ligand_dict['frags_t']
        # assert np.allclose(ligand_dict['frags_t'].sum(0), 0.), ligand_dict['frags_t']

        ligand_dict.update(index)
        data = FragLinkerData(**torchify_dict(ligand_dict))
        f1_local_pos = global_to_local(data.frags_R[0], data.frags_t[0], data.pos[data.fragment_mask == 1])
        f2_local_pos = global_to_local(data.frags_R[1], data.frags_t[1], data.pos[data.fragment_mask == 2])

        data.frags_local_pos = torch.cat([f1_local_pos.squeeze(0), f2_local_pos.squeeze(0)])
        data.frags_d = data.frags_t[1][0] - data.frags_t[0][0]

        # anchor
        link_indices = (data.linker_mask == 1).nonzero()[:, 0].tolist()
        frag_indices = (data.linker_mask == 0).nonzero()[:, 0].tolist()
        anchor_indices = [j for i, j in zip(*data.bond_index.tolist()) if i in link_indices and j in frag_indices]
        # assert len(anchor_indices) == 2 # mols in PROTAC-DB may have >2 anchors
        data.anchor_indices = anchor_indices
        anchor_mask = torch.zeros_like(data.fragment_mask)
        anchor_mask[anchor_indices] = 1
        data.anchor_mask = anchor_mask
        data.nbh_list = {i.item(): [j.item() for k, j in enumerate(data.bond_index[1])
                                    if data.bond_index[0, k].item() == i] for i in data.bond_index[0]}
        return data

    def process_single_data(self, task):
        index = task['entry']
        data = self.process_data(index)
        return {
            'data': data,
            'subset': task['subset']
        }

    def _process(self):
        with open(self.index_path, 'rb') as f:
            all_index = pickle.load(f)

        if self.dset_name == 'zinc':
            tasks = []
            for subset in all_index.keys():
                for entry in all_index[subset]:
                    tasks.append({
                        'id': entry['id'],
                        'entry': entry,
                        'subset': subset,
                    })
            data_list = joblib.Parallel(
                n_jobs=max(joblib.cpu_count() // 2, 1),
            )(
                joblib.delayed(self.process_single_data)(task)
                for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess')
            )
            db = lmdb.open(
                self.processed_path,
                map_size=10 * (1024 * 1024 * 1024),  # 10GB
                create=True,
                subdir=False,
                readonly=False,  # Writable
            )
            num_skipped = 0
            num_data = 0
            split = {k: [] for k in all_index.keys()}
            with db.begin(write=True, buffers=True) as txn:
                for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                    if data is None:
                        num_skipped += 1
                        continue
                    txn.put(
                        key=f'{num_data:08d}'.encode(),
                        value=pickle.dumps(data['data'])
                    )
                    split[data['subset']].append(num_data)
                    num_data += 1
            db.close()
            print(f'Processed {num_data}; Skipped {num_skipped}')
            torch.save(split, self.split_path)
        elif self.dset_name == 'protac':
            data_list = []
            dset_info = dataset_info(self.dset_name)
            num_skipped = 0
            num_data = 0
            for idx, index in enumerate(tqdm(all_index)):
                new_index = preprocess_protac_index(index)
                data = self.process_data(new_index)
                if data is None:
                    num_skipped += 1
                    continue
                # filter out invalid atom types
                valid = True
                for e, v, c in zip(data.element, data.valence, data.charge):
                    pt = Chem.GetPeriodicTable()
                    atom_str = "%s%i(%i)" % (pt.GetElementSymbol(int(e)), int(v), int(c))
                    if atom_str not in dset_info['atom_types']:
                        valid = False
                        print('Found invalid atom type: ', atom_str)
                        num_skipped += 1
                        break
                if valid:
                    num_data += 1
                    data_list.append(data)

            db = lmdb.open(
                self.processed_path,
                map_size=1 * (1024 * 1024 * 1024),  # 1GB
                create=True,
                subdir=False,
                readonly=False,  # Writable
            )
            with db.begin(write=True, buffers=True) as txn:
                for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                    txn.put(
                        key=f'{num_data:08d}'.encode(),
                        value=pickle.dumps(data)
                    )
            db.close()
            print(f'Processed {num_data}; Skipped {num_skipped}')
        else:
            raise ValueError(self.dset_name)

    def _split_dataset(self):
        if self.dset_name == 'protac':
            if self.db is None:
                self._connect_db()

            warheads, ligases, linkers = [], [], []
            for idx in range(len(self.keys)):
                key = self.keys[idx]
                data = pickle.loads(self.db.begin().get(key))
                warheads.append(data.smi_warhead)
                ligases.append(data.smi_ligase)
                linkers.append(data.smi_linker)

            split = {'train': [], 'test': []}
            if self.split_mode == 'warhead':
                test_warheads = list(set(warheads))[-10:]
                for idx in range(len(self.keys)):
                    key = self.keys[idx]
                    data = pickle.loads(self.db.begin().get(key))
                    if data.smi_warhead in test_warheads:
                        split['test'].append(idx)
                    else:
                        split['train'].append(idx)
            elif self.split_mode == 'ligase':
                test_ligases = list(set(ligases))[-5:]
                for idx in range(len(self.keys)):
                    key = self.keys[idx]
                    data = pickle.loads(self.db.begin().get(key))
                    if data.smi_ligase in test_ligases:
                        split['test'].append(idx)
                    else:
                        split['train'].append(idx)
            elif self.split_mode == 'random':
                all_ids = list(range(len(self.keys)))
                random.Random(2023).shuffle(all_ids)
                split['train'] = all_ids[:-100]
                split['test'] = all_ids[-100:]
            split['val'] = split['test']
        else:
            raise ValueError(self.dset_name)
        torch.save(split, self.split_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data.id = idx
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_raw_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data.id = idx
        return data


def preprocess_protac_index(index):
    rdmol = index['mol']
    new_mol = Chem.RenumberAtoms(rdmol, list(index['atom_indices_warhead']) + list(index['atom_indices_ligase']) + list(
        index['atom_indices_linker']))
    num_atoms_f1 = len(index['atom_indices_warhead'])
    num_atoms_f2 = len(index['atom_indices_ligase'])
    atom_indices_f1 = list(range(num_atoms_f1))
    atom_indices_f2 = list(range(num_atoms_f1, num_atoms_f1 + num_atoms_f2))
    fragment_mask = torch.zeros(rdmol.GetNumAtoms()).long()
    fragment_mask[atom_indices_f1] = 1
    fragment_mask[atom_indices_f2] = 2
    linker_mask = (fragment_mask == 0)

    # extract frag mol directly from new_mol, in case the Kekulize error
    bond_ids = []
    for bond_idx, bond in enumerate(new_mol.GetBonds()):
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if (fragment_mask[start] > 0) == (fragment_mask[end] == 0):
            bond_ids.append(bond_idx)
    break_mol = Chem.FragmentOnBonds(new_mol, bond_ids, addDummies=False)
    submols = Chem.GetMolFrags(break_mol, asMols=True)
    assert len(submols) == 3
    link_mol = submols[-1]

    G1 = frag_utils.topology_from_rdkit(link_mol)
    G2 = frag_utils.topology_from_rdkit(Chem.MolFromSmiles(index['smi_linker']))
    GM = isomorphism.GraphMatcher(G1, G2)
    flag = GM.is_isomorphic()
    if not flag:
        print('Linkers mismatch!')
        return None
    frag_mol = Chem.CombineMols(submols[0], submols[1])

    new_index = {
        'smiles': Chem.MolToSmiles(new_mol),
        'smi_warhead': index['smi_warhead'],
        'smi_ligase': index['smi_ligase'],
        'smi_linker': index['smi_linker'],
        'mol': new_mol,
        'frag_mol': frag_mol,
        'link_mol': link_mol,
        'atom_indices_f1': atom_indices_f1,
        'atom_indices_f2': atom_indices_f2,
        'fragment_mask': fragment_mask,
        'linker_mask': linker_mask,
    }
    return new_index
