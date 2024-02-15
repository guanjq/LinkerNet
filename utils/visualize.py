import py3Dmol
import os
import copy
from rdkit import Chem
from rdkit.Chem import Draw


def visualize_complex(pdb_block, sdf_block, show_protein_surface=True, show_ligand=True, show_ligand_surface=True):
    view = py3Dmol.view()

    # Add protein to the canvas
    view.addModel(pdb_block, 'pdb')
    if show_protein_surface:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'}, {'model': -1})
    else:
        view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
    view.setStyle({'model': -1}, {"cartoon": {"style": "edged", 'opacity': 0}})

    # Add ligand to the canvas
    if show_ligand:
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model': -1}, {'stick': {}})
        # view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
        if show_ligand_surface:
            view.addSurface(py3Dmol.VDW, {'opacity': 0.8}, {'model': -1})

    view.zoomTo()
    return view


def visualize_complex_with_frags(pdb_block, all_frags, show_protein_surface=True, show_ligand=True, show_ligand_surface=True):
    view = py3Dmol.view()

    # Add protein to the canvas
    view.addModel(pdb_block, 'pdb')
    if show_protein_surface:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'}, {'model': -1})
    else:
        view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
    view.setStyle({'model': -1}, {"cartoon": {"style": "edged", 'opacity': 0}})

    # Add ligand to the canvas
    if show_ligand:
        for frag in all_frags:
            sdf_block = Chem.MolToMolBlock(frag)
            view.addModel(sdf_block, 'sdf')
            view.setStyle({'model': -1}, {'stick': {}})
            # view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
            if show_ligand_surface:
                view.addSurface(py3Dmol.VDW, {'opacity': 0.8}, {'model': -1})

    view.zoomTo()
    return view

def visualize_complex_highlight_pocket(pdb_block, sdf_block,
                                       pocket_atom_idx, pocket_res_idx=None, pocket_chain=None,
                                       show_ligand=True, show_ligand_surface=True):
    view = py3Dmol.view()

    # Add protein to the canvas
    view.addModel(pdb_block, 'pdb')
    view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'}, {'model': -1})
    if pocket_atom_idx:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'red'}, {'model': -1, 'serial': pocket_atom_idx})
    if pocket_res_idx:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'red'},
                        {'model': -1, 'chain': pocket_chain, 'resi': list(set(pocket_res_idx))})
    # color_map = ['red', 'yellow', 'blue', 'green']
    # for idx, pocket_atom_idx in enumerate(all_pocket_atom_idx):
    #     print(pocket_atom_idx)
    #     view.addSurface(py3Dmol.VDW, {'opacity':0.7, 'color':color_map[idx]}, {'model': -1, 'serial': pocket_atom_idx})
    # view.addSurface(py3Dmol.VDW, {'opacity':0.7,'color':'red'}, {'model': -1, 'resi': list(set(pocket_residue))})

    # view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
    view.setStyle({'model': -1}, {"cartoon": {"style": "edged", 'opacity': 0}})
    # view.setStyle({'model': -1, 'serial': atom_idx},  {'cartoon': {'color': 'red'}})
    # view.setStyle({'model': -1, 'resi': [482, 484]},  {'cartoon': {'color': 'green'}})

    # Add ligand to the canvas
    if show_ligand:
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model': -1}, {'stick': {}})
        # view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
        if show_ligand_surface:
            view.addSurface(py3Dmol.VDW, {'opacity': 0.8}, {'model': -1})

    view.zoomTo()
    return view


def visualize_mol_highlight_fragments(mol, match_list):
    all_target_atm = []
    for match in match_list:
        target_atm = []
        for atom in mol.GetAtoms():
            if atom.GetIdx() in match:
                target_atm.append(atom.GetIdx())
        all_target_atm.append(target_atm)

    return Draw.MolsToGridImage([mol for _ in range(len(match_list))], highlightAtomLists=all_target_atm,
                                subImgSize=(400, 400), molsPerRow=4)


def visualize_generated_xyz_v2(atom_pos, atom_type, protein_path, ligand_path=None, show_ligand=False, show_protein_surface=True):
    ptable = Chem.GetPeriodicTable()

    num_atoms = len(atom_pos)
    xyz = "%d\n\n" % (num_atoms,)
    for i in range(num_atoms):
        symb = ptable.GetElementSymbol(atom_type[i])
        x, y, z = atom_pos[i]
        xyz += "%s %.8f %.8f %.8f\n" % (symb, x, y, z)

    # print(xyz)

    with open(protein_path, 'r') as f:
        pdb_block = f.read()

    view = py3Dmol.view()
    # Generated molecule
    view.addModel(xyz, 'xyz')
    view.setStyle({'model': -1}, {'sphere': {'radius': 0.3}, 'stick': {}})

    # Pocket
    view.addModel(pdb_block, 'pdb')
    if show_protein_surface:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'}, {'model': -1})
    else:
        view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
    view.setStyle({'model': -1}, {"cartoon": {"style": "edged", 'opacity': 0}})

    # Focus on the generated
    view.zoomTo()

    # Ligand
    if show_ligand:
        with open(ligand_path, 'r') as f:
            sdf_block = f.read()
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model': -1}, {'stick': {}})

    return view


def visualize_generated_xyz(data, root, show_ligand=False):
    ptable = Chem.GetPeriodicTable()

    num_atoms = data.ligand_context_element.size(0)
    xyz = "%d\n\n" % (num_atoms,)
    for i in range(num_atoms):
        symb = ptable.GetElementSymbol(data.ligand_context_element[i].item())
        x, y, z = data.ligand_context_pos[i].clone().cpu().tolist()
        xyz += "%s %.8f %.8f %.8f\n" % (symb, x, y, z)

    # print(xyz)

    protein_path = os.path.join(root, data.protein_filename)
    ligand_path = os.path.join(root, data.ligand_filename)

    with open(protein_path, 'r') as f:
        pdb_block = f.read()
    with open(ligand_path, 'r') as f:
        sdf_block = f.read()

    view = py3Dmol.view()
    # Generated molecule
    view.addModel(xyz, 'xyz')
    view.setStyle({'model': -1}, {'sphere': {'radius': 0.3}, 'stick': {}})
    # Focus on the generated
    view.zoomTo()

    # Pocket
    view.addModel(pdb_block, 'pdb')
    view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
    # Ligand
    if show_ligand:
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model': -1}, {'stick': {}})

    return view


def visualize_generated_sdf(data, protein_path, ligand_path, show_ligand=False, show_protein_surface=True):
    # protein_path = os.path.join(root, data.protein_filename)
    # ligand_path = os.path.join(root, data.ligand_filename)

    with open(protein_path, 'r') as f:
        pdb_block = f.read()

    view = py3Dmol.view()
    # Generated molecule
    mol_block = Chem.MolToMolBlock(data.rdmol)
    view.addModel(mol_block, 'sdf')
    view.setStyle({'model': -1}, {'sphere': {'radius': 0.3}, 'stick': {}})
    # Focus on the generated
    # view.zoomTo()

    # Pocket
    view.addModel(pdb_block, 'pdb')
    if show_protein_surface:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'}, {'model': -1})
        view.setStyle({'model': -1}, {"cartoon": {"style": "edged", 'opacity': 0}})
    else:
        view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
    # Ligand
    if show_ligand:
        with open(ligand_path, 'r') as f:
            sdf_block = f.read()
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model': -1}, {'stick': {}})
    view.zoomTo()
    return view


def visualize_generated_arms(data_list, protein_path, ligand_path, show_ligand=False, show_protein_surface=True):
    # protein_path = os.path.join(root, data.protein_filename)
    # ligand_path = os.path.join(root, data.ligand_filename)

    with open(protein_path, 'r') as f:
        pdb_block = f.read()

    view = py3Dmol.view()
    # Generated molecule
    for data in data_list:
        mol_block = Chem.MolToMolBlock(data.rdmol)
        view.addModel(mol_block, 'sdf')
        view.setStyle({'model': -1}, {'sphere': {'radius': 0.3}, 'stick': {}})
    # Focus on the generated
    # view.zoomTo()

    # Pocket
    view.addModel(pdb_block, 'pdb')
    if show_protein_surface:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'}, {'model': -1})
        view.setStyle({'model': -1}, {"cartoon": {"style": "edged", 'opacity': 0}})
    else:
        view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
    # Ligand
    if show_ligand:
        with open(ligand_path, 'r') as f:
            sdf_block = f.read()
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model': -1}, {'stick': {}})
    view.zoomTo()
    return view


def visualize_ligand(mol, size=(300, 300), style="stick", surface=False, opacity=0.5, viewer=None):
    """Draw molecule in 3D

    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    assert style in ('line', 'stick', 'sphere', 'carton')
    if viewer is None:
        viewer = py3Dmol.view(width=size[0], height=size[1])
    if isinstance(mol, list):
        for i, m in enumerate(mol):
            mblock = Chem.MolToMolBlock(m)
            viewer.addModel(mblock, 'mol' + str(i))
    elif len(mol.GetConformers()) > 1:
        for i in range(len(mol.GetConformers())):
            mblock = Chem.MolToMolBlock(mol, confId=i)
            viewer.addModel(mblock, 'mol' + str(i))
    else:
        mblock = Chem.MolToMolBlock(mol)
        viewer.addModel(mblock, 'mol')
    viewer.setStyle({style: {}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer


def visualize_full_mol(frags_mol, linker_pos, linker_type):
    ptable = Chem.GetPeriodicTable()

    num_atoms = len(linker_pos)
    xyz = "%d\n\n" % (num_atoms,)
    for i in range(num_atoms):
        symb = ptable.GetElementSymbol(linker_type[i])
        x, y, z = linker_pos[i]
        xyz += "%s %.8f %.8f %.8f\n" % (symb, x, y, z)

    view = py3Dmol.view()
    # Generated molecule
    view.addModel(xyz, 'xyz')
    view.setStyle({'model': -1}, {'sphere': {'radius': 0.3}, 'stick': {}})

    mblock = Chem.MolToMolBlock(frags_mol)
    view.addModel(mblock, 'sdf')
    view.setStyle({'model': -1}, {'stick': {}})
    view.zoomTo()
    return view


def mol_with_atom_index(mol):
    mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol
