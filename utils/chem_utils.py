# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

'''
beta version
'''
import math
from copy import copy, deepcopy
from tqdm import tqdm
from queue import Queue

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import calinski_harabasz_score

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw, rdFMCS
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import Descriptors, rdmolops
from rdkit.Chem.rdMolAlign import CalcRMS
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate

MAX_VALENCE = {'B': 3, 'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':3, 'O':2, 'P':5, 'S':6} #, 'Se':4, 'Si':4}
Bond_List = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


class AtomVocab:
    def __init__(self):
        # atom
        self.idx2atom = list(MAX_VALENCE.keys())
        self.atom2idx = { atom: i for i, atom in enumerate(self.idx2atom) }
        # bond
        self.idx2bond = copy(Bond_List)
        self.bond2idx = { bond: i for i, bond in enumerate(self.idx2bond) }
        
    def idx_to_atom(self, idx):
        return self.idx2atom[idx]
    
    def atom_to_idx(self, atom):
        return self.atom2idx[atom]

    def idx_to_bond(self, idx):
        return self.idx2bond[idx]
    
    def bond_to_idx(self, bond):
        return self.bond2idx[bond]
    
    def num_atom_type(self):
        return len(self.idx2atom)
    
    def num_bond_type(self):
        return len(self.idx2bond)

    

def smi2mol(smiles: str, kekulize=False, sanitize=True):
    '''turn smiles to molecule'''
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if kekulize:
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol, True)
    return mol


def mol2smi(mol):
    return Chem.MolToSmiles(mol)


def canonical_order(smiles: str):
    mol = smi2mol(smiles)
    smi = Chem.MolToSmiles(mol, canonical=True)
    return smi


def fingerprint(mol):
    return AllChem.GetMorganFingerprint(mol, 2)


def similarity(mol1, mol2):
    if isinstance(mol1, str):
        mol1 = smi2mol(mol1)
    if isinstance(mol2, str):
        mol2 = smi2mol(mol2)
    fps1 = fingerprint(mol1)
    fps2 = fingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fps1, fps2)


def fingerprint2numpy(fingerprint):
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint,arr)
    return arr


def pkd(kd):
    return -math.log(kd, 10)


def draw_molecules(*mols, canvas_size=(500, 500), useSVG=False, molsPerRow=3,
                   highlightAtomLists=None, highlightBondLists=None, legends=None,
                   save_path=None):
    rdkit_mols = []
    for mol in mols:
        if isinstance(mol, str): # smiles
            mol = smi2mol(mol)
        rdkit_mols.append(mol)
    image = Draw.MolsToGridImage(rdkit_mols, molsPerRow=molsPerRow, subImgSize=canvas_size, useSVG=useSVG,
                                 highlightAtomLists=highlightAtomLists, highlightBondLists=highlightBondLists,
                                 legends=legends)
    if save_path is not None:
        img_type = 'SVG' if useSVG else 'PNG'
        image.save(save_path, img_type)
    return image


def diversity(mols):
    '''
    \frac{2}{n(n-1)} \sum_{i < j} (1 - similarity(mol_i, mol_j))
    '''
    mols = [smi2mol(mol) if isinstance(mol, str) else mol for mol in mols]
    n = len(mols)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(similarity(mols[i], mols[j]))
    assert len(sims) == n * (n - 1) / 2
    diversity = 1 - sum(sims) / len(sims)
    return diversity


def calculate_1dqsar_repr(mol):
    # 从 SMILES 字符串创建分子对象
    if isinstance(mol, str):
        mol = smi2mol(mol)
    # 计算分子的分子量
    mol_weight = Descriptors.MolWt(mol)
    # 计算分子的 LogP 值
    log_p = Descriptors.MolLogP(mol)
    # 计算分子中的氢键供体数量
    num_h_donors = Descriptors.NumHDonors(mol)
    # 计算分子中的氢键受体数量
    num_h_acceptors = Descriptors.NumHAcceptors(mol)
    # 计算分子的表面积极性
    tpsa = Descriptors.TPSA(mol)
    # 计算分子中的可旋转键数量
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    # 计算分子中的芳香环数量
    num_aromatic_rings = Descriptors.NumAromaticRings(mol)
    # 计算分子中的脂环数量
    num_aliphatic_rings = Descriptors.NumAliphaticRings(mol)
    # 计算分子中的饱和环数量
    num_saturated_rings = Descriptors.NumSaturatedRings(mol)
    # 计算分子中的杂原子数量
    num_heteroatoms = Descriptors.NumHeteroatoms(mol)
    # 计算分子中的价电子数量
    num_valence_electrons = Descriptors.NumValenceElectrons(mol)
    # 计算分子中的自由基电子数量
    num_radical_electrons = Descriptors.NumRadicalElectrons(mol)
    # 计算分子的 QED 值
    qed = Descriptors.qed(mol)
    # 返回所有计算出的属性值
    return [mol_weight, log_p, num_h_donors, num_h_acceptors, tpsa, num_rotatable_bonds, num_aromatic_rings,
            num_aliphatic_rings, num_saturated_rings, num_heteroatoms, num_valence_electrons, num_radical_electrons,qed]


def find_cliques(mol):
    '''
    Cluster:
      1. a rotatable bond which is not in a ring
      2. rings in the smallest set of smallest rings (SSSR)
    '''
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1: #special case
        return [(0,)], [[0]]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append( (a1,a2) )

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    atom_cls = [[] for i in range(n_atoms)]
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls


def clustering(smiles_list, K=None, sim_mat=None, random_state=None):

    if sim_mat is None:
        mols = [smi2mol(smi) for smi in smiles_list]
        fps = [fingerprint(mol) for mol in mols]

        sim_mat = [[0 for _ in range(len(fps))] for _ in range(len(fps))]

        # 1D similarity
        qsars = np.array([calculate_1dqsar_repr(mol) for mol in mols])  # [N, n_feat]
        # normalize
        mean, std = np.mean(qsars, axis=0), np.std(qsars, axis=0) # [n_feat]
        qsars = (qsars - mean[None, :]) / (1e-16 + std[None, :]) # [N, n_feat]
        sim_qsar = cosine_similarity(qsars, qsars)

        # 2D similarity
        for i in tqdm(range(len(fps))):
            sim_mat[i][i] = 1.0
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                sim_mat[i][j] = sim_mat[j][i] = sim

        sim_mat = (np.array(sim_mat) + sim_qsar) / 2  # average
    
    if K is None:
        # try to find the best K
        best_K, best_score, best_labels = 0, 0, None
        K, patient = 2, 20
        eigen_values, eigen_vectors = np.linalg.eigh(sim_mat)
        vecs = eigen_vectors[:, -100:]
        while True:
            cls_labels = SpectralClustering(K, affinity='precomputed', random_state=random_state).fit_predict(sim_mat)
            score = calinski_harabasz_score(vecs, cls_labels)
            print(f'Trying K = {K}, CH-score = {score}, patience = {patient}')
            if score > best_score:
                best_K, best_score, best_labels = K, score, cls_labels
                patient = 20
            else:
                patient -= 1
            if patient < 0:
                break
            K += 1
        print(f'Best K = {best_K}, score = {best_score}')
        cls_labels = best_labels
    else:
        cls_labels = SpectralClustering(K, affinity='precomputed', random_state=random_state).fit_predict(sim_mat)

    # if visualize:
    #     eigen_values, eigen_vectors = np.linalg.eigh(sim_mat)
    #     high_dim_scatterplot({'vec': eigen_vectors[:, -20:], 'cluster': [str(c) for c in cls_labels]},
    #                          vec='vec', hue='cluster', save_path=os.path.join(out_dir, 'cluster_vis.png'),
    #                          hue_order=[str(i) for i in range(10)])

    return cls_labels, sim_mat


def get_R_N_sub(smiles, beta_aa=False):
    mol = smi2mol(smiles)
    frame = 'OC(=O)CCN' if beta_aa else 'OC(=O)CN'
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(frame))
    assert len(matches) == 1, f'{smiles}: number of amino-bond {len(matches)}'
    
    # find N and R carbon
    frame_atom_idx = { atom_idx: True for atom_idx in matches[0] }
    N_idx, R_C_idx = None, None
    for atom_idx in matches[0]:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() == 7: # is N
            N_idx = atom_idx
            break
    
    for atom_idx in matches[0]:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() == 6: # is C
            bonds = atom.GetBonds()
            for bond in bonds:
                if bond.GetBeginAtomIdx() == N_idx or bond.GetEndAtomIdx() == N_idx:
                    R_C_idx = atom_idx
                    break
            if R_C_idx is not None:
                break
    
    # N substitute
    N = mol.GetAtomWithIdx(N_idx)
    N_sub_bond = None
    for bond in N.GetBonds():
        if bond.GetBeginAtomIdx() != R_C_idx and bond.GetEndAtomIdx() != R_C_idx:
            assert N_sub_bond is None
            N_sub_bond = bond

    cycle = False
    if N_sub_bond is None: # no substituent on Nitrogen
        N_sub = None
    else:
        frags = Chem.GetMolFrags(Chem.FragmentOnBonds(Chem.Mol(mol), [N_sub_bond.GetIdx()], addDummies=False), asMols=True)
        if len(frags) == 1:
            cycle = True
        else:
            matches = frags[0].GetSubstructMatches(Chem.MolFromSmarts(frame))
            N_sub = frags[0] if len(matches) == 0 else frags[1]
    
    # R group
    RC = mol.GetAtomWithIdx(R_C_idx)
    R_bonds = []
    for bond in RC.GetBonds():
        if bond.GetBeginAtomIdx() not in frame_atom_idx or bond.GetEndAtomIdx() not in frame_atom_idx:
            R_bonds.append(bond)

    if len(R_bonds) == 0: # no R group
        R = None
    else:
        if not cycle:
            frags = Chem.GetMolFrags(Chem.FragmentOnBonds(Chem.Mol(mol), [b.GetIdx() for b in R_bonds], addDummies=False), asMols=True)
            matches = frags[0].GetSubstructMatches(Chem.MolFromSmarts(frame))
            R = frags[0] if len(matches) == 0 else frags[1]
    if cycle and len(R_bonds) == 1: # one cycle
        frags = Chem.GetMolFrags(Chem.FragmentOnBonds(Chem.Mol(mol), [N_sub_bond.GetIdx()] + [b.GetIdx() for b in R_bonds], addDummies=False), asMols=True)
        assert len(frags) == 2, smiles
        matches = frags[0].GetSubstructMatches(Chem.MolFromSmarts(frame))
        N_sub = R = (frags[0] if len(matches) == 0 else frags[1])
    elif cycle and len(R_bonds) > 1: # one cycle and an R group
        frags = Chem.GetMolFrags(Chem.FragmentOnBonds(Chem.Mol(mol), [b.GetIdx() for b in R_bonds], addDummies=False), asMols=True)
        matches = frags[0].GetSubstructMatches(Chem.MolFromSmarts(frame))
        R = (frags[0] if len(matches) == 0 else frags[1])
        frags = Chem.GetMolFrags(Chem.FragmentOnBonds(Chem.Mol(mol), [N_sub_bond.GetIdx()] + [b.GetIdx() for b in R_bonds], addDummies=False), asMols=True)
        R_smi = mol2smi(R)
        R_idx = None
        for i, frag in enumerate(frags):
            if mol2smi(frag) == R_smi:
                R_idx = i
                break
        frags = [f for i, f in enumerate(frags) if i != R_idx]
        assert len(frags) == 2
        matches = frags[0].GetSubstructMatches(Chem.MolFromSmarts(frame))
        N_sub = (frags[0] if len(matches) == 0 else frags[1])

    return (N_sub_bond, N_sub), (R_bonds, R)


def get_submol(mol, atom_indices, kekulize=False):
    if len(atom_indices) == 1:
        atom_symbol = mol.GetAtomWithIdx(atom_indices[0]).GetSymbol()
        atom_symbol = f'[{atom_symbol}]' # single atoms
        return smi2mol(atom_symbol, kekulize)
    aid_dict = { i: True for i in atom_indices }
    edge_indices = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        begin_aid = bond.GetBeginAtomIdx()
        end_aid = bond.GetEndAtomIdx()
        if begin_aid in aid_dict and end_aid in aid_dict:
            edge_indices.append(i)
    mol = Chem.PathToSubmol(mol, edge_indices)
    return mol


def get_submol_atom_map(mol, submol, group, kekulize=False):
    if len(group) == 1:
        return { group[0]: 0 }
    # turn to smiles order
    smi = mol2smi(submol)
    submol = smi2mol(smi, kekulize, sanitize=False)
    # # special with N+ and N-
    # for atom in submol.GetAtoms():
    #     if atom.GetSymbol() != 'N':
    #         continue
    #     if (atom.GetExplicitValence() == 3 and atom.GetFormalCharge() == 1) or atom.GetExplicitValence() < 3:
    #         atom.SetNumRadicalElectrons(0)
    #         atom.SetNumExplicitHs(2)
    
    matches = mol.GetSubstructMatches(submol)
    old2new = { i: 0 for i in group }  # old atom idx to new atom idx
    found = False
    for m in matches:
        hit = True
        for i, atom_idx in enumerate(m):
            if atom_idx not in old2new:
                hit = False
                break
            old2new[atom_idx] = i
        if hit:
            found = True
            break
    assert found
    return old2new


def cnt_atom(smi, return_dict=False):
    atom_dict = { atom: 0 for atom in MAX_VALENCE }
    for i in range(len(smi)):
        symbol = smi[i].upper()
        next_char = smi[i+1] if i+1 < len(smi) else None
        if symbol == 'B' and next_char == 'r':
            symbol += next_char
        elif symbol == 'C' and next_char == 'l':
            symbol += next_char
        if symbol in atom_dict:
            atom_dict[symbol] += 1
    if return_dict:
        return atom_dict
    else:
        return sum(atom_dict.values())


def bond_to_valence(bond_type):
    if bond_type == BondType.SINGLE:
        return 1
    elif bond_type == BondType.DOUBLE:
        return 2
    elif bond_type == BondType.TRIPLE:
        return 3
    elif bond_type == BondType.AROMATIC:
        return 1.5
    return 0


def get_atom_valence(mol, idx):
    val = 0
    for bond in mol.GetAtomWithIdx(idx).GetBonds():
        val += bond_to_valence(bond.GetBondType())
    return val


def valence_check(mol, idx1, idx2, bond_type):
    new_valence = bond_to_valence(bond_type)
    if new_valence == 0:
        return False
    atom1 = mol.GetAtomWithIdx(idx1).GetSymbol()
    atom2 = mol.GetAtomWithIdx(idx2).GetSymbol()
    c1 = mol.GetAtomWithIdx(idx1).GetFormalCharge()
    c2 = mol.GetAtomWithIdx(idx2).GetFormalCharge()
    a1_val = get_atom_valence(mol, idx1)
    a2_val = get_atom_valence(mol, idx2)
    # # special for S as S is likely to have either 2 or 6 valence
    # if (atom1 == 'S' and a1_val == 2) or (atom2 == 'S' and a2_val == 2):
    #     return False
    if (atom1 not in MAX_VALENCE) or (atom2 not in MAX_VALENCE): # rare atoms, such as Se
        return False
    return a1_val + new_valence + abs(c1) <= MAX_VALENCE[atom1] and \
           a2_val + new_valence + abs(c2) <= MAX_VALENCE[atom2]


def shortest_path_len(mol, i, j):
    """
    Find the shortest path between two atoms in a molecule.

    Args:
        mol (rdkit.Chem.Mol): The molecule object.
        i (int): Index of the start atom.
        j (int): Index of the target atom.

    Returns:
        tuple: (length of the shortest path, list of atom indices in the path)
               Returns (None, None) if no path is found.
    """
    queue = Queue()
    queue.put((mol.GetAtomWithIdx(i), [i]))
    visited = {}
    visited[i] = True
    while not queue.empty():
        atom, path = queue.get()
        for nei in atom.GetNeighbors():
            idx = nei.GetIdx()
            if idx == j:
                return len(path) + 1, path + [idx]
            if idx not in visited:
                visited[idx] = True
                queue.put((mol.GetAtomWithIdx(idx), path + [idx]))
    return None, None


def cycle_check(mol, idx1, idx2, bond_type):
    '''
        Check cycle constraints if we connect atom 1 to atom 2
    '''
    try: Chem.SanitizeMol(mol)
    except Exception: return False
    new_cycle_len, path = shortest_path_len(mol, idx1, idx2)

    # single ring check
    if new_cycle_len is None: return True # not connected, no new ring will form by connecting this bond
    elif new_cycle_len <= 4: return False # forbid 3-ring or 4-ring for interconnection (some fragment may contain common 4-ring)
    # elif new_cycle_len == 4:
    #     # drop 4-ring with double/triple/aromatic bonds
    #     if bond_type != BondType.SINGLE: return False
    #     for i, begin_idx in enumerate(path[:-1]):
    #         end_idx = path[i + 1]
    #         bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
    #         if bond.GetBondType() != BondType.SINGLE: return False  
    
    # get current rings
    ring2atom, atom2ring = {}, {}
    for ring_idx, atom_idxs in enumerate(mol.GetRingInfo().AtomRings()):
        ring2atom[ring_idx] = atom_idxs
        for a in atom_idxs:
            if a not in atom2ring: atom2ring[a] = []
            atom2ring[a].append(ring_idx)
    ring1s = set(atom2ring.get(idx1, []))
    ring2s = set(atom2ring.get(idx2, []))
    
    # share ring check
    share_ring = ring1s.intersection(ring2s)
    if len(share_ring) > 1: return False # already in bridged rings
    elif len(share_ring) == 1: # atom 1 and 2 are already in the same ring
        return len(ring2atom[share_ring.pop()]) > 5 # sharing one cycle, at least 6-ring can have two atoms connected
    else: # atom 1 and atom 2 are not sharing rings
        # atoms in the new cycle cannot already be in a bridged cycle
        for atom_idx in path:
            if atom_idx not in atom2ring: continue
            if len(atom2ring[atom_idx]) > 1: return False # this atom is in a bridged atom
        if new_cycle_len > 4: return True # forming at least 5-ring
        else: # forming 3-ring or 4-ring, may have stress
            # for each atom in the newly form ring, it cannot be in another 3-ring or 4-ring
            for atom_idx in path:
                if atom_idx not in atom2ring: continue  # this atom is not in any ring
                for ring_idx in atom2ring[atom_idx]:
                    ring_size = len(ring2atom[ring_idx])
                    if ring_size < 5: return False
            return True


def _single_side_sp2_check(mol, idx1, idx2, all_coords, tolerance=np.pi/36):
    '''
        check whether atom 1 is sp2 C and whether atom 2 is on the plane
        return True if the bond between atom 1 and 2 passes the check
        tolerance: default 5 degree
    '''
    atom1, atom2 = mol.GetAtomWithIdx(idx1), mol.GetAtomWithIdx(idx2)
    if atom1.GetSymbol() != 'C': return True    # only check C
    is_sp2 = False
    end_atoms = []
    for bond in atom1.GetBonds():
        if bond.GetBondType() == BondType.AROMATIC or bond.GetBondType() == BondType.DOUBLE:
            is_sp2 = True
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        end_atoms.append(end_idx if begin_idx == idx1 else begin_idx)
    
    if not is_sp2: return True  # not sp2 carbon
    if len(end_atoms) < 2: return True  # the plane is not established yet (only one connected atom)
    if len(end_atoms) > 2: return False    # one double bond and at least two single bonds, no room for another one

    pos1, pos2 = np.array(all_coords[idx1]), np.array(all_coords[idx2])
    end_pos1, end_pos2 = np.array(all_coords[end_atoms[0]]), np.array(all_coords[end_atoms[1]])

    u1, u2 = (pos1 - end_pos1), (pos2 - end_pos2)
    u1, u2 = u1 / np.linalg.norm(u1), u2 / np.linalg.norm(u2)
    ideal_direction = u1 + u2
    ideal_direction = ideal_direction / np.linalg.norm(ideal_direction)
    
    model_direction = (pos2 - pos1)
    model_direction = model_direction / np.linalg.norm(model_direction)

    angle = np.arccos(np.clip(np.dot(ideal_direction, model_direction), -1.0, 1.0))

    return np.abs(angle) < tolerance


def sp2_check(mol, idx1, idx2, all_coords):
    return _single_side_sp2_check(mol, idx1, idx2, all_coords) & _single_side_sp2_check(mol, idx2, idx1, all_coords)


def connect_fragments(mol, atom_coords):
    """
    Connects fragments in an RDKit molecule by adding bonds between atoms
    with unfulfilled valencies based on approximate atom coordinates.
    
    Args:
        mol (rdkit.Chem.RWMol): RDKit molecule object with potential fragments.
        atom_coords (list): Approximate 3D coordinates of each atom in mol.
        
    Returns:
        rdkit.Chem.Mol: RDKit molecule with added inter-fragment bonds.
    """
    atom_coords = np.array(atom_coords)

    # Identify initial fragments in the molecule
    fragments = list(rdmolops.GetMolFrags(mol, asMols=False))
    
    if len(fragments) == 1:
        return mol, []
    added_bonds = []
    unfulfilled_atoms = []  # Stores atoms with open valencies for each fragment
    idx_to_frag = {}

    # Step 1: Collect atoms with open valencies from each fragment
    delta_valence_map = {}
    for frag_idx, frag in enumerate(fragments):
        for atom_idx in frag:
            idx_to_frag[atom_idx] = frag_idx
            # Check if atom has unfulfilled valency (e.g., valency not satisfied)
            atom = mol.GetAtomWithIdx(atom_idx)
            # we prefer not to manually connect rare atoms to other atoms
            delta_valence = MAX_VALENCE.get(atom.GetSymbol(), 0) - (get_atom_valence(mol, atom_idx) + abs(atom.GetFormalCharge()))
            if delta_valence > 0:
                unfulfilled_atoms.append((atom_idx, frag_idx))
                delta_valence_map[atom_idx] = delta_valence
    
    # Step 2: Pair unfulfilled atoms between fragments based on distance and valency
    pairs_to_bond = []
    for i, (idx1, frag1) in enumerate(unfulfilled_atoms):
        for j, (idx2, frag2) in enumerate(unfulfilled_atoms):
            if i >= j: continue # avoid repeated bonds
            if frag1 != frag2:  # Only consider inter-fragment bonds
                dist = np.linalg.norm(atom_coords[idx1] - atom_coords[idx2])
                # Heuristic: distance must be reasonably small
                if dist < 3.0:  # Adjust threshold as needed
                    pairs_to_bond.append((idx1, idx2, dist))
    
    # Step 3: Sort pairs by distance and add bonds until all fragments are connected
    pairs_to_bond = sorted(pairs_to_bond, key=lambda x: x[2])  # Sort by distance
    for idx1, idx2, _ in pairs_to_bond:
        if idx_to_frag[idx1] == idx_to_frag[idx2]: continue # now they are in the same fragment
        # check if the valency is still ok
        if delta_valence_map[idx1] <= 0 or delta_valence_map[idx2] <= 0:
            continue
        # if not sp2_check(mol, idx1, idx2, atom_coords): continue
        # TODO: check whether the two fragments are still disconnected
        # Add bond between the closest eligible pair
        mol.AddBond(idx1, idx2, Chem.BondType.SINGLE)
        delta_valence_map[idx1] -= 1
        delta_valence_map[idx2] -= 1
        added_bonds.append((idx1, idx2, Chem.BondType.SINGLE))
        # Update fragments after adding each bond
        fragments = list(rdmolops.GetMolFrags(mol, asMols=False))
        if len(fragments) == 1:
            break  # Stop if all fragments are connected
        # Update fragment assignment
        idx_to_frag = {}
        for frag_idx, frag in enumerate(fragments):
            for atom_idx in frag: idx_to_frag[atom_idx] = frag_idx

    return mol, added_bonds

def _get_mol_from_sdf(sdf_path:str,removeHs:bool = True) -> Chem.Mol:
    mol = None
    with Chem.SDMolSupplier(sdf_path, removeHs=removeHs) as suppl:
        mol = suppl[0]
    if mol is None:
        with Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=removeHs) as suppl:
            mol = suppl[0]

    return mol

def _neutralize_atoms(mol):
    """Add and remove hydrogens to neutralize charges ignoring overall charge."""
    # https://www.rdkit.org/docs/Cookbook.html#neutralizing-charged-molecules
    # stronger than rdkit.Chem.MolStandardize.rdMolStandardize.Uncharger
    try:
        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
    except:
        return mol
    
def _remove_all_charges_and_hydrogens(mol):
    """
    Remove all charges and hydrogens from molecule
    Borrowed from https://github.com/maabuu/posebusters/blob/main/posebusters/tools/molecules.py#L77
    """
    try:
        # rdkit keeps hydrogens that define sterochemistry so remove stereo first
        Chem.RemoveStereochemistry(mol)
        mol = Chem.RemoveHs(mol)
        # remove charges
        mol = _neutralize_atoms(mol)
    except:
        return mol

def get_rdkit_rmsd(sdf_path_gen:str, sdf_path_ref:str) -> float:
    '''
    Compute RMSD by rdkit built-in functions, borrowed from: https://github.com/maabuu/posebusters/blob/48e3542c7dfd9a462c140706c9a32efeaa3a90f7/posebusters/modules/rmsd.py
    '''
    mol_gen = _get_mol_from_sdf(sdf_path_gen, removeHs=True)
    mol_ref = _get_mol_from_sdf(sdf_path_ref, removeHs=True) 
    mol_gen, mol_ref = deepcopy(mol_gen), deepcopy(mol_ref)
    # Round1. calculate directly
    try:
        rmsd = CalcRMS(mol_gen, mol_ref)
    except:
        rmsd = np.nan
    if not np.isnan(rmsd):
        return rmsd
    # Round2. igmoring charges or tautomers
    try:
        mol_ref_uncharged = _remove_all_charges_and_hydrogens(mol_ref)
        mol_gen_uncharged = _remove_all_charges_and_hydrogens(mol_gen)
        rmsd = CalcRMS(mol_gen_uncharged, mol_ref_uncharged)
    except:
        rmsd = np.nan
    if not np.isnan(rmsd):
        return rmsd
    # Round3. Assign Bond Orders
    try:
        mol_gen_with_bo = AssignBondOrdersFromTemplate(mol_ref, mol_gen)
        rmsd = CalcRMS(mol_gen_with_bo, mol_ref)
    except:
        rmsd = np.nan
    
    return rmsd

def get_rmsd_from_coord(sdf_path_gen:str, sdf_path_ref:str) -> float:
    """
    Get coordinates of both molecules, then mapping atoms by searching Max Common Substructure
    Modified from https://github.com/rdkit/rdkit/blob/1b9d0921b144671678640e52430fd299cf4b0f00/Contrib/CalcLigRMSD/CalcLigRMSD.py
    """
    mol_gen = _get_mol_from_sdf(sdf_path_gen, removeHs=True)
    mol_ref = _get_mol_from_sdf(sdf_path_ref, removeHs=True) 
    mol_gen, mol_ref = deepcopy(mol_gen), deepcopy(mol_ref)
    # Extract coordinates
    coordinates_lig2 = mol_gen.GetConformer().GetPositions()
    coordinates_lig1 = mol_ref.GetConformer().GetPositions()
    # FindMCS
    res = rdFMCS.FindMCS([mol_gen, mol_ref])
    mcs_mol = Chem.MolFromSmarts(res.smartsString) # extract MCS as rdkit.Chem.Mol
    mas1 = list(mol_ref.GetSubstructMatch(mcs_mol)) # match reference first
    mas2_list = mol_gen.GetSubstructMatches(mcs_mol, uniquify=False) # match mol_gen then 
    if len(mas2_list) == 0: # Fail to match (seems impossible)
        return np.nan 
    # Reorder coordinates leveraging MCS 
    coordinates_lig1 = coordinates_lig1[mas1]
    list_rmsd = []
    for match1 in mas2_list:
        coordinates_lig2_tmp = coordinates_lig2[list(match1)]
        diff = coordinates_lig2_tmp - coordinates_lig1
        list_rmsd.append(np.sqrt((diff * diff).sum() / len(coordinates_lig2_tmp)))
    # Return the minimum RMSD
    lig_rmsd = min(list_rmsd)
    return lig_rmsd

# stability

atom_encoder = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17}
atom_decoder = {v: k for k, v in atom_encoder.items()}

# Bond lengths from http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92, 'P': 144, 'S': 134, 'Cl': 127},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135, 'P': 184, 'S': 182, 'Cl': 177},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136, 'P': 177, 'S': 168, 'Cl': 175},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142, 'P': 163, 'S': 151, 'Cl': 164},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142, 'P': 156, 'S': 158, 'Cl': 166},
          'P': {'H': 144, 'C': 184, 'N': 177, 'O': 163, 'F': 156, 'P': 221, 'S': 210, 'Cl': 203},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'F': 158, 'P': 210, 'S': 204, 'Cl': 207},
          'Cl': {'H': 127, 'C': 177, 'N': 175, 'O': 164, 'F': 166, 'P': 203, 'S': 207, 'Cl': 199}
          }

bonds2 = {'H': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'C': {'H': -1, 'C': 134, 'N': 129, 'O': 120, 'F': -1, 'P': -1, 'S': 160, 'Cl': -1},
          'N': {'H': -1, 'C': 129, 'N': 125, 'O': 121, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'O': {'H': -1, 'C': 120, 'N': 121, 'O': 121, 'F': -1, 'P': 150, 'S': -1, 'Cl': -1},
          'F': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'P': {'H': -1, 'C': -1, 'N': -1, 'O': 150, 'F': -1, 'P': -1, 'S': 186, 'Cl': -1},
          'S': {'H': -1, 'C': 160, 'N': -1, 'O': -1, 'F': -1, 'P': 186, 'S': -1, 'Cl': -1},
          'Cl': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          }

bonds3 = {'H': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'C': {'H': -1, 'C': 120, 'N': 116, 'O': 113, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'N': {'H': -1, 'C': 116, 'N': 110, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'O': {'H': -1, 'C': 113, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'F': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'P': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'S': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'Cl': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          }
stdv = {'H': 5, 'C': 1, 'N': 1, 'O': 2, 'F': 3}
margin1, margin2, margin3 = 10, 5, 3
allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'P': 5, 'S': 4, 'Cl': 1}


def get_bond_order(atom1, atom2, distance):
    distance = 100 * distance  # We change the metric

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < bonds1[atom1][atom2] + margin1:
        thr_bond2 = bonds2[atom1][atom2] + margin2
        if distance < thr_bond2:
            thr_bond3 = bonds3[atom1][atom2] + margin3
            if distance < thr_bond3:
                return 3
            return 2
        return 1
    return 0


def check_stability(positions, atom_type, debug=False, hs=False, return_nr_bonds=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[
                atom_type[j]]
            order = get_bond_order(atom1, atom2, dist)
            # if i == 0:
            #     print(j, order)
            nr_bonds[i] += order
            nr_bonds[j] += order

    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        if hs:
            is_stable = allowed_bonds[atom_decoder[atom_type_i]] == nr_bonds_i
        else:
            is_stable = (allowed_bonds[atom_decoder[atom_type_i]] >= nr_bonds_i > 0)
        if is_stable == False and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    if return_nr_bonds:
        return molecule_stable, nr_stable_bonds, len(x), nr_bonds
    else:
        return molecule_stable, nr_stable_bonds, len(x)


def find_amino_acid_terminals(mol): # terminals should be marked with dummay atoms: e.g. NC1=CC=C(C=C1)C=C(C#N)C2=CC=CC(C(C(=O)*)N*)=C2
    if isinstance(mol, str):    # a smiles is given
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            raise ValueError("Invalid SMILES for the amino acid")

    # SMARTS for N–Cα–C(=O)O
    pattern = Chem.MolFromSmarts("[*]-N[C][C](=O)-[*]")
    matches = mol.GetSubstructMatches(pattern)

    if not matches:
        raise ValueError("Backbone pattern not found")

    # Match is: (N_dummy_idx, N_idx, CA_idx, C_idx, O_idx, C_dummy_idx)
    n_dummy_idx = matches[0][0]
    n_idx = matches[0][1]
    ca_idx = matches[0][2]
    c_idx = matches[0][3]
    o_idx = matches[0][4]
    c_dummay_idx = matches[0][5]

    return {
        "N_dummy_idx": n_dummy_idx,
        "N_term_idx": n_idx,
        "CA_idx": ca_idx,
        "C_term_idx": c_idx,
        "O_terminal_idx": o_idx,
        'C_dummy_idx': c_dummay_idx
    }


def find_dummies(mol, skip_dummies=None):
    dummies, connects = [], []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != '*': continue
        if (skip_dummies is not None) and (atom.GetIdx() in skip_dummies): continue
        dummies.append(atom.GetIdx())
        neighbors = atom.GetNeighbors()
        assert len(neighbors) == 1, 'Dummy atoms should only have one connected atom!'
        connects.append(neighbors[0].GetIdx())
    return dummies, connects


def find_term_and_delete_dummy(smiles, include_non_term_dummy=True):
    mol = Chem.MolFromSmiles(smiles)
    ori_idx = find_amino_acid_terminals(mol)
    if include_non_term_dummy:
        dummies, connects = [], []  # maintain other dummies and delete N/C-term dummies in the amino acid
    else:
        dummies, connects = find_dummies(mol, skip_dummies=[ori_idx['N_dummy_idx'], ori_idx['C_dummy_idx']])

    # delete dummy
    atoms_to_remove = sorted([ori_idx['N_dummy_idx'], ori_idx['C_dummy_idx']] + dummies, reverse=True)  # reverse to avoid reindexing issues
    # Use editable mol to remove dummy atoms
    editable_mol = Chem.RWMol(mol)
    for idx in atoms_to_remove:
        editable_mol.RemoveAtom(idx)
    # Get final mol
    mol_clean = editable_mol.GetMol()
    # Map old indices to new
    def map_old_to_new(removed_indices, original_idx):
        shift = sum(1 for i in removed_indices if i < original_idx)
        return original_idx - shift

    n_idx_new = map_old_to_new(atoms_to_remove, ori_idx['N_term_idx'])
    c_idx_new = map_old_to_new(atoms_to_remove, ori_idx['C_term_idx'])

    if include_non_term_dummy:
        return mol_clean, n_idx_new, c_idx_new
    # other atoms connected to dummy
    other_connects = [map_old_to_new(atoms_to_remove, i) for i in connects]
    return mol_clean, n_idx_new, c_idx_new, other_connects
