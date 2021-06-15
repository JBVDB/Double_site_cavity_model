import argparse
import os
import sys
from typing import List, Dict

import Bio
import Bio.PDB
import Bio.PDB.vectors
import numpy as np
import simtk
import simtk.openmm
import simtk.openmm.app
import simtk.unit

from scipy.spatial.distance import cdist

basepath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, basepath)

import grid


def extract_atomic_features(pdb_filename: str):
    """Extract atomic features from pdb."""

    # Save atomic features in a dictionary + residue features
    features = {}
    features["atom_names"] = []
    features["res_indices"] = []
    features["xyz"] = []

    chain_boundary_indices = [0]
    chain_ids = []
    aa_indices = []
    res_pdb_ids = []

    # Parse structure with OpenMM
    pdb_simtk = simtk.openmm.app.PDBFile(pdb_filename)

    for chain in pdb_simtk.getTopology().chains():
        chain_ids.append(chain.id) # name of chain in pdb file ("A", "D", ...)
        for res in chain.residues():
            try:
                aa_index = Bio.PDB.Polypeptide.three_to_index(res.name)
            except:
                aa_index = 20
            aa_indices.append(aa_index)
            res_pdb_ids.append(res.id) # id in pdb file

            for atom in res.atoms():
                # Extract atom names
                features["atom_names"].append(atom.name)
                # Extract res indices (no reset across chains, starts from 0)
                features["res_indices"].append(res.index)

        # Save the index of the next chain start
        chain_boundary_indices.append(res.index+1)

    # Convert valid lists to numpy arrays (for saving later on in .npz format)
    features["atom_names"] = np.array(features["atom_names"], dtype="U5")
    features["res_indices"] = np.array(features["res_indices"], dtype=np.int)
    features["xyz"] = (
            pdb_simtk.getPositions(True)
            .value_in_unit(simtk.unit.angstrom)
            .astype(np.float32())
        )

    return (features,
            np.array(aa_indices, np.int),
            chain_ids,
            chain_boundary_indices,
            np.array(res_pdb_ids, np.int)
            )


def extract_pairs_res(
    features,
    max_radius: float=4.5,
    min_radius: float=0.,
    ):
    """
    Return for each residue a tuple of two lists (both inner possible order)
    of residue indices within a certain distance (min_radius, max_radius, in Angstrom).

    Parameters
    ----------
    features: dict["atom_names": np.array((n_atoms,), dtype="U5"),
                   "res_indices": np.array((n_atoms,), dtype=np.int),
                   "xyz": np.array((n_atoms,)), dtype=np.float32()]
        Dict of all atomic features.
    max_radius: float
        Max radius from residue heavy atoms, for collecting pairs of residues
    min_radius: float
        Min radius from residue heavy atoms, for collecting pairs of residues
    """
    res_indices_glob = features["res_indices"] # atomic res indices, redundant
    res_indices_uniq = np.unique(res_indices_glob)

    all_res_heavy_atoms = []
    for atom_res_index in res_indices_uniq:
        # Select all heavy atoms.
        if np.logical_and(
                res_indices_glob == atom_res_index,
                ~np.char.startswith(features["atom_names"], prefix="H")
            ).any():

            mask = np.logical_and(
                res_indices_glob == atom_res_index,
                ~np.char.startswith(features["atom_names"], prefix="H")
            )
            # Save heavy atom coordinates of each residue.
            all_res_heavy_atoms.append(features["xyz"][mask])
        else:
            # Store None to maintain residue indices.
            all_res_heavy_atoms.append(None)
            continue

    # Get candidates for each residue and save.
    all_res1 = []
    all_res2 = []
    for i, res in enumerate(all_res_heavy_atoms):
        # Check all possible pairs, only once (i+1, fixed inner order).
        for j, other_res in enumerate(all_res_heavy_atoms[i+1:]):
            if other_res is None:
                print(f"no atom for residue {j}")
                continue
            # Compute heavy atoms' distances between each residue pairs.
            res1_res2_distances = cdist(res, other_res, metric="euclidean")
            if np.any(np.logical_and(min_radius < res1_res2_distances,
                                    res1_res2_distances <= max_radius)):
                all_res1.append(i)
                all_res2.append(i+1+j)

    all_pairs_res_indices = np.vstack([all_res1, all_res2]).T
    all_pairs_res_indices_r = np.vstack([all_res2, all_res1]).T

    return (all_pairs_res_indices,
            all_pairs_res_indices_r)



def extract_pairs_res_ca_ca_dist(
    features,
    ca_ca_cutoff: float=7.0,
    ):
    """
    Return for each residue a tuple of two lists (both inner possible order)
    of residue indices within a certain CA-CA distance
    (ca_ca_cutoff, in Angstrom).

    Parameters
    ----------
    features: dict["atom_names": np.array((n_atoms,), dtype="U5"),
                   "res_indices": np.array((n_atoms,), dtype=np.int),
                   "xyz": np.array((n_atoms,)), dtype=np.float32()]
        Dict of all atomic features.
    ca_ca_cutoff: float
        max CA-CA distance for pairs of residue selection
    """
    res_indices_glob = features["res_indices"] # atomic res indices, redundant
    res_indices_uniq = np.unique(res_indices_glob)

    all_res_heavy_atoms = []
    for atom_res_index in res_indices_uniq:
        # Select all CA atoms.
        if np.logical_and(
                res_indices_glob == atom_res_index,
                np.char.startswith(features["atom_names"], prefix="CA")
            ).any():

            mask = np.logical_and(
                res_indices_glob == atom_res_index,
                np.char.startswith(features["atom_names"], prefix="CA")
            )
            # Save CA atom's coordinates of each residue.
            all_res_heavy_atoms.append(features["xyz"][mask])
        else:
            # Store None to maintain residue indices.
            all_res_heavy_atoms.append(None)
            continue

    # Get candidates for each residue and save.
    all_res1 = []
    all_res2 = []
    for i, res in enumerate(all_res_heavy_atoms):
        # Check all possible pairs, only once (i+1, fixed inner order).
        for j, other_res in enumerate(all_res_heavy_atoms[i+1:]):
            if other_res is None:
                print(f"no atom for residue {j}")
                continue
            # Compute heavy atoms' distances between each residue pairs.
            if cdist(res, other_res)[0][0] <= ca_ca_cutoff:
                all_res1.append(i)
                all_res2.append(i+1+j)

    all_pairs_res_indices = np.vstack([all_res1, all_res2]).T
    all_pairs_res_indices_r = np.vstack([all_res2, all_res1]).T

    return (all_pairs_res_indices,
            all_pairs_res_indices_r)


def extract_pair_res_features(
    all_pairs_res_indices: np.array,
    features: Dict,
    aa_indices: np.array,
    res_pdb_ids: np.array,
    max_width_x=4, max_width_y=4, max_height=8,
    ):
    """
    Define a local reference system based on CA-CA axis, N and C atom positions,
    one for each residue of the pair, and return a tuple of all relevant info 
    for constructing sa PairResEnvironment object.
    
    Parameters
    ----------
    all_pairs_res_indices: np.array((n_pair_res, 2), dtype=np.int)
        Array with each row representing a pair of residue indices.
    features: dict["atom_names": np.array((n_atoms,), dtype="U5"),
                   "res_indices": np.array((n_atoms,), dtype=np.int),
                   "xyz": np.array((n_atoms,)), dtype=np.float32()]
        Dict of all atomic features.
    res_pdb_ids: np.array((n_res,), dtype=np.int)
        Vector of global residue indices
    max_width_x, _y, _height: int, int, int
        Max x, y, z width from CA-CA center in Angstrom.
    """

    # Keep backbone atoms of the residues of each pair
    atoms_backbone = ["N", "CA", "C", "O", "H", "HA"]

    res_indices_glob = features["res_indices"]

    pair_res_atoms_list = [] # selected atoms per pair_res evt
    pair_res_local_xyz_list = [] # locally-defined atomic coord per pair_res evt

    for (res1_index, res2_index) in all_pairs_res_indices:
        # Get CA, N, C positions of res1.
        if (
            np.logical_and(
                res_indices_glob == res1_index, features["atom_names"] == "N"
            ).any()
            and np.logical_and(
                res_indices_glob == res1_index, features["atom_names"] == "CA"
            ).any()
            and np.logical_and(
                res_indices_glob == res1_index, features["atom_names"] == "C"
            ).any()
            ):

            # Masks
            N_mask_res1 = np.logical_and(
                res_indices_glob == res1_index, features["atom_names"] == "N"
            ).nonzero()[0][0]
            
            CA_mask_res1 = np.logical_and(
                res_indices_glob == res1_index, features["atom_names"] == "CA"
            ).nonzero()[0][0]
            C_mask_res1 = np.logical_and(
                res_indices_glob == res1_index, features["atom_names"] == "C"
            ).nonzero()[0][0]
            # Positions
            pos_N_res1 = features["xyz"][N_mask_res1]
            pos_CA_res1 = features["xyz"][CA_mask_res1]
            pos_C_res1 = features["xyz"][C_mask_res1]
        else:
            # Store None to maintain indices.
            pair_res_atoms_list.append(None)
            pair_res_local_xyz_list.append(None)
            continue

        # Get CA position of res2.
        if np.logical_and(
            res_indices_glob == res2_index,
            features["atom_names"] == "CA"
            ).any():

            CA_mask_res2 = np.logical_and(
                res_indices_glob == res2_index, features["atom_names"] == "CA"
            ).nonzero()[0][0]

            pos_CA_res2 = features["xyz"][CA_mask_res2]

        # Define a rotation matrix.
        rot_matrix = grid.define_coordinate_system(pos_N_res1,
                                                pos_CA_res1,
                                                pos_C_res1,
                                                pos_CA_res2) # CA-CA axis upwards

        # Calculate coordinates relative to origin.
        center_ca_ca = np.mean([pos_CA_res1, pos_CA_res2], axis=0)
        xyz = features["xyz"] - center_ca_ca

        # Rotate to the local reference
        xyz = np.dot(rot_matrix, xyz.T).T

        # Select atoms that should be included in the rectangular cuboid,
        # -> new strategy: keep backbone

        res1_atoms = np.where(res_indices_glob == res1_index)[0]
        res1_atoms = res1_atoms[np.isin(features["atom_names"][res1_atoms],
                                        atoms_backbone)]
        res2_atoms = np.where(res_indices_glob == res2_index)[0]
        res2_atoms = res2_atoms[np.isin(features["atom_names"][res2_atoms],
                                        atoms_backbone)]

        selector = np.where(
                        (np.abs(xyz[:, 0]) < max_width_x) &
                        (np.abs(xyz[:, 1]) < max_width_y) &
                        (np.abs(xyz[:, 2]) < max_height) &
                        (res_indices_glob != res1_index) &
                        (res_indices_glob != res2_index)
                        )[0]
        selector = np.concatenate((res1_atoms, res2_atoms, selector))

        pair_res_local_xyz_list.append(xyz[selector]) # list of coordinates
        pair_res_atoms_list.append(selector)

    # Get the max number of atoms per pair_res to make sure all pair_res fit the
    # same array (sparse).
    max_selector = max(
    [len(selector) for selector in pair_res_atoms_list if selector is not None]
    )

    # 1) Save atomic INDICES per pair_res evt in array (-1 is arbitrary).
    pair_res_atoms_arr = np.full(
        shape=(len(pair_res_atoms_list), max_selector),
        fill_value=-1,
        dtype=np.int32
        ) # shape (n_pairs, max_n_neighbors)
    for i, selector in enumerate(pair_res_atoms_list):
        if selector is not None:
            pair_res_atoms_arr[i, :len(selector)] = selector.astype(np.int32)

    # 2) Save atomic TYPES per resi_evt in array.
    atom_type_list = ["C", "N", "O", "H", "S", "P"]
    atom_types_numeric = np.array(
        [atom_type_list.index(x[0]) for x in features["atom_names"]]
    )  # Zero refers to the first letter of atom name

    # 3) Save atomic COORDINATES per resi_evt in array (-99 is arbitrary).
    pair_res_local_xyz_arr = np.full(
        shape=[len(pair_res_local_xyz_list), max_selector, 3],
        fill_value=[-99, -99, -99], 
        dtype=np.float32
    ) # shape (n_pairs, max_n_neighbors, 3)
    for i, xyz_selected in enumerate(pair_res_local_xyz_list): # LIST, not ARR!
        if xyz_selected is not None:
            pair_res_local_xyz_arr[i, :xyz_selected.shape[0], :] = xyz_selected

    # 4) Save pair_res aa indices one-hot encoded along a vector 20*20 (before: 21*21)
    # Trick used: convert a tuple of index arrays into an array of flat indices
    # for each of these pairs_evt, that is used to mask a 20*20 array.
    aa_indices = np.array(aa_indices)
    mask_onehot = np.ravel_multi_index(aa_indices[all_pairs_res_indices].T,
                                    dims=(20, 20)) # previously: 21x21
    pair_onehot = np.zeros((len(all_pairs_res_indices), 20*20))
    pair_onehot[np.arange(len(all_pairs_res_indices)), mask_onehot] = 1
    # np.unravel_index(mask_onehot, shape=(20, 20)) # to retrieve pairs of idx

    # 5) Save and Reformat res_pdb_ids (residue_number real) as (-1, 2).
    pair_res_pdb_ids = res_pdb_ids[np.array(all_pairs_res_indices)]

    return (
        pair_res_local_xyz_arr,
        atom_types_numeric,
        pair_res_atoms_arr,
        pair_onehot,
        pair_res_pdb_ids
    )


def extract_environments(
    pdb_filename: str,
    pdb_id: str,
    max_radius: float=4.5,
    min_radius: float=0.,
    out_dir: str = "./",
    max_width_x=4, max_width_y=4, max_height=8,
    ca_ca_cutoff: float = 7.0,
    ca_ca_dist_based=False,
    ):
    """
    Extract residue environments from PDB file. Outputs .npz file.

    Parameters
    ----------
    pdb_filename: str
        PDB filename to extract environments from
    pdb_id: str
        PDBID. Used as a prefix for the output file, and does not have to follow
        the standard 4 character nomenclature
    max_radius: float, default=4.5
        Max radius from residue heavy atoms, for collecting pairs of residues
    min_radius: float, default=0
        Min radius from residue heavy atoms, for collecting pairs of residues
    max_width_x, _y, _height: int, int, int
        Max x, y, z width from CA-CA center in Angstrom.
    out_dir: str
        Output directory.
    ca_ca_cutoff: float
        Max CA-CA distance allowed between pairs of residues
    ca_ca_dist_based: bool
        Whether to base selection on CA-CA distance rather than inter-heavy-atom distance.
        Default: False
    """

    # Extract atomic features and other relevant info.
    (
        features,
        aa_indices,
        chain_ids,
        chain_boundary_indices,
        res_pdb_ids
     ) = extract_atomic_features(pdb_filename)

    # Extract 2 lists of pairs of residues (each possible order), whose at least
    # one of their pairwise heavy atom distance is higher than min_radius and at most most max_radius.
    
    if not ca_ca_dist_based:
        (
            all_pairs_res_indices,
            all_pairs_res_indices_r
         ) = extract_pairs_res(
            features,
            max_radius,
            min_radius
            )
    else:
        (
            all_pairs_res_indices,
            all_pairs_res_indices_r
         ) = extract_pairs_res_ca_ca_dist(
            features,
            ca_ca_cutoff
            )    

    # Extract relevant local coordinates for each pair_res and relevant info for
    # PairResEnvironment object.
    for i, all_pairs_res_indices_ in enumerate([all_pairs_res_indices,
                                                all_pairs_res_indices_r]):

        (
            pair_res_local_xyz_arr,
            atom_types_numeric,
            pair_res_atoms_arr,
            pair_onehot,
            pair_res_pdb_ids
         ) = extract_pair_res_features(
                all_pairs_res_indices_,
                features,
                aa_indices,
                res_pdb_ids,
                max_width_x,
                max_width_y,
                max_height
            )
        # Save as .npz
        np.savez_compressed(
            out_dir + f"/{pdb_id}_{i}_pair_res_features",
            atom_types_numeric=atom_types_numeric,
            positions=pair_res_local_xyz_arr,
            selector=pair_res_atoms_arr,
            pair_aa_onehot=pair_onehot,
            chain_boundary_indices=chain_boundary_indices,
            chain_ids=chain_ids,
            pair_res_indices=all_pairs_res_indices_ # pair_res_pdb_ids prevented us from retrieving chain info
        )


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_in", type=str)
    parser.add_argument("--max_radius", type=float, default=4.5)
    parser.add_argument("--min_radius", type=float, default=0.)
    parser.add_argument("--out_dir", type=str, default="./")
    parser.add_argument("--max_width_x", type=int, default=4)
    parser.add_argument("--max_width_y", type=int, default=4)
    parser.add_argument("--max_height", type=int, default=8)
    parser.add_argument("--ca_ca_cutoff", type=float, default=7.0)
    parser.add_argument("--ca_ca_dist_based", type=bool, default=False)
    args_dict = vars(parser.parse_args()) # get __dict__ attribute

    # Settings
    pdb_filename = args_dict["pdb_in"]
    pdb_id = os.path.basename(pdb_filename).split(".")[0]
    max_radius = args_dict["max_radius"]
    min_radius = args_dict["min_radius"]
    out_dir = args_dict["out_dir"]
    max_width_x = args_dict["max_width_x"]
    max_width_y = args_dict["max_width_y"]
    max_height = args_dict["max_height"]
    ca_ca_cutoff = args_dict["ca_ca_cutoff"]
    ca_ca_dist_based = args_dict["ca_ca_dist_based"]

    # Extract
    extract_environments(pdb_filename, pdb_id,
                         max_radius, min_radius, out_dir,
                         max_width_x, max_width_y, max_height,
                         ca_ca_cutoff, ca_ca_dist_based)