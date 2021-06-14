import Bio
import numpy as np


def define_coordinate_system(
    pos_N_res1,
    pos_CA_res1,
    pos_C_res1,
    pos_CA_res2):
    """Defines a local reference system based on N, CA, and C atom positions"""

    # Get grid (eigenbasis).
    e1 = pos_CA_res1 -  pos_CA_res2 # points towards CA_res2
    e1 /= np.linalg.norm(e1)

    e2 = pos_C_res1 - pos_N_res1 # N -> C axis
    e2 /= np.linalg.norm(e2)

    e3 = np.cross(e1, e2)
    e3 /= np.linalg.norm(e3)

    # Readjust e2 for being perpendicular to e1 and e3.

    # The magnitude of the product equals the area of a parallelogram with the
    # vectors for sides; in particular, the magnitude of the product of two 
    # perpendicular vectors is the product of their lengths.

    # Since e1 and e3 are eigenvectors: we ensure e2 norm is 1.
    e2 = np.cross(e1, -e3) # anticommutativity of cross product

    # Define a rotation matrix.
    rot_matrix = np.array([e2, e3, e1]) # CA-CA axis upwards
    return rot_matrix
