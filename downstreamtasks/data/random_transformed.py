import os
import numpy as np
import argparse
from Bio.PDB import PDBParser, PDBIO
from scipy.spatial.transform import Rotation as R

def random_transformation(matrix):
    """ Apply random rotation and translation to the coordinate matrix """
    # Random rotation
    rotation = R.random().as_matrix()  # Create a random rotation matrix
    transformed = np.dot(matrix, rotation)  # Apply rotation

    # Random translation
    translation = np.random.uniform(-10, 10, 3)  # Random translation vector
    transformed += translation  # Apply translation

    return transformed

def process_pdb_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    parser = PDBParser()
    io = PDBIO()

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".pdb"):
            file_path = os.path.join(input_dir, file_name)
            structure = parser.get_structure(file_name, file_path)

            for model in structure:
                for chain in model:
                    coords = np.array([atom.get_coord() for atom in chain.get_atoms()])
                    transformed_coords = random_transformation(coords)

                    # Update atom coordinates with transformed coordinates
                    for atom, new_coord in zip(chain.get_atoms(), transformed_coords):
                        atom.set_coord(new_coord)

            # Write the transformed structure to a new file
            output_file_path = os.path.join(output_dir, file_name)
            io.set_structure(structure)
            io.save(output_file_path)

def main():
    parser = argparse.ArgumentParser(description="Transform PDB files with random rotations and translations")
    parser.add_argument('input_directory', type=str, help='Path to the input directory containing PDB files')
    parser.add_argument('output_directory', type=str, help='Path to the output directory for saving transformed PDB files')
    args = parser.parse_args()

    process_pdb_files(args.input_directory, args.output_directory)

if __name__ == "__main__":
    main()
