#!/usr/bin/env python3
import argparse
import os
import pickle
import time
from alphafold.relax import relax
from alphafold.common import protein
from absl import logging
logging.set_verbosity(logging.INFO)

parser = argparse.ArgumentParser(description='Relax a protein model with amber.')
parser.add_argument('--unrelaxed_pdb_path', default=None, type=str, help='Path of unrelaxed protein')
parser.add_argument('--model_name', default=None, type=str, help='Model name')
parser.add_argument('--relaxed_output_path', default=None, type=str, help='Name for relaxed protein')
parser.add_argument('--output_dir', default=None, type=str, help='Path to output directory')
parser.add_argument('--relax_results_pkl', default=None, type=str, help='Path to a results pickle file')
parser.add_argument('--use_gpu_relax', default=True, type=bool, help='Whether to use gpu for relax')
args = parser.parse_args()


RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3

amber_relaxer = relax.AmberRelaxation(
    max_iterations=RELAX_MAX_ITERATIONS,
    tolerance=RELAX_ENERGY_TOLERANCE,
    stiffness=RELAX_STIFFNESS,
    exclude_residues=RELAX_EXCLUDE_RESIDUES,
    max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
    use_gpu=args.use_gpu_relax)

def run(unrelaxed_pdb_path, model_name, relaxed_output_path, relaxed_pdbs, relax_metrics):

    if os.path.exists(unrelaxed_pdb_path):
        with open(unrelaxed_pdb_path, 'r') as f:
            unrelaxed_protein = protein.from_pdb_string(f.read())
        if not os.path.exists(relaxed_output_path):
            logging.info(f"Running relax on {unrelaxed_pdb_path}")
            # Relax the prediction.
            t_0 = time.time()
            relaxed_pdb_str, _, violations = amber_relaxer.process(
                prot=unrelaxed_protein)
            relax_metrics[model_name] = {
                'remaining_violations': violations,
                'remaining_violations_count': sum(violations)
            }
            runtime = time.time() - t_0

            relaxed_pdbs[model_name] = relaxed_pdb_str
            with open(relaxed_output_path, 'w') as f:
                f.write(relaxed_pdb_str)
            logging.info(f"Saved relaxed model to {relaxed_output_path}")
        else:
            logging.error(f"{relaxed_output_path} already exists. Skipping relaxation step.")
    else:
        logging.error(f"{unrelaxed_pdb_path} does not exist.")


if os.path.exists(args.relax_results_pkl):
    with open(args.relax_results_pkl, 'rb') as f:
        relax_results = pickle.load(f)
        relaxed_pdbs = relax_results['relaxed_pdbs']
        relax_metrics = relax_results['relax_metrics']
else:
    relax_metrics = {}
    relaxed_pdbs = {}

run(args.unrelaxed_pdb_path, args.model_name, args.relaxed_output_path, relaxed_pdbs, relax_metrics)


relax_results = {'relax_metrics': relax_metrics, 'relaxed_pdbs': relaxed_pdbs}
with open(args.relax_results_pkl, 'wb') as f:
    pickle.dump(relax_results, f)