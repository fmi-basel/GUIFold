#!/usr/bin/env python3
# Copyright 2021 DeepMind Technologies Limited
# Copyright 2022 Friedrich Miescher Institute for Biomedical Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Georg Kempf, Friedrich Miescher Institute for Biomedical Research

"""Full AlphaFold protein structure prediction script."""
from copy import deepcopy
import json
import multiprocessing
from multiprocessing.managers import ListProxy
from multiprocessing import Manager, synchronize
from multiprocessing.synchronize import Lock
from operator import itemgetter
import os
import pathlib
import pickle
import random
import re
import shutil
from subprocess import PIPE, Popen
import sys
import time
import traceback
from typing import Dict, Union, Optional
from alphafold.data.msa_identifiers import _UNIPROT_PATTERN
from alphafold.data import msa_identifiers
import nvidia_smi

from absl import app
from absl import flags
from absl import logging
from guifold.afeval import EvaluationPipeline, EvaluationPipelineBatch

from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import pipeline_batch
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch

from alphafold.relax import relax
from alphafold.data import parsers

import tensorflow as tf

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import gzip


from alphafold.model import config
from alphafold.model import model
from alphafold.model import data

from sqlalchemy import create_engine, Column, Integer, String, Sequence
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


# Internal import (7716).


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
os.environ['OPENBLAS_NUM_THREADS'] = '1'

logging.set_verbosity(logging.INFO)

flags.DEFINE_string('fasta_path', None, 'Path to a single fasta file.')
flags.DEFINE_string('data_dir', None, 'Path to the directory containing model weights.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('jackhmmer_binary_path', shutil.which('jackhmmer'),
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', shutil.which('hhblits'),
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', shutil.which('hhsearch'),
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', shutil.which('hmmsearch'),
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', shutil.which('hmmbuild'),
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('hhalign_binary_path', None,
                    'Path to the hhalign executable.')
flags.DEFINE_string('mmseqs_binary_path', None,
                    'Path to the mmseqs executable.')
flags.DEFINE_string('kalign_binary_path', None,
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('uniref90_mmseqs_database_path', None, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('colabfold_envdb_database_path', None, 'Path to the colabfold_envdb '
                                                    'database for use by MMSeqs2.')
flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniref30_database_path', None, 'Path to the UniRef30 '
                    'database for use by HHblits.')
flags.DEFINE_string('uniref30_mmseqs_database_path', None, 'Path to the UniRef30 '
                                                    'database for use by MMseqs.')
flags.DEFINE_string('uniprot_database_path', None, 'Path to the Uniprot '
                    'database for use by JackHMMer.')
flags.DEFINE_string('uniprot_mmseqs_database_path', None, 'Path to the Uniprot '
                    'database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', None, 'Path to the PDB '
                    'seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_enum('db_preset', 'full_dbs',
                  ['full_dbs', 'reduced_dbs', 'colabfold_local', 'colabfold_web'],
                  'Choose preset MSA database configuration - '
                  'full_dbs: uniref30_bfd:hhblits, mgnify:jackhmmer, uniref90:jackhmmer, uniprot:jackhmmer '
                  'reduced_dbs: uniref30:jackhmmer, small_bfd:jackhmmer, uniref90:jackhmmer, uniprot:jackhmmer '
                  'colabfold_local: uniref30:mmseqs, colabfold_envdb:mmseqs, uniref90:mmseqs, uniprot:mmseqs '
                  'colabfold_webserver: uniref30:mmseqs (server), colabfold_envdb:mmseqs (server), uniref90:jackhmmer, unirpot:jackhmmer')
flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('num_multimer_predictions_per_model', 1, 'How many '
                     'predictions (each with a different random seed) will be '
                     'generated per model. E.g. if this is 2 and there are 5 '
                     'models then there will be 10 predictions per input. '
                     'Note: this FLAG only applies if model_preset=multimer')
flags.DEFINE_boolean('use_precomputed_msas', False, 'Whether to read MSAs that '
                     'have been written to disk instead of running the MSA '
                     'tools. The MSA files are looked up in the output '
                     'directory, so it must stay the same between multiple '
                     'runs that are to reuse the MSAs. WARNING: This will not '
                     'check if the sequence, database or configuration have '
                     'changed.')
flags.DEFINE_boolean('run_relax', False, 'Whether to run the final relaxation '
                     'step on the predicted models. Turning relax off might '
                     'result in predictions with distracting stereochemical '
                     'violations but might help in case you are having issues '
                     'with the relaxation stage.')
flags.DEFINE_boolean('use_gpu_relax', False, 'Whether to relax on GPU. '
                     'Relax on GPU can be much faster than CPU, so it is '
                     'recommended to enable if possible. GPUs must be available'
                     ' if this setting is enabled.')
flags.DEFINE_list('no_msa_list', False, 'Optional. If the use of MSAs should be disabled for a sequence'
                                   'a boolean needs to be given for each sequence in the same order '
                                   'as sequences are given in the fasta file.')
flags.DEFINE_list('no_template_list', False, 'Optional. If the use of templates should be disabled for a sequence'
                                                'a boolean needs to be given for each sequence in the same order '
                                                'as sequences are given in the fasta file.')
flags.DEFINE_list('custom_template_list', None, 'Optional. If a custom template should be used for one or'
                                                    ' more sequences, a comma-separated list of file paths or None needs to be given  '
                                                'in the same order as sequences are given in the fasta file.')
flags.DEFINE_list('precomputed_msas_list', None, 'Optional. Comma-separated list of paths to precomputed msas folders or None, given'
                                                 ' in the same order as input sequences.')
flags.DEFINE_string('custom_tempdir', None, 'Define a custom tempdir other than /tmp')
flags.DEFINE_integer('num_recycle', 3, 'Define maximum number of model recycles.')
#flags.DEFINE_bool('only_features', False, 'Stop after Feature pipeline. Useful for splitting up the job into CPU and GPU resources.')
#flags.DEFINE_bool('continue_from_features', False, 'Continue from features.pkl file.'
#                                                   ' Useful for splitting up the job into CPU and GPU resources.')
flags.DEFINE_integer('num_cpu', 1, 'Number of CPUs to use for feature generation.')
flags.DEFINE_string('precomputed_msas_path', None, 'Path to a directory with precomputed MSAs (job_dir/msas)')
#flags.DEFINE_boolean('batch_msas', False, 'Runs the monomer feature pipeline for all sequences in the input MSA file.')
flags.DEFINE_enum('pipeline', 'full', [
                'full', 'only_features', 'batch_msas', 'continue_from_features', 'all_vs_all', 'first_vs_all', 'first_n_vs_rest', 'grouped_bait_vs_preys', 'grouped_all_vs_all', 'only_relax'],
                'Choose preset pipeline configuration - '
                'full pipeline or '
                'stop after feature generation (only features) or '
                'calculate MSAs and find templates for given batch of sequences, uses template search based on monomer/multimer preset or'
                'continue from features.pkl file (continue_from_features)')
flags.DEFINE_enum('prediction', 'alphafold', ['alphafold'],
                  'Choose preset prediction configuration - AlphaFold.')
flags.DEFINE_boolean('debug', False, 'Enable debugging output.')
flags.DEFINE_integer('first_n_seq', None, 'Parameter needed for first_n_vs_rest protocol to define the first N sequences that will be kept constant in a screening.')
flags.DEFINE_integer('num_gpu', 1, 'Number of GPUs.')
flags.DEFINE_list('model_list', '1,2,3,4,5', 'List of indices defining which Alphafold models to use.')
flags.DEFINE_string('predictions_dir', None, 'Name of predictions output dir.')
flags.DEFINE_string('features_dir', None, 'Name of features output dir.')
flags.DEFINE_string('job_status_log_file', 'job_status.txt', 'Job status log path')
flags.DEFINE_integer('batch_max_sequence_length', 5000, 'Maximum total sequence length for a pairwise prediction')
flags.DEFINE_enum('msa_pairing', 'paired', ['paired', 'paired+unpaired'], 'Choose pairing strategy. paired = default AF protocol. paired+unpaired = all sequences are added again below the paired sequences')
flags.DEFINE_string('multichain_template_path', None, 'Path to a multichain template')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3
BATCH_PREDICTION_MODES = ['first_vs_all', 'all_vs_all', 'first_n_vs_rest', 'grouped_bait_vs_preys', 'grouped_all_vs_all']

class AccessionSpeciesDB(Base):
    __tablename__ = 'accession_species_mapping'
    id = Column(Integer, Sequence('accession_species_mapping_id_seq'), primary_key=True)
    accession_id = Column(String)
    species_id = Column(String)

class InputError(Exception):
    pass


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set and not bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')
  
def _write_job_status_log(path, msg):
    with open(path, 'a') as f:
        f.write(f'{msg}\n')


def create_accession_species_db(uniprot_db, db_session, output_path):
    """Maps uniprot accession ids to species ids and stores it in a database file"""
    try:
        accession_seq_id_mapping = {}


        chunk_size = 2**22

        with open(uniprot_db, 'r') as f:
            accession_seq_id_mapping = {}
            line_count = 0
            no_match_count = 0
            match_count = 0
            line_count_match = 0
            while True:
                lines = f.readlines(2**24)

                if not lines:
                    #print(lines)
                    #print(f'break after {line_count}')
                    break
                insert_objects = []
                for l in lines:
                    line_count += 1
                    if l.startswith('>'):
                        line_count_match += 1
                        sequence_identifier = l[1:].split()[0]
                        matches = re.search(_UNIPROT_PATTERN, sequence_identifier.strip())
                        if matches:
                            insert_objects.append(AccessionSpeciesDB(accession_id=matches.group('AccessionIdentifier'), species_id=matches.group('SpeciesIdentifier')))
                            match_count += 1
                        else:
                            no_match_count += 1
                db_session.bulk_save_objects(insert_objects)
                db_session.commit()
                logging.debug(f"Line count {line_count_match}")
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        os.remove(output_path)
        raise SystemExit
    

def batch_minimization(output_dir, gpu_relax):
    unrelaxed_pdbs, relaxed_pdbs = [], []
    for root, folders, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".pdb") and not file.endswith("_aligned.pdb"):
                if file.startswith("unrelaxed"): 
                    unrelaxed_pdbs.append(os.path.join(root, file))
                elif file.startswith("relaxed"):
                    relaxed_pdbs.append(os.path.join(root, file))
    
    unrelaxed_pdbs = [x for x in unrelaxed_pdbs if not x.replace("unrelaxed", "relaxed") in relaxed_pdbs]
    logging.info("unrelaxed pdbs for minimization:")
    logging.info(unrelaxed_pdbs)
    for unrelaxed_pdb in unrelaxed_pdbs:
        model_name = re.search("unrelaxed_(\w+_\d+).*.pdb", unrelaxed_pdb).group(1)
        model_dir = os.path.dirname(unrelaxed_pdb)

        relaxed_output_path = os.path.join(model_dir, f'relaxed_{model_name}.pdb')
        #Run in a new process to prevent CUDA initialitation error of openmm  
        relax_results_pkl = os.path.join(model_dir, 'relax_results.pkl')
        try:
            cmd = f"run_relaxation.py --unrelaxed_pdb_path {unrelaxed_pdb} --relaxed_output_path {relaxed_output_path} --model_name {model_name} --relax_results_pkl {relax_results_pkl}"
            if FLAGS.use_gpu_relax:
                cmd = f"{cmd} --use_gpu_relax"
            logging.debug(cmd)
            p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
            output, error = p.communicate()
            if output:
                logging.info(output.decode())
            if error:
                logging.error(error.decode())
        except Exception as e:
            logging.error(f"Relaxation failed: {e}.")

def custom_sort(item):
    return item[1]      

def plot_multichain_mask(mask):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    im = ax.imshow(mask, cmap='binary', interpolation='nearest')
    ax.set_title("Multichain mask")
    ax.set_xlabel('Chain Index')
    ax.set_ylabel('Chain Index')
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
    plt.savefig(f'multichain_mask.png')
    plt.close('all')

def predict_structure(
    fasta_path: str,
    protein_names: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seed: int,
    is_multimer: bool,
    no_msa_list: Optional[bool] = None,
    no_template_list: Optional[bool] = None,
    custom_template_list: Optional[str] = None,
    precomputed_msas_list: Optional[str] = None,
    prediction_pipeline: str = 'alphafold',
    feature_pipeline: str = 'full_dbs',
    batch_prediction: Optional[bool] = False,
    scores: Optional[tuple] = None,
    batch_mmseqs: bool = False,
    flags: object = None,
    results_queue: multiprocessing.Queue = None,
    score_dict: Optional[dict] = None,
    file_lock: Optional[Lock] = None,
    use_existing_features: bool = False,
    multichain_template_list: list = None):
    """Predicts structure using AlphaFold for the given sequence."""
    logging.info('Generating input features')
    load_existing_features = False
    fasta_name = os.path.splitext(os.path.basename(fasta_path))[0]
    timings = {}
    if not flags['pipeline'] == 'only_features' and not flags['pipeline'] == 'batch_msas':
        if flags['predictions_dir']:
            predictions_output_dir = os.path.join(output_dir_base, "predictions", flags['predictions_dir'])
        else:
            predictions_output_dir = os.path.join(output_dir_base, "predictions", prediction_pipeline)
        results_dir = os.path.join(predictions_output_dir, protein_names)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)   
    if flags['features_dir']:
        features_output_dir = os.path.join(output_dir_base, "features", flags['features_dir'])
    else:
        features_output_dir = os.path.join(output_dir_base, "features", feature_pipeline)

    msa_output_dir = features_output_dir
    features_output_path = os.path.join(features_output_dir, f'features_{protein_names}.pkl')
    msa_stats_output_path = os.path.join(features_output_dir, f'msa_stats_{protein_names}.json')
    if not os.path.exists(msa_output_dir):
        os.makedirs(msa_output_dir, exist_ok=True)

    # Get features.
    if not flags['pipeline'] == 'continue_from_features':
        if batch_mmseqs:
            #Calculate MSAs in batch
            logging.info("Running batch mmseqs pipeline")
            data_pipeline.process_batch_mmseqs(input_fasta_path=fasta_path,
                                          msa_output_dir=msa_output_dir,
                                          num_cpu=flags['num_cpu'])
            #Use the MSAs computed in the previous step
            data_pipeline.set_use_precomputed_msas(True)
            data_pipeline.set_precomputed_msas_path(msa_output_dir)

        if use_existing_features and os.path.exists(features_output_path):
            load_existing_features = True
        else:
            t_0 = time.time()
            feature_dict = data_pipeline.process(
                    input_fasta_path=fasta_path,
                    msa_output_dir=msa_output_dir,
                    no_msa=no_msa_list,
                    no_template=no_template_list,
                    custom_template_path=custom_template_list,
                    precomputed_msas_path=precomputed_msas_list,
                    num_cpu=flags['num_cpu'],
                    file_lock=file_lock)
            timings['features'] = time.time() - t_0
            msa_stats = data_pipeline.msa_stats

            # Write out features as a pickled dictionary.
            if not flags['pipeline'] == 'batch_msas':
                features_output_path = os.path.join(features_output_dir, f'features_{protein_names}.pkl')
                logging.info(f"Writing features to {features_output_path}")
                with open(features_output_path, 'wb') as f:
                    pickle.dump(feature_dict, f, protocol=4)
                with open(msa_stats_output_path, 'w') as f:
                    json.dump(msa_stats, f)

    #Stop here if only_msa flag is set
    if not flags['pipeline'] == 'only_features' and not flags['pipeline'] == 'batch_msas':
        logging.info('Predicting %s', protein_names)
        if flags['pipeline'] == 'continue_from_features' or load_existing_features:
            #Backward compatibility
            if not os.path.exists(features_output_path) and not os.path.exists(f"{features_output_path}.gz"):
                features_output_path_alternative = os.path.join(output_dir_base, fasta_name, 'features.pkl')
                if os.path.exists(features_output_path_alternative):
                    with open(features_output_path_alternative, 'rb') as f:
                        feature_dict = pickle.load(f)
                elif os.path.exists(f"{features_output_path}.gz"):
                    with gzip.open(f"{features_output_path}.gz", 'rb') as f:
                        feature_dict = pickle.load(f)
                elif not use_existing_features:
                    raise Exception(f"Continue_from_features requested but no feature pickle file found in expected location: {features_output_path} or {features_output_path_alternative}.")
            else:
                with open(features_output_path, 'rb') as f:
                    feature_dict = pickle.load(f)
            if os.path.exists(msa_stats_output_path):
                with open(msa_stats_output_path, 'r') as f:
                    msa_stats = json.load(f)
            else:
                msa_stats = None


        unrelaxed_pdbs = {}
        relaxed_pdbs = {}
        relax_metrics = {}
        ranking_confidences = {}

        # Run the models.
        num_models = len(model_runners)
        feature_dict_initial = {k: v for k, v in feature_dict.items()}
        for model_index, (model_name, model_runner) in enumerate(
            model_runners.items()):
            
            t_0 = time.time()
            model_random_seed = model_index + random_seed * num_models
            model_name = model_name.replace("_pred_0", "")
            unrelaxed_pdb_path = os.path.join(results_dir, f'unrelaxed_{model_name}.pdb')

            logging.info('Running model %s on %s', model_name, protein_names)
            result_output_path = os.path.join(results_dir, f'result_{model_name}.pkl')
            #Skip if model already exists
            logging.debug(f"Model exists: {os.path.exists(unrelaxed_pdb_path)}")
            if not os.path.exists(unrelaxed_pdb_path):
                logging.info(feature_dict.keys())
                logging.info(f"Final MSA size: {feature_dict['num_alignments']}")
                t_0 = time.time()
                model_random_seed = model_index + random_seed * num_models
                processed_feature_dict = model_runner.process_features(
                    feature_dict, random_seed=model_random_seed)
                timings[f'process_features_{model_name}'] = time.time() - t_0
                #if flags['multimer_template']:
                #    feature_dict['no_multichain_mask'] = True
                t_0 = time.time()
                if any([x == True for x in multichain_template_list]) and is_multimer:
                    logging.info(f"Multichain template list: {', '.join([str(item) for item in multichain_template_list])}: Multimer mode: {is_multimer}")
                    multichain_mask = processed_feature_dict['asym_id'][:, None] == processed_feature_dict['asym_id'][None, :]
                    #Make a pairwise matrix for aysm_id pairs
                    asym_id_pairwise_matrix = pairwise_matrix = np.empty((len(processed_feature_dict['asym_id']), len(processed_feature_dict['asym_id'])), dtype=object)
                    for i in range(len(processed_feature_dict['asym_id'])):
                        for j in range(len(processed_feature_dict['asym_id'])):
                            asym_id_pairwise_matrix[i, j] = (processed_feature_dict['asym_id'][i], processed_feature_dict['asym_id'][j])
                    # Update the mask based on the multichain_template_list which indicates which templates belong to a multichain template
                    allowed_pairs = []
                    for i, i_allowed in enumerate(multichain_template_list):
                        for j, j_allowed in enumerate(multichain_template_list):
                            if i_allowed and j_allowed:
                                if not (i+1,j+1) in allowed_pairs:
                                    allowed_pairs.append((i+1,j+1))
                                if not (j+1,i+1) in allowed_pairs:
                                    allowed_pairs.append((j+1,i+1))
                    indices = []

                    for i in range(len(asym_id_pairwise_matrix)):
                        for j in range(len(asym_id_pairwise_matrix[i])):
                            if asym_id_pairwise_matrix[i, j] in allowed_pairs:
                                indices.append((i, j))
                    # Update corresponding positions in multichain_mask
                    multichain_mask[indices] = True
                    plot_multichain_mask(multichain_mask)
                    logging.info("Multichain mask was adapted to provided multichain template to allow interchain contacts. The mask was saved to multichain_mask.png for inspection.")
                    processed_feature_dict['multichain_mask'] = multichain_mask

                prediction_result = model_runner.predict(processed_feature_dict,
                                                        random_seed=model_random_seed)
                t_diff = time.time() - t_0
                timings[f'predict_and_compile_{model_name}'] = t_diff
                logging.info(
                    'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
                    model_name, protein_names, t_diff)

                if benchmark:
                    t_0 = time.time()
                    model_runner.predict(processed_feature_dict,
                                        random_seed=model_random_seed)
                    t_diff = time.time() - t_0
                    timings[f'predict_benchmark_{model_name}'] = t_diff
                    logging.info(
                        'Total JAX model %s on %s predict time (excludes compilation time): %.1fs',
                        model_name, protein_names, t_diff)

                plddt = prediction_result['plddt']
                ranking_confidences[model_name] = prediction_result['ranking_confidence']

                # Save the model outputs.
                with open(result_output_path, 'wb') as f:
                    pickle.dump(prediction_result, f, protocol=4)

                # Add the predicted LDDT in the b-factor column.
                # Note that higher predicted LDDT value means higher model confidence.
                plddt_b_factors = np.repeat(
                    plddt[:, None], residue_constants.atom_type_num, axis=-1)

                unrelaxed_protein = protein.from_prediction(
                    features=processed_feature_dict,
                    result=prediction_result,
                    b_factors=plddt_b_factors,
                    remove_leading_feature_dimension=not model_runner.multimer_mode)
                unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)

                unrelaxed_pdb_path = os.path.join(results_dir, f'unrelaxed_{model_name}.pdb')
                with open(unrelaxed_pdb_path, 'w') as f:
                    f.write(unrelaxed_pdbs[model_name])
            else:
                if not os.path.exists(result_output_path):
                    result_output_path = result_output_path.replace("result_", "result_reduced_")
                with open(result_output_path, 'rb') as f:
                    prediction_result = pickle.load(f)
                logging.info(f"Skipping prediction because {unrelaxed_pdb_path} already exists.")
                with open(unrelaxed_pdb_path, 'r') as f:
                    unrelaxed_protein = protein.from_pdb_string(f.read())

            if amber_relaxer:
                relaxed_output_path = os.path.join(
                    results_dir, f'relaxed_{model_name}.pdb')
                #Run in a new process to prevent CUDA initialitation error of openmm  
                relax_results_pkl = os.path.join(results_dir, 'relax_results.pkl')
                try:
                    cmd = f"run_relaxation.py --unrelaxed_pdb_path {unrelaxed_pdb_path} --relaxed_output_path {relaxed_output_path} --model_name {model_name} --relax_results_pkl {relax_results_pkl}"
                    if flags['use_gpu_relax']:
                        cmd = f"{cmd} --use_gpu_relax"
                    logging.debug(cmd)
                    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
                    output, error = p.communicate()
                    if output:
                        logging.info(output.decode())
                    if error:
                        logging.error(error.decode())
                except Exception as e:
                    logging.error(f"Relaxation failed: {e}.")
                
                if os.path.exists(relax_results_pkl):
                    with open(relax_results_pkl, 'rb') as f:
                        relax_results = pickle.load(f)
                        relaxed_pdbs = relax_results['relaxed_pdbs']
                        relax_metrics = relax_results['relax_metrics']
                #Skip if relaxed model already exists


        # Rank by model confidence and write out relaxed PDBs in rank order.
        ranked_order = []
        for idx, (model_name, _) in enumerate(
            sorted(ranking_confidences.items(), key=custom_sort, reverse=True)):
            ranked_order.append(model_name)
            ranked_output_path = os.path.join(results_dir, f'ranked_by_plddt_{idx}.pdb')
            with open(ranked_output_path, 'w') as f:
                if amber_relaxer:
                    f.write(relaxed_pdbs[model_name])
                else:
                    f.write(unrelaxed_pdbs[model_name])
        ranking_output_path = os.path.join(results_dir, 'ranking_debug.json')
        with open(ranking_output_path, 'w') as f:
            label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
            f.write(json.dumps(
                {label: ranking_confidences, 'order': ranked_order}, indent=4))

        logging.info('Final timings for %s: %s', protein_names, timings)

        timings_output_path = os.path.join(results_dir, 'timings.json')
        with open(timings_output_path, 'w') as f:
            f.write(json.dumps(timings, indent=4))
        if amber_relaxer:
            relax_metrics_path = os.path.join(results_dir, 'relax_metrics.json')
            with open(relax_metrics_path, 'w') as f:
                f.write(json.dumps(relax_metrics, indent=4))

        evaluation = EvaluationPipeline(fasta_path=fasta_path,
                                         results_dir=results_dir,
                                           features_dir=msa_output_dir,
                                           prediction_pipeline=prediction_pipeline,
                                           batch_prediction=batch_prediction,
                                           msa_stats=msa_stats)
        evaluation.run_pipeline()

        if batch_prediction:
            msg = "Task finished"
            logging.info(msg)
            _write_job_status_log(flags['job_status_log_file'], msg)
            score_dict = evaluation.get_scores(score_dict)
            results_queue.put(score_dict)
        else:
            msg = "Alphafold pipeline completed. Exit code 0"
            logging.info(msg)
            _write_job_status_log(flags['job_status_log_file'], msg)

    else:
        msg = "Alphafold pipeline completed with feature generation. Exit code 0"
        logging.info(msg)
        _write_job_status_log(flags['job_status_log_file'], msg)


def parse_fasta(fasta_path):
    with open(fasta_path) as f:
        input_fasta_str = f.read()
    description_sequence_dict = {}
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    for i, desc in enumerate(input_descs):
        description_sequence_dict[desc] = input_seqs[i]
    return description_sequence_dict

def write_prediction_results(scores, results_file):
    with open(results_file, 'w') as f:
        f.write(f"{','.join([x for x in scores[0].keys()])}\n")
        for score in scores:
            logging.debug(score)
            f.write(f"{','.join([str(x) for x in score.values()])}\n")

def get_prediction_results(scores, results_file):
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            lines = f.readlines()
        lines = [l.strip('\n') for l in lines]
        if len(lines) > 1:
            keys = lines[0].split(',')
        for line in lines[1:]:
            values = line.split(',')
            values = [None if value == 'None' else value for value in values]
            dict_ = {key: value for (key, value) in zip(keys, values)}
            if not dict_['protein_names'] is None:
                scores.append(dict_)
    return scores

def find_index(list_of_dicts, value):
    for index, dictionary in enumerate(list_of_dicts):
        if value in list(dictionary.values()):
            return index
    return None

def check_batch_prediction_task(description_1, description_2, combinations, scores, prediction_pipeline, flags):
    if isinstance(description_1, list):
        description_1 = '_'.join(description_1)
    
    #Skip task if the complementary pair already exists
    if (description_2, description_1) in combinations:
        return False
    
    #Skip task if prediction already exists in score dict and all scores are found
    combinations.append((description_1, description_2))
    protein_names = f"{description_1}_{description_2}"
    if protein_names in [x['protein_names'] if 'protein_names' in x else None for x in scores]:
        score_index = find_index(scores, protein_names)
        logging.info([item for item in scores[score_index].values()])
        if not None in [item for item in scores[score_index].values()]:
            msg = f"Prediction pair {protein_names} already exists. Task finished. Skipping."
            logging.info(msg)
            _write_job_status_log(flags.job_status_log_file, msg)
            return False
        else:
            logging.info(f"Prediction pair {protein_names} already exists but not all evaluation scores found. Adding to task list.")
            return True
    else:
        logging.info(f"Prediction pair {protein_names} not found in score list. Adding to task list.")
        score_dict = {'protein_names': protein_names,
                            'min_pae_value': None, 'min_pae_model_name': None,
                            'max_ptm_value': None, 'max_ptm_model_name': None,
                            'max_iptm_value': None, 'max_iptm_model_name': None,
                            'max_multimer_score_value': None, 'max_multimer_score_model_name': None}
        scores.append(score_dict)
        return True
    
def create_input_fasta(description_1, description_2, sequence_1, sequence_2):
    #Create input FASTA file
    sequence_path = os.path.join(FLAGS.output_dir, "sequences")
    if not os.path.exists(sequence_path):
        os.mkdir(sequence_path)
    if isinstance(description_1, list):
        joined_description_1 = '_'.join(description_1)
        protein_names = f"{joined_description_1}_{description_2}"
        fasta_path = os.path.join(sequence_path, f"{joined_description_1}_{description_2}.fasta")
    else:
        protein_names = f"{description_1}_{description_2}"
        fasta_path = os.path.join(sequence_path, f"{description_1}_{description_2}.fasta")
    logging.debug(f"Writing input fasta: {fasta_path}")
    if isinstance(description_1, list):
        with open(fasta_path, 'w') as f:
            for i, desc_1 in enumerate(description_1):
                f.write(f">{desc_1}\n")
                seq_1 = sequence_1[i]
                f.write(f"{seq_1}\n\n")
            f.write(f">{description_2}\n")
            f.write(sequence_2)
    else:
        with open(fasta_path, 'w') as f:
            f.write(f">{description_1}\n")
            f.write(sequence_1)
            f.write(f"\n\n>{description_2}\n")
            f.write(sequence_2)
    return protein_names, fasta_path

def predict_structure_wrapper(kwargs, flags, gpu_id) -> bool:
    #Restrict task to specific GPU
    logging.info("Starting batch prediction")
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    logging.info(f"GPU id for this process is {os.environ['CUDA_VISIBLE_DEVICES']}")
    run_multimer_system = 'multimer' in flags['model_preset']
    if flags['model_preset'] == 'monomer_casp14':
        num_ensemble = 8
    else:
        num_ensemble = 1
    if run_multimer_system:
        num_predictions_per_model = flags['num_multimer_predictions_per_model']
    else:
        num_predictions_per_model = 1
    if kwargs['prediction_pipeline'] == 'alphafold':
        model_runners = {}
        model_names = config.MODEL_PRESETS[flags['model_preset']]
        for model_name in model_names:
            logging.debug(f'Model name: {model_name}')
            index = re.match('^model_(\d{1}).*', model_name).group(1)
            model_list = flags['model_list']
            if index in model_list:
                model_config = config.model_config(model_name, flags['num_recycle'])
                if run_multimer_system:
                    model_config.model.num_ensemble_eval = num_ensemble
                else:
                    model_config.data.eval.num_ensemble = num_ensemble
                model_params = data.get_model_haiku_params(
                    model_name=model_name, data_dir=flags['data_dir'])
                model_runner = model.RunModel(model_config, model_params)
                for i in range(num_predictions_per_model):
                    model_runners[f'{model_name}_pred_{i}'] = model_runner

        logging.info('Found %d models: %s', len(model_runners),
                    list(model_runners.keys()))
        kwargs['model_runners'] = model_runners
    else:
        kwargs['model_runners'] = {}

    random_seed = flags['random_seed']
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize // len(kwargs['model_runners']))
    kwargs['random_seed'] = random_seed
    logging.info('Using random seed %d for the data pipeline', random_seed)
    kwargs['flags'] = flags
    predict_structure(**kwargs)

def get_num_available_gpus():
    if FLAGS.prediction == 'alphafold':
        nvidia_smi.nvmlInit()
        devices = nvidia_smi.nvmlDeviceGetCount()
        if devices > 0:
            num_gpus = devices
        else:
            num_gpus = 0
        logging.info(f'Found {devices} GPUs')
    else:
        raise SystemExit(f"{FLAGS.prediction} pipeline unknown.")
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        if os.environ['CUDA_VISIBLE_DEVICES'] == "":
            num_gpus = 0
    return int(num_gpus)

def split_into_subsequences(input_string, max_length):
    """Split into equally sized chunks"""
    num_parts = (len(input_string) + max_length - 1) // max_length
    part_size = len(input_string) // num_parts
    parts = [input_string[i * part_size:(i + 1) * part_size] for i in range(num_parts - 1)]
    parts.append(input_string[(num_parts - 1) * part_size:])
    logging.info(f"Splitted {input_string} into:")
    logging.info(parts)
    return parts

def merge_baits(prediction_groups_mapping, max_batch_sequence_length):
    #Put all baits into one sequence if total bait sequence length smaller than threshold
    new_bait = None
    for id, group in prediction_groups_mapping.items():
        all_seqs_len = []
        all_seqs = []
        all_descs = []
        all_indices = []
        for desc, (seq, index) in group['baits'].items():
            logging.debug(group['baits'])
            all_seqs_len.append(len(seq))
            all_seqs.append(seq)
            all_descs.append(desc)
            all_indices.append(index)

        total_seq_len = sum(all_seqs_len)
        if total_seq_len < int(max_batch_sequence_length) and len(all_seqs_len) > 1:
            logging.info("Concatenating bait sequences")
            if len(set(all_indices)) > 1:
                logging.warning("Some settings for baits are not equal. The settings for the first bait sequence will be used for merging")
            concat_seq = ''.join(all_seqs)
            concat_desc = '_'.join(all_descs)
            new_bait = {'baits': {concat_desc: (concat_seq, all_indices[0])}}
        else:
            new_bait = None
        if new_bait:
            prediction_groups_mapping[id]['baits'] = new_bait['baits']
    return prediction_groups_mapping

def split_sequences(prediction_groups_mapping, max_batch_sequence_length):
    #{'baits': {}, 'preys': {desc: (seq, index)}}
    """For batch pairwise predictions a max_batch_sequence_length
      for baits and preys can be used to split bait and prey sequences if they exceed the threshold. To keep things simple baits and preys are split
      N times into equally sized chunks if they exceed the threshold. This will ensure that the total sequence length of a pair will always be
        below the max_batch_sequence_length. However it can also lead to significantly smaller total sequence length.  """
    for id, group in deepcopy(prediction_groups_mapping).items():
        descs_to_delete = {}
        for bait_prey in group.keys():
            for desc, (seq, index) in group[bait_prey].items():
                descs_to_delete[bait_prey] = []
                if len(seq) > max_batch_sequence_length:
                    logging.info(f"Sequence length {len(seq)} of {desc} larger than limit: {max_batch_sequence_length}. Splitting sequence.")
                    split_sequences = split_into_subsequences(seq, max_batch_sequence_length)
                    split_descs = [f"{desc}_split{i}" for i in range(len(split_sequences))]
                    split_indices = [index for _ in range(len(split_sequences))]
                    for i, new_desc in enumerate(split_descs):
                        prediction_groups_mapping[id][bait_prey][new_desc] = (split_sequences[i], split_indices[i])
                    del prediction_groups_mapping[id][bait_prey][desc]

    logging.debug(prediction_groups_mapping)
    return prediction_groups_mapping


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)
    logging.info("Alphafold pipeline starting...")
    if FLAGS.precomputed_msas_list is None:
        FLAGS.precomputed_msas_list = [FLAGS.precomputed_msas_list]
    if not FLAGS.precomputed_msas_path in ['None', None] or any([not item in ('None', None) for item in FLAGS.precomputed_msas_list]):
        FLAGS.use_precomputed_msas = True

    #Do not check for MSA tools when MSA already exists.
    global run_multimer_system
    run_multimer_system = 'multimer' in FLAGS.model_preset
    prediction_pipeline = FLAGS.prediction
    feature_pipeline = FLAGS.db_preset
    use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
    use_mmseqs_local = FLAGS.db_preset == 'colabfold_local'
    use_mmseqs_api = FLAGS.db_preset == 'colabfold_web'
    if FLAGS.precomputed_msas_path and FLAGS.precomputed_msas_list:
        logging.warning("Flags --precomputed_msas_path and --precomputed_msas_list selected at the same time. "
                        "MSAs from --precomputed_msas_list get priority over MSAs from --precomputed_msas_path.")
    if not FLAGS.pipeline == 'continue_from_features':
        for tool_name in (
            'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
            if not FLAGS[f'{tool_name}_binary_path'].value:
                raise ValueError(f'Path to "{tool_name}" binary not set.')
            if use_mmseqs_local:
                if not FLAGS.mmseqs_binary_path:
                    raise ValueError(f'Path to mmseqs binary not set.')
        _check_flag('small_bfd_database_path', 'db_preset',
                    should_be_set=FLAGS.db_preset=='reduced_dbs')
        _check_flag('bfd_database_path', 'db_preset',
                    should_be_set=FLAGS.db_preset=='full_dbs')
        _check_flag('mgnify_database_path', 'db_preset',
                    should_be_set=FLAGS.db_preset=='full_dbs' or FLAGS.db_preset=='reduced_dbs')
        _check_flag('uniref30_database_path', 'db_preset',
                    should_be_set=FLAGS.db_preset=='full_dbs')
        _check_flag('colabfold_envdb_database_path', 'db_preset',
                    should_be_set=FLAGS.db_preset=='colabfold_local')
        _check_flag('uniref30_mmseqs_database_path', 'db_preset',
                    should_be_set=FLAGS.db_preset=='colabfold_local')
        _check_flag('uniref90_mmseqs_database_path', 'db_preset',
                    should_be_set=FLAGS.db_preset=='colabfold_local')
        _check_flag('uniprot_mmseqs_database_path', 'db_preset',
                    should_be_set=FLAGS.db_preset=='colabfold_local')
    global num_ensemble
    if FLAGS.model_preset == 'monomer_casp14':
        
        num_ensemble = 8
    else:
        num_ensemble = 1

    #Only one fasta file allowed
    #protein_names = pathlib.Path(FLAGS.fasta_path).stem


    template_searcher_hmm = hmmsearch.Hmmsearch(
        binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        hhalign_binary_path=FLAGS.hhalign_binary_path,
        database_path=FLAGS.pdb_seqres_database_path,
        custom_tempdir=FLAGS.custom_tempdir)
    template_featurizer_hmm = templates.HmmsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path,
        custom_tempdir=FLAGS.custom_tempdir,
        strict_error_check=True)

    template_searcher_hhr = hhsearch.HHSearch(
        binary_path=FLAGS.hhsearch_binary_path,
        hhalign_binary_path=FLAGS.hhalign_binary_path,
        databases=[FLAGS.pdb70_database_path],
        custom_tempdir=FLAGS.custom_tempdir)
    template_featurizer_hhr = templates.HhsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path,
        custom_tempdir=FLAGS.custom_tempdir,
        strict_error_check=True)

    accession_species_db = None
    if FLAGS.db_preset in ['colabfold_local', 'colabfold_web']:
        accession_species_db = os.path.join(os.path.dirname(FLAGS.uniprot_database_path), 'accession_species.db')
        if not os.path.exists(accession_species_db):
            logging.info(f"Creating database with accession to species identifier mapping from uniprot in {accession_species_db}. This takes ~30 min and is only done if the file is missing.")
            engine = create_engine(f'sqlite:///{accession_species_db}', echo=False)
            Base.metadata.create_all(engine)
            Session = sessionmaker(bind=engine)
            db_session = Session()
            create_accession_species_db(FLAGS.uniprot_database_path, db_session, accession_species_db)
            db_session.close()
        else:
            logging.info(f"Database for accession to species identifier mapping already exists in {accession_species_db}")

    monomer_data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        hhblits_binary_path=FLAGS.hhblits_binary_path,
        mmseqs_binary_path=FLAGS.mmseqs_binary_path,
        uniref90_database_path=FLAGS.uniref90_database_path,
        uniref90_mmseqs_database_path=FLAGS.uniref90_mmseqs_database_path,
        mgnify_database_path=FLAGS.mgnify_database_path,
        bfd_database_path=FLAGS.bfd_database_path,
        uniref30_database_path=FLAGS.uniref30_database_path,
        uniref30_mmseqs_database_path=FLAGS.uniref30_mmseqs_database_path,
        uniprot_database_path=FLAGS.uniprot_database_path,
        uniprot_mmseqs_database_path=FLAGS.uniprot_mmseqs_database_path,
        small_bfd_database_path=FLAGS.small_bfd_database_path,
        colabfold_envdb_database_path=FLAGS.colabfold_envdb_database_path,
        template_searcher_hhr=template_searcher_hhr,
        template_searcher_hmm=None,
        template_featurizer_hhr=template_featurizer_hhr,
        template_featurizer_hmm=None,
        db_preset=FLAGS.db_preset,
        use_precomputed_msas=FLAGS.use_precomputed_msas,
        custom_tempdir=FLAGS.custom_tempdir,
        precomputed_msas_path=FLAGS.precomputed_msas_path,
        accession_species_db=accession_species_db,
        multimer=False)
    

    #Calculates all MSAs needed for monomer or multimer pipeline
    if FLAGS.pipeline == 'batch_msas':
      logging.debug("Adjusting template searcher and featurizer for batch_msas pipeline.")
      monomer_data_pipeline.template_searcher_hmm = template_searcher_hmm
      monomer_data_pipeline.template_featurizer_hmm = template_featurizer_hmm
      #Calculates uniprot hits
      monomer_data_pipeline.multimer = True
      data_pipeline = pipeline_batch.DataPipeline(monomer_data_pipeline=monomer_data_pipeline, batch_mmseqs=FLAGS.db_preset=='colabfold_local')
      global num_predictions_per_model
      num_predictions_per_model = 1
    elif run_multimer_system and not FLAGS.pipeline == 'batch_msas':
        logging.debug("Adjusting template searcher and featurizer for multimer pipeline.")
        monomer_data_pipeline.template_searcher_hhr = None
        monomer_data_pipeline.template_featurizer_hhr = None
        monomer_data_pipeline.template_searcher_hmm = template_searcher_hmm
        monomer_data_pipeline.template_featurizer_hmm = template_featurizer_hmm
        monomer_data_pipeline.multimer = True
        num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
        data_pipeline = pipeline_multimer.DataPipeline(
            monomer_data_pipeline=monomer_data_pipeline,
            jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
            uniprot_database_path=FLAGS.uniprot_database_path,
            pairing=FLAGS.msa_pairing)
    else:
        logging.debug("Using monomer pipeline.")
        num_predictions_per_model = 1
        data_pipeline = monomer_data_pipeline

    if not FLAGS.pipeline in BATCH_PREDICTION_MODES:
        model_runners = {}
        model_names = config.MODEL_PRESETS[FLAGS.model_preset]
        for model_name in model_names:
            logging.debug(f'Model name: {model_name}')
            index = re.match('^model_(\d{1}).*', model_name).group(1)
            model_list = FLAGS.model_list
            if index in model_list:
                model_config = config.model_config(model_name, FLAGS.num_recycle)
                if run_multimer_system:
                    model_config.model.num_ensemble_eval = num_ensemble
                else:
                    model_config.data.eval.num_ensemble = num_ensemble
                model_params = data.get_model_haiku_params(
                    model_name=model_name, data_dir=FLAGS.data_dir)
                model_runner = model.RunModel(model_config, model_params)
                for i in range(num_predictions_per_model):
                    model_runners[f'{model_name}_pred_{i}'] = model_runner

        logging.info('Found %d models: %s', len(model_runners),
                    list(model_runners.keys()))
        random_seed = FLAGS.random_seed
        if random_seed is None:
            random_seed = random.randrange(sys.maxsize // len(model_runners))
        logging.info('Using random seed %d for the data pipeline', random_seed)


    if FLAGS.run_relax:
        amber_relaxer = relax.AmberRelaxation(
            max_iterations=RELAX_MAX_ITERATIONS,
            tolerance=RELAX_ENERGY_TOLERANCE,
            stiffness=RELAX_STIFFNESS,
            exclude_residues=RELAX_EXCLUDE_RESIDUES,
            max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
            use_gpu=FLAGS.use_gpu_relax)
    else:
        amber_relaxer = None




    #Code adaptions to handle custom template, no MSA, no template
    #Check that no_msa_list has same number of elements as in fasta_sequence,
    #and convert to bool.
    description_sequence_dict = parse_fasta(FLAGS.fasta_path)
    seq_titles = list(description_sequence_dict.keys())
    if FLAGS.no_msa_list:
        if len(FLAGS.no_msa_list) != len(description_sequence_dict):
            raise ValueError('--no_msa_list must either be omitted or match '
                            'number of sequences.')
        no_msa_list = []
        for s in FLAGS.no_msa_list:
            if s.lower() == 'true':
                no_msa_list.append(True)
            elif s.lower() == 'false':
                no_msa_list.append(False)
            else:
                raise ValueError('--no_msa_list must contain comma separated '
                                    'true or false values.')
    else:
        no_msa_list = [False] * len(description_sequence_dict)


    if FLAGS.custom_template_list:
        if len(FLAGS.custom_template_list) != len(description_sequence_dict):
            raise ValueError('--custom_template_list must either be omitted or match '
                            'number of sequences.')
        custom_template_list = []
        for s in FLAGS.custom_template_list:
            if s in ["None", "none"]:
                custom_template_list.append(None)
            else:
                custom_template_list.append(s)
    else:
        custom_template_list = [None] * len(description_sequence_dict)

    #Split multichain template into individual files for feature processing. The features will be combined into a single template before prediction
    multichain_template_list = [False] * len(description_sequence_dict)
    if not FLAGS.multichain_template_path in ["None", None]:
        logging.info("Found multichain_template. This will overwrite any other custom templates")
        multichain_custom_template_list = templates.split_multichain_template(description_sequence_dict.values(), FLAGS.multichain_template_path, FLAGS.kalign_binary_path, FLAGS.custom_tempdir)
        logging.info("Updated custom_template_list:")
        #Merge with singlechain custom templates:
        for i, sc_template in enumerate(custom_template_list):
            if not multichain_custom_template_list[i] is None:
                custom_template_list[i] = multichain_custom_template_list[i]
                multichain_template_list[i] = True

        logging.info(custom_template_list)

    if FLAGS.no_template_list:
        if len(FLAGS.no_template_list) != len(description_sequence_dict):
            raise ValueError('--no_template_list must either be omitted or match '
                            'number of sequences.')
        no_template_list = []
        for s in FLAGS.no_template_list:
            if s.lower() == 'true':
                no_template_list.append(True)
            elif s.lower() == 'false':
                no_template_list.append(False)
            else:
                raise ValueError('--no_template_list must contain comma separated '
                                'true or false values.')
    else:
        no_template_list = [False] * len(description_sequence_dict)

    if FLAGS.precomputed_msas_list and not FLAGS.precomputed_msas_list == [None]:
        if len(FLAGS.precomputed_msas_list) != len(description_sequence_dict):
            raise ValueError('--precomputed_msas_list must either be omitted or match number of sequences.')

        precomputed_msas_list = []
        for s in FLAGS.precomputed_msas_list:
            if s in ["None", "none"]:
                precomputed_msas_list.append(None)
            else:
                precomputed_msas_list.append(s)
    else:
        precomputed_msas_list = [None] * len(description_sequence_dict)


    #Find matching MSAs in case of monomer pipeline. In case of multimer pipeline this is done in pipeline_multimer.py
    if not run_multimer_system and not FLAGS.pipeline == 'batch_msas' and FLAGS.precomputed_msas_path:
        #Only search for MSAs in precomputed_msas_path if no direct path is given in precomputed_msas_list
        if precomputed_msas_list[0] in [None, "None", "none"]:
            pcmsa_map = pipeline.get_pcmsa_map(FLAGS.precomputed_msas_path,
                                                                    description_sequence_dict,
                                                                    FLAGS.db_preset)
            logging.debug("Precomputed MSAs map")
            logging.debug(pcmsa_map)
            if len(pcmsa_map) == 1:
                precomputed_msas_list = list(pcmsa_map.values())[0]
            elif len(pcmsa_map) > 1:
                logging.warning("Found more than one precomputed MSA for given sequence. Will use the first one in the list.")
                precomputed_msas_list = list(pcmsa_map.values())[0]
    elif FLAGS.pipeline == 'batch_msas' and FLAGS.precomputed_msas_path:
        pcmsa_map = pipeline.get_pcmsa_map(FLAGS.precomputed_msas_path,
                                                        description_sequence_dict,
                                                        FLAGS.db_preset)
        if len(pcmsa_map) == len(precomputed_msas_list):
            precomputed_msas_list = list(pcmsa_map.values())

    #Batch predictions
    results_file_pairwise_predictions = os.path.join(FLAGS.output_dir, "predictions", FLAGS.predictions_dir, "pairwise_prediction_results.csv")
    if FLAGS.pipeline == 'first_n_vs_rest':
        first_n_seq = FLAGS.first_n_seq
    else:
        first_n_seq = None
    combinations = []
    results_queue = multiprocessing.Queue()
    task_queue = multiprocessing.Queue()
    # Get previous predictions from csv file
    scores = []
    scores = get_prediction_results(scores, results_file_pairwise_predictions)
    tasks = []
    kwargs_common = {
                            'output_dir_base': FLAGS.output_dir,
                            'data_pipeline': data_pipeline,
                            'amber_relaxer': amber_relaxer,
                            'no_msa_list': no_msa_list,
                            'no_template_list': no_template_list,
                            'custom_template_list': custom_template_list,
                            'precomputed_msas_list': precomputed_msas_list,
                            'prediction_pipeline': prediction_pipeline,
                            'feature_pipeline': feature_pipeline,
                            'is_multimer': run_multimer_system,
                            'batch_prediction': True,
                            'benchmark': FLAGS.benchmark,
                            'use_existing_features': True
                            }
    flag_dict = {}
    for attr, flag_obj in FLAGS.__flags.items():
        flag_dict[attr] = flag_obj.value
    #All vs all workflow
    # if FLAGS.pipeline == 'all_vs_all':
    #     for i, (desc_1, seq_1) in enumerate(description_sequence_dict.items()):
    #         for o, (desc_2, seq_2) in enumerate(description_sequence_dict.items()):
    #             kwargs = {k: v for (k,v) in kwargs_common.items()}
    #             desc_1_prev = desc_1
    #             if desc_1 == desc_2:
    #                 desc_1 = f"{desc_1}_1"
    #                 desc_2 = f"{desc_2}_2"
    #             kwargs['no_msa_list'] = [no_msa_list[i], no_msa_list[o]]
    #             kwargs['no_template_list'] =  [no_template_list[i], no_template_list[o]]
    #             kwargs['custom_template_list'] = [custom_template_list[i], custom_template_list[o]]
    #             kwargs['precomputed_msas_list'] = [precomputed_msas_list[i], precomputed_msas_list[o]]
    #             if check_batch_prediction_task(desc_1, desc_2, combinations, scores, prediction_pipeline):
    #                 kwargs['protein_names'], kwargs['fasta_path'] = create_input_fasta(desc_1, desc_2, seq_1, seq_2)
    #                 index = find_index(scores, kwargs['protein_names'])
    #                 kwargs['score_dict'] = scores[index]
    #                 tasks.append(kwargs)
    #             desc_1 = desc_1_prev



    prediction_groups_mapping = {}
    no_msa_list_full = no_msa_list.copy()
    no_template_list_full = no_template_list.copy()
    custom_template_list_full = custom_template_list.copy()
    precomputed_msas_list_full = precomputed_msas_list.copy()
    multichain_template_list_full = multichain_template_list.copy()
    if FLAGS.pipeline in ['first_vs_all', 'first_n_vs_rest', 'all_vs_all']:
        prediction_groups_mapping = {'group1': {}}
        if FLAGS.pipeline == 'first_vs_all':
            bait_stop_index = 1
        elif FLAGS.pipeline == 'first_n_vs_rest':
            bait_stop_index = FLAGS.first_n_seq
        else:
            bait_stop_index = None
        if bait_stop_index:
            if bait_stop_index >= len(description_sequence_dict):
                raise InputError("first_n_seq needs to be smaller than the total number of given sequences.")
        for i, (desc, seq) in enumerate(description_sequence_dict.items()):
            logging.debug(desc)
            if bait_stop_index:
                if not 'group1' in prediction_groups_mapping:
                    prediction_groups_mapping['group1'] = {}
                if not 'baits' in prediction_groups_mapping['group1']:
                    prediction_groups_mapping['group1']['baits'] = {}
                if not 'preys' in prediction_groups_mapping['group1']:
                    prediction_groups_mapping['group1']['preys'] = {}
                if i < bait_stop_index:
                    prediction_groups_mapping['group1']['baits'][desc] = (seq, i)
                else:
                    prediction_groups_mapping['group1']['preys'][desc] = (seq, i)
            else:
                if not 'group1' in prediction_groups_mapping:
                    prediction_groups_mapping['group1'] = {}
                if not 'preys' in prediction_groups_mapping['group1']:
                    prediction_groups_mapping['group1']['preys'] = {}
                prediction_groups_mapping['group1']['preys'][desc] = (seq, i)

    elif FLAGS.pipeline in ['grouped_bait_vs_preys', 'grouped_all_vs_all']:
        for index, (desc, seq) in enumerate(description_sequence_dict.items()):
            regex_bait = r'bait_((?:(?!_split\d+).)*)(?:_split\d+)?'
            regex_prey = r'prey_((?:(?!_split\d+).)*)(?:_split\d+)?'
            if re.search(regex_bait, desc):
                id = re.search(regex_bait, desc).groups(1)[0]
                desc = desc.replace(f'_bait_{id}', '')
                if not id in prediction_groups_mapping:
                    prediction_groups_mapping[id] = {'baits': {desc: (seq, index)}, 'preys': {}}
                else:
                    prediction_groups_mapping[id]['baits'][desc] = (seq, index)
            elif re.search(regex_prey, desc):
                id = re.search(regex_prey, desc).groups(1)[0]
                desc = desc.replace(f'_prey_{id}', '')
                if not id in prediction_groups_mapping:
                    prediction_groups_mapping[id] = {'baits': {}, 'preys': {desc: (seq, index)}}
                else:
                    prediction_groups_mapping[id]['preys'][desc] = (seq, index)
            else:
                logging.error(f"{desc} has the wrong format for grouped workflow. Expected format: ProteinName_[bait|prey]_unique_identifier")

    logging.debug(prediction_groups_mapping)
        
    subunit_list = []
    if FLAGS.pipeline in ['grouped_bait_vs_preys', 'first_vs_all', 'first_n_vs_rest']:
        #Bait vs preys
        #Merge bait complex if possible
        #prediction_groups_mapping = merge_baits(prediction_groups_mapping, max_batch_sequence_length=FLAGS.batch_max_sequence_length)
        #Split sequences if too long
        #prediction_groups_mapping = split_sequences(prediction_groups_mapping, max_batch_sequence_length=FLAGS.batch_max_sequence_length)
        for id, group in prediction_groups_mapping.items():
            desc_1_list, seq_1_list, index_1_list = [], [], []
            for desc_1, (seq_1, index_1) in group['baits'].items():
                desc_1_prev = desc_1
                desc_1_list.append(desc_1)
                seq_1_list.append(seq_1)
                index_1_list.append(index_1)
            for desc_2, (seq_2, index_2) in group['preys'].items():
                desc_2_prev = desc_2
                kwargs = {k: v for (k,v) in kwargs_common.items()}
                if len(desc_1_list) == 1:
                    desc_1 = desc_1_list[0]
                    desc_1_prev = desc_1
                    if desc_1 == desc_2:
                        desc_1 = f"{desc_1}_1"
                        desc_2 = f"{desc_2}_2"
                else:
                    desc_1_prev = None
                no_msa_list = [no_msa_list_full[index_1] for index_1 in index_1_list] + [no_msa_list_full[index_2]]
                no_template_list = [no_template_list_full[index_1] for index_1 in index_1_list] + [no_template_list_full[index_2]]
                custom_template_list = [custom_template_list_full[index_1] for index_1 in index_1_list] + [custom_template_list_full[index_2]]
                precomputed_msas_list = [precomputed_msas_list_full[index_1] for index_1 in index_1_list] + [precomputed_msas_list_full[index_2]]
                multichain_template_list = [multichain_template_list_full[index_1] for index_1 in index_1_list] + [multichain_template_list_full[index_2]]
                kwargs['no_msa_list'] = no_msa_list
                kwargs['no_template_list'] =  no_template_list
                kwargs['custom_template_list'] = custom_template_list
                kwargs['precomputed_msas_list'] = precomputed_msas_list
                kwargs['multichain_template_list'] = multichain_template_list
                if check_batch_prediction_task(desc_1_list, desc_2, combinations, scores, prediction_pipeline, FLAGS):
                    kwargs['protein_names'], kwargs['fasta_path'] = create_input_fasta(desc_1_list, desc_2, seq_1_list, seq_2)
                    index = find_index(scores, kwargs['protein_names'])
                    kwargs['score_dict'] = scores[index]
                    if not kwargs in tasks:
                        tasks.append(kwargs)
                    else:
                        logging.warning(f"{kwargs['protein_names']} already in task list. Not added again.")
                    for desc_1 in desc_1_list:
                        subunit_list.append(desc_1)
                    subunit_list.append(desc_2)
                desc_1 = desc_1_prev
                desc_2 = desc_2_prev
        
    elif FLAGS.pipeline in ['grouped_all_vs_all', 'all_vs_all']:
        logging.info(f"Task list at beginning")
        logging.info(tasks)
        #Split sequences if too long
        #prediction_groups_mapping = split_sequences(prediction_groups_mapping, max_batch_sequence_length=FLAGS.batch_max_sequence_length)
        
        for id, group in prediction_groups_mapping.items():
            merged_dict = deepcopy(group['preys'])
            if 'baits' in group:
                if len(group['baits']) > 0:
                    merged_dict.update(deepcopy(group['baits']))
            for desc_1, (seq_1, index_1) in merged_dict.items():
                desc_1_prev = desc_1
                for desc_2, (seq_2, index_2) in merged_dict.items():
                    desc_2_prev = desc_2
                    kwargs = {k: v for (k,v) in kwargs_common.items()}
                    if desc_1 == desc_2:
                        desc_1 = f"{desc_1}_1"
                        desc_2 = f"{desc_2}_2"
                    no_msa_list = [no_msa_list_full[index_1]] + [no_msa_list_full[index_2]]
                    no_template_list = [no_template_list_full[index_1]] + [no_template_list_full[index_2]]
                    custom_template_list = [custom_template_list_full[index_1]] + [custom_template_list_full[index_2]]
                    precomputed_msas_list = [precomputed_msas_list_full[index_1]] + [precomputed_msas_list_full[index_2]]
                    multichain_template_list = [multichain_template_list_full[index_1]] + [multichain_template_list_full[index_2]]
                    kwargs['no_msa_list'] = no_msa_list
                    kwargs['no_template_list'] =  no_template_list
                    kwargs['custom_template_list'] = custom_template_list
                    kwargs['precomputed_msas_list'] = precomputed_msas_list        
                    kwargs['multichain_template_list'] = multichain_template_list                    
                    if check_batch_prediction_task(desc_1, desc_2, combinations, scores, prediction_pipeline, FLAGS):
                        kwargs['protein_names'], kwargs['fasta_path'] = create_input_fasta(desc_1, desc_2, seq_1, seq_2)
                        index = find_index(scores, kwargs['protein_names'])
                        kwargs['score_dict'] = scores[index]
                        if not kwargs in tasks:
                            tasks.append(kwargs)
                        else:
                            logging.warning(f"{kwargs['protein_names']} already in task list. Not added again.")
                            logging.warning(tasks)
                        subunit_list.append(desc_1)
                        subunit_list.append(desc_2)
                    desc_1 = desc_1_prev
                    desc_2 = desc_2_prev

    #Only relaxation step if predictions available
    elif FLAGS.pipeline == 'only_relax':
        batch_minimization(FLAGS.output_dir, FLAGS.use_gpu_relax)
        msg = "Alphafold pipeline completed. Exit code 0"
        logging.info(msg)
        _write_job_status_log(FLAGS.job_status_log_file, msg)
        

    #Single prediction workflow
    else:
        protein_names = '_'.join(list(description_sequence_dict.keys()))
        msg = f'Total number of tasks: 1'
        _write_job_status_log(FLAGS.job_status_log_file, msg)
        predict_structure(
            fasta_path=FLAGS.fasta_path,
            protein_names=protein_names,
            output_dir_base=FLAGS.output_dir,
            data_pipeline=data_pipeline,
            model_runners=model_runners,
            amber_relaxer=amber_relaxer,
            benchmark=FLAGS.benchmark,
            random_seed=random_seed,
            is_multimer=run_multimer_system,
            no_msa_list=no_msa_list,
            no_template_list=no_template_list,
            custom_template_list=custom_template_list,
            precomputed_msas_list=precomputed_msas_list,
            prediction_pipeline=prediction_pipeline,
            feature_pipeline=feature_pipeline,
            batch_mmseqs=FLAGS.db_preset=='colabfold_local',
            multichain_template_list=multichain_template_list,
            flags=flag_dict)

    if FLAGS.pipeline in BATCH_PREDICTION_MODES:
        file_lock_dict = {}
        with Manager() as manager:
            for subunit in subunit_list:
                if not subunit in file_lock_dict:
                    file_lock_dict[subunit] = manager.Lock()
            file_lock = file_lock_dict
            

            logging.info("Starting batch predictions")
            #To run prediction tasks in parallel, multiprocessing is used to avoid cuda errors and memory leaks
            active_processes = []

            available_gpus = get_num_available_gpus()
            if available_gpus == 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                FLAGS.num_gpu = 0
                gpu_available = False
                #Can only use one CPU otherwise too many threads are created https://github.com/google/jax/issues/1539
                max_tasks = 1
                logging.warning(f"No GPUs found. Switching to CPU.")
            elif available_gpus < FLAGS.num_gpu:
                FLAGS.num_gpu = available_gpus
                logging.warning(f"{FLAGS.num_gpu} GPUs requested but only {available_gpus} found. Adjusting number.")
                max_tasks = available_gpus
                gpu_available = True
            else:
                max_tasks = FLAGS.num_gpu
                gpu_available = True
                logging.info(f"Using {FLAGS.num_gpu} GPUs.")

            msg = f'Total number of tasks: {len(tasks)}'
            logging.info(msg)
            _write_job_status_log(FLAGS.job_status_log_file, msg)
            results_with_error = 0
            for kwargs in tasks:
                logging.info("Tasks:")
                logging.info(kwargs['protein_names'])
                kwargs['file_lock'] = file_lock
                task_queue.put(kwargs)

            while task_queue.qsize() > 0:
                logging.info(f"{task_queue.qsize()} tasks left.")
                #Start processes in parallel
                active_processes = []
                for _ in range(max_tasks):
                    logging.info(f"Max tasks: {max_tasks}")
                    if task_queue.qsize() > 0:
                        kwargs = task_queue.get()
                        logging.info(f"{task_queue.qsize()} tasks left.")
                        kwargs['results_queue'] = results_queue
                        logging.debug(f'active processes {len(active_processes)} < max_tasks {max_tasks}')
                        if gpu_available:
                            gpu_id = len(active_processes)
                        else:
                            gpu_id = None
                        logging.debug(f"Current GPU ID: {gpu_id}")
                        p = multiprocessing.Process(target=predict_structure_wrapper, args=(kwargs, flag_dict, gpu_id))
                        logging.info("Starting new process")
                        if not gpu_id is None:
                            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                        else:
                            os.environ["CUDA_VISIBLE_DEVICES"] = ""
                        p.start()
                        active_processes.append(p)
                    else:
                        break
                logging.debug(f'active processes {len(active_processes)} >= max_tasks {max_tasks}')
                for p in active_processes:
                    p.join()

                time.sleep(10)
                
                #Collect results
                while results_queue.qsize() > 0:
                    result = results_queue.get()
                    if None in list(result.values()):
                        logging.error(f'Got None for {result["protein_names"]}')
                        results_with_error += 1
                    value = result['protein_names']
                    index = find_index(scores, value)
                    scores[index] = result
                    logging.debug(f'Index for {value} is {index}')
                    logging.debug(scores[index])              
                write_prediction_results(scores, results_file_pairwise_predictions)
                if results_with_error > 0:
                    logging.warning(f"{results_with_error} results with failed evaluation.")
            
        #Run minimization if requested
        if FLAGS.run_relax:
            batch_minimization(FLAGS.output_dir, FLAGS.use_gpu_relax)

        #Run evaluation
        evaluation_batch = EvaluationPipelineBatch(FLAGS.output_dir, scores, seq_titles)
        evaluation_batch.run()
        msg = "Alphafold pipeline completed. Exit code 0"
        logging.info(msg)
        _write_job_status_log(FLAGS.job_status_log_file, msg)

def sigterm_handler(_signo, _stack_frame):
    raise KeyboardInterrupt

if __name__ == '__main__':
  # Needed so that multiprocessing.Pool workers don't crash.
  __spec__ = None
  flags.mark_flags_as_required([
      'fasta_path',
      'output_dir',
      'data_dir',
      'uniref90_database_path',
      'mgnify_database_path',
      'template_mmcif_dir',
      'max_template_date',
      'obsolete_pdbs_path',
      'use_gpu_relax',
  ])
  try:
    import signal
    signal.signal(signal.SIGTERM, sigterm_handler)
    app.run(main)
  except KeyboardInterrupt:
      msg = "Alphafold pipeline was aborted. Exit code 2"
      logging.error(msg)
      _write_job_status_log(FLAGS.job_status_log_file, msg)
  except Exception:
      logging.info(traceback.print_exc())
      msg = "Alphafold pipeline finished with an error. Exit code 1"
      logging.error(msg)
      _write_job_status_log(FLAGS.job_status_log_file, msg)

