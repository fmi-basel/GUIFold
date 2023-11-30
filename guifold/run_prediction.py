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
# Parts from https://github.com/hpcaitech/FastFold

"""Full AlphaFold protein structure prediction script."""
import json
import multiprocessing
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
from alphafold.model import config
from alphafold.model import model
from alphafold.relax import relax
from alphafold.data import parsers
from alphafold.model import data



import numpy as np
import gzip


# Internal import (7716).


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

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
flags.DEFINE_string('pdb100_database_path', None, 'Path to the PDB100 '
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
                'full', 'only_features', 'batch_msas', 'continue_from_features', 'all_vs_all', 'first_vs_all', 'first_n_vs_rest', 'only_relax'],
                'Choose preset pipeline configuration - '
                'full pipeline or '
                'stop after feature generation (only features) or '
                'calculate MSAs and find templates for given batch of sequences, uses template search based on monomer/multimer preset or'
                'continue from features.pkl file (continue_from_features)')
flags.DEFINE_enum('prediction', 'alphafold', ['alphafold', 'fastfold', 'rosettafold'],
                  'Choose preset prediction configuration - AlphaFold or FastFold implementation.')
flags.DEFINE_boolean('debug', False, 'Enable debugging output.')
flags.DEFINE_integer('first_n_seq', None, 'Parameter needed for first_n_vs_rest protocol to define the first N sequences that will be kept constant in a screening.')
flags.DEFINE_integer('num_gpu', 1, 'Number of GPUs.')
flags.DEFINE_integer('chunk_size', None, 'Chunk size.')
flags.DEFINE_boolean('inplace', False, 'Inplace.')
flags.DEFINE_list('model_list', '1,2,3,4,5', 'List of indices defining which Alphafold models to use.')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3
BATCH_PREDICTION_MODES = ['first_vs_all', 'all_vs_all', 'first_n_vs_rest']



def flag_specific_imports():
    if FLAGS.prediction == 'fastfold':
        import torch
        import torch.multiprocessing as mp
        from fastfold.model.hub import AlphaFold
        import fastfold
        from fastfold.config import model_config as ff_model_config
        from fastfold.common import protein as ff_protein
        from fastfold.data import feature_pipeline
        from fastfold.model.nn.triangular_multiplicative_update import set_fused_triangle_multiplication
        from fastfold.model.fastnn import set_chunk_size
        from fastfold.model.nn.triangular_multiplicative_update import set_fused_triangle_multiplication
        from fastfold.utils.inject_fastnn import inject_fastnn
        from fastfold.utils.import_weights import import_jax_weights_
        from fastfold.utils.tensor_utils import tensor_tree_map

        if int(torch.__version__.split(".")[0]) >= 1 and int(torch.__version__.split(".")[1]) > 11:
            torch.backends.cuda.matmul.allow_tf32 = True

    if FLAGS.prediction == 'rosettafold':
        from rosettafold.network import predict
        from collections import namedtuple
        from rosettafold.network.ffindex import read_index, read_data


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set and not bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')

def inference_model(rank, world_size, result_q, batch, model_name, chunk_size, inplace, model_preset, data_dir):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()

    torch.cuda.set_device(rank)
    config = ff_model_config(model_name)
    if chunk_size:
        if chunk_size > 0:
          config.globals.chunk_size = chunk_size
    config.globals.inplace = inplace
    config.globals.is_multimer = model_preset == 'multimer'
    set_fused_triangle_multiplication()
    set_fused_triangle_multiplication()
    model = AlphaFold(config)
    import_jax_weights_(model, data_dir, version=model_name)

    model = inject_fastnn(model)
    model = model.eval()
    model = model.cuda()

    set_chunk_size(model.globals.chunk_size)

    with torch.no_grad():
        batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}

        t = time.perf_counter()
        out = model(batch)
        logging.info(f"Inference time: {time.perf_counter() - t}")

    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

    result_q.put(out)

    torch.distributed.barrier()
    torch.cuda.synchronize()
    del os.environ['RANK']
    del os.environ['LOCAL_RANK']
    del os.environ['WORLD_SIZE']

def reextract_msa(msa, sequences, msa_output_path):
    """Re-extract the final MSA from the feature dict for rosettafold2"""
    ID_TO_HHBLITS_AA = residue_constants.ID_TO_HHBLITS_AA
    new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    reversed_msa = np.argsort(np.array(new_order_list))[msa]
    new_rows = [[ID_TO_HHBLITS_AA[aa] for aa in row] for row in reversed_msa]

    positions_to_insert = []
    msa_query_seq = list(new_rows[0])
    for seq in sequences:
        if re.search(seq, ''.join(msa_query_seq)):
            m = re.search(seq, ''.join(msa_query_seq))
            pos = m.end()
            msa_query_seq.insert(pos, "/")
            positions_to_insert.append(pos)
        else:
            logging.error(f"Mismatch between input sequences and MSAs. Cannot continue.\n{seq} not found in {msa_query_seq}")
            raise SystemExit

    rows = new_rows
    new_rows = []
    for row in rows:
        if row:
            row = list(row)
            for pos in positions_to_insert[:-1]:
                row.insert(pos, '/')
            new_rows.append(''.join(row))

    msa_output_path = os.path.join(msa_output_path, 'paired_concatenated_msas.a3m')
    with open(msa_output_path, 'w') as f:
        for i, row in enumerate(new_rows):
            f.write(f">{i}\n")
            f.write(f"{row}\n")

    return msa_output_path

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
    batch_mmseqs: bool = False):
    """Predicts structure using AlphaFold for the given sequence."""
    logging.info('Predicting %s', protein_names)
    fasta_name = os.path.splitext(os.path.basename(fasta_path))[0]
    timings = {}
    predictions_output_dir = os.path.join(output_dir_base, "predictions", prediction_pipeline)
    features_output_dir = os.path.join(output_dir_base, "features", feature_pipeline)
    results_dir = os.path.join(predictions_output_dir, protein_names)
    msa_output_dir = features_output_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)    
    if not os.path.exists(msa_output_dir):
        os.makedirs(msa_output_dir, exist_ok=True)
    features_output_path = os.path.join(features_output_dir, f'features_{protein_names}.pkl')
    # Get features.
    if not FLAGS.pipeline == 'continue_from_features':
        if batch_mmseqs:
            #Calculate MSAs in batch
            logging.info("Running batch mmseqs pipeline")
            data_pipeline.process_batch_mmseqs(input_fasta_path=fasta_path,
                                          msa_output_dir=msa_output_dir,
                                          num_cpu=FLAGS.num_cpu)
            #Use the MSAs computed in the previous step
            data_pipeline.set_use_precomputed_msas(True)
            data_pipeline.set_precomputed_msas_path(msa_output_dir)
        t_0 = time.time()
        feature_dict = data_pipeline.process(
                input_fasta_path=fasta_path,
                msa_output_dir=msa_output_dir,
                no_msa=no_msa_list,
                no_template=no_template_list,
                custom_template_path=custom_template_list,
                precomputed_msas_path=precomputed_msas_list,
                num_cpu=FLAGS.num_cpu)
        timings['features'] = time.time() - t_0


        # Write out features as a pickled dictionary.
        if not FLAGS.pipeline == 'batch_msas':
            logging.info(f"Writing features to {features_output_path}")
            with open(features_output_path, 'wb') as f:
                pickle.dump(feature_dict, f, protocol=4)

    #Stop here if only_msa flag is set
    if not FLAGS.pipeline == 'only_features' and not FLAGS.pipeline == 'batch_msas':
        if FLAGS.pipeline == 'continue_from_features':
            #Backward compatibility
            if not os.path.exists(features_output_path) and not os.path.exists(f"{features_output_path}.gz"):
                features_output_path = os.path.join(output_dir_base, fasta_name, 'features.pkl')
            if os.path.exists(features_output_path):
                with open(features_output_path, 'rb') as f:
                    feature_dict = pickle.load(f)
            elif os.path.exists(f"{features_output_path}.gz"):
                with gzip.open(f"{features_output_path}.gz", 'rb') as f:
                    feature_dict = pickle.load(f)
            else:
                raise("Continue_from_features requested but no feature pickle file found in this directory.")


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
            if prediction_pipeline == 'fastfold':
               model_name = re.match('(model_\d{1}(_ptm|_multimer){0,1}).*', model_name).group(1)
            model_name = model_name.replace("_pred_0", "")
            unrelaxed_pdb_path = os.path.join(results_dir, f'unrelaxed_{model_name}.pdb')

            logging.info('Running model %s on %s', model_name, protein_names)
            result_output_path = os.path.join(results_dir, f'result_{model_name}.pkl')
            #Skip if model already exists
            logging.debug(f"Model exists: {os.path.exists(unrelaxed_pdb_path)}")
            if not os.path.exists(unrelaxed_pdb_path) or not os.path.exists(result_output_path):
                logging.info(feature_dict.keys())
                if prediction_pipeline == 'rosettafold':
                    logging.info("Starting prediction pipeline for rosettafold2")

                    params_path = os.path.join(FLAGS.data_dir, "params", "RF2_apr23.pt")
                    if (torch.cuda.is_available()):
                        logging.info("Running on GPU")
                        pred = predict.Predictor(params_path, torch.device("cuda:0"))
                    else:
                        logging.info("Running on CPU")
                        pred = predict.Predictor(params_path, torch.device("cpu"))

                    with open(fasta_path) as f:
                        input_fasta_str = f.read()
                    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
                    msa_path = reextract_msa(feature_dict['msa'], input_seqs, msa_output_dir)
                    logging.info(f"Reextracted MSA path: {msa_path}")

                    inputs = msa_path
                    FFDB = FLAGS.pdb100_database_path
                    FFindexDB = namedtuple("FFindexDB", "index, data")
                    ffdb = FFindexDB(read_index(FFDB+'_pdb.ffindex'),
                                    read_data(FFDB+'_pdb.ffdata'))
                    logging.info(f"Inputs from run_prediciton {inputs}")
                    logging.info(f"Running prediction")
                    prediction_result = pred.predict(
                                            inputs=inputs,
                                            output_dir=results_dir, 
                                            symm='C1', 
                                            n_recycles=FLAGS.num_recycle, 
                                            n_models=1, 
                                            subcrop=1, 
                                            nseqs=256, 
                                            nseqs_full=2048, 
                                            ffdb=ffdb)
                    assert len(prediction_result) == 1
                    prediction_result = prediction_result[0]
                    model_output_path = os.path.join(results_dir, f"unrelaxed_model_{model_index}.pdb")
                    if not os.path.exists(model_output_path):
                        logging.error("No output model found!")
                        raise SystemExit
                    with open(model_output_path, 'r') as f:
                        content = f.read()
                        unrelaxed_pdbs[model_name] = content 
                    
                    

                elif prediction_pipeline == 'fastfold':
                    feature_dict = {}
                    #Fix some differences between openfold and alphafold feature_dict
                    for k, v in feature_dict_initial.items():
                        if k == 'template_all_atom_masks':
                            feature_dict['template_all_atom_mask'] = v
                        elif k == 'seq_length' and is_multimer:
                            feature_dict['seq_length'] = [v*v]
                        else:
                            feature_dict[k] = v
                    
                    ff_config = ff_model_config(model_name)
                    feature_processor = feature_pipeline.FeaturePipeline(ff_config.data)
                    processed_feature_dict = feature_processor.process_features(
                        feature_dict, mode='predict', is_multimer=is_multimer,
                    )
                    timings[f'process_features_{model_name}'] = time.time() - t_0


                    t_0 = time.time()
                    batch = processed_feature_dict

                
                    with mp.Manager() as manager:
                        result_q = manager.Queue()
                        chunk_size = FLAGS.chunk_size
                        inplace = FLAGS.inplace
                        num_gpu = FLAGS.num_gpu

                        if is_multimer:
                            params_file = f"params_{model_name}_v3.npz"
                        else:
                            params_file = f"params_{model_name}.npz"
                        params_path = os.path.join(FLAGS.data_dir, "params", params_file)
                        print(params_path)
                        model_preset = FLAGS.model_preset
                        torch.multiprocessing.spawn(inference_model, nprocs=num_gpu, args=(num_gpu, result_q, batch, model_name, chunk_size, inplace, model_preset, params_path))

                        prediction_result = result_q.get()

                        batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)

                        t_diff = time.time() - t_0
                        timings[f'predict_and_compile_{model_name}'] = t_diff
                        logging.info(
                            'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
                            model_name, protein_names, t_diff)

                        if benchmark:
                            pass
                elif prediction_pipeline == 'alphafold':
                    t_0 = time.time()
                    model_random_seed = model_index + random_seed * num_models
                    processed_feature_dict = model_runner.process_features(
                        feature_dict, random_seed=model_random_seed)
                    timings[f'process_features_{model_name}'] = time.time() - t_0

                    t_0 = time.time()
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
                if prediction_pipeline == 'alphafold':
                    ranking_confidences[model_name] = prediction_result['ranking_confidence']

                # Save the model outputs.
                
                with open(result_output_path, 'wb') as f:
                    pickle.dump(prediction_result, f, protocol=4)

                # Add the predicted LDDT in the b-factor column.
                # Note that higher predicted LDDT value means higher model confidence.
                plddt_b_factors = np.repeat(
                    plddt[:, None], residue_constants.atom_type_num, axis=-1)
                
                if prediction_pipeline == 'fastfold':
                    unrelaxed_protein = ff_protein.from_prediction(features=batch,
                                                        result=prediction_result,
                                                        b_factors=plddt_b_factors)
                    unrelaxed_pdbs[model_name] = ff_protein.to_pdb(unrelaxed_protein)
                elif prediction_pipeline == 'alphafold':
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
                with open(result_output_path, 'rb') as f:
                    prediction_result = pickle.load(f)
                logging.info(f"Skipping prediction because {unrelaxed_pdb_path} already exists.")
                with open(unrelaxed_pdb_path, 'r') as f:
                    if prediction_pipeline == 'fastfold':
                        unrelaxed_protein = ff_protein.from_pdb_string(f.read())
                    elif prediction_pipeline == 'alphafold':
                        unrelaxed_protein = protein.from_pdb_string(f.read())
                    elif prediction_pipeline == 'rosettafold':
                        unrelaxed_protein = protein.from_pdb_string(f.read())

            if amber_relaxer:
                relaxed_output_path = os.path.join(
                    results_dir, f'relaxed_{model_name}.pdb')
                #Run in a new process to prevent CUDA initialitation error of openmm  
                relax_results_pkl = os.path.join(results_dir, 'relax_results.pkl')
                try:
                    cmd = f"run_relaxation.py --unrelaxed_pdb_path {unrelaxed_pdb_path} --relaxed_output_path {relaxed_output_path} --model_name {model_name} --relax_results_pkl {relax_results_pkl}"
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
                
                if os.path.exists(relax_results_pkl):
                    with open(relax_results_pkl, 'rb') as f:
                        relax_results = pickle.load(f)
                        relaxed_pdbs = relax_results['relaxed_pdbs']
                        relax_metrics = relax_results['relax_metrics']
                #Skip if relaxed model already exists


        # Rank by model confidence and write out relaxed PDBs in rank order.
        if not prediction_pipeline == 'fastfold':
            ranked_order = []
            for idx, (model_name, _) in enumerate(
                sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)):
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
                                           prediction_pipeline=prediction_pipeline)
        evaluation.run_pipeline()

        if batch_prediction:
            logging.info("Task finished")
            evaluation.get_scores(scores)
            #protein_names, model_name, value = evaluation.get_min_inter_pae(evaluation.get_pae_results_unsorted(), fasta_name))
            #max_iptm_list.append(evaluation.get_max_iptm())
            #max_ptm_list.append(evaluation.get_max_ptm())
            #return min_inter_pae_list, max_iptm_list, max_ptm_list
        else:
            logging.info("Alphafold pipeline completed. Exit code 0")

    else:
        logging.info("Alphafold pipeline completed with feature generation. Exit code 0")


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
            dict_ = {key: value for (key, value) in zip(keys, values)}
            scores.append(dict_)
        logging.info("Found the following scores from previous jobs:")
        logging.info(scores)
    return scores

def find_index(list_of_dicts, key):
    for index, dictionary in enumerate(list_of_dicts):
        if key in dictionary:
            return index
    return -1

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)
    flag_specific_imports()
    logging.info("Alphafold pipeline starting...")
    if FLAGS.precomputed_msas_list is None:
        FLAGS.precomputed_msas_list = [FLAGS.precomputed_msas_list]
    if not FLAGS.precomputed_msas_path in ['None', None] or any([not item in ('None', None) for item in FLAGS.precomputed_msas_list]):
        FLAGS.use_precomputed_msas = True

    #Do not check for MSA tools when MSA already exists.
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
        multimer=False)

    #Calculates all MSAs needed for monomer or multimer pipeline
    if FLAGS.pipeline == 'batch_msas':
      logging.debug("Adjusting template searcher and featurizer for batch_msas pipeline.")
      monomer_data_pipeline.template_searcher_hmm = template_searcher_hmm
      monomer_data_pipeline.template_featurizer_hmm = template_featurizer_hmm
      #Calculates uniprot hits
      monomer_data_pipeline.multimer = True
      data_pipeline = pipeline_batch.DataPipeline(monomer_data_pipeline=monomer_data_pipeline, batch_mmseqs=FLAGS.db_preset=='colabfold_local')
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
            uniprot_database_path=FLAGS.uniprot_database_path)
    else:
        logging.debug("Using monomer pipeline.")
        num_predictions_per_model = 1
        data_pipeline = monomer_data_pipeline

    if prediction_pipeline == 'rosettafold':
        model_runners = {'model_0': ""}
    else:
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

    random_seed = FLAGS.random_seed
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize // len(model_runners))
    logging.info('Using random seed %d for the data pipeline', random_seed)



    #Code adaptions to handle custom template, no MSA, no template
    #Check that no_msa_list has same number of elements as in fasta_sequence,
    #and convert to bool.
    description_sequence_dict = parse_fasta(FLAGS.fasta_path)
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
            logging.info("Precomputed MSAs map")
            logging.info(pcmsa_map)
            if len(pcmsa_map) == 1:
                precomputed_msas_list = list(pcmsa_map.values())[0]
            elif len(pcmsa_map) > 1:
                logging.warning("Found more than one precomputed MSA for given sequence. Will use the first one in the list.")
                precomputed_msas_list = list(pcmsa_map.values())[0]
    elif FLAGS.pipeline == 'batch_msas' and FLAGS.precomputed_msas_path:
        logging.warning("Precomputed MSAs will not be copied when running batch features.")

    results_file_pairwise_predictions = f"pairwise_prediction_results_{FLAGS.prediction}.csv"
    if FLAGS.pipeline == 'all_vs_all':
        combinations = []
        manager = multiprocessing.Manager()
        #min_inter_pae_list, max_iptm_list, max_ptm_list = manager.list(), manager.list(), manager.list()
        scores = manager.list()
        
        scores = get_prediction_results(scores, results_file_pairwise_predictions)
        for i, (desc_1, seq_1) in enumerate(description_sequence_dict.items()):
            for o, (desc_2, seq_2) in enumerate(description_sequence_dict.items()):
                logging.debug(f"protein names {desc_1} {desc_2}")
                #Check if sequence names are identical
                if desc_1 == desc_2:
                    prev_desc_1 = desc_1
                    prev_desc_2 = desc_2
                    desc_1 = f"{desc_1}_1"
                    desc_2 = f"{desc_2}_2"
                else:
                    prev_desc_1, prev_desc_2 = None, None
                #Skip if the complementary pair already exists
                if (desc_2, desc_1) in combinations:
                    if prev_desc_1:
                        desc_1 = prev_desc_1
                    if prev_desc_2:
                        desc_2 = prev_desc_2
                    continue
                combinations.append((desc_1, desc_2))
                protein_names = f"{desc_1}_{desc_2}"
                if protein_names in [x['protein_names'] if 'protein_names' in x else None for x in scores]:
                    score_index = find_index(scores, protein_names)
                    if not None in [item for item in scores[score_index].values()]:
                        logging.info(f"Prediction pair {protein_names} already exists. Task finished. Skipping.")
                        if prev_desc_1:
                            desc_1 = prev_desc_1
                        if prev_desc_2:
                            desc_2 = prev_desc_2
                        continue
                    else:
                        logging.info(f"Prediction pair {protein_names} already exists but not all evaluation scores found.")
                else:
                    logging.info(f"Prediction pair {protein_names} not found in score list. Running prediction.")
                    if prediction_pipeline == 'rosettafold':
                        score_dict = {'protein_names': protein_names,
                                        'min_pae_value': None, 'min_pae_model_name': None}
                    else:
                        score_dict = {'protein_names': protein_names,
                                            'min_pae_value': None, 'min_pae_model_name': None,
                                        'max_ptm_value': None, 'max_ptm_model_name': None,
                                        'max_iptm_value': None, 'max_iptm_model_name': None}
                    scores.append(score_dict)
                #Check if reversed combination already exists   
                protein_names = f"{desc_1}_{desc_2}"
                fasta_path = os.path.join(FLAGS.output_dir, f"{desc_1}_{desc_2}.fasta")
                with open(fasta_path, 'w') as f:
                    f.write(f">{desc_1}\n")
                    f.write(seq_1)
                    f.write(f"\n\n>{desc_2}\n")
                    f.write(seq_2)

                kwargs = {
                        "fasta_path": fasta_path,
                        "protein_names": protein_names,
                        "output_dir_base": FLAGS.output_dir,
                        "data_pipeline": data_pipeline,
                        "model_runners": model_runners,
                        "amber_relaxer": amber_relaxer,
                        "benchmark": FLAGS.benchmark,
                        "random_seed": random_seed,
                        "is_multimer": run_multimer_system,
                        "no_msa_list": no_msa_list,
                        "no_template_list": [no_template_list[i], no_template_list[o]],
                        "custom_template_list": [custom_template_list[i], custom_template_list[o]],
                        "precomputed_msas_list": [precomputed_msas_list[i], precomputed_msas_list[o]],
                        "batch_prediction": True,
                        "scores": scores,
                        "prediction_pipeline": prediction_pipeline,
                        "feature_pipeline": feature_pipeline}
                
                if prediction_pipeline == 'fastfold':
                    logging.info("Starting FastFold in a new process.")
                    process = multiprocessing.Process(target=predict_structure, kwargs=kwargs)
                    process.start()
                    process.join()
                else:
                    predict_structure(**kwargs)

                #reset
                if prev_desc_1:
                    desc_1 = prev_desc_1
                if prev_desc_2:
                    desc_2 = prev_desc_2
                logging.debug(f"protein names after reset: {desc_1} {desc_2}")
                if not None in [item for item in scores[-1].values()]:
                    logging.debug("Scores dict")
                    logging.debug(scores)
                    write_prediction_results(scores, results_file_pairwise_predictions)
                else:
                    logging.error("Found None in scores list.")
                    logging.error(scores)
                    sys.exit()

        if FLAGS.run_relax:
            batch_minimization(FLAGS.output_dir, FLAGS.use_gpu_relax)
        evaluation_batch = EvaluationPipelineBatch(FLAGS.output_dir, scores)
        evaluation_batch.run()
        logging.info("Alphafold pipeline completed. Exit code 0")

    elif FLAGS.pipeline == 'first_vs_all':
        combinations = []
        manager = multiprocessing.Manager()
        scores = manager.list()
        #min_inter_pae_list, max_iptm_list, max_ptm_list = manager.list(), manager.list(), manager.list()
        desc_1 = list(description_sequence_dict.keys())[0]
        seq_1 = description_sequence_dict[desc_1]
        scores = get_prediction_results(scores, results_file_pairwise_predictions)
        for i, (desc_2, seq_2) in enumerate(description_sequence_dict.items()):
            if desc_1 == desc_2:
                prev_desc_1 = desc_1
                desc_1 = f"{desc_1}_1"
                desc_2 = f"{desc_2}_2"
            else:
                prev_desc_1 = None

            protein_names = f"{desc_1}_{desc_2}"
            if protein_names in [x['protein_names'] if 'protein_names' in x else None for x in scores]:
                score_index = find_index(scores, protein_names)
                if not None in [item for item in scores[score_index].values()]:
                    logging.info(f"Prediction pair {protein_names} already exists. Task finished. Skipping.")
                    if prev_desc_1:
                        desc_1 = prev_desc_1
                    continue
                else:
                    logging.info(f"Prediction pair {protein_names} already exists but not all evaluation scores found.")
            else:
                logging.info(f"Prediction pair {protein_names} not found in score list. Running prediction.")
                if prediction_pipeline == 'rosettafold':
                    score_dict = {'protein_names': protein_names,
                                    'min_pae_value': None, 'min_pae_model_name': None}
                else:
                    score_dict = {'protein_names': protein_names,
                                        'min_pae_value': None, 'min_pae_model_name': None,
                                    'max_ptm_value': None, 'max_ptm_model_name': None,
                                    'max_iptm_value': None, 'max_iptm_model_name': None}
            scores.append(score_dict)
            protein_names = f"{desc_1}_{desc_2}"
            fasta_path = os.path.join(FLAGS.output_dir, f"{desc_1}_{desc_2}.fasta")
            with open(fasta_path, 'w') as f:
                f.write(f">{desc_1}\n")
                f.write(seq_1)
                f.write(f"\n\n>{desc_2}\n")
                f.write(seq_2)
 

            logging.info(f"Number items ins scores: {len(scores)}")

            kwargs = {
                "fasta_path": fasta_path,
                "protein_names": protein_names,
                "output_dir_base": FLAGS.output_dir,
                "data_pipeline": data_pipeline,
                "model_runners": model_runners,
                "amber_relaxer": amber_relaxer,
                "benchmark": FLAGS.benchmark,
                "random_seed": random_seed,
                "is_multimer": run_multimer_system,
                "no_msa_list": no_msa_list,
                "no_template_list": [no_template_list[0], no_template_list[i]],
                "custom_template_list": [custom_template_list[0], custom_template_list[i]],
                "precomputed_msas_list": [precomputed_msas_list[0], precomputed_msas_list[i]],
                "batch_prediction": True,
                "scores": scores,
                "prediction_pipeline": prediction_pipeline,
                "feature_pipeline": feature_pipeline}

            if prediction_pipeline == 'fastfold':
                process = multiprocessing.Process(target=predict_structure, kwargs=kwargs)
                process.start()
                process.join()
            else:
                predict_structure(**kwargs)

            if prev_desc_1:
                desc_1 = prev_desc_1

            if not None in [item for item in scores[-1].values()]:
                write_prediction_results(scores, results_file_pairwise_predictions)
            else:
                logging.debug("Found None in scores list.")
                logging.debug(scores)
                sys.exit()

        if FLAGS.run_relax:
            batch_minimization(FLAGS.output_dir, FLAGS.use_gpu_relax)
        evaluation_batch = EvaluationPipelineBatch(FLAGS.output_dir, scores)
        evaluation_batch.run()
        logging.info("Alphafold pipeline completed. Exit code 0")

    elif FLAGS.pipeline == 'first_n_vs_rest':
        first_n_seq = FLAGS.first_n_seq
        if first_n_seq is None:
            logging.error("FLAG first_n needs to be defined when using the first_n_vs_rest protocol")
            raise SystemExit
        combinations = []
        manager = multiprocessing.Manager()
        scores = manager.list()
        #min_inter_pae_list, max_iptm_list, max_ptm_list = manager.list(), manager.list(), manager.list()
        first_n_desc_list = list(description_sequence_dict.keys())[:first_n_seq]
        first_n_seq_list = list(description_sequence_dict.values())[:first_n_seq]
        desc_1 = '_'.join(first_n_desc_list)
        scores = get_prediction_results(scores, results_file_pairwise_predictions)
        for i, (desc_2, seq_2) in enumerate(description_sequence_dict.items()):
            if i > first_n_seq:
                protein_names = f"{desc_1}_{desc_2}"
                if protein_names in [x['protein_names'] if 'protein_names' in x else None for x in scores]:
                    score_index = find_index(scores, protein_names)
                    if not None in [item for item in scores[score_index].values()]:
                        logging.info(f"Prediction pair {protein_names} already exists. Task finished. Skipping.")
                        continue
                    else:
                        logging.info(f"Prediction pair {protein_names} already exists but not all evaluation scores found.")
                else:
                    logging.info(f"Prediction pair {protein_names} not found in score list. Running prediction.")
                    if prediction_pipeline == 'rosettafold':
                        score_dict = {'protein_names': protein_names,
                                        'min_pae_value': None, 'min_pae_model_name': None}
                    else:
                        score_dict = {'protein_names': protein_names,
                                            'min_pae_value': None, 'min_pae_model_name': None,
                                        'max_ptm_value': None, 'max_ptm_model_name': None,
                                        'max_iptm_value': None, 'max_iptm_model_name': None}
                    scores.append(score_dict)
                protein_names = f"{desc_1}_{desc_2}"
                fasta_path = os.path.join(FLAGS.output_dir, f"{desc_1}_{desc_2}.fasta")
                with open(fasta_path, 'w') as f:
                    for desc_1 in first_n_desc_list:
                        f.write(f">{desc_1}\n")
                        seq_1 = description_sequence_dict[desc_1]
                        f.write(f"{seq_1}\n\n")
                    f.write(f"{desc_2}\n")
                    f.write({seq_2})
    

                logging.info(f"Number items ins scores: {len(scores)}")

                kwargs = {
                    "fasta_path": fasta_path,
                    "protein_names": protein_names,
                    "output_dir_base": FLAGS.output_dir,
                    "data_pipeline": data_pipeline,
                    "model_runners": model_runners,
                    "amber_relaxer": amber_relaxer,
                    "benchmark": FLAGS.benchmark,
                    "random_seed": random_seed,
                    "is_multimer": run_multimer_system,
                    "no_msa_list": no_msa_list,
                    "no_template_list": [no_template_list[0], no_template_list[i]],
                    "custom_template_list": [custom_template_list[0], custom_template_list[i]],
                    "precomputed_msas_list": [precomputed_msas_list[0], precomputed_msas_list[i]],
                    "batch_prediction": True,
                    "scores": scores,
                    "prediction_pipeline": prediction_pipeline,
                    "feature_pipeline": feature_pipeline}

                if prediction_pipeline == 'fastfold':
                    process = multiprocessing.Process(target=predict_structure, kwargs=kwargs)
                    process.start()
                    process.join()
                else:
                    predict_structure(**kwargs)

                if not None in [item for item in scores[-1].values()]:
                    write_prediction_results(scores, results_file_pairwise_predictions)
                else:
                    logging.debug("Found None in scores list.")
                    logging.debug(scores)
                    sys.exit()
        if FLAGS.run_relax:
            batch_minimization(FLAGS.output_dir, FLAGS.use_gpu_relax)
        evaluation_batch = EvaluationPipelineBatch(FLAGS.output_dir, scores)
        evaluation_batch.run()
        logging.info("Alphafold pipeline completed. Exit code 0")        


    elif FLAGS.pipeline == 'only_relax':
        batch_minimization(FLAGS.output_dir, FLAGS.use_gpu_relax)
        logging.info("Alphafold pipeline completed. Exit code 0")

    else:
        protein_names = '_'.join(list(description_sequence_dict.keys()))
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
            batch_mmseqs=FLAGS.db_preset=='colabfold_local')

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
      logging.error("Alphafold pipeline was aborted. Exit code 2")
  except Exception:
      logging.info(traceback.print_exc())
      logging.error("Alphafold pipeline finished with an error. Exit code 1")
