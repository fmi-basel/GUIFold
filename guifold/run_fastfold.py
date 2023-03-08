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
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
import traceback
from typing import Dict, Union, Optional

from absl import app
from absl import flags
from absl import logging
from guifold.afeval import EvaluationPipeline
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import pipeline_batch
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config as af_config
from alphafold.model import model
from alphafold.relax import relax
from alphafold.data import parsers
import numpy as np
import inspect
import gzip

from alphafold.model import data

import torch
import torch.multiprocessing as mp
from fastfold.model.hub import AlphaFold
import fastfold
from fastfold.config import model_config as ff_model_config
from fastfold.common import protein as ff_protein
from fastfold.data import feature_pipeline
from fastfold.model.fastnn import set_chunk_size
from fastfold.model.nn.triangular_multiplicative_update import set_fused_triangle_multiplication
from fastfold.utils.inject_fastnn import inject_fastnn
from fastfold.utils.import_weights import import_jax_weights_
from fastfold.utils.tensor_utils import tensor_tree_map
# Internal import (7716).

if int(torch.__version__.split(".")[0]) >= 1 and int(torch.__version__.split(".")[1]) > 11:
    torch.backends.cuda.matmul.allow_tf32 = True

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

logging.set_verbosity(logging.INFO)

flags.DEFINE_string('fasta_path', None, 'Path to a single fasta file.')
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
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
                  ['full_dbs', 'reduced_dbs', 'colabfold'],
                  'Choose preset MSA database configuration - '
                  'smaller genetic database config (reduced_dbs) or '
                  'full genetic database config  (full_dbs) or '
                  'colabfold database config (uniref30, colabfold_envdb) in combination with jackhmmer+mmseqs')
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
                'full', 'only_features', 'batch_msas', 'continue_from_features'],
                'Choose preset pipeline configuration - '
                'full pipeline or '
                'stop after feature generation (only features) or '
                'calculate MSAs and find templates for given batch of sequences, ignore monomer/multimer preset (batch features) or'
                'continue from features.pkl file (continue_from_features)')
flags.DEFINE_integer('num_gpu', 1, 'Number of GPUs.')
flags.DEFINE_integer('chunk_size', None, 'Chunk size.')
flags.DEFINE_boolean('inplace', False, 'Inplace.')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
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
        config.globals.chunk_size = chunk_size
    config.globals.inplace = inplace
    config.globals.is_multimer = model_preset == 'multimer'
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
        print(f"Inference time: {time.perf_counter() - t}")

    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

    result_q.put(out)

    torch.distributed.barrier()
    torch.cuda.synchronize()

def predict_structure(
    fasta_path: str,
    fasta_name: str,
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
    precomputed_msas_list: Optional[str] = None):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  features_output_path = os.path.join(output_dir, 'features.pkl')
  # Get features.
  if not FLAGS.pipeline == 'continue_from_features':
      t_0 = time.time()
      feature_dict = data_pipeline.process(
            input_fasta_path=fasta_path,
            msa_output_dir=msa_output_dir,
            no_msa=no_msa_list,
            no_template=no_template_list,
            custom_template=custom_template_list,
            precomputed_msas=precomputed_msas_list,
            num_cpu=FLAGS.num_cpu)
      timings['features'] = time.time() - t_0


      # Write out features as a pickled dictionary.
      if not FLAGS.pipeline == 'batch_features':
          with open(features_output_path, 'wb') as f:
            pickle.dump(feature_dict, f, protocol=4)

  #Stop here if only_msa flag is set
  if not FLAGS.pipeline == 'only_features' and not FLAGS.pipeline == 'batch_msas':
      if FLAGS.pipeline == 'continue_from_features':
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
        logging.info('Running model %s on %s', model_name, fasta_name)
        #Fix some differences between openfold and alphafold feature_dict
        feature_dict = {}
        for k, v in feature_dict_initial.items():
            if k == 'template_all_atom_masks':
                feature_dict['template_all_atom_mask'] = v
            elif k == 'seq_length' and is_multimer:
                feature_dict['seq_length'] = [v*v]
            else:
                feature_dict[k] = v
        t_0 = time.time()
        #model_random_seed = model_index + random_seed * num_models
        #processed_feature_dict = model_runner.process_features(
        #      feature_dict, random_seed=model_random_seed)

        import re
        model_name = re.match('(model_\d{1}(_ptm|_multimer){0,1}).*', model_name).group(1)
        model_num = re.match('model_(\d{1}).*', model_name).group(1)
        ff_config = ff_model_config(model_name)
        feature_processor = feature_pipeline.FeaturePipeline(ff_config.data)

        print(feature_dict.keys())

        print(is_multimer)
        processed_feature_dict = feature_processor.process_features(
            feature_dict, mode='predict', is_multimer=is_multimer,
        )
        timings[f'process_features_{model_name}'] = time.time() - t_0


        t_0 = time.time()
        batch = processed_feature_dict

        manager = mp.Manager()
        result_q = manager.Queue()
        chunk_size = FLAGS.chunk_size
        inplace = FLAGS.inplace
        num_gpu = FLAGS.num_gpu


        if is_multimer:
          params_file = f"params_{model_name}_v3.npz"
        else:
          params_file = f"params_{model_name}.npz"
        params_path = os.path.join(FLAGS.data_dir, "params", params_file)
        model_preset = FLAGS.model_preset
        torch.multiprocessing.spawn(inference_model, nprocs=num_gpu, args=(num_gpu, result_q, batch, model_name, chunk_size, inplace, model_preset, params_path))

        prediction_result = result_q.get()

        batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)

        t_diff = time.time() - t_0
        timings[f'predict_and_compile_{model_name}'] = t_diff
        logging.info(
            'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
            model_name, fasta_name, t_diff)

        if benchmark:
          pass

        plddt = prediction_result['plddt']
        #ranking_confidences[model_name] = prediction_result['ranking_confidence']

        # Save the model outputs.
        result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
        with open(result_output_path, 'wb') as f:
          pickle.dump(prediction_result, f, protocol=4)

        # Add the predicted LDDT in the b-factor column.
        # Note that higher predicted LDDT value means higher model confidence.
        plddt_b_factors = np.repeat(
            plddt[:, None], residue_constants.atom_type_num, axis=-1)
        unrelaxed_protein = ff_protein.from_prediction(features=batch,
                                                    result=prediction_result,
                                                    b_factors=plddt_b_factors)

        unrelaxed_pdbs[model_name] = ff_protein.to_pdb(unrelaxed_protein)
        unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
        with open(unrelaxed_pdb_path, 'w') as f:
          f.write(unrelaxed_pdbs[model_name])

        if amber_relaxer:
          # Relax the prediction.
          t_0 = time.time()
          relaxed_pdb_str, _, violations = amber_relaxer.process(
              prot=unrelaxed_protein)
          relax_metrics[model_name] = {
              'remaining_violations': violations,
              'remaining_violations_count': sum(violations)
          }
          timings[f'relax_{model_name}'] = time.time() - t_0

          relaxed_pdbs[model_name] = relaxed_pdb_str

          # Save the relaxed PDB.
          relaxed_output_path = os.path.join(
              output_dir, f'relaxed_{model_name}.pdb')
          with open(relaxed_output_path, 'w') as f:
            f.write(relaxed_pdb_str)

      # Rank by model confidence and write out relaxed PDBs in rank order.
      # ranked_order = []
      # for idx, (model_name, _) in enumerate(
      #     sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)):
      #   ranked_order.append(model_name)
      #   ranked_output_path = os.path.join(output_dir, f'ranked_by_plddt_{idx}.pdb')
      #   with open(ranked_output_path, 'w') as f:
      #     if amber_relaxer:
      #       f.write(relaxed_pdbs[model_name])
      #     else:
      #       f.write(unrelaxed_pdbs[model_name])
      #
      # ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
      # with open(ranking_output_path, 'w') as f:
      #   label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
      #   f.write(json.dumps(
      #       {label: ranking_confidences, 'order': ranked_order}, indent=4))

      logging.info('Final timings for %s: %s', fasta_name, timings)

      timings_output_path = os.path.join(output_dir, 'timings.json')
      with open(timings_output_path, 'w') as f:
        f.write(json.dumps(timings, indent=4))
      if amber_relaxer:
        relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
        with open(relax_metrics_path, 'w') as f:
          f.write(json.dumps(relax_metrics, indent=4))

      evaluation = EvaluationPipeline(FLAGS.fasta_path)
      evaluation.run_pipeline()
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

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if not FLAGS.precomputed_msas_path in ['None', None] or any([not item in ('None', None) for item in FLAGS.precomputed_msas_list]):
      FLAGS.use_precomputed_msas = True

  #Do not check for MSA tools when MSA already exists.
  run_multimer_system = 'multimer' in FLAGS.model_preset
  use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
  use_mmseqs = FLAGS.db_preset == 'colabfold'
  if FLAGS.precomputed_msas_path and FLAGS.precomputed_msas_list:
      logging.warning("Flags --precomputed_msas_path and --precomputed_msas_list selected at the same time."
                      "MSAs from --precomputed_msas_list get priority over MSAs from --precomputed_msas_path.")
  if not FLAGS.pipeline == 'continue_from_features':
      for tool_name in ('jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
        if not FLAGS[f'{tool_name}_binary_path'].value:
          raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                           'sure it is installed on your system.')
        if FLAGS.db_preset == 'colabfold':
            if not FLAGS.mmseqs_binary_path:
                raise ValueError(f'Could not find path to mmseqs2 binary. Make sure it is installed on your system.')
      _check_flag('small_bfd_database_path', 'db_preset',
                  should_be_set=use_small_bfd)
      _check_flag('bfd_database_path', 'db_preset',
                  should_be_set=not use_small_bfd)
      _check_flag('uniref30_database_path', 'db_preset',
                  should_be_set=not use_small_bfd and not use_mmseqs)
      _check_flag('pdb70_database_path', 'model_preset',
                  should_be_set=not run_multimer_system)
      _check_flag('pdb_seqres_database_path', 'model_preset',
                  should_be_set=run_multimer_system)
      _check_flag('uniprot_database_path', 'model_preset',
                  should_be_set=run_multimer_system)
      _check_flag('colabfold_envdb_database_path', 'db_preset',
                  should_be_set=use_mmseqs)
      _check_flag('uniref30_mmseqs_database_path', 'db_preset',
                  should_be_set=use_mmseqs)

  if FLAGS.model_preset == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1

  #Only one fasta file allowed
  fasta_name = pathlib.Path(FLAGS.fasta_path).stem

  if run_multimer_system:
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        hhalign_binary_path=FLAGS.hhalign_binary_path,
        database_path=FLAGS.pdb_seqres_database_path,
        custom_tempdir=FLAGS.custom_tempdir)
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path,
        custom_tempdir=FLAGS.custom_tempdir,
        strict_error_check=True)
  else:
    template_searcher = hhsearch.HHSearch(
        binary_path=FLAGS.hhsearch_binary_path,
        hhalign_binary_path=FLAGS.hhalign_binary_path,
        databases=[FLAGS.pdb70_database_path],
        custom_tempdir=FLAGS.custom_tempdir)
    template_featurizer = templates.HhsearchHitFeaturizer(
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
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniref30_database_path=FLAGS.uniref30_database_path,
      uniref30_mmseqs_database_path=FLAGS.uniref30_mmseqs_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      colabfold_envdb_database_path=FLAGS.colabfold_envdb_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=use_small_bfd,
      use_precomputed_msas=FLAGS.use_precomputed_msas,
      use_mmseqs=use_mmseqs,
      custom_tempdir=FLAGS.custom_tempdir,
      precomputed_msas_path=FLAGS.precomputed_msas_path)

  if FLAGS.pipeline == 'batch_msas':
      data_pipeline = pipeline_batch.DataPipeline(monomer_data_pipeline)
      num_predictions_per_model = 1
  elif run_multimer_system and not FLAGS.pipeline == 'batch_msas':
    num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        uniprot_database_path=FLAGS.uniprot_database_path)
  else:
    num_predictions_per_model = 1
    data_pipeline = monomer_data_pipeline


  model_runners = {}
  model_names = af_config.MODEL_PRESETS[FLAGS.model_preset]
  for model_name in model_names:
      model_config = af_config.model_config(model_name, FLAGS.num_recycle)
      if run_multimer_system:
          model_config.model.num_ensemble_eval = num_ensemble
      else:
          model_config.data.eval.num_ensemble = num_ensemble
      model_params = data.get_model_haiku_params(
          model_name=model_name, data_dir=FLAGS.data_dir)
      model_runner = model.RunModel(model_config, model_params)
      for i in range(num_predictions_per_model):
          model_runners[f'{model_name}_pred_{i}'] = model_runner

  logging.info('Have %d models: %s', len(model_names),
               list(model_names))

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
    random_seed = random.randrange(sys.maxsize // len(model_names))
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

  if FLAGS.precomputed_msas_list:
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

  if not run_multimer_system and not FLAGS.pipeline == 'batch_msas' and FLAGS.precomputed_msas_path:
      pcmsa_map = pipeline.get_pcmsa_map(FLAGS.precomputed_msas_path,
                                                             description_sequence_dict)
      logging.info("Precomputed MSAs map")
      logging.info(pcmsa_map)
      if len(pcmsa_map) > 0:
          precomputed_msas_list = list(pcmsa_map.values())[0]
      elif len(pcmsa_map) > 1:
          logging.warning("Found more than one precomputed MSA for given sequence. Will use the first one in the list.")
          precomputed_msas_list = list(pcmsa_map.values())[0]
  elif FLAGS.pipeline == 'batch_msas' and FLAGS.precomputed_msas_path:
      logging.warning("Precomputed MSAs will not be copied when running batch features.")


  predict_structure(
        fasta_path=FLAGS.fasta_path,
        fasta_name=fasta_name,
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
        precomputed_msas_list=precomputed_msas_list)

def sigterm_handler(_signo, _stack_frame):
    raise KeyboardInterrupt

if __name__ == '__main__':
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
