#!/usr/bin/env python

import configparser
from contextlib import closing
from multiprocessing import Pool
from subprocess import Popen
import os
from shutil import rmtree, copyfile
import argparse
import pkg_resources


config_file = pkg_resources.resource_filename('guifold.config', 'guifold.conf')
config = configparser.ConfigParser()
config.read(config_file)
test_data_dir = config['TESTING']['TEST_DATA_DIR']


monomer_fasta = os.path.join(test_data_dir, "monomer.fasta")
multimer_fasta = os.path.join(test_data_dir, "multimer.fasta")
no_msa_list_multimer = "False,True"
no_template_list_multimer = "False,True"
precomputed_msas_list_multimer = f"None,{os.path.join(test_data_dir, 'calculated/SRP14')}"
custom_template_list_multimer = f"None,{os.path.join(test_data_dir, 'calculated/2W9J')}"


test_jobs = {'simple_monomer': {'MODEL_PRESET': 'monomer_ptm',
                                                            'FASTA_PATH': monomer_fasta,
                                                            'PRECOMPUTED_MSAS_LIST': "None",
                                                            'NO_MSA_LIST': "False",
                                                            "NO_TEMPLATE_LIST": "False",
                                                            "CUSTOM_TEMPLATE_LIST": "None",
                                                            "DB_PRESET": "full_dbs",
                                                            "PRECOMPUTED_MSAS_PATH": "None",
                                                            "PIPELINE": "full",
                                                            "PREDICTION": "alphafold"},
                                        'simple_monomer_fastfold': {'MODEL_PRESET': 'monomer_ptm',
                                                            'FASTA_PATH': monomer_fasta,
                                                            'PRECOMPUTED_MSAS_LIST': "None",
                                                            'NO_MSA_LIST': "False",
                                                            "NO_TEMPLATE_LIST": "False",
                                                            "CUSTOM_TEMPLATE_LIST": "None",
                                                            "DB_PRESET": "full_dbs",
                                                            "PRECOMPUTED_MSAS_PATH": "None",
                                                            "PIPELINE": "full",
                                                            "PREDICTION": "fastfold"},
                                        'simple_monomer_rosettafold': {'MODEL_PRESET': 'monomer_ptm',
                                                            'FASTA_PATH': monomer_fasta,
                                                            'PRECOMPUTED_MSAS_LIST': "None",
                                                            'NO_MSA_LIST': "False",
                                                            "NO_TEMPLATE_LIST": "False",
                                                            "CUSTOM_TEMPLATE_LIST": "None",
                                                            "DB_PRESET": "full_dbs",
                                                            "PRECOMPUTED_MSAS_PATH": "None",
                                                            "PIPELINE": "full",
                                                            "PREDICTION": "rosettafold"},
                                        'simple_monomer_small_bfd': {'MODEL_PRESET': 'monomer_ptm',
                                                            'FASTA_PATH': monomer_fasta,
                                                            'PRECOMPUTED_MSAS_LIST': "None",
                                                            'NO_MSA_LIST': "False",
                                                            "NO_TEMPLATE_LIST": "False",
                                                            "CUSTOM_TEMPLATE_LIST": "None",
                                                            "DB_PRESET": "reduced_dbs",
                                                            "PRECOMPUTED_MSAS_PATH": "None",
                                                            "PIPELINE": "full",
                                                            "PREDICTION": "alphafold"},
                                        'simple_multimer': {'MODEL_PRESET': 'multimer',
                                                            'FASTA_PATH': multimer_fasta,
                                                            'PRECOMPUTED_MSAS_LIST': "None,None",
                                                            'NO_MSA_LIST': "False,False",
                                                            "NO_TEMPLATE_LIST": "False,False",
                                                            "CUSTOM_TEMPLATE_LIST": "None,None",
                                                            "DB_PRESET": "full_dbs",
                                                            "PRECOMPUTED_MSAS_PATH": "None,None",
                                                            "PIPELINE": "full",
                                                            "PREDICTION": "alphafold"},
                                        'simple_multimer_fastfold': {'MODEL_PRESET': 'multimer',
                                                            'FASTA_PATH': multimer_fasta,
                                                            'PRECOMPUTED_MSAS_LIST': "None,None",
                                                            'NO_MSA_LIST': "False,False",
                                                            "NO_TEMPLATE_LIST": "False,False",
                                                            "CUSTOM_TEMPLATE_LIST": "None,None",
                                                            "DB_PRESET": "full_dbs",
                                                            "PRECOMPUTED_MSAS_PATH": "None,None",
                                                            "PIPELINE": "full",
                                                            "PREDICTION": "fastfold"},
                                        'simple_multimer_rosettafold': {'MODEL_PRESET': 'multimer',
                                                            'FASTA_PATH': multimer_fasta,
                                                            'PRECOMPUTED_MSAS_LIST': "None,None",
                                                            'NO_MSA_LIST': "False,False",
                                                            "NO_TEMPLATE_LIST": "False,False",
                                                            "CUSTOM_TEMPLATE_LIST": "None,None",
                                                            "DB_PRESET": "full_dbs",
                                                            "PRECOMPUTED_MSAS_PATH": "None,None",
                                                            "PIPELINE": "full",
                                                            "PREDICTION": "rosettafold"},
                                        'simple_multimer_colabfold*_web': {'MODEL_PRESET': 'multimer',
                                                            'FASTA_PATH': multimer_fasta,
                                                            'PRECOMPUTED_MSAS_LIST': "None,None",
                                                            'NO_MSA_LIST': "False,False",
                                                            "NO_TEMPLATE_LIST": "False,False",
                                                            "CUSTOM_TEMPLATE_LIST": "None,None",
                                                            "DB_PRESET": "colabfold*_web",
                                                            "PRECOMPUTED_MSAS_PATH": "None,None",
                                                            "PIPELINE": "full",
                                                            "PREDICTION": "alphafold"},
                                        'batch_msas_colabfold*_local': {'MODEL_PRESET': 'multimer',
                                                            'FASTA_PATH': multimer_fasta,
                                                            'PRECOMPUTED_MSAS_LIST': "None,None",
                                                            'NO_MSA_LIST': "False,False",
                                                            "NO_TEMPLATE_LIST": "False,False",
                                                            "CUSTOM_TEMPLATE_LIST": "None,None",
                                                            "DB_PRESET": "colabfold*_local",
                                                            "PRECOMPUTED_MSAS_PATH": "None,None",
                                                            "PIPELINE": "batch_msas",
                                                            "PREDICTION": "alphafold"},
                                        'multimer_no_template_no_msa': {'MODEL_PRESET': 'multimer',
                                                            'FASTA_PATH': multimer_fasta,
                                                            'PRECOMPUTED_MSAS_LIST': "None,None",
                                                            'NO_MSA_LIST': "True,False",
                                                            "NO_TEMPLATE_LIST": "True,False",
                                                            "CUSTOM_TEMPLATE_LIST": "None,None",
                                                            "DB_PRESET": "full_dbs",
                                                            "PRECOMPUTED_MSAS_PATH": "None,None",
                                                            "PIPELINE": "full",
                                                            "PREDICTION": "alphafold"},
                                        'multimer_precomputed_msas_custom_template': {'MODEL_PRESET': 'multimer',
                                                            'FASTA_PATH': multimer_fasta,
                                                            'PRECOMPUTED_MSAS_LIST': precomputed_msas_list_multimer,
                                                            'NO_MSA_LIST': "False,False",
                                                            "NO_TEMPLATE_LIST": "False,False",
                                                            "CUSTOM_TEMPLATE_LIST": custom_template_list_multimer,
                                                            "DB_PRESET": "full_dbs",
                                                            "PRECOMPUTED_MSAS_PATH": test_data_dir,
                                                            "PIPELINE": "full",
                                                            "PREDICTION": "alphafold"},
                                        'multimer_precomputed_msas_subsequence': {'MODEL_PRESET': 'multimer',
                                                           'FASTA_PATH': multimer_fasta,
                                                           'PRECOMPUTED_MSAS_LIST': precomputed_msas_list_multimer,
                                                           'NO_MSA_LIST': "False,False",
                                                           "NO_TEMPLATE_LIST": "False,False",
                                                           "CUSTOM_TEMPLATE_LIST": "None,None",
                                                           "DB_PRESET": "full_dbs",
                                                           "PRECOMPUTED_MSAS_PATH": test_data_dir,
                                                           "PIPELINE": "full",
                                                            "PREDICTION": "alphafold"},
                                        'multimer_first_vs_all': {'MODEL_PRESET': 'multimer',
                                                            'FASTA_PATH': multimer_fasta,
                                                            'PRECOMPUTED_MSAS_LIST': "None,None",
                                                            'NO_MSA_LIST': "False,False",
                                                            "NO_TEMPLATE_LIST": "False,False",
                                                            "CUSTOM_TEMPLATE_LIST": "None,None",
                                                            "DB_PRESET": "full_dbs",
                                                            "PRECOMPUTED_MSAS_PATH": "None,None",
                                                            "PIPELINE": "first_vs_all",
                                                            "PREDICTION": "alphafold"},
                                        'multimer_all_vs_all': {'MODEL_PRESET': 'multimer',
                                                            'FASTA_PATH': multimer_fasta,
                                                            'PRECOMPUTED_MSAS_LIST': "None,None",
                                                            'NO_MSA_LIST': "False,False",
                                                            "NO_TEMPLATE_LIST": "False,False",
                                                            "CUSTOM_TEMPLATE_LIST": "None,None",
                                                            "DB_PRESET": "full_dbs",
                                                            "PRECOMPUTED_MSAS_PATH": "None,None",
                                                            "PIPELINE": "first_vs_all",
                                                            "PREDICTION": "alphafold"},
                                        'multimer_first_n_vs_rest': {'MODEL_PRESET': 'multimer',
                                                            'FASTA_PATH': multimer_fasta,
                                                            'PRECOMPUTED_MSAS_LIST': "None,None",
                                                            'NO_MSA_LIST': "False,False",
                                                            "NO_TEMPLATE_LIST": "False,False",
                                                            "CUSTOM_TEMPLATE_LIST": "None,None",
                                                            "DB_PRESET": "full_dbs",
                                                            "PRECOMPUTED_MSAS_PATH": "None,None",
                                                            "FIRST_N": "2",
                                                            "PIPELINE": "first_n_vs_rest",
                                                            "PREDICTION": "alphafold"},
}

parser = argparse.ArgumentParser()
parser.add_argument('--nproc', default=1, type=int, help='Number of parallel processes.')
parser.add_argument('--tests', default=None, help=f'List of tests to run. Available tests are {test_jobs.keys()}.')
args = parser.parse_args()

if not args.tests:
    tests_to_run = list(test_jobs.keys())
else:
    tests_to_run = args.tests.split(',')

config_file = pkg_resources.resource_filename('guifold.config', 'guifold.conf')
config = configparser.ConfigParser()
config.read(config_file)
config_keys = config.keys()

                         

cmd = """export CUDA_VISIBLE_DEVICES=""; run_prediction.py\\
 --model_preset {MODEL_PRESET}\\
 --output_dir {OUTPUT_DIR}\\
 --fasta_path {FASTA_PATH}\\
 --no_msa_list {NO_MSA_LIST}\\
 --no_template_list {NO_TEMPLATE_LIST}\\
 --custom_template_list {CUSTOM_TEMPLATE_LIST}\\
 --precomputed_msas_list {PRECOMPUTED_MSAS_LIST}\\
 --precomputed_msas_path {PRECOMPUTED_MSAS_PATH}\\
 --run_relax \\
 --num_multimer_predictions_per_model 1\\
 --db_preset {DB_PRESET}\\
 --max_template_date 2022-12-06\\
 --num_recycle 1\\
 --pipeline {PIPELINE}\\
 --num_cpu 8\\
 --jackhmmer_binary_path {JACKHMMER_BINARY_PATH}\\
 --hhblits_binary_path {HHBLITS_BINARY_PATH}\\
 --mmseqs_binary_path {MMSEQS_BINARY_PATH}\\
 --hhsearch_binary_path {HHSEARCH_BINARY_PATH}\\
 --hmmsearch_binary_path {HMMSEARCH_BINARY_PATH}\\
 --hmmbuild_binary_path {HMMBUILD_BINARY_PATH}\\
 --hhalign_binary_path {HHALIGN_BINARY_PATH}\\
 --kalign_binary_path {KALIGN_BINARY_PATH}\\
 --data_dir {DATA_DIR}\\
 --uniref90_database_path {UNIREF90_DATABASE_PATH}\\
 --small_bfd_database_path {SMALL_BFD_DATABASE_PATH}\\
 --bfd_database_path {BFD_DATABASE_PATH}\\
 --uniref30_database_path {UNIREF30_DATABASE_PATH}\\
 --uniref30_mmseqs_database_path {UNIREF30_MMSEQS_DATABASE_PATH}\\
 --uniref90_mmseqs_database_path {UNIREF90_MMSEQS_DATABASE_PATH}\\
 --uniprot_mmseqs_database_path {UNIPROT_MMSEQS_DATABASE_PATH}\\
 --uniprot_database_path {UNIPROT_DATABASE_PATH}\\
 --pdb_seqres_database_path {PDB_SEQRES_DATABASE_PATH}\\
 --colabfold_envdb_database_path {COLABFOLD_ENVDB_DATABASE_PATH}\\
 --mgnify_database_path {MGNIFY_DATABASE_PATH}\\
 --pdb70_database_path {PDB70_DATABASE_PATH}\\
 --template_mmcif_dir {TEMPLATE_MMCIF_DIR}\\
 --obsolete_pdbs_path {OBSOLETE_PDBS_PATH}\\
 --custom_tempdir {CUSTOM_TEMPDIR}\\
 --first_n_seq {FIRST_N}\\
 --model_list 1\\
 --debug &> {OUTPUT_DIR}/testing.log
 """

cmd_list = []
for title, params in test_jobs.items():
    if title in tests_to_run:
        print(f"Running test {title}")
        output_dir = os.path.join(test_data_dir, title)
        print(f"Output dir: {output_dir}")
        if os.path.exists(output_dir):
            rmtree(output_dir)
        os.mkdir(output_dir)
        copyfile(params['FASTA_PATH'], os.path.join(output_dir, os.path.basename(params['FASTA_PATH'])))
        fasta_path = os.path.join(output_dir, os.path.basename(params['FASTA_PATH']))
        if not 'FIRST_N' in params:
            params['FIRST_N'] = 2
        formatted_cmd = cmd.format(MODEL_PRESET=params['MODEL_PRESET'],
                OUTPUT_DIR=output_dir,
                FASTA_PATH=fasta_path,
                PRECOMPUTED_MSAS_LIST=params['PRECOMPUTED_MSAS_LIST'],
                NO_MSA_LIST=params['NO_MSA_LIST'],
                NO_TEMPLATE_LIST=params['NO_TEMPLATE_LIST'],
                CUSTOM_TEMPLATE_LIST=params['CUSTOM_TEMPLATE_LIST'],
                DB_PRESET=params['DB_PRESET'],
                PRECOMPUTED_MSAS_PATH=params['PRECOMPUTED_MSAS_PATH'],
                PIPELINE=params['PIPELINE'],
                PREDICTION=params['PREDICTION'],
                FIRST_N=params['FIRST_N'],
                JACKHMMER_BINARY_PATH=config['TESTING']['jackhmmer_binary_path'],
                HHBLITS_BINARY_PATH=config['TESTING']['hhblits_binary_path'],
                MMSEQS_BINARY_PATH=config['TESTING']['mmseqs_binary_path'],
                HHSEARCH_BINARY_PATH=config['BINARIES']['hhsearch_binary_path'],
                HMMSEARCH_BINARY_PATH=config['BINARIES']['hmmsearch_binary_path'],
                HMMBUILD_BINARY_PATH=config['BINARIES']['hmmbuild_binary_path'],
                HHALIGN_BINARY_PATH=config['BINARIES']['hhalign_binary_path'],
                KALIGN_BINARY_PATH=config['BINARIES']['kalign_binary_path'],
                DATA_DIR=config['DATABASES']['data_dir'],
                UNIREF90_DATABASE_PATH=config['DATABASES']['uniref90_database_path'],
                SMALL_BFD_DATABASE_PATH=config['DATABASES']['small_bfd_database_path'],
                BFD_DATABASE_PATH=config['DATABASES']['bfd_database_path'],
                UNIREF30_DATABASE_PATH=config['DATABASES']['uniref30_database_path'],
                UNIPROT_DATABASE_PATH=config['DATABASES']['uniprot_database_path'],
                PDB_SEQRES_DATABASE_PATH=config['DATABASES']['pdb_seqres_database_path'],
                UNIREF30_MMSEQS_DATABASE_PATH=config['DATABASES']['uniref30_mmseqs_database_path'],
                COLABFOLD_ENVDB_DATABASE_PATH=config['DATABASES']['colabfold_envdb_database_path'],
                UNIREF90_MMSEQS_DATABASE_PATH=config['DATABASES']['uniref90_mmseqs_database_path'],
                UNIPROT_MMSEQS_DATABASE_PATH=config['DATABASES']['uniprot_mmseqs_database_path'],
                MGNIFY_DATABASE_PATH=config['DATABASES']['mgnify_database_path'],
                PDB70_DATABASE_PATH=config['DATABASES']['pdb70_database_path'],
                TEMPLATE_MMCIF_DIR=config['DATABASES']['template_mmcif_dir'],
                OBSOLETE_PDBS_PATH=config['DATABASES']['obsolete_pdbs_path'],
                CUSTOM_TEMPDIR=config['OTHER']['custom_tempdir'])
        
        cmd_list.append((formatted_cmd, title))

for cmd, title in cmd_list:
    print(title)
    print(cmd)

def run_cmd(cmd, title):
    print("Formatted cmd is:")
    print(cmd)
    p = Popen(cmd, shell=True)
    p.communicate()

print(f"Running with {args.nproc} processes.")
with closing(Pool(args.nproc)) as pool:
    
    p = pool.starmap(run_cmd, cmd_list)

