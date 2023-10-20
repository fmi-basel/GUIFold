#!/usr/bin/env python
import configparser
import os
import sys
import re
from shutil import copyfile
from time import sleep

import pkg_resources

config_file = pkg_resources.resource_filename('guifold.config', 'guifold.conf')
config = configparser.ConfigParser()
config.read(config_file)
test_data_dir = config['TESTING']['TEST_DATA_DIR'] #testing"

path_sequence_dict = {os.path.join(test_data_dir, "calculated/SRP9"): 'MPQYQTWEEFSRAAEKLYLADPMKARVVLKYRHSDGNLCVKVTDDLVCLVYKTDQAQDVKKIEKFHSQLMRLMVAKEARNVTMETE',
                      os.path.join(test_data_dir, "calculated/SRP14"): 'MVLLESEQFLTELTRLFQKCRTSGSVYITLKKYDGRTKPIPKKGTVEGFEPADNKCLLRATDGKKKISTVVSSKEVNKFQMAYSNLLRANMDGLKKRDKKNKTKKTKAAAAAAAAAPAAAATAPTTAATTAATAAQ'}

def get_msa_path_by_sequence(input_file):
    print(f"Input file {input_file}")
    path_list = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        print(f"get msa {list(line)}")
        if line.startswith(">"):
            continue
        else:
            for path_, sequence in path_sequence_dict.items():
                if line.strip('\n') == sequence:
                    print(f"Line {line} == {sequence}: path {path_}")

                    path_list.append(path_)
    print(path_list)
    if len(path_list) == 1:
        return path_list[0]
    else:
        return path_list

def hhblits(args):
    print(args)
    print("Running dummy hhblits")
    bfd_found, uniclust_found = False, False
    for i, arg in enumerate(args):
        print(arg)
        if arg == '-i':
            input_file = args[i+1]
        elif arg == '-oa3m':
            output_path = args[i+1]
        elif re.search('bfd_metaclust', arg):
            bfd_found = True
        elif re.search('uniref30', arg):
            uniclust_found = True
        if bfd_found and uniclust_found:
            file_to_copy = 'bfd_uniref_hits.a3m'
    source_path = os.path.join(get_msa_path_by_sequence(input_file), file_to_copy)
    print(f"Copying {source_path} to {output_path}")
    copyfile(source_path, output_path)

def mmseqs_get_num(dir):
    print(f"Files in {dir}")
    print(os.listdir(dir))
    nums = [x.replace('.a3m', '').replace('input_sequence_', '') for x in os.listdir(dir) if x.startswith('input_sequence') and x.endswith('.a3m')]
    nums = [int(x) for x in nums]
    print("Num list:")
    print(nums)
    if len(nums) > 0:
        max_num = max(nums) + 1
    else:
        max_num = 0
    return max_num

def mmseqs(args):
    print("Running dummy mmseqs")
    print(args)
    file_to_copy = None
    arg_string = ' '.join(args)
    for i, arg in enumerate(args):
        #Copy file 
        if arg == 'createdb':
            input_path = args[i+1]
            output_dir = args[i+2]
            print(f"Input path {input_path}, output dir {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            max_num = mmseqs_get_num(output_dir)
            output_path = os.path.join(output_dir, f"input_sequence_{max_num}.a3m")
            copyfile(input_path, output_path)
            print(f"Copied {input_path} to {output_path}")
        elif arg == 'search' and not re.search('prof_res', arg_string):
            input_dir = args[i+1]
            max_num = int(mmseqs_get_num(input_dir)) - 1
            input_path = os.path.join(input_dir, f"input_sequence_{max_num}.a3m")
            if re.search('uniref30_22', arg_string):
                file_to_copy = 'uniref30_colabfold_envdb_mmseqs_hits.a3m'
            elif re.search('uniref90', arg_string):
                file_to_copy = 'uniref90_mmseqs_hits.a3m'
            elif re.search('uniprot', arg_string):
                file_to_copy = 'uniprot_mmseqs_hits.a3m'

            if file_to_copy:
                msa_path = get_msa_path_by_sequence(input_path)
                if not isinstance(msa_path, list):
                    msa_path = [msa_path]
                if not isinstance(file_to_copy, list):
                    files_to_copy = [file_to_copy]
                for name in files_to_copy:
                    for i, path in enumerate(msa_path):
                        source_path = os.path.join(path, name)
                        output_path = os.path.join(input_dir.replace('/qdb', ''), f"{i}.a3m")
                        print(f"Copying {source_path} to {output_path}")
                        copyfile(source_path, output_path)
            else:
                print(f"Not copying at this stage.")


def jackhmmer(args):
    print("Running dummy jackhmmer")
    print(args)
    for i, arg in enumerate(args):
        if arg == '-N':
            input_file = args[i+2]
        elif arg == '-A':
            output_path = args[i+1]
        elif re.search('mgy_clusters_2022_05.fa', arg):
            file_to_copy = 'mgnify_hits.sto'
        elif re.search('uniref90.fasta', arg):
            file_to_copy = 'uniref90_hits.sto'
        elif re.search('uniprot.fasta', arg):
            file_to_copy = 'uniprot_hits.sto'
        elif re.search('bfd-first_non_consensus_sequences.fasta', arg):
            file_to_copy = 'small_bfd_hits.sto'
        else:
            file_to_copy = ''.join(args)


    source_path = os.path.join(get_msa_path_by_sequence(input_file), file_to_copy)
    print(f"Copying {source_path} to {output_path}")
    copyfile(source_path, output_path)
