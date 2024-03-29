# Copyright 2022 Friedrich Miescher Institute for Biomedical Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Georg Kempf, Friedrich Miescher Institute for Biomedical Research

import copy
import json
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import pickle
import pkg_resources
import os
import sys
import argparse
import numpy as np
from absl import logging
from Bio.PDB import PDBParser, Superimposer, PDBIO
from Bio import SeqIO
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from jinja2 import Template, Environment, FileSystemLoader
from alphafold.data import parsers
import pickle
import re
import typing

import jax
import jax.numpy as jnp



class EvaluationPipeline:
    def __init__(self, fasta_path: str = None, results_dir: str = None, features_dir: str = None, continue_from_existing_results: bool = False, custom_spacing: bool = False,
                 custom_start_residue_list: str = None, custom_axis_label_list: str = None, batch_prediction: str = False, prediction_pipeline: str = 'alphafold'):
        self.sequence_file = fasta_path
        self.continue_from_existing_results = continue_from_existing_results
        self.output_dir_base = os.path.split(os.path.realpath(self.sequence_file))[0]
        _, seq_titles = self.parse_input_sequence(self.sequence_file)
        self.protein_names = '_'.join(seq_titles)
        output_dir_predictions = os.path.join(self.output_dir_base, "predictions")
        output_dir_features = os.path.join(self.output_dir_base, "features")
        self.features_dir = features_dir
        self.features_path = os.path.join(self.features_dir, f'features_{self.protein_names}.pkl')
        self.results_dir = results_dir
        self.custom_spacing = custom_spacing
        self.custom_start_residue_list = custom_start_residue_list
        self.custom_axis_label_list = custom_axis_label_list
        self.pae_results_unsorted = None
        self.prediction = prediction_pipeline
        self.batch = batch_prediction
        
        
        #Check if input path/files exist
        if not os.path.exists(self.results_dir):
            error_msg = f"Results directory {self.results_dir} not found"
            logging.error(error_msg)
            raise SystemExit(error_msg)
        if not os.path.exists(self.features_path):
            error_msg = f"Features pickle file {self.features_path} not found"
            logging.error(error_msg)
            raise SystemExit(error_msg)
        
    def round_float(self, data):
        decimal_places = 2
        if isinstance(data, float):
            return round(data, decimal_places)
        elif isinstance(data, tuple):
            return tuple(self.round_float(item) for item in data)
        elif isinstance(data, list):
            return [self.round_float(item) for item in data]
        else:
            return data

    def run_pipeline(self):
        logging.info("Running evaluation pipeline.")
        pae_list, plddt_list, iptm_list, ptm_list = [], [], [], []
        multimer = False
        no_pae = False

        input_sequences, seq_titles = self.parse_input_sequence(self.sequence_file)
        self.seq_titles = seq_titles
        seq_len_dict = self.get_sequence_len(input_sequences)
        indices = self.get_indices(seq_len_dict)
        results_pickle_path = os.path.join(self.results_dir, 'results.pkl')

        if len(input_sequences) > 1:
            multimer = True

        if os.path.exists(results_pickle_path) and self.continue_from_existing_results:
            logging.info("Results pickle exists and continue from existing results requested.")
            try:
                with open(results_pickle_path, 'rb') as handle:
                    plddt_list, average_pae_list, iptm_list, ptm_list = pickle.load(handle)
            except ValueError:
                with open(results_pickle_path, 'rb') as handle:
                    plddt_list, average_pae_list = pickle.load(handle)
                    iptm_list, ptm_list = None, None
            if self.check_none(average_pae_list):
                no_pae = True
        else:
            logging.info(f"Extracting results from prediction pipeline {self.prediction}.")
            if self.prediction == 'alphafold':
                for i, mdl in enumerate([os.path.join(self.results_dir, x) for x in os.listdir(self.results_dir)
                                        if x.startswith("result_") and x.endswith(".pkl")]):
                    if not i > 6:
                        logging.debug(mdl)
                        with open(mdl, 'rb') as f:
                            pkl_data = pickle.load(f)
                            for k, v in pkl_data.items():
                                logging.debug(k)
                            if 'predicted_aligned_error' in pkl_data:
                                pae = pkl_data['predicted_aligned_error']
                            else:
                                pae = None
                            if 'plddt' in pkl_data:
                                plddt = np.mean(pkl_data['plddt'])
                            else:
                                plddt = None
                            if 'iptm' in pkl_data:
                                iptm = pkl_data['iptm']
                            else:
                                iptm = None
                            if 'ptm' in pkl_data:
                                ptm = pkl_data['ptm']
                            elif 'predicted_tm_score' in pkl_data:
                                ptm = pkl_data['predicted_tm_score']
                            else:
                                ptm = None
                            #print(plddt)
                            mdl_name = os.path.splitext(os.path.basename(mdl))[0]
                            pae_list.append((pae, mdl_name))
                            plddt_list.append((plddt, mdl_name))
                            iptm_list.append((iptm, mdl_name))
                            ptm_list.append((ptm, mdl_name))
                        if not self.prediction == 'rosettafold':
                            self.save_confidence_json(pkl_data, mdl.replace('.pkl', '.json'))
                logging.debug(pae_list)
                logging.debug("Check none:")
                logging.debug(self.check_none(pae_list))



                if multimer:
                    if not self.check_none(pae_list):
                        pae_list = self.get_best_prediction_for_model_by_pae(pae_list)
                    else:
                        no_pae = True
                    plddt_list = self.get_best_prediction_for_model_by_plddt(plddt_list)
                logging.debug(pae_list)
            elif self.prediction == 'rosettafold':
                for i, mdl in enumerate([os.path.join(self.results_dir, x) for x in os.listdir(self.results_dir)
                                if x.startswith("result_") and x.endswith(".npz")]):
                    with open(mdl, 'rb') as f:
                        data = np.load(f)
                        logging.debug("Loaded data PAE:")
                        logging.debug(data['pae'])
                        if 'pae' in data:
                            pae = data['pae']
                        else:
                            pae = None
                        if 'lddt' in data:
                            plddt = np.mean(data['lddt'])
                        else:
                            plddt = None
                        #print(plddt)
                        mdl_name = os.path.splitext(os.path.basename(mdl))[0]
                        pae_list.append((pae, mdl_name))
                        logging.debug("PAE list:")
                        logging.debug(pae_list)
                        plddt_list.append((plddt, mdl_name))

            if not self.check_none(pae_list):
                average_pae_list = self.analyse_pae(pae_list,
                                                indices,
                                                seq_titles)
                self.plot_pae(pae_list,
                               indices,
                                 seq_titles)
            else:
                no_pae = True
                average_pae_list = [(model_name, None) for pae, model_name in pae_list]
            plddt_list = sorted(plddt_list, key=lambda x: x[0], reverse=True)
            if not iptm_list is None:
                if not self.check_none(iptm_list):
                    logging.info("Sorting iptm list")
                    logging.info(iptm_list)
                    iptm_list = sorted(iptm_list, key=lambda x: x[0], reverse=True)
                else:
                    iptm_list = None
            if not ptm_list is None:
                if not self.check_none(ptm_list):
                    ptm_list = sorted(ptm_list, key=lambda x: x[0], reverse=True)
                else:
                    ptm_list = None
            with open(results_pickle_path, 'wb') as handle:
                pickle.dump((plddt_list, average_pae_list, iptm_list, ptm_list), handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.iptm_list, self.ptm_list = iptm_list, ptm_list




        logging.debug("Before sort")
        logging.debug(average_pae_list)
        self.pae_results_unsorted = average_pae_list
        if not no_pae:
            average_pae_list = self.sort_results(average_pae_list)
            logging.debug(average_pae_list)
            images = self.get_image_path_list(average_pae_list)
            pae_messages = self.get_pae_messages(average_pae_list)
        else:
            images = None
            pae_messages = None
        pdb_path_list = self.align_models(average_pae_list)
        chain_color = self.get_chain_color()



        #with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
        #                       "templates", "results.html"), 'r') as f:
        #    template = Template(f.read())

        for path, _ in pdb_path_list:
            logging.debug(path)

        templates_path = pkg_resources.resource_filename("guifold", "templates")
        pae_examples_path = pkg_resources.resource_filename("guifold", "images/pae_examples.png")
        logging.debug(templates_path)

        #Round floats to two decimals
        for item in [average_pae_list, plddt_list, iptm_list, ptm_list]:
            item = self.round_float(item)

        ### Make MSA coverage plot
        msa_coverage_path = self.msa_coverage()
        
        ### Render template
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('results.html')
        if no_pae:
            average_pae_list = None
        rendered = template.render(pae_results=average_pae_list,
                                   plddt_list=plddt_list,
                                   iptm_list=iptm_list,
                                   ptm_list =ptm_list,
                                   images=images,
                                   pae_messages=pae_messages,
                                   pdb_path_list=pdb_path_list,
                                   chain_color=chain_color,
                                   templates_path=templates_path,
                                   multimer=multimer,
                                   pae_examples_path=pae_examples_path,
                                   msa_coverage_path=msa_coverage_path,
                                   use_model_viewer=False,
                                   )

        html_path = os.path.join(self.results_dir, "results.html")
        with open(html_path, "w") as f_out:
            f_out.write(rendered)
        rendered = template.render(pae_results=average_pae_list,
                                   plddt_list=plddt_list,
                                   iptm_list=iptm_list,
                                   ptm_list=ptm_list,
                                   images=images,
                                   pae_messages=pae_messages,
                                   pdb_path_list=pdb_path_list,
                                   chain_color=chain_color,
                                   templates_path=templates_path,
                                   multimer=multimer,
                                   pae_examples_path=pae_examples_path,
                                   msa_coverage_path=msa_coverage_path,
                                   use_model_viewer=True)
        html_path = os.path.join(self.results_dir, "results_model_viewer.html")
        with open(html_path, "w") as f_out:
            f_out.write(rendered)
        #plt.tight_layout()
        logging.info(f"Finished. Results written to {self.results_dir}.")

    def msa_coverage(self):
        """Adapted from https://github.com/sokrypton/ColabFold/blob/0d63cbd596fe938e3c6724761497d739820508eb/colabfold/colabfold.py 
        and https://github.com/jasperzuallaert/VIBFold/blob/main/visualize_alphafold_results.py"""
        try:
            feature_dict = pickle.load(open(self.features_path, 'br'))
        except OSError:
            raise SystemExit(f"Feature file {self.features_path} not found")

        #Get subunit boundaries
        if 'asym_id' in feature_dict:
            #multimer prediction
            _, num_residues = np.unique(feature_dict['asym_id'], return_counts=True)
        else:
            #monomer prediction
            num_residues = feature_dict['msa'][0]
        cumulative_num_residues = []
        cumulative_num = 0
        for num_res in num_residues:
            cumulative_num += num_res
            cumulative_num_residues.append(cumulative_num)
        msa = feature_dict['msa']
        seq_identity = np.mean(msa[0] == msa, axis=-1)
        #seq_identity_full = (msa[0] == msa)
        seq_identity_indices = np.argsort(seq_identity)
        msa_by_identity = np.where(msa == 21, np.nan, 1.0) * seq_identity[:, np.newaxis]
        msa_by_identity = msa_by_identity[seq_identity_indices]
        #print(msa_by_identity)

        fig, ax = plt.subplots(figsize=(14, 4), dpi=100)
        ax.set_title(f"Sequence coverage ()")
        im = ax.imshow(msa_by_identity, interpolation='nearest', aspect='auto',
                    cmap="rainbow", vmin=0, vmax=1, origin='lower')
        ax.plot(np.sum(msa != 21, axis=0), color='black')
        #Plot vertical lines after each subunit
        ax.vlines(cumulative_num_residues, 0, msa.shape[0], color='black', linewidth=2)
        ax.set_xlim(-0.5, msa.shape[1] - 0.5)
        ax.set_ylim(-0.5, msa.shape[0] - 0.5)
        fig.colorbar(im, label="Sequence identity to query")
        ax.set_xlabel("Positions")
        ax.set_ylabel("Sequences")
        msa_coverage_path = os.path.join(self.results_dir, 'msa_coverage.png')
        fig.savefig(msa_coverage_path)
        return msa_coverage_path    

    def check_none(self, nested_list):
        if isinstance(nested_list, (list, tuple)):
            for x in nested_list:
                if self.check_none(x):
                    return True
            return False
        else:
            if nested_list is None:
                return True
            else:
                return False

    def parse_input_sequence(self, input_seq):
        input_sequences = []
        seq_titles = []
        record_dict = SeqIO.index(input_seq, "fasta")
        for record in record_dict.values():
            input_sequences.append(str(record.seq))
            seq_titles.append(str(record.description))
        return input_sequences, seq_titles

    def get_sequence_len(self, input_seq):
        seq_len_dict = {}
        for i, seq in enumerate(input_seq):
            seq_len_dict[i] = len(seq)
        return seq_len_dict

    def get_indices(self, seq_len_dict):
        indices = []
        prev = 0
        added = 0
        for k, v in seq_len_dict.items():
            added += v
            indices.append((prev, added))
            prev = added
        return indices

    def get_best_prediction_for_model_by_plddt(self, plddt_list):
        best_plddt = {}
        new_plddt_list = []
        for plddt, model_name in plddt_list:
            model_num = re.search("_model_(\d+)_", model_name).group(1)
            if not model_num in best_plddt:
                best_plddt[model_num] = [plddt, model_name]
            else:
                if np.mean(plddt) < best_plddt[model_num][0]:
                    best_plddt[model_num] = [plddt, model_name]
        for model_num in best_plddt.keys():
            new_plddt_list.append([best_plddt[model_num][0], best_plddt[model_num][1]])
        return new_plddt_list

    def get_best_prediction_for_model_by_pae(self, pae_list):
        """Make a new pae list including only best prediction for a model"""
        best_pae = {}
        new_pae_list = []
        for pae, model_name in pae_list:
            model_num = re.search("_model_(\d+)_", model_name).group(1)
            if not model_num in best_pae:
                best_pae[model_num] = [pae, model_name, np.mean(pae)]
            else:
                if np.mean(pae) < best_pae[model_num][2]:
                    best_pae[model_num] = [pae, model_name, np.mean(pae)]
        for model_num in best_pae.keys():
            new_pae_list.append([best_pae[model_num][0], best_pae[model_num][1]])
        return new_pae_list

    def get_major_tick_spacing(self, num_elements, len_seq, custom_spacing):
        if custom_spacing:
            return int(custom_spacing)
        elif len_seq <= 50:
            return 10
        elif num_elements < 4 and not len_seq >= 2000:
            return 100
        elif num_elements >= 4 or len_seq >= 2000:
            return 200
        
    def analyse_pae(self, pae_list, indices, seq_titles):
        num_seqs = len(indices)
        matrix_len = num_seqs * num_seqs
        u = 0
        y_labels = []
        for i, _ in enumerate(range(matrix_len)):
            y_labels.append(seq_titles[u])
            if (i + 1) % num_seqs == 0:
                u += 1
        x_labels = []
        for i, _ in enumerate(range(num_seqs)):
            x_labels.append(seq_titles[i])
        x_labels = x_labels*num_seqs
        average_pae_dict = {}
        for i,(pae, model_name) in enumerate(pae_list):
            average_pae_dict[model_name] = {"Overall": np.mean(pae)}
            count = 0
            for u in range(len(indices)):
                for v in range(len(indices)):
                    data = pae[indices[u][0]:indices[u][1],indices[v][0]:indices[v][1]]
                    average_pae = np.mean(data)
                    average_pae_dict[model_name][f"{x_labels[count]} vs {y_labels[count]}"] = average_pae
                    count += 1

        return average_pae_dict

    def plot_pae(self, pae_list, indices, seq_titles, custom_spacing=None, custom_aa_start=None, custom_labels=None):
        """Plots PAE. Adapted from https://github.com/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb"""
        num_models = len(pae_list)
        num_seqs = len(indices)
        matrix_len = num_seqs * num_seqs
        custom_aa_start_list = []
        if custom_aa_start:
            custom_aa_start_list = custom_aa_start.split(',')
            if not isinstance(custom_aa_start_list, list):
                custom_aa_start_list = [custom_aa_start_list]
            custom_aa_start_list = [int(x) for x in custom_aa_start_list] * num_seqs

            if len(custom_aa_start_list) != num_seqs*num_seqs:
                raise ValueError(f"Number of given starting residue values ({len(custom_aa_start_list)})"
                                 f" does not match the number of protein subunits ({num_seqs*num_seqs}).")

        if custom_labels:
            custom_labels_list = custom_labels.split(',')
            if not isinstance(custom_labels_list, list):
                custom_labels_list = [custom_labels_list]
            seq_titles = custom_labels_list

        u = 0
        y_labels = []
        for i, _ in enumerate(range(matrix_len)):
            y_labels.append(seq_titles[u])
            if (i + 1) % num_seqs == 0:
                u += 1
        x_labels = []
        for i, _ in enumerate(range(num_seqs)):
            x_labels.append(seq_titles[i])
        x_labels = x_labels*num_seqs
        titles_raw = zip(y_labels, x_labels)
        titles = []
        for title in titles_raw:
            title_1 = title[0]
            title_2 = title[1]
            title = f"{title_1} against {title_2}"
            titles.append(title)
        plt.rc('xtick',labelsize=6)
        plt.rc('ytick',labelsize=6)
        plt.rc('font',**{'family':'sans-serif', 'size': 8})

        #average_paes = {}
        for i,(pae, model_name) in enumerate(pae_list):
            heights, widths = [], []
            #average_paes[model_name] = {"Overall": np.mean(pae)}
            #logging.debug(average_paes)
            logging.debug(f"Number of inidices {len(indices)}")
            for u in range(len(indices)):
                for v in range(len(indices)):
                    data = pae[indices[u][0]:indices[u][1],indices[v][0]:indices[v][1]]
                    num_rows = data.shape[0]
                    num_cols = data.shape[1]
                    heights.append(num_rows)
                    widths.append(num_cols)

            logging.debug(f"Len heights {len(heights)}")
            logging.debug(heights)
            max_height = max(heights)
            #heights = list(dict.fromkeys(heights))
            logging.debug(heights)
            heights = [h / max_height for i, h in enumerate(heights) if i % len(indices) == 0]
            logging.debug(heights)
            fig, axs = plt.subplots(num_seqs, num_seqs, squeeze=False,
                                    gridspec_kw={'height_ratios': heights, 'width_ratios': heights},
                                    constrained_layout=True)#, sharex=True, sharey=True)
            count = 0
            num_rows_cols_list = []
            maps = []
            for u in range(len(indices)):
                for v in range(len(indices)):
                    data = pae[indices[u][0]:indices[u][1],indices[v][0]:indices[v][1]]
                    #average_pae = np.mean(data)
                    #average_paes[model_name][f"{x_labels[count]} vs {y_labels[count]}"] = average_pae
                    num_rows = data.shape[0]
                    num_cols = data.shape[1]
                    num_rows_cols_list.append((num_rows, num_cols))
                    im = axs[u, v].pcolor(data,cmap="bwr", vmin=0, vmax=30, rasterized=True)#,
                              #extent=(0, num_cols, num_rows, 0),
                              #interpolation='nearest'
                              #)
                    axs[u, v].invert_yaxis()
                    #Show labels only for bottom row
                    if u == len(indices) - 1:
                        logging.debug(f"logging.debug x label {u} {v}")
                        axs[u, v].set_xlabel(f"{x_labels[count]} [aa]")

                    if v % len(indices) == 0:
                        logging.debug(f"logging.debug y label {u} {v}")
                        axs[u, v].set_ylabel(f"{y_labels[count]} [aa]")

                    if len(custom_aa_start_list) > 0:
                        start_tick = custom_aa_start_list[count]
                        major_start_tick_rounded = round(start_tick/100)*100
                        minor_start_tick_rounded = round(start_tick/10)*10
                    else:
                        minor_start_tick_rounded = major_start_tick_rounded = start_tick = 0
                    logging.debug(f"u: {u}\nv: {v}\nx label {x_labels[count]}\ny label {y_labels[count]}\nlen_data: {len(data)}\nmajor tick spacing: {self.get_major_tick_spacing(len(indices), len(data), custom_spacing)}")
                    logging.debug(f"len_data_x: {np.size(data, 1)} len_data_y: {np.size(data, 0)}")
                    len_y_axis = np.size(data, 0)
                    len_x_axis = np.size(data, 1)
                    major_tick_spacing_x = self.get_major_tick_spacing(len(indices), len_x_axis, custom_spacing)
                    major_tick_spacing_y = self.get_major_tick_spacing(len(indices), len_y_axis, custom_spacing)
                    major_tick_positions_x = np.arange(major_start_tick_rounded - start_tick,
                                                       len_x_axis,
                                                       major_tick_spacing_x)
                    minor_tick_positions_x = np.arange(minor_start_tick_rounded - start_tick,
                                                       len_x_axis,
                                                       major_tick_spacing_x / 10)
                    major_tick_positions_y = np.arange(major_start_tick_rounded - start_tick,
                                                       len_y_axis,
                                                       major_tick_spacing_y)
                    minor_tick_positions_y = np.arange(minor_start_tick_rounded - start_tick,
                                                        len_y_axis,
                                                        major_tick_spacing_y / 10)
                    tick_labels_x = np.arange(major_start_tick_rounded,
                                              len_x_axis + start_tick,
                                              major_tick_spacing_x)
                    tick_labels_y = np.arange(major_start_tick_rounded,
                                              len_y_axis + start_tick,
                                              major_tick_spacing_y)
                    axs[u,v].set_xticks(major_tick_positions_x)
                    axs[u,v].set_xticks(minor_tick_positions_x, minor=True)
                    axs[u,v].set_xticklabels(tick_labels_x)
                    axs[u,v].set_yticks(major_tick_positions_y)
                    axs[u,v].set_yticks(minor_tick_positions_y, minor=True)
                    axs[u,v].set_yticklabels(tick_labels_y)

                        #plt.sca(axs[u, v])
                        #plt.xticks(np.arange(len(data)), np.arange(custom_aa_start_list[count],
                        #                                         len(data)+custom_aa_start_list[count]))

                    #axs[u, v].xaxis.set_major_locator(MultipleLocator(
                    #    self.get_major_tick_spacing(len(indices), len(data), custom_spacing)))
                    #axs[u, v].xaxis.set_major_formatter(FormatStrFormatter('%d'))
                    #axs[u, v].xaxis.set_minor_locator(MultipleLocator(50))
                    #axs[u, v].yaxis.set_major_locator(MultipleLocator(100))
                    #axs[u, v].yaxis.set_major_formatter(FormatStrFormatter('%d'))
                    #axs[u, v].yaxis.set_minor_locator(MultipleLocator(50))
                    count += 1
            #fig.subplots_adjust(hspace=0.6, wspace=0.3)
            #adjustw(ims, wspace=1)
            #divider = make_axes_locatable(plt.gca())
            #cax = divider.append_axes("right", "5%", pad="3%")
            cbar = plt.colorbar(im, ax=axs, cmap="bwr", shrink=0.8)
            cbar.set_label('Predicted aligned error (PAE) [$\AA$]', rotation=90, labelpad=10)
            #plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'pae_{model_name}.png'), dpi=300)
            plt.savefig(os.path.join(self.results_dir, f'pae_{model_name}.svg'), dpi=300)
        #logging.debug(average_paes)
        #return average_paes
        #plt.show()
        #plt.savefig(os.path.join(output_dir, f'pae_{model_name}.png'))#, bbox_inches = 'tight')
        #plt.clf()



    def get_image_path_list(self, results):
        images = []
        for item in results:
            images.append((item[0], f'pae_{item[0]}.png'))
        return images

    def align_pdb(self, reference, mobile):
        pdb_parser = PDBParser(QUIET = True)
        logging.debug(f"Reference model: {reference} Mobile model {mobile}")

        ref_structure = pdb_parser.get_structure("reference", reference)
        mobile_structure = pdb_parser.get_structure("mobile", mobile)

        ref_model = ref_structure[0]
        mobile_model = mobile_structure[0]


        ref_atoms = []
        mobile_atoms = []

        #ref_chains = [str(c.get_id()) for c in ref_model.get_chains()][0]
        mobile_chains = [str(c.get_id()) for c in mobile_model.get_chains()]
        logging.debug(f"mobile chains {mobile_chains}")

        #Get the number of residues of each chain to find the largest chain for alignment
        num_res_per_chain = {str(c.get_id()): len([r for r in c.get_residues()]) for c in ref_model.get_chains()}
        num_res_per_chain = {k: v for k, v in sorted(num_res_per_chain.items(), key=lambda item: item[1])}
        ref_chain = mobile_chain = list(num_res_per_chain.keys())[-1]
        assert ref_chain in mobile_chains
        logging.debug(f"Ref chain {ref_chain} Mobile chain {mobile_chain}")

        for ref_res in ref_model[ref_chain]:
            ref_atoms.append(ref_res['CA'])

        for mobile_res in mobile_model[mobile_chain]:
            mobile_atoms.append(mobile_res['CA'])

        super_imposer = Superimposer()
        super_imposer.set_atoms(ref_atoms, mobile_atoms)
        super_imposer.apply(mobile_model.get_atoms())
        name = os.path.splitext(os.path.basename(mobile))[0]
        name_aligned = f"{name}_aligned.pdb"
        io = PDBIO()
        io.set_structure(mobile_structure)
        out_path = os.path.join(self.results_dir, name_aligned)
        if not os.path.exists(out_path):
            io.save(out_path)
        logging.debug(out_path)
        return name_aligned

    def align_models(self, results):
        pdbs = []
        files_unrelaxed = [os.path.join(self.results_dir, x) for x in os.listdir(self.results_dir)
                           if x.startswith("unrelaxed") and x.endswith(".pdb") and not re.search("aligned", x)]
        reference = None
        for item in results:
            logging.debug(item[0])
            model_index = re.search("model_(\d+)", item[0]).group(1)
            if re.search("_pred_", item[0]):
                pred_index = re.search("_pred_(\d+)", item[0]).group(1)
            else:
                pred_index = None

            for f in files_unrelaxed:
                logging.debug(f"Index {model_index} {f}")
                if model_index == re.search("model_(\d+)", os.path.basename(f)).group(1):
                    if not pred_index is None:
                        if not pred_index == re.search("_pred_(\d+)", os.path.basename(f)).group(1):
                            continue
                    if reference is None:
                        reference = f
                    aligned = self.align_pdb(reference, f)
                    logging.debug(aligned)
                    with open(os.path.join(self.results_dir, aligned), 'r') as fin:
                        content = fin.readlines()
                    content = ''.join([x for x in content if x.startswith("ATOM")])
                    pdbs.append((aligned, content))
        logging.debug(files_unrelaxed)
        return pdbs

    def sort_results(self, results: dict):
        return sorted(results.items(),
                      key = lambda x: x[1]['Overall'],
                      reverse=False)

    def get_pae_results_unsorted(self):
        return self.pae_results_unsorted

    def get_min_inter_pae(self, results) -> tuple:
        logging.info(results)
        results = sorted(results.items(),
                      key = lambda x: x[1][list(x[1].keys())[2]],
                      reverse=False)
        logging.info(results)
        protein_name = list(results[0][1].keys())[3]
        model_name = results[0][0]
        min_pae = results[0][1][protein_name]
        return (protein_name.replace(" vs ", "_"), min_pae, model_name)
    
    def get_max_iptm(self) -> tuple:
        #return model_name, value
        protein_names = '_'.join(self.seq_titles)
        max_iptm = self.iptm_list[0][0].tolist()
        max_iptm_model_name = self.iptm_list[0][1]
        return (protein_names, max_iptm, max_iptm_model_name)
    
    def get_max_ptm(self):
        protein_names = '_'.join(self.seq_titles)
        max_ptm = self.ptm_list[0][0].tolist()
        max_ptm_model_name = self.ptm_list[0][1]
        return (protein_names, max_ptm, max_ptm_model_name)
    

    def find_index_by_protein_name(self, data_list, search_value):
        for index, item in enumerate(data_list):
            if 'protein_names' in item and item['protein_names'] == search_value:
                return index
        return -1

    def get_scores(self, score: dict):
        protein_names_pae, min_pae, model_name_min_pae = self.get_min_inter_pae(self.get_pae_results_unsorted())
        if not self.prediction == 'rosettafold':
            protein_names_ptm, max_ptm, model_name_max_ptm = self.get_max_ptm()
            protein_names_iptm, max_iptm, model_name_max_iptm = self.get_max_iptm()
            logging.debug(f"Check if protein names are equal: {protein_names_pae},{protein_names_ptm},{protein_names_iptm},{score['protein_names']}")
            assert protein_names_pae == protein_names_ptm == protein_names_iptm == score['protein_names']
            logging.debug(f"model_name_max_ptm: {model_name_max_ptm}")
        logging.debug(f"model_name_min_pae: {model_name_min_pae}")
        score['min_pae_model_name'] = model_name_min_pae
        logging.debug(f"min_pae: {min_pae}")
        score['min_pae_value'] = min_pae
        if not self.prediction == 'rosettafold':
            score['max_ptm_model_name'] = model_name_max_ptm
            logging.debug(f"min ptm: {max_ptm}")
            score['max_ptm_value'] = max_ptm
            logging.debug(f"model_name_max_iptm: {model_name_max_iptm}")
            score['max_iptm_model_name'] = model_name_max_iptm
            logging.debug(f"max_iptm_value: {max_iptm}")
            score['max_iptm_value'] = max_iptm
        logging.debug(score['min_pae_model_name'])
        logging.debug("Updated scores dict")
        logging.debug(score)
        return score

    def get_pae_messages(self, results):
        best_result = float(results[0][1]['Overall'])
        if best_result >= 20.0:
            messages = f"The best overall PAE is higher than 20 A." \
                       " This does not indicate a high confidence." \
                       " The prediction is likely incorrect."
        elif best_result > 10.0 and best_result < 20.0:
            messages = f"The best overall PAE is between 10 and 20 A." \
                       " This indicates an intermediate confidence" \
                       " and prediction might be close to reality."
        elif best_result < 10.0:
            messages = f"The best overall PAE is lower than 10 A." \
                       " This indicates a high confidence" \
                       " and prediction is likely correct."

        return messages


    def get_chain_color(self):
        structure = [os.path.join(self.results_dir, x) for x in os.listdir(self.results_dir)
                           if x.startswith("unrelaxed") and x.endswith(".pdb")][0]
        pdb = PDBParser().get_structure("structure", structure)[0]
        chains = [str(c.get_id()) for c in pdb.get_chains()]
        return list(zip(chains,
            ["lime","cyan","magenta","yellow","salmon","white","blue","orange"]))
    

    def convert_to_list(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, (jnp.ndarray, jax.interpreters.xla.DeviceArray)):
            return jax.device_get(obj).tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def save_confidence_json(self, data, file_name):
        keys = ['predicted_aligned_error',
                            'plddt',
                            'predicted_tm_score',
                            'ptm',
                            'iptm',
                            'predicted_lddt',
                            'num_recycles']
        for key in keys:
            confidence_data = {key: data[key] for key in keys if key in data}
        confidence_data = self.convert_to_list(confidence_data)
        with open(os.path.join(self.results_dir, file_name), 'w') as f:
            json.dump(confidence_data, f)

class EvaluationPipelineBatch:
    def __init__(self, results_dir: str, scores: dict):
        self.results_dir = results_dir
        self.scores = scores
        
    def run(self):
        self.write_html()

    def write_html(self):
        templates_path = pkg_resources.resource_filename("guifold", "templates")
        logging.debug(templates_path)
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('batch_summary.html')
        logging.debug("Writing to summary.html")
        logging.debug(self.scores)

        rendered = template.render(scores=self.scores)
        
        html_path = os.path.join(self.results_dir, "batch_summary.html")
        with open(html_path, "w") as fout:
            fout.write(rendered)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('--fasta_path', default=None, required=True, help='Path to input FASTA file (must be located below the results folder).')
    parser.add_argument('--results_dir', default=None, required=True, help='Path to results.')
    parser.add_argument('--features_dir', default=None, required=True, help='Path to features.')
    parser.add_argument('--prediction_pipeline', default='alphafold', help='(Optional) Prediciton pipeline')
    parser.add_argument('--debug', default=False, action='store_true', help='(Optional) Debug mode.')
    parser.add_argument('--continue_from_existing_results', default=False, action='store_true',
                        help='(Optional) Continue from previously extracted PAE values.')
    parser.add_argument('--custom_spacing', default=None, help='(Optional) Custom value for spacing of major tick labels.')
    parser.add_argument('--custom_start_residue_list', default=None, help='(Optional) Define custom axes minimum number. If the prediction was done with internal segments of the proteins,'
                                                            'the actual residue numbering can be restored in the plot by'
                                                            'giving the starting residue numbers of the segments separated by a comma.'
                                                            'For example, if there are two subunits with predicted segments'
                                                            '15-100 and 50-300, the list would be --custom_start_residue_list 15,50')
    parser.add_argument('--custom_axis_label_list', default=None, help='(Optional) Define custom axes labels.'
                                                          ' Labels for different subunits need to be separated by a comma'
                                                          ' and given in the same order as the sequences in the fasta file. Example: --custom_axis_label_list \"Protein A\", \"Protein B\"')
    args, unknown = parser.parse_known_args()
    if not args.debug:
        logging.set_verbosity(logging.INFO)
    else:
        logging.set_verbosity(logging.DEBUG)

    EvaluationPipeline(fasta_path=args.fasta_path,
                        results_dir=args.results_dir,
                        features_dir=args.features_dir,
                        continue_from_existing_results=args.continue_from_existing_results,
                        custom_spacing=args.custom_spacing,
                        custom_start_residue_list=args.custom_start_residue_list,
                        custom_axis_label_list=args.custom_axis_label_list,
                        prediction_pipeline=args.prediction_pipeline).run_pipeline()