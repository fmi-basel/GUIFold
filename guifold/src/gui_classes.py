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

from __future__ import absolute_import
import pkg_resources
import datetime
import sys
import os
import re
import logging
import socket
from Bio import SeqIO
from PyQt5 import QtWidgets, QtGui, QtCore, QtWebEngineWidgets
from jinja2 import Template, Environment, meta, PackageLoader
from sqlalchemy.orm.exc import NoResultFound
import traceback
from datetime import date
from shutil import copyfile, rmtree
import configparser
from subprocess import check_output
import nvidia_smi
import math
import shutil
from Bio.PDB import MMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
logger = logging.getLogger('guifold')
#logger.setLevel(logging.DEBUG)

class WebViewer(QtWebEngineWidgets.QWebEngineView):

    def __init__(self, parent=None):
        super(WebViewer, self).__init__(parent)

    def closeEvent(self, qclose_event):
        """Overwrite QWidget method"""
        # Sets the accept flag of the event object, indicates that the event receiver wants the event.
        qclose_event.accept()
        # Schedules this object for deletion, QObject
        self.page().deleteLater()

ctrl_type_dict = {'dsb': QtWidgets.QDoubleSpinBox,
                  'sbo': QtWidgets.QSpinBox,
                  'lei': QtWidgets.QLineEdit,
                  'tbl': QtWidgets.QTableWidget,
                  'cmb': QtWidgets.QComboBox,
                  'pte': QtWidgets.QPlainTextEdit,
                  'chk': QtWidgets.QCheckBox,
                  'web': QtWebEngineWidgets.QWebEngineView,
                  'pro': QtWidgets.QProgressBar,
                  'tre': QtWidgets.QTreeWidget}

install_path = os.path.dirname(os.path.realpath(sys.argv[0]))


class Variable:
    """Basic model for GUI controls and DB tables"""

    def __init__(self, var_name=None,
                 type=None,
                 db=True,
                 ctrl_type=None,
                 db_primary_key=False,
                 db_foreign_key=None,
                 db_relationship=None,
                 db_backref=None,
                 cmb_dict=None,
                 cmd=False,
                 required=False):
        self.var_name = var_name
        self.value = None
        self.type = type
        self.ctrl_type = ctrl_type
        self.ctrl = None
        self.db = db
        self.db_relationship = db_relationship
        self.db_backref = db_backref
        self.db_foreign_key = db_foreign_key
        self.db_primary_key = db_primary_key
        self.cmb_dict = cmb_dict
        self.cmd = cmd
        self.selected_item = None
        self.required = required

    def get_value(self):
        return self.value

    def set_ctrl_type(self):
        try:
            self.ctrl_type = self.ctrl.__class__.__name__
        except Exception:
            logger.warning("Cannot set control {}".format(self.ctrl.__class__.__name__), exc_info=True)

    def set_value(self, value):
        self.value = value

    def set_cmb_by_text(self, value):
        if self.ctrl_type == 'cmb':
            for k, v in self.cmb_dict.items():
                logger.debug(f"Setting cmb, {value} in dict {v}")
                if value == v:
                    index = self.ctrl.findText(str(v), QtCore.Qt.MatchFixedString)
                    if index >= 0:
                        self.ctrl.setCurrentIndex(index)
                    else:
                        logger.debug("Item not found")
        else:
            logger.debug("Not a cmb box.")

    def set_control(self, result_obj, result_var):
        if result_obj == 'None' or result_obj is None:
            result_obj = ''
        if self.ctrl_type == 'dsb':
            self.ctrl.setValue(float(result_obj))
        elif self.ctrl_type == 'lei':
            self.ctrl.setText(str(result_obj))
        elif self.ctrl_type == 'pte':
            self.ctrl.setPlainText(str(result_obj))
        elif self.ctrl_type == 'sbo':
            self.ctrl.setValue(int(result_obj))
        elif self.ctrl_type == 'cmb':
            for k, v in self.cmb_dict.items():
                logger.debug(f"Setting cmb, result_obj {result_obj} value in dict {v}")
                if result_obj == v:
                    index = self.ctrl.findText(str(v), QtCore.Qt.MatchFixedString)
                    if index >= 0:
                        self.ctrl.setCurrentIndex(index)
                    else:
                        logger.debug("Item not found")
        elif self.ctrl_type == 'chk':
            if not isinstance(result_obj, bool):
                if result_obj.lower() == 'true':
                    result_obj = True
                elif result_obj.lower() == 'false':
                    result_obj = False
                else:
                    raise ValueError("Value of controlbox must either be true or false!")
            self.ctrl.setChecked(result_obj)
        else:
            logger.error("Control type {} not found for control {}!".format(self.ctrl_type, result_var))
        logger.debug(f"Setting {result_var} to {result_obj}")

    def unset_control(self):
        if hasattr(self, 'ctrl_type') and not self.ctrl_type is None:
            self.ctrl = None

    def is_set(self):
        if isinstance(self.value, list):
            if not self.value == []:
                return True
            else:
                return False
        else:
            if not self.value in [None, "None", "none", ""]:
                return True
            else:
                return False

    def list_like_str_not_all_none(self):
        if self.value:
            lst = self.value.split(',')
            for e in lst:
                if not e in [None, "None", "none", ""]:
                    return True
        return False



    def update_from_self(self):
        if not self.value is None and not self.ctrl is None:
            logger.debug(f"Set value of control {self.var_name} to {self.value}")
            self.set_control(self.value, self.var_name)

    def update_from_db(self, db_result):
        """ Set/update variable values from DB """
        logger.debug("Update from DB")
        if not db_result is None and not db_result == []:
            for result_var in vars(db_result):
                #logger.debug("DB var: {}\nClass var: {}".format(result_var, self.var_name))
                if result_var == self.var_name:
                    result_obj = getattr(db_result, result_var)
                    self.value = result_obj
                    logger.debug("Variable name: {}\nValue: {}".format(result_var, result_obj))
                    if not self.ctrl is None:
                        if not result_obj in ['tbl', None]:
                            self.set_control(result_obj, result_var)
        else:
            logger.warning("DB result empty. Nothing to update.")

    def update_from_gui(self):
        """ Set/update attribute values (self.value) from GUI controls """
        if not self.ctrl is None:
            if self.ctrl_type in ['sbo', 'dsb']:
                    self.value = self.ctrl.value()
            elif self.ctrl_type == 'lei':
                self.value = self.ctrl.text()
            elif self.ctrl_type == 'pte':
                self.value = self.ctrl.toPlainText()
            elif self.ctrl_type == 'chk':
                self.value = self.ctrl.isChecked()
            elif self.ctrl_type == 'cmb':  # and not self.selected_item is None:
                self.value = self.ctrl.currentText()
            #if self.value == "":
            #    self.value = None
            logger.debug(f"Value of {self.var_name}: {self.value}")
        else:
            logger.debug(f"ctrl of {self.var_name} is not bound.")


    def reset_ctrl(self):
        """ Clear variable values and GUI controls """
        logger.debug("Resetting GUI controls ")
        if not self.ctrl is None:
            logger.debug(f"resetting {self.var_name}")
            if not self.ctrl_type in ['chk', 'tbl']:
                self.ctrl.clear()
                self.value = None

            elif self.ctrl_type == 'chk':  # and not self.selected_item is None:
                self.ctrl.setChecked(False)
                self.value = False

            elif self.ctrl_type == 'tbl':
                while self.ctrl.rowCount() > 0:
                    self.ctrl.removeRow(0);
        else:
            logger.debug(f"ctrl of {self.var_name} is not bound.")


class TblCtrlJobs(Variable):
    """GUI list control which shows jobs for project"""
    def __init__(self, var_name=None, type=None, db=True, ctrl_type=None, db_primary_key=False, db_foreign_key=None):
        super().__init__(var_name, type, db, ctrl_type, db_primary_key, db_foreign_key)
        self.selected_item = None


class WebCtrlEvaluation(Variable):
    """GUI list control which shows the results in a webbrowser"""
    def __init__(self, var_name=None, type=None, db=True, ctrl_type=None, db_primary_key=False, db_foreign_key=None):
        super().__init__(var_name, type, db, ctrl_type, db_primary_key, db_foreign_key)


class SelectCustomTemplateWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(SelectCustomTemplateWidget, self).__init__(parent)

        # add your buttons
        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        lei_custom_template = QtWidgets.QLineEdit()
        btn_custom_template = QtWidgets.QToolButton()
        hbox.addWidget(lei_custom_template)
        hbox.addWidget(btn_custom_template)
        self.lei_custom_template = lei_custom_template
        self.btn_custom_template = btn_custom_template
        self.setLayout(hbox)

class SelectPrecomputedMsasWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(SelectPrecomputedMsasWidget, self).__init__(parent)

        # add your buttons
        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        lei_precomputed_msas = QtWidgets.QLineEdit()
        btn_precomputed_msas = QtWidgets.QToolButton()
        hbox.addWidget(lei_precomputed_msas)
        hbox.addWidget(btn_precomputed_msas)
        self.lei_precomputed_msas = lei_precomputed_msas
        self.btn_precomputed_msas = btn_precomputed_msas
        self.setLayout(hbox)


class TblCtrlSequenceParams(Variable):
    """GUI list control which shows validation report"""
    def __init__(self, var_name=None, type=None, db=True, ctrl_type=None, db_primary_key=False, db_foreign_key=None):
        super().__init__(var_name, type, db, ctrl_type, db_primary_key, db_foreign_key)
        self.sequence_params_template =  {"custom_template_list": [],
                                          "precomputed_msas_list": [],
                                        "no_msa_list": [],
                                        "no_template_list": []}
        self.sequence_params_values = {k: [] for k in self.sequence_params_template.keys()}
        self.sequence_params_widgets = {k: [] for k in self.sequence_params_template.keys()}


    def register(self, parameter):
        self.sequence_dict[parameter.var_name] = parameter

    def get_seq_names(self, sequence_str):
        seq_names = []
        logger.debug(sequence_str)
        lines = sequence_str.split('\n')
        o = -1
        for line in lines:
            if line.startswith('>'):
                line = line.replace(" ", "_")
                name = re.sub(r'[\W]*', '', line)
                if name in seq_names:
                    name = f"{name}_dup"
                seq_names.append(name)
        return seq_names

    def sanitize_sequence_str(self, sequence_str):
        new_lines, error_msgs = [], []
        seq_name_line = False
        num_seq_names = 0
        num_seqs = 0
        lines = sequence_str.split('\n')
        for line in lines:
            if line.startswith('>'):
                seq_name_line = True
                num_seq_names += 1
                line = line.replace(" ", "_")
                #Exclude > from check
                if re.search(r'[\W]', line[1:]):
                    logger.debug("Found a non standard char in the sequence name")
                    logger.debug(line)
                    error_msgs.append("Non standard characters found in the sequence name!")
                new_lines.append(line)
            else:
                if seq_name_line:
                    num_seqs += 1
                    seq_name_line = False
                #Any blanks in the sequence will be replaced before writing the fasta file
                line = line.replace(" ", "")
                if re.search(r'[\W]', line):
                    logger.debug("Found a non standard char in the sequence")
                    logger.debug(line)
                    error_msgs.append("Non standard characters found in the sequence!")
                new_lines.append(line)
        if num_seq_names == 0:
            error_msgs.append("No lines starting with > found. Add a >SomeProteinName line above each sequence.")
        if num_seq_names != num_seqs:
            error_msgs.append("Number of sequence names not matching sequences. Add a >SomeProteinName line above each sequence.")
        new_lines = '\n'.join(new_lines)
        return new_lines, error_msgs

    def init_gui(self):
        self.ctrl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ctrl.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.ctrl.setColumnCount(5)
        self.ctrl.setHorizontalHeaderLabels(("Sequence name",
                                             "Custom template",
                                             "Precomputed MSAs",
                                             "No MSA",
                                             "No templates"))
        self.ctrl.setColumnWidth(0, 200)
        self.ctrl.setColumnWidth(1, 300)
        self.ctrl.setColumnWidth(2, 300)
        self.ctrl.setColumnWidth(3, 200)
        self.ctrl.setColumnWidth(4, 200)
        self.ctrl.setStyleSheet("item { align: center; padding: 0px; margin: 0px;}")

    def OnBtnSelectFile(self, lei):
        dlg = QtWidgets.QFileDialog()
        if dlg.exec_():
            path = dlg.selectedFiles()[0]
            logger.debug(path)
            lei.setText(path)

    def OnBtnSelectFolder(self, lei, widget):
        path = QtWidgets.QFileDialog.getExistingDirectory(widget, 'Select Folder')
        lei.setText(path)

    def create_widgets(self, seq_names):
        self.sequence_params_widgets = {k: [] for k in self.sequence_params_template.keys()}
        for i, _ in enumerate(seq_names):
            custom_template_widget = SelectCustomTemplateWidget()
            custom_template_widget.btn_custom_template.clicked.connect(
                lambda checked, a=custom_template_widget.lei_custom_template: self.OnBtnSelectFile(a))
            self.sequence_params_widgets["custom_template_list"].append(custom_template_widget)

            chk_nomsa = QtWidgets.QCheckBox()
            chk_nomsa.setStyleSheet("text-align: center; margin-left:50%; margin-right:50%;")
            self.sequence_params_widgets["no_msa_list"].append(chk_nomsa)

            chk_notemplate = QtWidgets.QCheckBox()
            chk_notemplate.setStyleSheet("text-align: center; margin-left:50%; margin-right:50%;")
            self.sequence_params_widgets["no_template_list"].append(chk_notemplate)

            precomputed_msas_widget = SelectPrecomputedMsasWidget()
            precomputed_msas_widget.btn_precomputed_msas.clicked.connect(
                lambda checked, a=precomputed_msas_widget.lei_precomputed_msas, b=precomputed_msas_widget: self.OnBtnSelectFolder(a, b))
            self.sequence_params_widgets["precomputed_msas_list"].append(precomputed_msas_widget)
        logger.debug(self.sequence_params_widgets)

    def add_widgets_to_cells(self, seq_names):
        logger.debug(self.sequence_params_widgets['custom_template_list'])
        for i, seq_name in enumerate(seq_names):
            self.ctrl.setItem(i, 0, QtWidgets.QTableWidgetItem(seq_name))
            self.ctrl.setCellWidget(i, 1, self.sequence_params_widgets['custom_template_list'][i])
            self.ctrl.setCellWidget(i, 2, self.sequence_params_widgets['precomputed_msas_list'][i])
            self.ctrl.setCellWidget(i, 3, self.sequence_params_widgets['no_msa_list'][i])
            self.ctrl.setCellWidget(i, 4, self.sequence_params_widgets['no_template_list'][i])
            self.ctrl.resizeColumnsToContents()

    def read_sequences(self, sequence_ctrl, sess=None):
        self.reset_ctrl()
        self.sequence_params_values = {k: [] for k in self.sequence_params_template.keys()}
        self.sequence_params_widgets = {k: [] for k in self.sequence_params_template.keys()}
        sequence_str = sequence_ctrl.toPlainText()
        sequence_str, error_msgs = self.sanitize_sequence_str(sequence_str)
        seq_names = self.get_seq_names(sequence_str)

        self.ctrl.setRowCount(len(seq_names))

        row_labels = tuple(seq_names)
        self.ctrl.setVerticalHeaderLabels(row_labels)

        logger.debug("Row Labels")
        logger.debug(row_labels)

        # Create Columns
        self.create_widgets(seq_names)
        self.add_widgets_to_cells(seq_names)

        return seq_names, error_msgs

    def get_from_table(self, seq_names):
        self.sequence_params_values = {k: [] for k in self.sequence_params_template.keys()}
        for i, _ in enumerate(seq_names):
            for key in self.sequence_params_widgets.keys():
                if key == "custom_template_list":
                    value = self.sequence_params_widgets[key][i].lei_custom_template.text()
                    self.sequence_params_values[key].append(value)
                elif key == "precomputed_msas_list":
                    value = self.sequence_params_widgets[key][i].lei_precomputed_msas.text()
                    self.sequence_params_values[key].append(value)
                else:
                    self.sequence_params_values[key].append(str(self.sequence_params_widgets[key][i].isChecked()))



        return (','.join(['None' if x == "" else x for x in self.sequence_params_values['custom_template_list']]),
                ','.join(['None' if x == "" else x for x in self.sequence_params_values['precomputed_msas_list']]),
                ','.join(self.sequence_params_values['no_msa_list']),
                ','.join(self.sequence_params_values['no_template_list']),
                ','.join(seq_names))



    def set_values(self, seq_names):
        logger.debug("Set values")
        for i, _ in enumerate(seq_names):
            for key in self.sequence_params_values.keys():
                logger.debug(f"i {i} key {key} {self.sequence_params_values[key][i]}")
                if key == "custom_template_list":
                    self.sequence_params_widgets[key][i].lei_custom_template.setText(self.sequence_params_values[key][i])
                elif key == "precomputed_msas_list":
                    self.sequence_params_widgets[key][i].lei_precomputed_msas.setText(self.sequence_params_values[key][i])
                else:
                    val = self.sequence_params_values[key][i]
                    if val in ["1", "True"]:
                        val = True
                    else:
                        val = False
                    self.sequence_params_widgets[key][i].setChecked(val)



    def update_from_db(self, db_result, other=None):

        if not db_result.job_name is None:
            self.sequence_params_values = {k: [] for k in self.sequence_params_template.keys()}
            self.sequence_params_widgets = {k: [] for k in self.sequence_params_template.keys()}
            self.reset_ctrl()
            logger.debug(db_result)
            seq_names = db_result.seq_names.split(',')
            self.ctrl.setRowCount(len(seq_names))
            if not db_result.custom_template_list is None:
                self.sequence_params_values['custom_template_list'] = ['None' if x == '' else x for x in db_result.custom_template_list.split(',')]
            else:
                self.sequence_params_values['custom_template_list'] = ['None' for _ in seq_names]
            if not db_result.precomputed_msas_list is None:
                self.sequence_params_values['precomputed_msas_list'] = ['None' if x == '' else x for x in db_result.precomputed_msas_list.split(',')]
            else:
                self.sequence_params_values['precomputed_msas_list'] = ['None' for _ in seq_names]
            if not db_result.no_msa_list is None:
                self.sequence_params_values['no_msa_list'] = db_result.no_msa_list.split(',')
            if not db_result.no_template_list is None:
                self.sequence_params_values['no_template_list'] = db_result.no_template_list.split(',')
            self.create_widgets(seq_names)
            self.set_values(seq_names)
            self.add_widgets_to_cells(seq_names)



class GUIVariables:
    """ Functions shared by GUI associated variables. Inherited by """
    def set_controls(self, ui, db_table):
        logger.debug("Setting controls")
        for var in vars(self):
            logger.debug(var)
            
            obj = getattr(self, var)
            logger.debug(obj)
            if hasattr(obj, 'ctrl_type') and not obj.ctrl_type is None:
                logger.debug(f"{obj.ctrl_type}_{db_table}_{obj.var_name}")
                #if obj.ctrl is None:
                ctrl = ui.findChild(ctrl_type_dict[obj.ctrl_type],
                                      '{}_{}_{}'.format(obj.ctrl_type, db_table, obj.var_name))
                logger.debug(ctrl)
                if not ctrl is None:
                    obj.ctrl = ctrl
                    logger.debug(obj.ctrl)
                else:
                    logger.debug("ctrl not found in res")

    def unset_controls(self, ui, db_table):
        logger.debug("Unsetting controls")
        for var in vars(self):
            logger.debug(var)

            obj = getattr(self, var)
            logger.debug(obj)
            if hasattr(obj, 'ctrl_type') and not obj.ctrl_type is None:
                obj.ctrl = None

    def delete_controls(self, var_obj):
        logger.debug("Deleting controls")
        for var in vars(self):
            logger.debug(var)

            obj = getattr(self, var)
            if hasattr(obj, 'var_name'):
                logger.debug(f"self object {obj.var_name}")
            for var_ in vars(var_obj):
                logger.debug(f"db object {var_}")
                if hasattr(obj, 'ctrl'):
                    if obj.var_name == var_:
                        logger.debug(f"Deleting {obj.var_name}")
                        obj.ctrl = None

    def update_from_self(self):
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'ctrl') and hasattr(obj, 'value'):
                obj.update_from_self()


    # Read values from gui controls and update variables
    def update_from_gui(self):
        logger.debug("Update from GUI")
        for var in vars(self):
            logger.debug(f"Update {var}")
            obj = getattr(self, var)
            logger.debug(obj)
            if hasattr(obj, 'ctrl'):
                obj.update_from_gui()

    def reset_ctrls(self):
        logger.debug("Reset ctrls")
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'ctrl'):
                obj.reset_ctrl()

    # Get values from DB for respective job and update gui controls
    def update_from_db(self, db_result, other=None):
        logger.debug("========>>> Update from DB")
        if not db_result is None and not db_result == []:
            for var in vars(self):
                obj = getattr(self, var)
                if hasattr(obj, 'ctrl'):
                    #Update only if var is in other
                    if not other is None:
                        logger.debug("Updating only variables in \"other\" object")
                        for var_ in vars(other):
                            logger.debug(f"params var {var}, other var {var_}")
                            if var_ == var:
                                obj.update_from_db(db_result)
                    else:
                        obj.update_from_db(db_result)
        else:
            logger.warning("DB result empty. Nothing to update.")

    def update_from_default(self, default_values):
        logger.debug("========>>> Update from Default")
        if not default_values is None and not default_values == []:
            for var in vars(self):
                obj = getattr(self, var)
                if hasattr(obj, 'ctrl') and obj.var_name in vars(default_values):
                    obj.update_from_db(default_values)
        else:
            logger.warning("Default values empty. Nothing to update.")

    def get_dict_run_job(self):
        job_dict = {}
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'db') and (hasattr(obj, 'ctrl') or hasattr(obj, 'file_type')):
                if not obj.is_set():
                    obj.value = None
                job_dict[obj.var_name] = obj.value
        return job_dict

    def get_dict_cmd(self, foreign_obj=None):
        cmd_dict = {}
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'cmd'):
                if obj.cmd is True:
                    if hasattr(obj, 'ctrl_type'):
                        if obj.ctrl_type == 'chk':
                            if not isinstance(obj.value, bool):
                                if not obj.value is None:
                                    if obj.value.lower() == 'true':
                                        cmd_dict[obj.var_name] = ""
                            else:
                                if obj.value:
                                    cmd_dict[obj.var_name] = ""
                        else:
                            if obj.is_set():
                                cmd_dict[obj.var_name] = obj.value
                    else:
                        if obj.is_set():
                            cmd_dict[obj.var_name] = obj.value
        return cmd_dict

    def get_dict_db_insert(self, foreign_obj=None):
        insert_dict = {}
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'db'):
                if obj.db is True and obj.db_primary_key is False and obj.db_foreign_key is None and obj.db_relationship is None:
                    logger.debug(f"dict db insert var {var} value {obj.value}")
                    if obj.is_set():
                        insert_dict[obj.var_name] = obj.value
                elif not obj.db_relationship is None:
                    if not foreign_obj is None:
                        insert_dict[obj.var_name] = foreign_obj
        return [insert_dict]


class Evaluation(GUIVariables):
    def __init__(self):
        self.db = None
        self.db_table = 'evaluation'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.job_id = Variable('job_id', 'int', db_foreign_key='job.id')
        self.results_path = Variable('results_path', 'str')
        self.results = Variable("results", db=False, ctrl_type='web')
        self.pbar = Variable("pbar", db=False, ctrl_type='pro')

    def set_db(self, db):
        self.db = db

    def check_exists(self, job_id, sess):
        result = sess.query(self.db.Evaluation).filter_by(job_id=job_id).all()
        if result == [] or result == None:
            return False
        else:
            return True


    def get_dict_db_insert(self, foreign_obj=None):
        insert_dict = {}
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'db'):
                if obj.db is True and obj.db_primary_key is False and obj.db_foreign_key is None and obj.db_relationship is None:
                    logger.debug(f"dict db insert var {var} value {obj.value}")
                    if obj.is_set():
                        insert_dict[obj.var_name] = obj.value
                elif not obj.db_relationship is None:
                    if not foreign_obj is None:
                        insert_dict[obj.var_name] = foreign_obj
        return [insert_dict]

    def generate_db_object(self, data):
        if isinstance(data, list):
            return [self.db.Evaluation(**row) for row in data]
        else:
            return [self.db.Evaluation(**data)]

    def get_results_path_by_id(self, job_id, sess):
        logger.debug(f"get result for job id {job_id}")
        result = sess.query(self.db.Evaluation).filter_by(job_id=job_id).one()
        return result.results_path

    def print_page_info(self, ok):
        self.pbar.ctrl.hide()

    def print_load_started(self):
        logger.debug('_____________________________________________________started loading_____________________________________________________')
        self.pbar.ctrl.show()

    def print_load_percent(self, percent):
        self.pbar.ctrl.setValue(int(percent))
        logger.debug(percent)
        if percent == 100:
            self.pbar.ctrl.hide()


    def init_gui(self, gui_params, sess):
        if not gui_params['job_id'] is None:
            results_path = self.get_results_path_by_id(gui_params['job_id'], sess)
            if not results_path is None:
                logger.debug(results_path)
                self.results.ctrl.settings().setAttribute(QtWebEngineWidgets.QWebEngineSettings.WebGLEnabled, False)
                logger.debug(f"WebGL enabled: {self.results.ctrl.settings().testAttribute(QtWebEngineWidgets.QWebEngineSettings.WebGLEnabled)}")
                logger.debug(f"QMLSCENE_DEVICE: {os.environ['QMLSCENE_DEVICE']}")

                self.results.ctrl.loadStarted.connect(self.print_load_started)
                self.results.ctrl.loadProgress.connect(self.print_load_percent)
                self.results.ctrl.loadFinished.connect(self.print_page_info)
                self.results.ctrl.load(QtCore.QUrl(f'file://{results_path}'))
                #self.results.ctrl.setUrl(QtCore.QUrl(f'file://{results_path}'))

        return gui_params




class JobParams(GUIVariables):
    def __init__(self):
        self.db = None
        self.db_table = 'jobparams'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.job_id = Variable('job_id', 'int', db_foreign_key='job.id')
        #self.job = Variable('job', None, db_relationship='Job')
        self.job_name = Variable('job_name', 'str', ctrl_type='lei')
        self.output_dir = Variable('output_dir', 'str', cmd=True)
        self.sequences = Variable('sequences', 'str', ctrl_type='pte')
        self.seq_names = Variable('seq_names', 'str')
        self.fasta_path = Variable('fasta_path', 'str', cmd=True)
        self.sequence_params = TblCtrlSequenceParams('sequence_params',
                                                      None,
                                                      db=False,
                                                      ctrl_type='tbl')
        self.custom_template_list = Variable('custom_template_list', 'str', cmd=True)
        self.precomputed_msas_list = Variable('precomputed_msas_list', 'str', cmd=True)
        #self.use_precomputed_msas = Variable('use_precomputed_msas', 'bool', ctrl_type='chk', cmd=True)
        #self.continue_from_features = Variable('continue_from_features', 'bool', ctrl_type='chk', cmd=True)
        self.no_msa_list = Variable('no_msa_list', 'str', cmd=True)
        self.no_template_list = Variable('no_template_list', 'str', cmd=True)
        self.run_relax = Variable('run_relax', 'bool', ctrl_type='chk', cmd=True)
        self.num_multimer_predictions_per_model = Variable('num_multimer_predictions_per_model',
                                                           'int', ctrl_type='sbo', cmd=True)
        self.queue = Variable('queue', 'bool', ctrl_type='chk')
        self.db_preset_dict = {0: 'full_dbs',
                               1: 'reduced_dbs',
                               2: 'colabfold'}
        self.db_preset = Variable('db_preset', 'str', ctrl_type='cmb', cmb_dict=self.db_preset_dict, cmd=True)
        self.model_preset_dict = {0: 'automatic',
                                  1: 'monomer',
                                  2: 'monomer_casp14',
                                  3: 'monomer_ptm',
                                  4: 'multimer'}
        self.model_preset = Variable('model_preset', 'str', ctrl_type='cmb', cmb_dict=self.model_preset_dict, cmd=True)
        self.benchmark = Variable('benchmark', 'bool', ctrl_type='chk', cmd=True)
        self.random_seed = Variable('random_seed', 'str', ctrl_type='lei', cmd=True)
        self.max_template_date = Variable('max_template_date', 'str', ctrl_type='lei', cmd=True)
        self.precomputed_msas_path = Variable('precomputed_msas_path', 'str', ctrl_type='lei', cmd=True)
        #self.only_features = Variable('only_features', 'bool', ctrl_type='chk', cmd=True)
        self.force_cpu = Variable('force_cpu', 'bool', ctrl_type='chk')
        self.num_recycle = Variable('num_recycle', 'int', ctrl_type="sbo", cmd=True)
        #self.batch_features = Variable('batch_features', 'bool', ctrl_type='chk', cmd=True)
        self.pipeline_dict = {0: 'full',
                              1: 'only_features',
                              2: 'batch_features',
                              3: 'continue_from_msas',
                              4: 'continue_from_features'}
        self.pipeline = Variable('pipeline', 'str', ctrl_type='cmb', cmb_dict=self.pipeline_dict, cmd=True)


    def set_db(self, db):
        self.db = db

    def set_fasta_paths(self, job_path, job_name):
        self.fasta_path.set_value(os.path.join(job_path, f"{job_name}.fasta"))

    def write_fasta(self):
        with open(self.fasta_path.get_value(), 'w') as f:
            sequence_str = self.sequences.get_value()
            sequence_str = sequence_str.replace(" ", "")
            f.write(sequence_str)

    def db_insert_params(self, sess, data=None):
        assert isinstance(data, list)
        rows = [self.db.Jobparams(**row) for row in data]
        for row in rows:
            sess.merge(row)
        sess.commit()

    def generate_db_object(self, data=None):
        assert isinstance(data, list)
        return [self.db.Jobparams(**row) for row in data]

    def get_params_by_job_id(self, job_id, sess):
        logger.debug(f"get result for job id {job_id}")
        result = sess.query(self.db.Jobparams).filter_by(job_id=job_id).one()
        return result

    def parse_fasta(self, fasta_file):
        sequences = []
        if fasta_file is None:
            raise ValueError("No fasta file defined!")
        elif not os.path.exists(fasta_file):
            raise ValueError("Wrong fasta file path!")
        else:
            record_dict = SeqIO.index(fasta_file, "fasta")
            for record in record_dict.values():
                sequences.append(record.seq)
        return sequences

    def read_sequences(self):
        logger.debug(self.sequences.__dict__)
        logger.debug(self.sequence_params.__dict__)
        seq_names, error_msgs = self.sequence_params.read_sequences(self.sequences.ctrl)
        logger.debug(f"Sequences {self.sequences.get_value()}")
        logger.debug(error_msgs)
        self.custom_template_list.value,\
        self.precomputed_msas_list.value,\
        self.no_msa_list.value,\
        self.no_template_list.value,\
        self.seq_names.value = self.sequence_params.get_from_table(seq_names)
        logger.debug(self.custom_template_list.value)
        logger.debug(self.precomputed_msas_list.value)
        logger.debug(self.job_name.ctrl.text())
        job_name = self.job_name.ctrl.text()
        if self.job_name.ctrl.text() == "":
            job_name = self.seq_names.value.replace(',','_')
            self.job_name.ctrl.setText(job_name)
        self.job_name.set_value(job_name)
        return error_msgs

    #Make sure that the input file only contains one model and one chain and if the chain doesn't have id 'A' rename it.
    #Save the cif with a new 4 letter filename.
    def process_custom_template_files(self, job_dir):
        msgs = []
        new_custom_template_list = []
        for i, template in enumerate(self.custom_template_list.get_value().split(',')):
            logger.debug(f"{i} {template}")
            if not template in [None, "None"]:
                if not template.endswith('.cif'):
                    msgs.append("Custom template needs to be in cif format. Use https://mmcif.pdbj.org/converter to convert.")
                else:
                    parser = MMCIFParser(QUIET=True)
                    structure = parser.get_structure('structure', template)
                    models = list(structure.get_models())
                    if len(models) > 1:
                        msgs.append("More than one model found in custom template. Make sure that only one model and chain is present in the file.")
                    chains = [x.get_id() for x in structure[0].get_chains()]
                    if len(chains) > 1:
                        msgs.append("More than one chain found in custom template. Make sure only the chain that matches the target is present in the file.")
                    elif not 'A' in chains:
                        logger.debug("Chain IDs in template")
                        logger.debug(chains)
                        msgs.append("The template chain needs to have id 'A'.")

                    if i < 10:
                        out_name = f'cus{i}'
                        out_path = os.path.join(job_dir, f'cus{i}.cif')
                    elif i < 100:
                        out_name = f'cu{i}'
                        out_path = os.path.join(job_dir, f'cu{i}.cif')
                    else:
                        msg.append("Too many templates.")
                    new_custom_template_list.append(out_path)
                    new_lines = []
                    with open(template, 'r') as f:
                        lines = f.readlines()
                    for line in lines:
                        if line.startswith("_pdbx_database_status.entry_id"):
                            new_line = re.sub(r"(_pdbx_database_status.entry_id\s+)\S+", rf'\1{out_name}', line)
                            new_lines.append(new_line)
                        else:
                            new_lines.append(line)
                    revision_date_lines = [
                        '_pdbx_audit_revision_history.data_content_type\t"Structure model"\n',
                        '_pdbx_audit_revision_history.major_revision\t1\n',
                        '_pdbx_audit_revision_history.minor_revision\t0\n',
                        '_pdbx_audit_revision_history.revision_date\t1971-01-01\n',
                        '#']
                    new_lines.extend(revision_date_lines)
                    logger.debug(f"{template} saved as {out_path}")
                    with open(out_path, 'w') as f:
                        for line in new_lines:
                            f.write(line)
            else:
                new_custom_template_list.append("None")
        return ','.join(new_custom_template_list), msgs


    def update_from_sequence_table(self):
        self.custom_template_list.value, \
        self.precomputed_msas_list.value, \
        self.no_msa_list.value, \
        self.no_template_list.value, \
        self.seq_names.value = self.sequence_params.get_from_table(self.seq_names.value.split(','))


    def get_name_by_job_id(self, job_id, sess):
        result = sess.query(self.db.Jobparams.job_name).filter_by(job_id=job_id).one()
        logger.debug(result[0])
        logger.debug(result)
        return result[0]

    def set_db_preset(self):
        if not self.db_preset.is_set():
            self.db_preset.set_value('full_dbs')

    def set_max_template_date(self):
        #format 2021-11-02
        cur_date = date.today()
        if not self.max_template_date.is_set():
            self.max_template_date.set_value(cur_date)

    def init_gui(self, gui_params, sess):
        self.sequence_params.init_gui()
        return gui_params


class Job(GUIVariables):
    def __init__(self):
        self.db = None
        self.db_table = 'job'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.project_id = Variable('project_id', 'int', db_foreign_key='project.id')
        self.jobparams = Variable('jobparams', None, db_relationship='Jobparams', db_backref="Job")
        self.evaluation = Variable('evaluation', None, db_relationship='Evaluation', db_backref="Job")
        self.job_project_id = Variable('job_project_id', 'int', db=True)
        self.name = Variable('name', 'str')
        # self.params = Variable('params', 'int', db_relationship='Params')
        # self.name = Variable('name', 'str')
        self.list = TblCtrlJobs('list', None, db=False, ctrl_type='tbl')
        self.timestamp = Variable('timestamp', 'str')
        self.log = Variable('log', 'str', db=False, ctrl_type='pte')
        self.log_file = Variable('log_file', 'str', db=True)
        self.status = Variable('status', 'str', db=True)
        self.pid = Variable('pid', 'str', db=True)
        self.host = Variable('host', 'str', db=True)
        self.path = Variable('path', 'str', db=True)
        self.type = Variable('type', 'str', db=True)

    def set_db(self, db):
        self.db = db

    def get_status(self, job_id, sess):
        result = sess.query(self.db.Job).get(job_id)
        return result.status

    def update_status(self, status, job_id, sess):
        result = sess.query(self.db.Job).get(job_id)
        result.status = status
        sess.commit()

    def get_job_project_id(self, job_id, project_id, sess):
        result = sess.query(self.db.Job).filter_by(id=job_id, project_id=project_id).first()
        if result is None:
            return None
        else:
            return result.job_project_id

    def get_max_job_project_id(self, project_id, sess):
        result = sess.query(self.db.Job).filter_by(project_id=project_id).order_by(self.db.Job.job_project_id.desc()).first()
        if result is None:
            return None
        else:
            return result.job_project_id


    def get_next_job_project_id(self, project_id, sess):
        max_id = self.get_max_job_project_id(project_id, sess)
        if max_id is None:
            logger.debug("No job_project_id found.")
            max_id = 0
        return max_id + 1

    def set_next_job_project_id(self, project_id, sess):
        job_project_id = self.get_next_job_project_id(project_id, sess)
        self.job_project_id.value = job_project_id
        logger.debug(f"job_project_id is {job_project_id}")

    def get_job_id_by_job_project_id(self, job_project_id, project_id, sess):
        result = sess.query(self.db.Job).filter_by(project_id=project_id, job_project_id=job_project_id).first()
        return result.id

    def get_pid(self, job_id, sess):
        result = sess.query(self.db.Job).get(job_id)
        logger.debug(f"Getting PID for job_id {job_id} from DB. PID is {result.pid}")
        return result.pid

    def get_queue_job_id(self, log_file):
        queue_job_id = None
        with open(log_file, 'r') as f:
            lines = f.readlines()
        regex = re.compile("QUEUE_JOB_ID=(\d+)")
        for l in lines:
            if re.search(regex, l):
                queue_job_id = re.search(regex, l).group(1)
        return queue_job_id

    def get_host(self, job_id, sess):
        result = sess.query(self.db.Job).get(job_id)
        return result.host

    def get_jobs_by_project_id(self, project_id, sess):
        result = sess.query(self.db.Job).filter_by(project_id=project_id)
        return result

    def get_type(self, job_params):
        if job_params['pipeline'] in ['continue_from_msas', 'continue_from_features']:
            type = "prediction"
        elif job_params['pipeline'] in ['only_features', 'batch_features']:
            type = "features"
        else:
            type = "full"
        return type

    def get_type_by_job_id(self, job_id):
        result = sess.query(self.db.Job).get(job_id)
        return result.type

    def set_type(self, type):
        self.type.set_value(type)

    #ToDo: check usage and change to var.set()
    def set_project_id(self, project_id):
        self.project_id.value = project_id

    def update_pid(self, pid, job_id, sess):
        logger.debug(f"Updating pid {pid} for job id {job_id}")
        result = sess.query(self.db.Job).get(job_id)
        result.pid = pid
        sess.commit()

    # def convert_protocol(self, cmd_dict):
    #     conversion = {"FastRelax": "1",
    #                   "Backbone Minimization": "2",
    #                   "Allatom Minimization": "3",
    #                   "Automatic rebuilding": "4",
    #                   "Ramachandran-based rebuilding": "5",
    #                   "Only B-factor refinement": "6"}
    #     cmd_dict['protocol'] = conversion[cmd_dict['protocol']]

    def check_queue_submit_cmd(self, cmd):
        if shutil.which(cmd):
            return True
        else:
            return False

    def prepare_submit_script(self, job_params, job_args, estimated_gpu_mem, split_mem):
        msgs = []
        logger.debug("Preparing submit script")
        logger.debug(f"Estimated GPU mem: {estimated_gpu_mem}")
        command = ["run_alphafold.py\\\n"] + job_args
        command = ' '.join(command)
        logfile = job_params["log_file"]
        submit_script = f"{job_params['job_path']}/submit_script_{job_params['type']}.run"

        #templates = pkg_resources.resource_filename("guifold", "templates")
        env = Environment(loader=PackageLoader("guifold", "templates"))
        template = env.get_template('submit_script.j2')
        template_source = env.loader.get_source(env, 'submit_script.j2')
        parsed_content = env.parse(template_source)
        template_vars = meta.find_undeclared_variables(parsed_content)
        logger.debug("Template vars:")
        logger.debug(template_vars)
        to_render = {}

        if 'num_cpus' in template_vars:
            if job_params['pipeline'] == 'continue_from_features':
                to_render['num_cpus'] = 1
            else:
                to_render['num_cpus'] = job_params['num_cpus']
        if 'use_gpu' in template_vars:
            if job_params['pipeline'] == 'only_features' or job_params['force_cpu']:
                to_render['use_gpu'] = False
            else:
                to_render['use_gpu'] = True
            logger.debug(f"{to_render['use_gpu']} {job_params['force_cpu']}")
        if 'add_dependency' in template_vars:
            to_render['add_dependency'] = job_params['split_job']
            # logger.debug(f"only msa {job_params['only_features']}")
            #     if job_params['only_features']:
            #         if job_params['cpu_lane_list']:
            #             logger.debug(f"lanes {job_params['cpu_lane_list']}")
            #             to_render['cpu_lanes'] = job_params['cpu_lane_list']
            #         to_render['use_gpu'] = False
            #     else:
            #         to_render['use_gpu'] = True
            #         if job_params['gpu_lane_list']:
            #             to_render['gpu_lanes'] = job_params['gpu_lane_list']
            #         if 'num_cpus' in template_vars:
            #             to_render['num_cpus'] = 1
        else:
            if job_params['split_job']:
                msgs.append("Job splitting selected in the GUI"
                            " but the submission template is not configured correctly,"
                            " i.e. missing add_dependency parameter. See documentation.")
        if 'queue_job_id' in template_vars:
            to_render['queue_job_id'] = job_params['queue_pid']
        if 'mem' in template_vars:
            if estimated_gpu_mem < job_params['min_ram']:
                ram = job_params['min_ram']
            else:
                if job_params['force_cpu'] or job_params['pipeline'] == 'only_features':
                    ram = job_params['min_ram']
                else:
                    ram = estimated_gpu_mem
            to_render['mem'] = int(ram)
        if 'gpu_mem' in template_vars:
            to_render['gpu_mem'] = estimated_gpu_mem
        # if 'gpu_name' in template_vars:
        #     if not gpu_name is None:
        #         to_render['gpu_name'] = gpu_name
        #     else:
        #         msgs.append("GPU name variable specified in template but no GPU names found."
        #                     " Please specify available GPUs in the settings Dialog"
        #                     " or remove the variable from the template.")
        if 'account' in template_vars:
            if not job_params['queue_account'] is None:
                to_render['account'] = job_params['queue_account']
            else:
                msgs.append("Queue account name variable specified in template but no account found."
                            " Please specify account in the Settings dialog or remove the variable from the template.")
        if 'split_mem' in template_vars:
            to_render['split_mem'] = split_mem
        if 'total_sequence_length' in template_vars:
            to_render['total_sequence_length'] = job_params['total_seqlen']
        #Suitable cluster lanes/partitions should be defined only in the submission script
        # based on use_gpu and total_sequence_length
        # if 'lanes' in template_vars:
        #     if job_params['gpu_lane_list'] or job_params['cpu_lane_list']:
        #         if job_params['only_features'] or job_params['force_cpu']:
        #             if job_params['cpu_lane_list']:
        #                 to_render['lanes'] = job_params['cpu_lane_list']
        #         else:
        #             if job_params['gpu_lane_list']:
        #                 to_render['lanes'] = job_params['gpu_lane_list']
        #             else:
        #                 to_render['lanes'] = job_params['cpu_lane_list']
        #     else:
        #         msgs.append("Lanes variable specified in template but no lanes found. "
        #                     "Please specify GPU and/or CPU lanes in the Settings dialog or remove"
        #                     " the variable form the template.")

        rendered = template.render(**to_render,
                                   command=command,
                                   logfile=logfile)

        with open(submit_script, 'w') as f:
            f.write(rendered)
        return submit_script, msgs

    #Calculate required gpu memory in GB by sequence length
    def calculate_gpu_mem(self, total_seq_length):
        logger.debug(total_seq_length)
        mem = int(math.ceil(5.561*math.exp(0.00095881*total_seq_length)))
        logger.debug(f"Calculated memory: {mem}")
        return mem

    #Try to get GPU memory from host. Not available when submitted to queue.
    def get_gpu_mem(self):
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return float(info.total) / 10**9

    def get_max_ram(self):
        mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        mem_gb = int(mem/(1024.**3))
        return mem_gb

    def prepare_cmd(self, job_params, cmd_dict, split_job_step=None):
        #cmd_dict = job_params.copy()
        error_msgs, warn_msgs = [], []
        logger.debug(cmd_dict)
        #self.convert_protocol(cmd_dict)
        logger.debug(cmd_dict)
        job_args = []
        if 'model_preset' in cmd_dict:
            if any([cmd_dict['model_preset'] is None, cmd_dict['model_preset'] == 'automatic']):
                if job_params['multimer']:
                    job_args.append('--model_preset multimer')
                else:
                    job_args.append('--model_preset monomer_ptm')
            else:
                job_args.append(f"--model_preset {cmd_dict['model_preset']}")
            del cmd_dict['model_preset']
        else:
            if job_params['multimer']:
                job_args.append('--model_preset multimer')
            else:
                job_args.append('--model_preset monomer_ptm')
        job_args.extend(['--{} {}\\\n'.format(k, v) if i < len(cmd_dict)-1 else '--{} {}'.format(k, v) for i, (k, v) in enumerate(cmd_dict.items())])
        logger.debug(job_args)
        #job_args = [re.sub(r'\sTrue', '', x) for x in job_args if not x is None if not re.search(r'\sFalse$', x)]


        #Switch to unified memory if GPU memory is not enough for given sequence length
        estimated_gpu_mem = self.calculate_gpu_mem(job_params['total_seqlen'])
        #gpu_name = None
        #gpu_mem = None
        if not 'max_ram' in job_params:
            job_params['max_ram'] = self.get_max_ram()

        #Decide which GPUs to use if several are available on a cluster
        if not any([job_params['force_cpu'],
                    job_params['pipeline'] in ['only_features', 'batch_features']]):
            gpu_mem = None
        else:
            if job_params['queue']:
                gpu_mem = job_params['max_gpu_mem']
            else:
                gpu_mem = self.get_gpu_mem()

        #Increase RAM for mmseqs caching, approximately half of the database size should be sufficient
        if job_params['db_preset'] == 'colabfold':
            if job_params['pipeline'] in ['full', 'only_features', 'batch_features']:
               if job_params['max_ram'] < 500:
                   job_params['min_ram'] = job_params['max_ram']
               else:
                   job_params['min_ram'] = 500

        split_mem = None
        if not any([gpu_mem is None,
                    estimated_gpu_mem is None,
                    job_params['force_cpu'],
                    job_params['pipeline'] in ['only_features', 'batch_features']]):
            if estimated_gpu_mem > gpu_mem:
                if estimated_gpu_mem > job_params['max_ram']:
                    error_msgs.append(f"The estimated memory of {estimated_gpu_mem} GB for a total sequence length of {job_params['total_seqlen']}"
                                      f" is larger than the maximum availabe GPU memory ({gpu_mem}) GB"
                                      f" and system RAM ({job_params['max_ram']} GB).")
                else:
                    split_mem = estimated_gpu_mem / gpu_mem
                    warn_msgs.append(f"The estimated memory of {estimated_gpu_mem} GB for a total sequence length of {job_params['total_seqlen']}"
                                f" is larger than the availabe GPU memory ({gpu_mem} GB). Confirm to run the job with "
                                f"unified memory (slow; job can only be cancelled from command line)?")
        if job_params['queue']:
            queue_submit = job_params['queue_submit']
            submit_script, more_msgs = self.prepare_submit_script(job_params, job_args, estimated_gpu_mem, split_mem)
            error_msgs.extend(more_msgs)
            with open(job_params['log_file'], 'a') as log_handle, open(submit_script, 'r') as submit_script_handle:
                log_handle.write(submit_script_handle.read())
            cmd = [queue_submit, submit_script]
        else:
            cmd = []
            if not split_mem is None and not job_params['force_cpu']:
                cmd = [f'export TF_FORCE_UNIFIED_MEMORY=True; export XLA_PYTHON_CLIENT_MEM_FRACTION={split_mem}; ']
            #cmd = [f"/bin/bash -c \'echo test > {job_params['log_file']}\'"]
            if job_params['pipeline'] in ['only_features', 'batch_features'] or job_params['force_cpu']:
                cmd = ['export CUDA_VISIBLE_DEVICES=""; '] + cmd
            bin_path = os.path.join(sys.exec_prefix, 'bin')
            cmd += [f"run_alphafold.py\\\n"] + job_args + [f">> {job_params['log_file']} 2>&1"]
        logger.debug("Job command\n{}".format(cmd))
        return cmd, error_msgs, warn_msgs, estimated_gpu_mem

    def insert_evaluation(self, _evaluation, job_params, sess):
        if not _evaluation.check_exists(job_params['job_id'], sess):
            job_obj = self.get_job_by_id(job_params['job_id'], sess)
            
            
            results_path = os.path.join(job_params['job_path'],
                                                  job_params["job_name"],
                                                  "results.html")
            if os.path.exists(results_path):
                _evaluation.results_path.set_value(results_path)
                _evaluation.job_id.set_value(job_params['job_id'])
                evaluation_dict_db = _evaluation.get_dict_db_insert()
                
                evaluation_obj = _evaluation.generate_db_object(evaluation_dict_db)
                job_obj.evaluation_collection.extend(evaluation_obj)
                
                sess.commit()
                return True

            else:
                return False
        else:
            return True

    def get_job_status(self, log_file):
        exit_code = None
        status_dict = {"db_search_started": False,
                       "model_1_started": False,
                       "model_2_started": False,
                       "model_3_started": False,
                       "model_4_started": False,
                       "model_5_started": False,
                       "evaluation_started": False,
                        "finished": False}
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                pattern_exit_code = re.compile(r'Exit\scode\s(\d+)')
                cancelled_pattern = re.compile(r'CANCELLED')
                pattern_db_search =  re.compile(r"Predicting\s\w+")
                pattern_model_1 =  re.compile(r"Running\smodel\s\w+1")
                pattern_model_2 =  re.compile(r"Running\smodel\s\w+2")
                pattern_model_3 =  re.compile(r"Running\smodel\s\w+3")
                pattern_model_4 =  re.compile(r"Running\smodel\s\w+4")
                pattern_model_5 =  re.compile(r"Running\smodel\s\w+5")
                pattern_finished =  re.compile(r"Alphafold pipeline completed")
                if re.search(pattern_exit_code, line):
                    exit_code = int(re.search(pattern_exit_code, line).group(1))
                if re.search(cancelled_pattern, line):
                    exit_code = 2
                if re.search(pattern_db_search, line):
                    status_dict['db_search_started'] = True
                if re.search(pattern_model_1, line):
                    status_dict['model_1_started'] = True
                if re.search(pattern_model_2, line):
                    status_dict['model_2_started'] = True
                if re.search(pattern_model_3, line):
                    status_dict['model_3_started'] = True
                if re.search(pattern_model_4, line):
                    status_dict['model_4_started'] = True
                if re.search(pattern_model_5, line):
                    status_dict['model_5_started'] = True
                if re.search(pattern_model_5, line):
                    status_dict['evaluation_started'] = True
                if re.search(pattern_finished, line):
                    status_dict['finished'] = True
        except Exception:
            logger.debug(traceback.print_exc())
            pass

        return exit_code, status_dict

    def db_insert_job(self, sess=None, data=None):
        assert isinstance(data, list)
        rows = [self.db.Job(**row) for row in data]
        for row in rows:
            sess.merge(row)
        sess.commit()

    def delete_job(self, job_id, sess):
        sess.query(self.db.Job).filter_by(id=job_id).delete()
        #Cascading delete not yet working
        sess.query(self.db.Jobparams).filter_by(job_id=job_id).delete()
        sess.query(self.db.Evaluation).filter_by(job_id=job_id).delete()
        sess.commit()

    def delete_job_files(self, job_id, path, sess):
        self.delete_job(job_id, sess)
        rmtree(path)

    def set_timestamp(self):
        self.timestamp.value = datetime.datetime.now()

    def set_host(self):
        self.host.value = socket.gethostname()

    def hostname(self):
        return socket.gethostname()

    def generate_db_object(self, data=None):
        assert isinstance(data, list)
        return [self.db.Job(**row) for row in data]

    def get_job_by_id(self, job_id, sess):
        result = sess.query(self.db.Job).filter_by(id=job_id).one()
        return result

    def get_jobs(self, sess):
        result = sess.query(self.db.Job)
        return result

    def read_log(self, log_file):
        lines = []
        with open(log_file, 'r') as log:
            lines = log.readlines()
        return lines

    def update_log(self, gui_params):
        if 'log_file' in gui_params:
            logger.debug(f"Log file: {gui_params['log_file']}")
            if os.path.exists(gui_params['log_file']):
                self.log.reset_ctrl()
                lines = self.read_log(gui_params['log_file'])
                for line in lines:
                    self.log.ctrl.appendPlainText(line.strip('\n'))


    def get_path_by_project_id(self, project_id, sess):
        result = sess.query(self.db.Project.path).filter_by(id=project_id).first()
        return result[0]

    #Name of jobdir in project folder
    def get_job_dir(self, job_name):
        job_dir = job_name
        return job_dir

    #Full path to job dir
    def get_job_path(self, project_path, job_dir):
        job_path = os.path.join(project_path, job_dir)
        return job_path

    def build_log_file_path(self, project_path, job_name, type):
        job_dir = self.get_job_dir(job_name)
        job_path = self.get_job_path(project_path, job_dir)
        log_file = os.path.join(job_path, f"{job_name}_{type}.log")
        return log_file

    def get_log_file(self, job_id, sess):
        result = sess.query(self.db.Job.log_file).filter_by(id=job_id).first()
        return result[0]

    def get_queue_pid(self, log_file, job_id, sess):
        pid = None
        regex = re.compile(r'QUEUE_JOB_ID=(\d+)')
        with open(log_file, 'r') as f:
            content = f.read()
        if re.search(regex, content):
            pid = re.search(regex, content).group(1)
        if not pid is None:
            self.update_pid(pid, job_id, sess)
        logger.debug(f"pid from log file is {pid}.")
        return pid


    def get_log_file_by_id(self, job_id, sess):
        result = sess.query(self.db.Job).filter_by(id=job_id).one()
        return result.log_file

    def set_log_file(self, log_file):
        self.log_file.value = log_file

    def set_status(self, status):
        self.status.value = status

    def check_pid(self, pid):
        try:
            os.kill(int(pid), 0)
            logger.debug("PID exists")
            return True
        except Exception:
            logger.debug("PID not found")
            return False

    def reconnect_jobs(self, sess):
        jobs_running = []
        result = sess.query(self.db.Job).filter((self.db.Job.status=="running") | (self.db.Job.status=="started")).all()

        for job in result:
            jobparams = sess.query(self.db.Jobparams).filter(self.db.Jobparams.id == job.id).one()
            jobs_running.append({'job_id': job.id,
                                 'queue': jobparams.queue,
                                 'status': job.status,
                                 'job_path': job.path,
                                 'log_file': job.log_file,
                                 'pid': job.pid,
                                 'time_started': job.timestamp})
        return jobs_running

    def init_gui(self, gui_params, sess=None):
        logger.debug("=== Init Job list ===")
        # Clear Lists
        self.list.reset_ctrl()
        self.log.reset_ctrl()
        logger.debug("reset end")
        # Fill job list
        self.list.ctrl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.list.ctrl.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self.list.ctrl.setColumnCount(4)
        self.list.ctrl.verticalHeader().setVisible(False)
        self.list.ctrl.setHorizontalHeaderLabels(('ID', 'Name', 'Type', 'Status'))
        self.list.ctrl.setColumnWidth(0, 50)
        self.list.ctrl.setColumnWidth(1, 100)
        self.list.ctrl.setColumnWidth(2, 70)
        self.list.ctrl.setColumnWidth(3, 70)
        project_id = gui_params['project_id']
        if not project_id is None:
            gui_params['project_path'] = self.get_path_by_project_id(project_id, sess)
            jobs = self.get_jobs_by_project_id(project_id, sess)
            for job in jobs:
                status = self.get_status(job.id, sess)
                if status is None:
                    status = "unknown"


                rows = self.list.ctrl.rowCount()
                self.list.ctrl.insertRow(rows)
                self.list.ctrl.setItem(rows, 0, QtWidgets.QTableWidgetItem(str(job.job_project_id)))
                self.list.ctrl.setItem(rows, 1, QtWidgets.QTableWidgetItem(job.name))
                self.list.ctrl.setItem(rows, 2, QtWidgets.QTableWidgetItem(job.type.capitalize()))
                self.list.ctrl.setItem(rows, 3, QtWidgets.QTableWidgetItem(status))
            self.list.ctrl.scrollToItem(self.list.ctrl.item(self.list.ctrl.currentRow(), 0), QtWidgets.QAbstractItemView.PositionAtCenter)
            #self.list.ctrl.scrollToBottom()
        return gui_params


class Project(GUIVariables):
    def __init__(self):
        self.db = None
        self.db_table = 'project'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.jobs = Variable('jobs', None, db_relationship='Job')
        self.name = Variable('name', 'str', ctrl_type='lei')
        self.path = Variable('path', 'str', ctrl_type='lei')
        self.list = Variable('list', None, db=False, ctrl_type='cmb')

        self.active = Variable('active', 'bool', ctrl_type=None)

    def set_db(self, db):
        self.db = db

    def insert_project(self, data, sess):
        assert isinstance(data, list)
        rows = [self.db.Project(**row) for row in data]
        for row in rows:
            sess.merge(row)
        sess.commit()

    def check_if_exists(self, project_name, sess):
        exists = False
        result = self.get_projects(sess)
        for row in result:
            if project_name == row.name:
                exists = True
        return exists

    def delete_project(self, project_id, sess):
        sess.query(self.db.Project).filter(self.db.Project.id == project_id).delete()
        #Cascading delete not yet working
        job_ids = sess.query(self.db.Job.id).filter_by(project_id=project_id).all()
        sess.query(self.db.Job).filter_by(project_id=project_id).delete()
        logger.debug("Job IDs to delete")
        logger.debug(job_ids)
        for job_id in job_ids:
            job_id = job_id[0]
            sess.query(self.db.Jobparams).filter_by(job_id=job_id).delete()
            sess.query(self.db.Evaluation).filter_by(job_id=job_id).delete()
        sess.commit()

    def is_empty(self, sess):
        if sess.query(self.db.Project).all() == []:
            return True
        else:
            return False

    def get_project_by_id(self, project_id, sess):
        return sess.query(self.db.Project).get(project_id)

    def get_active_project(self, sess):
        try:
            project_name, project_id = sess.query(self.db.Project.name, self.db.Project.id).filter_by(
                active=True).one()
        except NoResultFound:
            project_name, project_id = None, None
        return project_name, project_id

    def get_projects(self, sess):
        result = sess.query(self.db.Project).all()
        return result

    def update_project(self, project_id, data, sess,):
        assert isinstance(data, list)
        result = sess.query(self.db.Project).get(project_id)
        for k, v in data[0].items():
            setattr(result, k, v)
        sess.commit()

    def change_active_project(self, new_project, sess, new_active_id=None):
        """
        Change ative project in DB
        :param new_project:
        :return:
        """

        try:
            last_active = sess.query(self.db.Project).filter_by(active=True).one()
            last_active_id = int(last_active.id)
        except:
            last_active_id = -1

        all = sess.query(self.db.Project).all()

        if new_active_id is None:
            new_active = sess.query(self.db.Project).filter_by(name=new_project).one()
            new_active_id = int(new_active.id)

        if not last_active_id == new_active_id:
            # if not last_active_id == -1:
            #    last_active.active = False
            for row in all:
                row.active = False
            new_active.active = True
        sess.commit()

        #result = sess.query(self.db.Project).all()

        return new_active_id

    def get_path_by_project_id(self, project_id, sess):
        result = sess.query(self.db.Project).filter_by(id=project_id).first()
        return result.path

    def init_gui(self, gui_params, sess=None):
        logger.debug("=== Init Projects ===")
        projects = self.get_projects(sess)
        # if projects == []:
        #     base_project = [{'name': 'base', 'path': os.getcwd(), 'active': True}]
        #     self.insert_project(base_project)
        #     projects = self.get_projects()
        # logger.debug("Projects")
        # logger.debug(projects)
        if not projects == []:
            self.list.reset_ctrl()
            for item in projects:
                if self.list.ctrl.findText(item.name) == -1:
                    self.list.ctrl.addItem(item.name)
                else:
                    logger.warning("{} not found.".format(item.name))
            name, id = self.get_active_project(sess)
            logger.debug(f"Active project {name}")
            #Handle case when active project was deleted
            if not name is None:
                gui_params['project_path'] = self.get_path_by_project_id(id, sess)
                projects = self.get_projects(sess)
                # name = projects[0].name
                # id = projects[0].id
                index = self.list.ctrl.findText(name, QtCore.Qt.MatchFixedString)
                logger.debug(f"currently selected project index {index}")
                if index >= 0:
                    self.list.ctrl.setCurrentIndex(index)
                gui_params['project_id'] = id
            return gui_params
        else:
            gui_params['project_id'] = None
            logger.debug("No project found.")
            return gui_params


class Settings(GUIVariables):
    def __init__(self):
        self.db = None
        self.db_table = 'settings'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.queue_submit = Variable('queue_submit', 'str', ctrl_type='lei')
        self.queue_cancel = Variable('queue_cancel', 'str', ctrl_type='lei')
        self.queue_account = Variable('queue_account', 'str', ctrl_type='lei')
        self.num_cpus = Variable('num_cpus', 'int', ctrl_type='sbo', cmd=True)
        self.max_gpu_mem = Variable('max_gpu_mem', 'int', ctrl_type='sbo')
        self.split_job = Variable('split_job', 'bool', ctrl_type='chk')
        self.min_ram = Variable('min_ram', 'int', ctrl_type='sbo')
        self.max_ram = Variable('max_ram', 'int', ctrl_type='sbo')
        #self.queue_submit_dialog = Variable('queue_submit_dialog', 'bool', ctrl_type='chk')
        self.queue_jobid_regex = Variable('queue_jobid_regex', 'str', ctrl_type='lei')
        self.queue_default = Variable('queue_default', 'bool', ctrl_type='chk')
        self.jackhmmer_binary_path =  Variable('jackhmmer_binary_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.hhblits_binary_path = Variable('hhblits_binary_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.mmseqs_binary_path = Variable('mmseqs_binary_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.hhsearch_binary_path = Variable('hhsearch_binary_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.hmmsearch_binary_path = Variable('hmmsearch_binary_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.hmmbuild_binary_path = Variable('hmmbuild_binary_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.hhalign_binary_path = Variable('hhalign_binary_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.kalign_binary_path = Variable('kalign_binary_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.data_dir = Variable('data_dir', 'str', ctrl_type='lei', cmd=True, required=True)
        self.uniref90_database_path = Variable('uniref90_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.uniref30_database_path = Variable('uniref30_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.colabfold_envdb_database_path = Variable('colabfold_envdb_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.mgnify_database_path = Variable('mgnify_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.bfd_database_path = Variable('bfd_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.small_bfd_database_path = Variable('small_bfd_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.uniclust30_database_path = Variable('uniclust30_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.uniprot_database_path = Variable('uniprot_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.pdb70_database_path = Variable('pdb70_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.pdb_seqres_database_path = Variable('pdb_seqres_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.template_mmcif_dir = Variable('template_mmcif_dir', 'str', ctrl_type='lei', cmd=True, required=True)
        self.obsolete_pdbs_path = Variable('obsolete_pdbs_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.custom_tempdir = Variable('custom_tempdir', 'str', ctrl_type='lei', cmd=True)
        self.use_gpu_relax = Variable('use_gpu_relax', 'bool', ctrl_type='chk', cmd=True)
        self.global_config = Variable('global_config', 'bool', ctrl_type='chk')

    def set_db(self, db):
        self.db = db

    def init_gui(self, gui_params, sess):
        return gui_params

    def get_from_db(self, sess):
        result = sess.query(self.db.Settings).get(1)
        return result

    def check_required_settings(self):
        unset_setting = False
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'required'):
                if obj.required:
                    if obj.value is None or obj.value == '':
                        unset_setting = True
                        logger.debug("obj.var_name is required but not set.")
        return unset_setting


    def get_slurm_account(self):
        account = None
        try:
            account_raw = check_output("sacctmgr show User $(whoami) -p -n &> /dev/null", shell=True)
            account_raw = account_raw.decode()
            logger.debug(f'Output from sacctmgr: {account_raw}')
            if not account_raw is None:
                account = account_raw.split("|")[1]
        except Exception as e:
            logger.debug("Could not retrieve SLURM account. You can ignore this if you are not using accounts or a different queueing system")
            logger.debug(traceback.print_exc())
        return account

    def set_slurm_account(self, account, sess):
        logger.debug(f"Updating slurm account {account}")
        result = sess.query(self.db.Settings).get(1)
        result.queue_account = account
        sess.commit()


    def update_from_global_config(self):
        msgs = []
        logger.debug("update from global config")
        config_file = pkg_resources.resource_filename('guifold.config', 'guifold.conf')
        logger.debug(config_file)
        if os.path.exists(config_file):
            config = configparser.ConfigParser()
            config.read(config_file)
            config_keys = config.keys()
            for section in ['QUEUE', 'DATABASES', 'BINARIES', 'OTHER']:
                if section in config_keys:
                    for key in config[section]:
                        for var in vars(self):
                            if key == var:
                                obj = getattr(self, var)
                                if config[section][key].lower() == 'true':
                                    obj.value = True
                                elif config[section][key].lower() == 'false':
                                    obj.value = False
                                elif not config[section][key] == '':
                                    obj.value = config[section][key]
                else:
                    error_msg = f"{section} section missing from config file"
                    logger.error(error_msg)
                    msgs.append(error_msg)

        return msgs

    # def get_alphafold_path(self, sess):
    #     result = self.get_from_db(sess)
    #     return result.alphafold_path


    # def check_executables(self, sess):
    #     messages = []
    #     settings = self.get_from_db(sess)
    #     if not settings is None:
    #         if not settings.alphafold_path == '':
    #             self.path.register('alphafold', settings.alphafold_path)
    #         else:
    #             self.path.register('alphafold', None)
    #     else:
    #         self.path.register('alphafold', None)
    #     try:
    #         logger.debug("check alphafold")
    #         self.path.set_exec('alphafold', 'run_alphafold.py')
    #         logger.debug("check alphafold")
    #     except (SystemExit, FileNotFoundError):
    #         messages.append("Alphafold executables not found. Check path in settings!")
    #     except (SystemExit, FileNotFoundError):
    #         pass



        exec_dict = {}
        #result = {'exists': False, 'executable': False}
        #
        #
        # results = {}
        # if self.path.get('alphafold') is None:
        #     messages.append("Path to phenix executables not set. Check in settings!")
        # else:
        #     results['alphafold_path'] = self.path.get('alphafold')
        #     self.update_path(self.path.get('alphafold'), 'alphafold', sess)
        # return messages

    def add_blank_entry(self, sess):
        settings = self.get_from_db(sess)
        if settings is None:
            logger.debug("Adding first entry to settings")
            blank = {}
            for var in vars(self):
                logger.debug(var)
                obj = getattr(self, var)
                try:
                    if hasattr(obj, db):
                        if obj.db and not obj.db_primary_key:
                            blank[obj] = ''
                except NameError:
                    pass
                except Exception as e:
                    traceback.print_exc()
            #blank.update({'global_config': False})
            self.db_insert_settings([blank], sess)
            return True
        else:
            return False


    def db_insert_settings(self, data, sess):
        logger.debug("Insert settings into DB")
        logger.debug(data)
        assert isinstance(data, list)
        rows = [self.db.Settings(**row) for row in data]
        for row in rows:
            sess.merge(row)
        sess.commit()

    # def update_queue_submit(self, value, sess):
    #     result = sess.query(self.db.Settings).get(1)
    #     result.queue_submit_dialog = value
    #     sess.commit()

    def update_settings(self, insert_dict, sess):
        logger.debug("Update settings")
        settings = self.get_from_db(sess)
        if settings is None:
            self.db_insert_settings(insert_dict, sess)
        else:
            for key, value in insert_dict[0].items():
                setattr(settings, key, value)
            sess.commit()

class DefaultValues:
    def __init__(self, other):
        self.job_name = None
        self.output_dir = None
        self.db_preset = 'full_dbs'
        self.pipeline = 'full'
        settings = other.settings.get_from_db(other.sess)
        if settings.queue_default:
            self.queue = True
