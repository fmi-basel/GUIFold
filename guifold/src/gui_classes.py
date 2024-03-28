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
from copy import deepcopy
import hashlib
from io import StringIO
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
from guifold.src.db_helper import DBHelper
from guifold.src.gui_dialogs import message_dlg, open_files_and_dirs_dlg
from sqlalchemy.orm.exc import NoResultFound
from guifold.src.gui_threads import LogThread
import traceback
from datetime import date
from shutil import copyfile, rmtree
import configparser
from subprocess import Popen
import nvidia_smi
import math
import shutil
from Bio.PDB import MMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
from typing import Dict, Optional, Tuple, Union, List
from guifold.src.gui_dlg_advanced_params import DefaultValues as AdvancedSettingsDefaults
import sqlalchemy.orm
logger = logging.getLogger('guifold')
#logger.setLevel(logger.DEBUG)

class WebViewer(QtWebEngineWidgets.QWebEngineView):

    def __init__(self, parent=None):
        super(WebViewer, self).__init__(parent)

    def closeEvent(self, qclose_event):
        """Overwrite QWidget method"""
        # Sets the accept flag of the event object, indicates that the event receiver wants the event.
        qclose_event.accept()
        # Schedules this object for deletion, QObject
        self.page().deleteLater()

class NonSelectableGroupItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None):
        super().__init__(parent)

    def flags(self):
        return super().flags() & QtCore.Qt.QNoItemFlags

    def mousePressEvent(self, event):
        pass  # Ignore mouse press events

ctrl_type_dict = {'dsb': QtWidgets.QDoubleSpinBox,
                  'sbo': QtWidgets.QSpinBox,
                  'lei': QtWidgets.QLineEdit,
                  'tbl': QtWidgets.QTableWidget,
                  'cmb': QtWidgets.QComboBox,
                  'pte': QtWidgets.QPlainTextEdit,
                  'chk': QtWidgets.QCheckBox,
                  'web': QtWebEngineWidgets.QWebEngineView,
                  'pro': QtWidgets.QProgressBar,
                  'tre': QtWidgets.QTreeWidget,
                  'lbl': QtWidgets.QLabel}

install_path = os.path.dirname(os.path.realpath(sys.argv[0]))

screening_protocol_names = ['first_vs_all', 'all_vs_all', 'first_n_vs_rest', 'grouped_bait_vs_preys', 'grouped_all_vs_all']

class Variable:
    """
        Base model for GUI controls and DB tables. 

        Arguments:

        var_name (str): the name of the variable.
        type (type or None): the data type of the variable.
        db (bool): a flag indicating if the variable should be saved in the database.
        ctrl_type (str): the type of the GUI control.
        db_primary_key (bool): a flag indicating if the variable is a primary key.
        db_foreign_key (str): the foreign key value.
        db_relationship (str): the type of the database relationship.
        db_backref (str): the name of the back-reference.
        cmb_dict (Dict): a dictionary representing the values of a combo box.
        cmd (bool): a flag indicating if the variable is a command.
        required (bool): a flag indicating if the variable is required.
    """
    def __init__(
            self,
            var_name: str = None,
            type_: Union[type, None] = None,
            db: bool = True,
            ctrl_type: str = None,
            db_primary_key: bool = False,
            db_foreign_key: str = None,
            db_relationship: str = None,
            db_backref: str = None,
            cmb_dict: Dict = None,
            cmd: bool = False,
            required: bool = False):

        self.var_name = var_name
        self.value = None
        self.type = type_
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

    def get_value(self) -> Union[list, str, int, float, bool, None]:
        """Get the value of the variable."""
        return self.value

    def set_ctrl_type(self) -> None:
        """Set the control type."""
        try:
            self.ctrl_type = self.ctrl.__class__.__name__
        except Exception:
            logger.warning("Cannot set control {}".format(self.ctrl.__class__.__name__), exc_info=True)

    def set_value(self, value: Union[list, str, int, float, bool, None]) -> None:
        """ Set the value of the variable."""
        self.value = value

    def set_cmb_by_text(self, value: str) -> None:
        """Set combo box value by text."""
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

    def set_control(self, result_val: Union[str, float, int, bool], result_var: str) -> None:
        """Set control value based on control type."""
        if result_val == 'None' or result_val is None:
            result_val = ''
        if self.ctrl_type == 'dsb':
            if result_val != '':
                self.ctrl.setValue(float(result_val))
        elif self.ctrl_type == 'lei':
            self.ctrl.setText(str(result_val))
        elif self.ctrl_type == 'pte':
            self.ctrl.setPlainText(str(result_val))
        elif self.ctrl_type == 'sbo':
            if result_val != '':
                self.ctrl.setValue(int(result_val))
        elif self.ctrl_type == 'cmb':
            for k, v in self.cmb_dict.items():
                logger.debug(f"Setting cmb, result_val {result_val} value in dict {v}")
                if result_val == v:
                    index = self.ctrl.findText(str(v), QtCore.Qt.MatchFixedString)
                    if index >= 0:
                        self.ctrl.setCurrentIndex(index)
                    else:
                        logger.debug("Item not found")
        elif self.ctrl_type == 'chk':
            if not isinstance(result_val, bool):
                if result_val.lower() == 'true':
                    result_val = True
                elif result_val.lower() == 'false':
                    result_val = False
                else:
                    raise ValueError("Value of controlbox must either be true or false!")
            self.ctrl.setChecked(result_val)
        else:
            logger.error("Control type {} not found for control {}!".format(self.ctrl_type, result_var))
        logger.debug(f"Setting {result_var} to {result_val}")

    def unset_control(self) -> None:
        """Unset control value."""
        if hasattr(self, 'ctrl_type') and not self.ctrl_type is None:
            self.ctrl = None

    def is_set(self) -> bool:
        """Check if a variable is set."""
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

    def list_like_str_not_all_none(self) -> bool:
        """Check if a string, that is list-like, contains at least one non-None value."""
        if self.value:
            lst = self.value.split(',')
            for e in lst:
                if not e in [None, "None", "none", ""]:
                    return True
        return False

    def list_like_str_not_all_false(self) -> bool:
        """Check if a string, that is list-like, contains at least one True value."""
        if self.value:
            lst = self.value.split(',')
            for e in lst:
                if not e in [False, "False", "false"]:
                    return True
        return False

    def update_from_self(self) -> None:
        """Set control value from variable value."""
        if not self.value is None and not self.ctrl is None:
            logger.debug(f"Set value of control {self.var_name} to {self.value}")
            self.set_control(self.value, self.var_name)

    def update_from_db(self, db_result: Union[str, float, int, bool]) -> None:
        """Set/update variable values from DB """
        logger.debug("Update from DB")
        if not db_result is None and not db_result == []:
            for result_var in vars(db_result):
                #logger.debug("DB var: {}\nClass var: {}".format(result_var, self.var_name))
                if result_var == self.var_name:
                    result_val = getattr(db_result, result_var)
                    self.value = result_val
                    logger.debug("Variable name: {}\nValue: {}".format(result_var, result_val))
                    if not self.ctrl is None:
                        if not result_val in ['tbl', 'tre']:
                            if result_val in [None, "None", "none"]:
                                if self.ctrl_type == 'chk':
                                    result_val = False
                                else:
                                    result_val = ""
                            self.set_control(result_val, result_var)
        else:
            logger.warning("DB result empty. Nothing to update.")

    def update_from_gui(self) -> None:
        """Set/update attribute values (self.value) from GUI controls"""
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


    def reset_ctrl(self) -> None:
        """Clear variable values and GUI controls """
        logger.debug("Resetting GUI controls ")
        if not self.ctrl is None:
            logger.debug(f"resetting {self.var_name}")
            if not self.ctrl_type in ['chk', 'tbl', 'tre']:
                self.ctrl.clear()
                self.value = None

            elif self.ctrl_type == 'chk':  # and not self.selected_item is None:
                self.ctrl.setChecked(False)
                self.value = False

            elif self.ctrl_type == 'tbl':
                while self.ctrl.rowCount() > 0:
                    self.ctrl.removeRow(0)
            elif self.ctrl_type == 'tre':
                # Get the model associated with the tree view
                model = self.ctrl.model()

                # Clear the model by removing all rows
                model.removeRows(0, model.rowCount())
        else:
            logger.debug(f"ctrl of {self.var_name} is not bound.")


class TblCtrlJobs(Variable):
    """GUI list control which shows jobs for project"""
    def __init__(self,
                var_name: str = None,
                type: str = None,
                db: bool = True,
                ctrl_type: str = None,
                db_primary_key: str = False,
                db_foreign_key: str = None) -> None:
        super().__init__(var_name,
                        type,
                        db,
                        ctrl_type,
                        db_primary_key,
                        db_foreign_key)
        self.selected_item = None


class WebCtrlEvaluation(Variable):
    """GUI list control which shows the results in a webbrowser"""
    def __init__(self,
                var_name: str = None,
                type: str = None,
                db: bool = True,
                ctrl_type: str = None,
                db_primary_key: str = False,
                db_foreign_key: str = None) -> None:
        super().__init__(var_name,
                        type,
                        db,
                        ctrl_type,
                        db_primary_key,
                        db_foreign_key)


class SelectCustomTemplateWidget(QtWidgets.QWidget):
    """Line edit and button to select custom templates"""
    def __init__(self, parent=None):
        super(SelectCustomTemplateWidget, self).__init__(parent)

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
    """Line edit and button to select precomputed MSAs"""
    def __init__(self, parent=None):
        super(SelectPrecomputedMsasWidget, self).__init__(parent)

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
    """Table view for managing enetered sequences. Inherits `Variable` class."""
    def __init__(self,
                var_name: str = None,
                type: str = None,
                db: bool = True, 
                ctrl_type: str = None,
                db_primary_key: bool = False, 
                db_foreign_key: str = None) -> None:
        super().__init__(var_name, type, db, ctrl_type, db_primary_key, db_foreign_key)
        self.sequence_params_template =  {"custom_template_list": [],
                                          "precomputed_msas_list": [],
                                        "no_msa_list": [],
                                        "no_template_list": []}
        self.sequence_params_values = {k: [] for k in self.sequence_params_template.keys()}
        self.sequence_params_widgets = {k: [] for k in self.sequence_params_template.keys()}

    def register(self, parameter: str) -> None:
        """Register a parameter in the sequence parameter table"""
        self.sequence_dict[parameter.var_name] = parameter

    def get_seq_names(self, sequence_str: str) -> list:
        """Get sequence names from a sequence string"""
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

    def sanitize_sequence_str(self, sequence_str: str) -> Tuple[str, list]:
        """Sanitize sequence string and return a list of error messages"""
        sanitized_sequence, error_msgs = [], []
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
                sanitized_sequence.append(line)
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
                sanitized_sequence.append(line)
        if num_seq_names == 0:
            error_msgs.append("No lines starting with > found. Add a >SomeProteinName line above each sequence.")
        if num_seq_names != num_seqs:
            error_msgs.append("Number of sequence names not matching sequences. Add a >SomeProteinName line above each sequence.")
        sanitized_sequence = '\n'.join(sanitized_sequence)
        return sanitized_sequence, error_msgs

    def init_gui(self, other: object = None, sess: sqlalchemy.orm.Session = None) -> None:
        """Initialize the GUI"""
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

    def OnBtnSelectFile(self, lei: QtWidgets.QLineEdit) -> None:
        """Select a file and write the path to the line edit"""
        dlg = QtWidgets.QFileDialog()
        if dlg.exec_():
            path = dlg.selectedFiles()[0]
            logger.debug(path)
            if len(path) > 1:
                message_dlg('error', 'Only one file or folder can be selected.')
            else:
                lei.setText(path)

    def OnBtnSelectFolder(self,
                        lei: QtWidgets.QLineEdit,
                        widget: QtWidgets.QWidget,
                        check: Union[str, None] = None) -> None:
        """Select a folder and write the path to the line edit"""
        path = QtWidgets.QFileDialog.getExistingDirectory(widget, 'Select Folder')
        if check == 'precomputed_msas':
            self.check_precomputed_msas_folder(path)
        lei.setText(path)

    def OnBtnSelectFolderOrFiles(self,
                                lei: QtWidgets.QLineEdit,
                                widget: QtWidgets.QWidget) -> None:
        """Select a folder or files and write the path to the line edit"""
        path = open_files_and_dirs_dlg(widget, 'Select File or Folder')
        if len(path) == 0:
            message_dlg('error', 'No files or folders selected.')
        elif len(path) > 1:
            message_dlg('error', 'Only one file or folder can be selected.')
        else:
            lei.setText(path[0])

    def create_widgets(self, seq_names: list) -> None:
        """Create widgets for the sequence parameter table"""
        self.sequence_params_widgets = {k: [] for k in self.sequence_params_template.keys()}
        for i, _ in enumerate(seq_names):
            custom_template_widget = SelectCustomTemplateWidget()
            custom_template_widget.btn_custom_template.clicked.connect(
                lambda checked, a=custom_template_widget.lei_custom_template, b=custom_template_widget: self.OnBtnSelectFolderOrFiles(a, b))
            self.sequence_params_widgets["custom_template_list"].append(custom_template_widget)

            chk_nomsa = QtWidgets.QCheckBox()
            chk_nomsa.setStyleSheet("text-align: center; margin-left:50%; margin-right:50%;")
            self.sequence_params_widgets["no_msa_list"].append(chk_nomsa)

            chk_notemplate = QtWidgets.QCheckBox()
            chk_notemplate.setStyleSheet("text-align: center; margin-left:50%; margin-right:50%;")
            self.sequence_params_widgets["no_template_list"].append(chk_notemplate)

            precomputed_msas_widget = SelectPrecomputedMsasWidget()
            precomputed_msas_widget.btn_precomputed_msas.clicked.connect(
                lambda checked, a=precomputed_msas_widget.lei_precomputed_msas, b=precomputed_msas_widget, c='precomputed_msas': self.OnBtnSelectFolder(a, b, c))
            self.sequence_params_widgets["precomputed_msas_list"].append(precomputed_msas_widget)
        logger.debug(self.sequence_params_widgets)

    def add_widgets_to_cells(self, seq_names: list):
        """Add widgets to the cells of the sequence parameter table"""
        logger.debug(self.sequence_params_widgets['custom_template_list'])
        for i, seq_name in enumerate(seq_names):
            self.ctrl.setItem(i, 0, QtWidgets.QTableWidgetItem(seq_name))
            self.ctrl.setCellWidget(i, 1, self.sequence_params_widgets['custom_template_list'][i])
            self.ctrl.setCellWidget(i, 2, self.sequence_params_widgets['precomputed_msas_list'][i])
            self.ctrl.setCellWidget(i, 3, self.sequence_params_widgets['no_msa_list'][i])
            self.ctrl.setCellWidget(i, 4, self.sequence_params_widgets['no_template_list'][i])
            self.ctrl.resizeColumnsToContents()

    def read_sequences(self,
                    sequence_ctrl: QtWidgets.QPlainTextEdit,
                    sess: sqlalchemy.orm.Session = None) -> Tuple[list, list]:
        """Read the sequences from the sequence parameter table"""
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

    def get_from_table(self,
                    seq_names: list) -> Tuple[str, str, str, str, str]:
        """Get the values from the sequence parameter table"""
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

    def set_values(self, seq_names: list) -> None:
        """Set the values of the sequence parameter table"""
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

    def update_from_db(self,
                    db_result: list,
                    other: Union[None, object] = None) -> None:
        """Update the sequence parameter table from the database"""
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

    def check_precomputed_msas_folder(self, path: str) -> None:
        """Check that the precomputed MSAs folder contains the expected files"""
        found = False
        for f in os.listdir(path):
            if re.search("uniref90_hits", f):
                found = True
        if not found:
            message_dlg('error', f'No suitable MSAs files found in {path}. Expected to find at least uniref90_hits.')

class GUIVariables:
    """This class contains common functions shared by GUI associated variables."""
    def set_controls(self, ui: QtWidgets.QMainWindow, db_table: str) -> None:
        """ Set QtWidget references for all `Variable` objects in the given scope."""
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

    def unset_controls(self, ui: QtWidgets.QMainWindow, db_table: str) -> None:
        """Remove QtWidget references from all `Variable` objects in the given scope."""
        logger.debug("Unsetting controls")
        for var in vars(self):
            logger.debug(var)
            obj = getattr(self, var)
            logger.debug(obj)
            if hasattr(obj, 'ctrl_type') and not obj.ctrl_type is None:
                obj.ctrl = None

    def delete_controls(self, var_obj: Variable) -> None:
        """Remove QtWidget references from a given `Variable` object in case it exists in the given scope."""
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

    def update_from_self(self) -> None:
        """Update QtWidget value from `value` paramter of `Variable` objects in the given scope."""
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'ctrl') and hasattr(obj, 'value'):
                obj.update_from_self()

    def update_from_gui(self) -> None:
        """Get QtWidget value and update `value` paramter of `Variable` objects in the given scope."""
        logger.debug("Update from GUI")
        for var in vars(self):
            logger.debug(f"Update {var}")
            obj = getattr(self, var)
            logger.debug(obj)
            if hasattr(obj, 'ctrl'):
                obj.update_from_gui()

    def reset_ctrls(self) -> None:
        """Reset all QtWidget controls in the given scope."""
        logger.debug("Reset ctrls")
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'ctrl'):
                obj.reset_ctrl()

    def update_from_db(self, db_result: list, other=None) -> None:
        """Get values from DB and update QtWidgets."""
        logger.debug("Update from DB")
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

    def update_from_default(self, default_values: object) -> None:
        """Update `value` paramter of `Variable` objects in the given scope to default values."""
        logger.debug("Update from Default")
        if not default_values is None and not default_values == []:
            for var in vars(self):
                obj = getattr(self, var)
                if hasattr(obj, 'ctrl') and obj.var_name in vars(default_values):
                    obj.update_from_db(default_values)
        else:
            logger.warning("Default values empty. Nothing to update.")

    def get_dict_run_job(self) -> dict:
        """Get dict with values from GUI controls and variables in the given scope."""
        job_dict = {}
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'db') and (hasattr(obj, 'ctrl') or hasattr(obj, 'file_type')):
                if not obj.is_set():
                    obj.value = None
                job_dict[obj.var_name] = obj.value
        return job_dict

    def get_dict_cmd(self, foreign_obj: object = None) -> dict:
        """Get dict with values that are required for running a command."""
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

    def get_dict_db_insert(self, foreign_obj: object = None) -> List[Dict]:
        """ Gather names and values from Variable objects and construct a mapping for DB insertion. """
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
    """ Class for managing job evaluation results. Inherits functions from GUIVariables.
    
    Attributes:
        db (str): Database name.
        db_table (str): Database table name.
        id (Variable): Evaluation ID.
        job_id (Variable): Job ID.
        results_path (Variable): Path to results.
        results (Variable): Webview widget.
        pbar (Variable): Progress bar.
        pairwise_combinations_label (Variable): Label for pairwise combinations.
        pairwise_combinations_list (Variable): List of pairwise combinations.
        """
    def __init__(self) -> None:
        self.db = None
        self.db_table = 'evaluation'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.job_id = Variable('job_id', 'int', db_foreign_key='job.id')
        self.results_path = Variable('results_path', 'str')
        self.results = Variable("results", db=False, ctrl_type='web')
        self.pbar = Variable("pbar", db=False, ctrl_type='pro')
        self.pairwise_combinations_label = Variable("pairwise_combinations_label", db=False, ctrl_type='lbl')
        self.pairwise_combinations_list = Variable("pairwise_combinations_list", db=False, ctrl_type='cmb')

    def set_db(self, db: DBHelper) -> None:
        """Set database helper object."""
        self.db = db

    def check_exists(self, job_id: int, sess: sqlalchemy.orm.Session) -> bool:
        """Check if evaluation for given job exists in DB."""
        result = sess.query(self.db.Evaluation).filter_by(job_id=job_id).all()
        if result == [] or result == None:
            return False
        else:
            return True

    def get_dict_db_insert(self, foreign_obj: object = None) -> List[dict]:
        """Gather names and values from Variable objects and construct a mapping for DB insertion."""
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

    def generate_db_object(self, data: list) -> list:
        """Generate DB object from data."""
        if isinstance(data, list):
            return [self.db.Evaluation(**row) for row in data]
        else:
            return [self.db.Evaluation(**data)]

    def get_results_path_by_id(self, job_id: int, sess: sqlalchemy.orm.Session) -> str:
        """Get results path by job ID."""
        logger.debug(f"get result for job id {job_id}")
        result_evaluation = sess.query(self.db.Evaluation).filter_by(job_id=job_id).one()
        result_job = sess.query(self.db.Job).filter_by(id=job_id).one()
        results_path = result_evaluation.results_path
        job_name = result_job.name
        project_id = result_job.project_id
        project_path = sess.query(self.db.Project.path).filter_by(id=project_id).one()
        #Path inside the project dir
        absolute_results_path = os.path.join(project_path, job_name, results_path)
        return absolute_results_path

    def print_page_info(self, ok) -> None:
        """Print page info."""
        self.pbar.ctrl.hide()

    def print_load_started(self) -> None:
        """Print load started."""
        logger.debug('Started loading')
        self.pbar.ctrl.show()

    def print_load_percent(self, percent: Union[str, int]) -> None:
        """Print load percent."""
        self.pbar.ctrl.setValue(int(percent))
        logger.debug(percent)
        if percent == 100:
            self.pbar.ctrl.hide()

    def get_combination_names(self, results_folder: str) -> List[str]:
        """Get pairwise target combination names from results folder."""
        combination_names = []
        logger.debug(results_folder)
        folders = [f for f in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, f))]
        logger.debug(folders)
        batch_summary_path = os.path.join(results_folder, "batch_summary.html")
        if not os.path.exists(batch_summary_path):
            logger.debug(f"{batch_summary_path} does not exist")
        if os.path.exists(batch_summary_path) and not "summary" in combination_names:
            combination_names.append("summary")
        for f in folders:
            if 'results.html' in os.listdir(os.path.join(results_folder, f)):
                combination_names.append(f)
        return combination_names

    def get_combination_results_path(self, results_folder: str, combination_name: str) -> str:
        """Get results path for given combination name."""
        results_file_path = None
        if combination_name == "summary":
            results_file_path = os.path.join(results_folder, 'batch_summary.html')
        else:
            folders = [f for f in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, f))]
            for f in folders:
                if f == combination_name:
                    results_file_path = os.path.join(results_folder, f, 'results.html')
            
        return results_file_path

    def init_gui(self, gui_params: dict, other: object = None, sess: sqlalchemy.orm.Session = None) -> dict:
        """Update GUI with evaluation results."""
        if not gui_params['job_id'] is None:
            if not gui_params['results_path'] is None:
                #results_path = self.get_results_path_by_id(gui_params['job_id'], sess)
                results_path = gui_params['results_path']
                logger.debug(f"Results path: {results_path}")
                #Backward compatibility
                if not os.path.exists(results_path):
                    results_path = os.path.join(gui_params['project_path'], gui_params['job_dir'])

                if not gui_params['pairwise_batch_prediction']:
                    self.pairwise_combinations_list.ctrl.setHidden(True)
                    self.pairwise_combinations_label.ctrl.setHidden(True)
                    results_path = os.path.join(results_path, "results.html")
                    #Backward compatibility
                    if not os.path.exists(results_path):
                        results_path = os.path.join(results_path, gui_params['job_dir'], "results.html")
                else:
                    self.pairwise_combinations_list.ctrl.setHidden(False)
                    self.pairwise_combinations_label.ctrl.setHidden(False)
                    combinations_list = self.get_combination_names(results_path)
                    self.pairwise_combinations_list.ctrl.clear()
                    for item in combinations_list:
                        self.pairwise_combinations_list.ctrl.addItem(item)
                    if gui_params['selected_combination_name'] is None:
                        if len(combinations_list) > 0:
                            results_path = self.get_combination_results_path(results_path, combinations_list[0])
                            gui_params['results_path_combination'] = os.path.join(gui_params['results_path'],  combinations_list[0])
                    else:
                        results_path = self.get_combination_results_path(results_path, gui_params['selected_combination_name'])
                        gui_params['results_path_combination'] = os.path.join(gui_params['results_path'],  gui_params['selected_combination_name'])
                logger.debug(results_path)
                self.results.ctrl.settings().setAttribute(QtWebEngineWidgets.QWebEngineSettings.WebGLEnabled, False)
                logger.debug(f"WebGL enabled: {self.results.ctrl.settings().testAttribute(QtWebEngineWidgets.QWebEngineSettings.WebGLEnabled)}")
                logger.debug(f"QMLSCENE_DEVICE: {os.environ['QMLSCENE_DEVICE']}")

                self.results.ctrl.loadStarted.connect(self.print_load_started)
                self.results.ctrl.loadProgress.connect(self.print_load_percent)
                self.results.ctrl.loadFinished.connect(self.print_page_info)
                self.results.ctrl.load(QtCore.QUrl(f'file://{results_path}'))
                #self.results.ctrl.setUrl(QtCore.QUrl(f'file://{results_path}'))
                gui_params['results_html'] = results_path

        return gui_params

class JobParams(GUIVariables):
    """ Class for managing job specific parameters. 
    
    Attributes:
        db (Database): Database object 
        db_table (str): Name of the database table
        id (Variable): ID of the jobparams entry
        job_id (Variable): ID of the job
        job_name (Variable): Name of the job
        output_dir (Variable): Output directory
        sequences (Variable): Sequences
        seq_names (Variable): Sequence names
        fasta_path (Variable): Path to fasta file (reflects AF argument)
        sequence_params (TblCtrlSequenceParams): Sequence parameters TableView widget
        custom_template_list (Variable): List-like comma-separated string with path to custom templates (reflects AF argument)
        precomputed_msas_list (Variable): List-like comma-separated string with path to precomputed MSAs (reflects AF argument)
        no_msas_list (Variable): List-like comma-separated string with True/False values (reflects AF argument)
        no_templates_list (Variable): List-like comma-separated string with True/False values (reflects AF argument)
        run_relax (Variable): Toggle alphafold relax step (reflects AF argument)
        num_multimer_predictions_per_model (Variable): Number of multimer predictions per model (reflects AF argument)
        queue (Variable): Whether to use queue submission
        db_preset_dict (dict): Dictionary with database presets
        db_preset (Variable): Database preset (reflects AF argument)
        model_preset_dict (dict): Dictionary with model presets
        model_preset (Variable): Model preset (reflects AF argument)
        benchmark (Variable): Toggle benchmarking (reflects AF argument)
        random_seed (Variable): Random seed (reflects AF argument)
        max_template_date (Variable): Maximum template date (reflects AF argument)
        precomputed_msas_path (Variable): Path to precomputed MSAs (reflects AF argument)
        force_cpu (Variable): Toggle CPU usage
        num_recycle (Variable): Number of recycling steps (reflects AF argument)
        pipeline_dict (dict): Dictionary with pipeline presets
        pipeline (Variable): Pipeline preset (reflects AF argument)
        prediction_dict (dict): Dictionary with prediction presets
        prediction (Variable): Prediction protocol preset
        num_gpu (Variable): Number of GPUs
        pairwise_batch_prediction (Variable): Toggle pairwise batch prediction of a set of sequences
        
        """
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
                               2: 'colabfold_local',
                               3: 'colabfold_web'}
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
        #self.batch_msas = Variable('batch_msas', 'bool', ctrl_type='chk', cmd=True)
        self.pipeline_dict = {0: 'full',
                              1: 'only_features',
                              2: 'batch_msas',
                              3: 'continue_from_features',
                              4: 'all_vs_all',
                              5: 'first_vs_all',
                              6: 'first_n_vs_rest',
                              7: 'grouped_bait_vs_preys',
                              8: 'grouped_all_vs_all',
                              9: 'only_relax'}
        self.pipeline = Variable('pipeline', 'str', ctrl_type='cmb', cmb_dict=self.pipeline_dict, cmd=True)
        self.prediction_dict = {0: 'alphafold'}
        self.prediction = Variable('prediction', 'str', ctrl_type='cmb', cmb_dict=self.prediction_dict, cmd=True)
        self.num_gpu = Variable('num_gpu', 'int', ctrl_type="sbo", cmd=True)
        self.pairwise_batch_prediction = Variable('pairwise_batch_prediction', 'bool')
        self.use_model_1 = Variable('use_model_1', 'bool', ctrl_type='chk')
        self.use_model_2 = Variable('use_model_2', 'bool', ctrl_type='chk')
        self.use_model_3 = Variable('use_model_3', 'bool', ctrl_type='chk')
        self.use_model_4 = Variable('use_model_4', 'bool', ctrl_type='chk')
        self.use_model_5 = Variable('use_model_5', 'bool', ctrl_type='chk')
        self.model_list = Variable('model_list', 'str', cmd=True)
        self.first_n_seq = Variable('first_n_seq', 'int', ctrl_type='sbo', cmd=True)
        self.features_dir = Variable('features_dir', 'str', db=True, cmd=True)
        self.predictions_dir = Variable('predictions_dir', 'str', db=True, cmd=True)
        self.batch_max_sequence_length = Variable('batch_max_sequence_length', 'int', ctrl_type='sbo', cmd=True)
        self.msa_pairing_dict = {0: 'paired',
                             1: 'paired+unpaired'}
        self.msa_pairing = Variable('msa_pairing', 'str', ctrl_type='cmb', cmb_dict=self.msa_pairing_dict, cmd=True)
        self.multichain_template_path = Variable('multichain_template_path', 'str', ctrl_type='lei', cmd=True)

    def set_db(self, db: DBHelper) -> None:
        """Set database helper object."""
        self.db = db

    def set_fasta_paths(self, job_path: str, job_name: str) -> None:
        """Set fasta path."""
        self.fasta_path.set_value(os.path.join(job_path, f"{job_name}.fasta"))

    def write_fasta(self):
        """Write fasta file."""
        logger.debug(f"Writing to file {self.fasta_path.get_value()}")
        with open(self.fasta_path.get_value(), 'w') as f:
            sequence_str = self.sequences.get_value()
            sequence_str = sequence_str.replace(" ", "")
            f.write(sequence_str)

    def db_insert_params(self, sess: sqlalchemy.orm.Session, data: list = None) -> None:
        """Insert params into database."""
        assert isinstance(data, list)
        rows = [self.db.Jobparams(**row) for row in data]
        for row in rows:
            sess.merge(row)
        sess.commit()

    def generate_db_object(self, data: list = None) -> List[sqlalchemy.orm.Query]:
        """Generate database object."""
        assert isinstance(data, list)
        return [self.db.Jobparams(**row) for row in data]

    def get_params_by_job_id(self, job_id, sess) -> sqlalchemy.orm.Query:
        """Get params by job id."""
        logger.debug(f"get result for job id {job_id}")
        if isinstance(job_id, int):
            result = sess.query(self.db.Jobparams).filter_by(job_id=job_id).one()
        elif isinstance(job_id, list):
            result = sess.query(self.db.Jobparams).filter(self.db.Jobparams.id.in_(job_id)).all()
        return result
    
    def get_protein_names(self) -> str:
        seq_names = self.seq_names.get_value()
        protein_names = seq_names.replace(',', '_')
        return protein_names

    def get_params_hash(self) -> str:
        #Get hash for all params that affect the prediction
        params_list = []
        params_list.append(self.no_msa_list.get_value())
        params_list.append(self.no_template_list.get_value())
        params_list.append(self.custom_template_list.get_value())
        params_list.append(self.num_recycle.get_value())
        params_list.append(self.max_template_date.get_value())
        params_list.append(self.random_seed.get_value())
        params_list.append(self.precomputed_msas_list.get_value())
        params_list.append(self.precomputed_msas_path.get_value())
        params_list.append(self.db_preset.get_value())
        params_list.append(self.model_preset.get_value())
        params_list = [str(x) for x in params_list]
        logger.debug("Generating hash for the following params:")
        logger.debug(params_list)
        params_str = ''.join(params_list)
        md5_hash = hashlib.md5(params_str.encode()).hexdigest()

        return md5_hash[:5]

    def parse_fasta(self, fasta_file: str) -> List[str]:
        """Parse fasta file."""
        sequences = []
        descriptions = []
        if fasta_file is None:
            raise ValueError("No fasta file defined!")
        elif not os.path.exists(fasta_file):
            raise ValueError("Wrong fasta file path!")
        else:
            record_dict = SeqIO.index(fasta_file, "fasta")
            for record in record_dict.values():
                sequences.append(record.seq)
                descriptions.append(record.id)
        return sequences, descriptions
    
    def parse_fasta_string(self, fasta_string: str)-> List[str]:
        """Parse fasta string."""
        sequences = []
        descriptions = []
        records = SeqIO.parse(StringIO(fasta_string), "fasta")
        for record in records:
            sequences.append(str(record.seq))
            descriptions.append(record.id)
        return sequences, descriptions

    def read_sequences(self) -> list:
        """Read sequences from fasta input."""
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
        job_name = self.seq_names.value.replace(',','_')
        self.job_name.ctrl.setText(job_name)
        self.job_name.set_value(job_name)
        return error_msgs
    
    def split_and_read_sequences(self) -> list:
        self.split_sequences()
        self.read_sequences()

    def split_into_subsequences(self, input_string, max_length):
        """Split into equally sized chunks"""
        num_parts = (len(input_string) + max_length - 1) // max_length
        part_size = len(input_string) // num_parts
        parts = [input_string[i * part_size:(i + 1) * part_size] for i in range(num_parts - 1)]
        parts.append(input_string[(num_parts - 1) * part_size:])
        logging.info(f"Splitted {input_string} into:")
        logging.info(parts)
        return parts

    def split_sequences(self):
        sequences, descriptions = self.parse_fasta_string(self.sequences.ctrl.toPlainText())
        split_sequence_list = []
        split_description_list = []
        for i, sequence in enumerate(sequences):
            split_sequences = self.split_into_subsequences(sequence, self.batch_max_sequence_length.get_value())
            split_descriptions = [f"{descriptions[i]}_split{o}" for o in range(len(split_sequences))]
            split_sequence_list.extend(split_sequences)
            split_description_list.extend(split_descriptions)
        fasta_list = []
        for i, seq in enumerate(split_sequence_list):
            fasta_list.append(f">{split_description_list[i]}\n{seq}\n\n")
        fasta_string = ''.join(fasta_list)
        self.sequences.ctrl.setPlainText(fasta_string)


    def process_single_template(self, template: str, msgs: list, output_folder: str, index: int, multichain: bool = False) -> bool:
        """Process single template.
          Make sure that the input file only contains one model and one chain and if the chain doesn't have id 'A' rename it.
            Save the cif with a new 4 letter filename."""
        template_name = os.path.basename(template)
        if not template.endswith('.cif'):
            msgs.append(f"Custom template {template_name} needs to be in cif format. Use https://mmcif.pdbj.org/converter to convert.")
            return False
        else:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('structure', template)
            models = list(structure.get_models())
            if len(models) > 1:
                msgs.append(f"More than one model found in {template_name}. Make sure that only one model and chain is present in the file.")
                return False
            chains = [x.get_id() for x in structure[0].get_chains()]
            if not multichain:
                if len(chains) > 1:
                    msgs.append(f"More than one chain found in {template_name}. Make sure only the chain that matches the target is present in the file.")
                    return False
                elif not 'A' in chains:
                    logger.debug("Chain IDs in template")
                    logger.debug(chains)
                    msgs.append(f"The template chain needs to have id 'A'in {template_name}.")
                    return False

            if index < 10:
                out_name = f'cus{index}'
                out_path = os.path.join(output_folder, f'cus{index}.cif')
            elif index < 100:
                out_name = f'cu{index}'
                out_path = os.path.join(output_folder, f'cu{index}.cif')
            elif index < 1000:
                out_name = f'c{index}'
                out_path = os.path.join(output_folder, f'c{index}.cif')
            else:
                msgs.append("Too many templates.")
                return False
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
            return True
            
    def process_custom_template_files(self, job_dir: str) -> Tuple[str, list]:
        """Process custom template files."""
        msgs = []
        new_custom_template_list = []
        multichain_template_path = None
        folder_count = 0
        for i, item in enumerate(self.custom_template_list.get_value().split(',')):
            custom_template_folder = os.path.join(job_dir, f"custom_templates_{folder_count}")
            logging.debug(f"Processing custom template: {item}")
            if not item in [None, "None"]:
                if not os.path.exists(custom_template_folder):
                    folder_count += 1
                    os.mkdir(custom_template_folder)
                if os.path.isdir(item):
                    cif_files = [f for f in os.listdir(item) if f.endswith(".cif")]
                    if len(cif_files) == 0:
                        msgs.append(f"No cif files found in {item}")
                    else:
                        for o, file in enumerate(cif_files):
                            template = os.path.join(item, file)
                            status = self.process_single_template(template, msgs, custom_template_folder, o)
                            if status is False:
                                break
                else:
                    template = item
                    self.process_single_template(template, msgs, custom_template_folder, 0)
                logger.debug(f"{i} {template}")
                new_custom_template_list.append(custom_template_folder)
            else:
                new_custom_template_list.append("None")
        
        if not self.multichain_template_path.get_value() in [None, "None"]:
            template = self.multichain_template_path.get_value()
            if os.path.exists(template):
                custom_template_folder = os.path.join(job_dir, f"custom_templates_{folder_count}")
                if not os.path.exists(custom_template_folder):
                    os.mkdir(custom_template_folder)
                self.process_single_template(template, msgs, custom_template_folder, 0, multichain=True)
            else:
                msgs.append(f"{template} does not exist!")
            multichain_template_path = os.path.join(custom_template_folder, f'cus0.cif')


            
        return ','.join(new_custom_template_list), multichain_template_path, msgs


    def update_from_sequence_table(self) -> None:
        """Update sequence params from sequence table."""
        self.custom_template_list.value, \
        self.precomputed_msas_list.value, \
        self.no_msa_list.value, \
        self.no_template_list.value, \
        self.seq_names.value = self.sequence_params.get_from_table(self.seq_names.value.split(','))


    def get_name_by_job_id(self, job_id: int, sess: sqlalchemy.orm.Session) -> str:
        """Get job name by job id."""
        result = sess.query(self.db.Jobparams.job_name).filter_by(job_id=job_id).one()
        logger.debug(result[0])
        logger.debug(result)
        return result[0]
    
    def get_pairwise_batch_prediction(self, job_id: id, sess: sqlalchemy.orm.Session) -> bool:
        """Get pairwise batch prediction."""
        result = sess.query(self.db.Jobparams.pairwise_batch_prediction).filter_by(job_id=job_id).one()
        return result[0]

    def set_db_preset(self) -> None:
        """Set db preset."""
        if not self.db_preset.is_set():
            self.db_preset.set_value('full_dbs')

    def set_max_template_date(self) -> None:
        """Set max template date."""
        #format 2021-11-02
        cur_date = date.today()
        if not self.max_template_date.is_set():
            self.max_template_date.set_value(cur_date)

    def init_gui(self, gui_params: dict, other: object = None, sess: sqlalchemy.orm.Session = None) -> dict:
        """Update GUI with sequence params."""
        self.sequence_params.init_gui()
        return gui_params


class Job(GUIVariables):
    """This class contains variables and functions for for handling jobs. It inherits from GUIVariables."""
    def __init__(self) -> None:
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
        self.list = TblCtrlJobs('list', None, db=False, ctrl_type='tre')
        self.timestamp = Variable('timestamp', 'str')
        self.log = Variable('log', 'str', db=False, ctrl_type='pte')
        self.log_file = Variable('log_file', 'str', db=True)
        self.job_status_log_file = Variable('job_status_log_file', 'str', db=True, cmd=True)
        self.status = Variable('status', 'str', db=True)
        self.pid = Variable('pid', 'str', db=True)
        self.host = Variable('host', 'str', db=True)
        self.path = Variable('path', 'str', db=True)
        self.type = Variable('type', 'str', db=True)
        self.tree_item_expanded = Variable('tree_item_expanded', 'bool', db=True)
        self.active = Variable('active', 'bool', db=True)

    #Getters

    def get_status(self, job_id: str, sess: sqlalchemy.orm.Session) -> str:
        result = sess.query(self.db.Job).get(job_id)
        return result.status

    def get_job_project_id(self, job_id: int, project_id: int, sess: sqlalchemy.orm.Session) -> Optional[int]:
        result = sess.query(self.db.Job).filter_by(id=job_id, project_id=project_id).first()
        if result is None:
            return None
        else:
            return result.job_project_id

    def get_project_id_by_job_id(self, job_id, sess):
        result = sess.query(self.db.Job).filter_by(id=job_id).first()
        return result.project_id
    
    def get_max_job_project_id(self, project_id: int, sess: sqlalchemy.orm.Session) -> Optional[int]:
        result = sess.query(self.db.Job).filter_by(project_id=project_id).order_by(self.db.Job.job_project_id.desc()).first()
        if result is None:
            return None
        else:
            return result.job_project_id
        
    def get_next_job_project_id(self, project_id: int, sess: sqlalchemy.orm.Session) -> int:
        max_id = self.get_max_job_project_id(project_id, sess)
        if max_id is None:
            logger.debug("No job_project_id found.")
            max_id = 0
        return max_id + 1
    
    def get_job_id_by_job_project_id(self, job_project_id: int, project_id: int, sess: sqlalchemy.orm.Session):
        result = sess.query(self.db.Job).filter_by(project_id=project_id, job_project_id=job_project_id).first()
        return result.id

    def get_pid(self, job_id: int, sess: sqlalchemy.orm.Session):
        result = sess.query(self.db.Job).get(job_id)
        logger.debug(f"Getting PID for job_id {job_id} from DB. PID is {result.pid}")
        return result.pid
    
    def get_queue_job_id(self, log_file: str) -> str:
        queue_job_id = None
        logger.debug(f"Reading from {log_file}")
        with open(log_file, 'r') as f:
            lines = f.readlines()
        regex = re.compile("QUEUE_JOB_ID=(\d+)")
        for l in lines:
            if re.search(regex, l):
                queue_job_id = re.search(regex, l).group(1)
        return queue_job_id
    
    def get_host(self, job_id: int, sess: sqlalchemy.orm.Session):
        result = sess.query(self.db.Job).get(job_id)
        return result.host
    
    def get_tree_item_expanded(self, job_name: str, project_id: int, sess: sqlalchemy.orm.Session) -> bool:
        result = sess.query(self.db.Job).filter_by(name=job_name, project_id=project_id).first()
        return result.tree_item_expanded
    
    def get_jobs_by_project_id(self, project_id: int, sess: sqlalchemy.orm.Session):
        result = sess.query(self.db.Job).filter_by(project_id=project_id)
        return result

    def get_type(self, job_params):
        if job_params['pipeline'] in ['continue_from_features', 'first_vs_all', 'all_vs_all', 'first_n_vs_rest', 'grouped_bait_vs_preys', 'grouped_all_vs_all']:
            type = "prediction"
        elif job_params['pipeline'] in ['only_features', 'batch_msas']:
            type = "features"
        elif job_params['pipeline'] == 'only_relax':
            type = "relax"
        else:
            type = "full"
        return type

    def get_type_by_job_id(self, job_id, sess):
        result = sess.query(self.db.Job).get(job_id)
        return result.type    

    def get_active_job_id(self, sess: sqlalchemy.orm.Session) -> int:
        result = sess.query(self.db.Job).filter_by(active=True).first()
        if result:
            return result.id
        else:
            return None
    
    #Setters

    def set_db(self, db) -> None:
        # Set the database to use for this connection.
        self.db = db

    def set_next_job_project_id(self, project_id: str, sess: sqlalchemy.orm.Session) -> None:
        job_project_id = self.get_next_job_project_id(project_id, sess)
        self.job_project_id.value = job_project_id
        logger.debug(f"job_project_id is {job_project_id}")

    def set_type(self, type):
        self.type.set_value(type)

    def set_job_active(self, job_id: int, sess: sqlalchemy.orm.Session) -> int:
        #Set all jobs inactive first
        active_jobs = sess.query(self.db.Job).filter_by(active=True).all()
        for i, item in enumerate(active_jobs):
            active_jobs[i].active = False
        #Set job active by job_id
        job = sess.query(self.db.Job).get(job_id)
        if job:
            job.active = True
        sess.commit()

    #ToDo: check usage and change to var.set()
    def set_project_id(self, project_id):
        self.project_id.value = project_id        

    #DB Updates

    def update_status(self, status: str, job_id: int, sess: sqlalchemy.orm.Session) -> None:
        result = sess.query(self.db.Job).get(job_id)
        result.status = status
        sess.commit()

    def update_tree_item_expanded(self, job_name: str, project_id: int, tree_item_expanded: bool, sess: sqlalchemy.orm.Session) -> None:
        result = sess.query(self.db.Job).filter_by(name=job_name, project_id=project_id).first()
        result.tree_item_expanded = tree_item_expanded
        sess.commit()

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
        if job_params['prediction'] in ['alphafold']:
            command = ["run_prediction.py\\\n"] + job_args
        command = ' '.join(command)
        logfile = os.path.join(job_params['job_path'], job_params["log_file"])
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

        if 'num_cpu' in template_vars:
            if job_params['pipeline'] == 'continue_from_features':
                to_render['num_cpu'] = 1
            else:
                to_render['num_cpu'] = job_params['num_cpu']
        if 'num_gpus' in template_vars:
            if 'num_gpu' in job_params:
                to_render['num_gpus'] = job_params['num_gpu']
            else:
                to_render['num_gpus'] = 1
        if 'use_gpu' in template_vars:
            if job_params['pipeline'] in ['only_features', 'batch_msas'] or job_params['force_cpu']:
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
            #         if 'num_cpu' in template_vars:
            #             to_render['num_cpu'] = 1
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
                if job_params['force_cpu'] or job_params['pipeline'] in ['only_features', 'batch_features']:
                    ram = job_params['min_ram']
                else:
                    ram = estimated_gpu_mem
            ram = job_params['min_ram']
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

        logger.debug(f"Writing to file {submit_script}")
        with open(submit_script, 'w') as f:
            f.write(rendered)
        return submit_script, msgs

    def calculate_gpu_mem_alphafold(self, total_seq_length: int) -> int:
        """Calculate required gpu memory in GB by sequence length for alphafold"""
        logger.debug(total_seq_length)
        #polynomal fit large dataset
        mem = 0.00000504*total_seq_length**2 - 0.00135796*total_seq_length + 5.55021461
        #exponential fit
        #mem = int(math.ceil(4.8898*math.exp(0.00077181*total_seq_length)))
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
        error_msgs, warn_msgs = [], []
        estimated_gpu_mem = None
        logger.debug(cmd_dict)
        logger.debug(cmd_dict)
        job_args = []
        log_file = os.path.join(job_params['job_path'], job_params['log_file'])
        job_status_log_file = job_params['job_status_log_file']
        job_status_log_file = os.path.join(job_params['job_path'], job_status_log_file)
        cmd_dict['job_status_log_file'] = job_status_log_file

        #Estimate memory
        estimated_gpu_mem = self.calculate_gpu_mem_alphafold(job_params['total_seqlen'])

        #Generate job argument list
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


        #gpu_name = None
        #gpu_mem = None
        if not 'max_ram' in job_params:
            job_params['max_ram'] = self.get_max_ram()

        #Decide which GPUs to use if several are available on a cluster
        if any([job_params['force_cpu'],
                    job_params['pipeline'] in ['only_features', 'batch_msas']]):
            gpu_mem = None
        else:
            if job_params['queue']:
                gpu_mem = job_params['max_gpu_mem']
            else:
                gpu_mem = self.get_gpu_mem()

        #Increase RAM for mmseqs caching, approximately half of the database size should be sufficient
        if job_params['db_preset'] == 'colabfold_local':
            if job_params['pipeline'] in ['full', 'only_features', 'batch_msas']:
                if job_params['max_ram'] < 800:
                   job_params['min_ram'] = job_params['max_ram']
                else:
                   job_params['min_ram'] = 800

        split_mem = None
        if not any([gpu_mem is None,
                    estimated_gpu_mem is None,
                    job_params['force_cpu'],
                    job_params['pipeline'] in ['only_features', 'batch_msas']]):
            logger.debug(f"\n\n\n\n\n==============Estimated GPU MEM {estimated_gpu_mem} Max GPU mem {gpu_mem}")
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
        else:
            logger.debug(f"Skipping GPU memory check")
            logger.debug([gpu_mem is None,
                          estimated_gpu_mem is None,
                          job_params['force_cpu'],
                          job_params['pipeline'] in ['only_features', 'batch_msas']])
        if job_params['queue']:
            queue_submit = job_params['queue_submit']
            submit_script, more_msgs = self.prepare_submit_script(job_params, job_args, estimated_gpu_mem, split_mem)
            error_msgs.extend(more_msgs)
            cmd = [queue_submit, submit_script]
        else:
            cmd = []
            if not split_mem is None and not job_params['force_cpu']:
                cmd = [f'export TF_FORCE_UNIFIED_MEMORY=True; export XLA_PYTHON_CLIENT_MEM_FRACTION={split_mem}; ']
            #cmd = [f"/bin/bash -c \'echo test > {job_params['log_file']}\'"]
            if job_params['pipeline'] in ['only_features', 'batch_msas'] or job_params['force_cpu']:
                cmd = ['export CUDA_VISIBLE_DEVICES=""; '] + cmd
            bin_path = os.path.join(sys.exec_prefix, 'bin')
            cmd += [f"run_prediction.py\\\n"]
            cmd += job_args + [f">> {log_file} 2>&1"]
        logger.debug("Job command\n{}".format(cmd))
        return cmd, error_msgs, warn_msgs, estimated_gpu_mem

    def insert_evaluation(self, _evaluation: object, job_params: dict, sess: sqlalchemy.orm.session.Session) -> bool:
        if not _evaluation.check_exists(job_params['job_id'], sess):
            job_obj = self.get_job_by_id(job_params['job_id'], sess)
            if job_params['pairwise_batch_prediction']:
                results_path = job_params['results_path']
            else:
                results_path = os.path.join(job_params['results_path'],
                                                            "results.html")
            if os.path.exists(os.path.join(job_params['project_path'], results_path)):
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
        
    def get_status_dict(self) -> dict:
        task_status_dict = {
                "num_tasks_finished": 0,
                "db_search_started": False,
                "model_1_started": False,
                "model_2_started": False,
                "model_3_started": False,
                "model_4_started": False,
                "model_5_started": False,
                "evaluation_started": False,
                }
        return task_status_dict

    def get_job_status_from_log(self, params):
        logger.debug(f"Updating job status params")
        exit_code = None
        task_status_dict = self.get_status_dict()
        job_status = None
        exit_code_script = None
        exit_code_queue = None
        #try:
        lines = []
        #logger.debug(params)
        log_file = os.path.join(params['job_path'], params['log_file'])
        job_status_log_file = os.path.join(params['job_path'], params['job_status_log_file'])
        if os.path.exists(job_status_log_file):
            task_status_dict['num_tasks_finished'] = 0
            logger.debug(f"Reading from {log_file}")
            with open(log_file, 'r') as f:
                lines = f.readlines()
        else:
            msg = f"Job status file {params['job_status_log_file']} does not exist."
            params['errors'].append(msg)
            if params['log_file_lines']:
                #Only lines added since last update
                lines = params['log_file_lines']
            elif os.path.exists(log_file):
                #All lines read again, reset counter
                task_status_dict['num_tasks_finished'] = 0
                logger.debug(f"Reading from {log_file}")
                with open(log_file, 'r') as f:
                    lines = f.readlines()
            else:
                msg = f'Log file {log_file} does not exist'
                params['errors'].append(msg)
                return params

        for line in lines:
            pattern_exit_code_script = re.compile(r'Exit\scode\s(\d+)')
            pattern_exit_code_queue = re.compile(r'Workflow\sfinished\swith\scode\s(\d+)')
            pattern_started = re.compile(r'Alphafold pipeline starting...')
            cancelled_pattern = re.compile(r'CANCELLED')
            pattern_db_search =  re.compile(r"Predicting\s\w+")
            pattern_model_1 =  re.compile(r"Running\smodel\smodel_1")
            pattern_model_2 =  re.compile(r"Running\smodel\smodel_2")
            pattern_model_3 =  re.compile(r"Running\smodel\smodel_3")
            pattern_model_4 =  re.compile(r"Running\smodel\smodel_4")
            pattern_model_5 =  re.compile(r"Running\smodel\smodel_5")
            pattern_task_finished = re.compile(r'Task finished')
            pattern_finished =  re.compile(r"Alphafold pipeline completed")
            pattern_tasks_done = re.compile(r"(\d+)/(\d+) tasks finished.")
            if re.search(pattern_tasks_done, line):
                g = re.search(pattern_tasks_done, line)
                task_status_dict['num_tasks_finished'] = g.groups(1)
                params['num_jobs'] = g.groups(2)
            elif re.search(pattern_task_finished, line):
                task_status_dict['num_tasks_finished'] += 1
                logger.debug(f"Task finished found. Current number: {task_status_dict['num_tasks_finished']}")
            if re.search(pattern_exit_code_script, line):
                exit_code_script = int(re.search(pattern_exit_code_script, line).group(1))
                logger.debug(f"Exit code from script found in log file {exit_code_script}")
            if re.search(pattern_exit_code_queue, line):
                exit_code_queue = int(re.search(pattern_exit_code_queue, line).group(1))
                logger.debug(f"Exit code from queue found in log file {exit_code_queue}")
            if re.search(cancelled_pattern, line):
                exit_code = 2
                logger.debug(f"Cancelled pattern found in log file {exit_code_queue}")
            if re.search(pattern_started, line):
                job_status = 'running'
            if re.search(pattern_db_search, line):
                task_status_dict['db_search_started'] = True
            if re.search(pattern_model_1, line):
                task_status_dict['model_1_started'] = True
            if re.search(pattern_model_2, line):
                task_status_dict['model_2_started'] = True
            if re.search(pattern_model_3, line):
                task_status_dict['model_3_started'] = True
            if re.search(pattern_model_4, line):
                task_status_dict['model_4_started'] = True
            if re.search(pattern_model_5, line):
                task_status_dict['model_5_started'] = True
            if re.search(pattern_model_5, line):
                task_status_dict['evaluation_started'] = True
            if re.search(pattern_finished, line):
                job_status = 'finished'
        
        logger.debug(f"exit_code_script {exit_code_script} exit_code_queue {exit_code_queue}")
        if exit_code_script == 1 or exit_code_queue == 1:
            logger.debug("Either exit code script or exit code_queue is 1")
            exit_code = 1
        elif exit_code_script == 2:
            logger.debug("Exit code script is 2")
            exit_code = 2
        elif exit_code_script == 0 and exit_code_queue == 0:
            logger.debug("Exit code script and exit_code_queue are 0")
            exit_code = 0
        elif exit_code_queue == 0 and exit_code_script is None:
            #Assume there is an error if there is no exit code from the script
            exit_code = 1
        elif exit_code_queue:
            if exit_code_queue > 2:
                exit_code = 1
        #except Exception as e:
        #    logger.debug(f"Error while updating job status params: {e}")
        #    logger.debug(traceback.print_exc())

        
        if not 'task_status' in params:
            params['task_status'] = {}
        params['task_status'] = task_status_dict
        if not 'status' in params:
            params['status'] = "unknown"
        if not job_status is None:
            params['status'] = job_status
        if not 'exit_code' in params:
            params['exit_code'] = None
        params['exit_code'] = exit_code
        logger.debug(params)
        return params


    def db_insert_job(self, sess: sqlalchemy.orm.session.Session = None, data: list = None) -> None:
        assert isinstance(data, list)
        rows = [self.db.Job(**row) for row in data]
        for row in rows:
            sess.merge(row)
        sess.commit()

    def delete_job(self, job_id: int, sess: sqlalchemy.orm.session.Session) -> None:
        sess.query(self.db.Job).filter_by(id=job_id).delete()
        #Cascading delete not yet working
        sess.query(self.db.Jobparams).filter_by(job_id=job_id).delete()
        sess.query(self.db.Evaluation).filter_by(job_id=job_id).delete()
        sess.commit()

    def delete_job_files(self, job_id: str, path: str, sess: sqlalchemy.orm.session.Session) -> None:
        self.delete_job(job_id, sess)
        rmtree(path)

    def set_timestamp(self) -> None:
        self.timestamp.value = datetime.datetime.now()

    def set_host(self) -> None:
        self.host.value = socket.gethostname()

    def hostname(self) -> str:
        return socket.gethostname()

    def generate_db_object(self, data: list = None) -> list:
        assert isinstance(data, list)
        return [self.db.Job(**row) for row in data]

    def get_job_by_id(self, job_id: int, sess: sqlalchemy.orm.session.Session) -> list:
        result = sess.query(self.db.Job).filter_by(id=job_id).one()
        return result

    def get_jobs(self, sess: sqlalchemy.orm.session.Session) -> sqlalchemy.orm.Query:
        result = sess.query(self.db.Job)
        return result

    def read_log(self, log_file: str) -> List[str]:
        lines = []
        logger.debug(f"Reading from {log_file}")
        with open(log_file, 'r') as log:
            lines = log.readlines()
        return lines

    def update_log(self, log_lines: str = None, log_file: str = None, job_id_active: int = None, job_id_thread: int = None, notebook_page: str = None, append: bool = False) -> None:
        """ Updates the log control """
        logger.debug("Updating log")
        if (job_id_active == job_id_thread) or job_id_thread is None:
            if not append:
                self.log.reset_ctrl()
            if log_file:
                if os.path.exists(log_file):
                    logger.debug(f"Reading from {log_file}")
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                else:
                    logger.error(f"Log file {log_file} does not exist!")
                    lines = []
            elif log_lines:
                logger.debug("Found log lines")
                lines = log_lines
            logger.debug("Appending lines")
            for line in lines:
                self.log.ctrl.appendPlainText(line.strip('\n'))
            #self.log_reader_thread = LogThread(log_file, log_lines)
            #self.log_reader_thread.log_updated.connect(self.append_log_lines)
            #self.log_reader_thread.log_updated.connect(self.update_job_status_params)
            #self.log_reader_thread.start()
        else:
            logger.debug("job ids do not match or LogTab not selected")

    def append_log_lines(self, lines: list):
        logger.debug("Appending line")


    def get_path_by_project_id(self, project_id: int, sess: sqlalchemy.orm.session.Session) -> Union[str, None]:
        result = sess.query(self.db.Project.path).filter_by(id=project_id).one()
        return result[0]


    #Name of jobdir in project folder
    def get_job_dir(self, job_name: str) -> str:
        job_dir = job_name
        return job_dir

    #Full path to job dir
    def get_job_path(self, project_path: str, job_dir: str) -> str:
        job_path = os.path.join(project_path, job_dir)
        return job_path

    def build_log_file_path(self, job_name: str, type: str, prediction: str, db_preset: str, pipeline: str, job_project_id: int) -> str:
        if type == 'prediction':
            if pipeline in screening_protocol_names:
                log_file = os.path.join(f"{job_name}_{type}_batch_{prediction}_{job_project_id}.log")
            else:
                log_file = os.path.join(f"{job_name}_{type}_{prediction}_{job_project_id}.log")
        elif type == 'features':
            if pipeline == "batch_msas":
                log_file = os.path.join(f"{job_name}_{type}_batch_{db_preset}_{job_project_id}.log")
            else:
                log_file = os.path.join(f"{job_name}_{type}_{db_preset}_{job_project_id}.log")
        elif type == 'full':
            log_file = os.path.join(f"{job_name}_{type}_{db_preset}_{prediction}_{job_project_id}.log")
        elif type == 'relax':
            log_file = os.path.join(f"{job_name}_{type}_{job_project_id}.log")
        else:
            logger.error(f"Unknown type {type}")
        return log_file

    #Backward compatibility
    def build_log_file_path_deprec(self, job_name: str, type: str) -> str:
        log_file = os.path.join(f"{job_name}_{type}.log")
        return log_file

    def get_log_file(self, job_id: int, sess: sqlalchemy.orm.session.Session) -> str:
        """Reconstruct the log path in case the project path was changed."""
        result_job = sess.query(self.db.Job).filter_by(id=job_id).first()
        result_jobparams = sess.query(self.db.Jobparams).filter_by(job_id=job_id).first()
        job_name = result_jobparams.job_name
        project_id = result_job.project_id
        type = result_job.type
        prediction = result_jobparams.prediction
        db_preset = result_jobparams.db_preset
        pipeline = result_jobparams.pipeline
        job_project_id = result_job.job_project_id
        project_path = self.get_path_by_project_id(project_id, sess)
        #Path inside the project dir
        log_file = self.build_log_file_path(job_name, type, prediction, db_preset, pipeline, job_project_id)
        log_path = os.path.join(project_path, job_name, log_file)
        #Backward compatibility
        if not os.path.exists(log_path):
            log_path = log_path.replace(f"_{job_project_id}", "")
        if not os.path.exists(log_path):
            log_path = os.path.join(project_path, job_name, self.build_log_file_path_deprec(job_name, type))
        return log_path
    
    def get_job_status_log_file(self, log_file: str) -> str:
        job_status_log_path = "job_status_" + log_file
        return job_status_log_path

    def get_queue_pid(self, log_file: str, job_id: int, sess: sqlalchemy.orm.session.Session) -> Union[str, None]:
        pid = None
        regex = re.compile(r'QUEUE_JOB_ID=(\d+)')
        logger.debug(f"Reading from {log_file}")
        with open(log_file, 'r') as f:
            content = f.read()
        if re.search(regex, content):
            pid = re.search(regex, content).group(1)
        if not pid is None:
            self.update_pid(pid, job_id, sess)
        logger.debug(f"pid from log file is {pid}.")
        return pid
    
    def get_log_file_by_id(self, job_id: int, sess: sqlalchemy.orm.session.Session):
        result = sess.query(self.db.Job).filter_by(id=job_id).one()
        return result.log_file

    def set_log_file(self, log_file: str) -> None:
        self.log_file.value = log_file

    def set_status(self, status: str) -> None:
        self.status.value = status

    def check_pid(self, pid: int) -> bool:
        try:
            os.kill(int(pid), 0)
            logger.debug("PID exists")
            return True
        except Exception:
            logger.debug("PID not found")
            return False
        
    def get_job_name_by_id(self, job_id, sess):
        result = sess.query(self.db.Jobparams.job_name).filter_by(id=job_id).first()
        return result[0]

    def reconnect_jobs(self, sess: sqlalchemy.orm.session.Session) -> list:
        jobs_running = []
        result = sess.query(self.db.Job).filter((self.db.Job.status=="running") | (self.db.Job.status=="starting") | (self.db.Job.status=="waiting")).all()

        for job in result:
            jobparams = sess.query(self.db.Jobparams).filter(self.db.Jobparams.id == job.id).one()
            jobs_running.append({'job_id': job.id,
                                 'job_project_id': job.job_project_id,
                                 'queue': jobparams.queue,
                                 'status': job.status,
                                 'job_path': job.path,
                                 'job_name': jobparams.job_name,
                                 'job_status_log_file': self.get_job_status_log_file(self.get_log_file(job.id, sess)),
                                 'log_file': self.get_log_file(job.id, sess),
                                 'pid': job.pid,
                                 'time_started': job.timestamp})
            logger.debug(f"Reconnect job {job.id}, {job.job_project_id}, {jobparams.job_name}, {job.status}")
        return jobs_running

    # def init_gui(self, gui_params: dict, sess=None) -> dict:
    #     """Init or update the GUI with the current job list."""
    #     logger.debug("=== Init Job list ===")
    #     # Clear Lists
    #     self.list.reset_ctrl()
    #     self.log.reset_ctrl()
    #     logger.debug("reset end")
    #     # Fill job list
    #     self.list.ctrl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    #     self.list.ctrl.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
    #     self.list.ctrl.setColumnCount(4)
    #     self.list.ctrl.verticalHeader().setVisible(False)
    #     self.list.ctrl.setHorizontalHeaderLabels(('ID', 'Name', 'Type', 'Status'))
    #     self.list.ctrl.setColumnWidth(0, 50)
    #     self.list.ctrl.setColumnWidth(1, 100)
    #     self.list.ctrl.setColumnWidth(2, 70)
    #     self.list.ctrl.setColumnWidth(3, 70)
    #     project_id = gui_params['project_id']
    #     if not project_id is None:
    #         gui_params['project_path'] = self.get_path_by_project_id(project_id, sess)
    #         jobs = self.get_jobs_by_project_id(project_id, sess)
    #         for job in jobs:
    #             status = self.get_status(job.id, sess)
    #             if status is None:
    #                 status = "unknown"


    #             rows = self.list.ctrl.rowCount()
    #             self.list.ctrl.insertRow(rows)
    #             self.list.ctrl.setItem(rows, 0, QtWidgets.QTableWidgetItem(str(job.job_project_id)))
    #             self.list.ctrl.setItem(rows, 1, QtWidgets.QTableWidgetItem(job.name))
    #             self.list.ctrl.setItem(rows, 2, QtWidgets.QTableWidgetItem(job.type.capitalize()))
    #             self.list.ctrl.setItem(rows, 3, QtWidgets.QTableWidgetItem(status))
    #         self.list.ctrl.scrollToItem(self.list.ctrl.item(self.list.ctrl.currentRow(), 0), QtWidgets.QAbstractItemView.PositionAtCenter)
    #         #self.list.ctrl.scrollToBottom()
    #     return gui_params
    

    def collect_group_subgroup_items(self, parent_item):
        for i in range(parent_item.childCount()):
            child_item = parent_item.child(i)
            item_text = child_item.text(0)
            if "Group" in item_text or "Subgroup" in item_text:
                print(item_text)
            self.collect_group_subgroup_items(child_item)


    #TreeView Implementation
    def init_gui(self, gui_params: dict, reset: bool = False, other: object = None, sess: sqlalchemy.orm.Session = None) -> dict:
        from PyQt5 import QtWidgets

        logger.debug("=== Init Job list ===")
        # Clear Lists
        if reset:
            logger.debug("Restting Job list")
            self.list.reset_ctrl()
        #self.log.reset_ctrl()
        logger.debug("reset end")

        # Fill job list
        self.list.ctrl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        header_labels = ["Name", "Status", "ID"]
        self.list.ctrl.setHeaderLabels(header_labels)
        #self.list.ctrl.setHeaderHidden(True)
        self.list.ctrl.setColumnWidth(0, 240)
        self.list.ctrl.setColumnWidth(1, 70)
        self.list.ctrl.setColumnWidth(2, 15)
        project_id = gui_params['project_id']

        if project_id is not None:
            job_groups = {}
            existing_items = {}  # Dictionary to store existing items by job ID

            # Iterate through existing top-level items in the tree widget and store them
            for i in range(self.list.ctrl.topLevelItemCount()):
                group_item = self.list.ctrl.topLevelItem(i)
                job_name = group_item.text(0)
                if not job_name in job_groups:
                    job_groups[job_name] = group_item

                # Store sub-items in the dictionary by job ID
                for j in range(group_item.childCount()):
                    sub_item = group_item.child(j)
                    job_project_id = str(sub_item.text(2))
                    existing_items[job_project_id] = sub_item

            jobs = self.get_jobs_by_project_id(project_id, sess)
            jobparams = other.jobparams.get_params_by_job_id([job.id for job in jobs], sess)
            jobparams = [row.__dict__ for row in jobparams]
            logger.debug("Existing items in job list")
            logger.debug(existing_items)


            # Iterate through the fetched job list
            for i, job in enumerate(jobs):
                jobparams_row = [row for row in jobparams if row['id']==job.id][0]
                job_id = job.id
                job_name = job.name
                job_type = job.type
                #jobparams_result = other.jobparams.get_params_by_job_id(job_id, sess=sess)
                prediction = jobparams_row['prediction']#jobparams_result.prediction
                db_preset = jobparams_row['db_preset']
                pipeline = jobparams_row['pipeline']

                if job_type == 'prediction':
                    if pipeline in screening_protocol_names:
                        job_type = f"{job_type}_[batch_{prediction}]"
                    else:
                        job_type = f"{job_type}_[{prediction}]"
                elif job_type == 'features':
                    if pipeline == 'batch_msas':
                        job_type = f"{job_type}_[batch_{db_preset}]"
                    else:
                        job_type = f"{job_type}_[{db_preset}]"
                elif job_type == 'relax':
                    pass
                elif job_type == 'full':
                    job_type= f"{job_type}_[{db_preset}_{prediction}]"

                job_project_id = str(job.job_project_id)

                # Check if job already has an item in the tree
                if job_project_id in existing_items:
                    logger.debug(f"{job_project_id} in existing_items")
                    sub_item = existing_items[job_project_id]
                    current_status = sub_item.text(1)
                    new_status = job.status
                    
                    # Update status if it has changed
                    if current_status != new_status:
                        sub_item.setText(1, new_status)

                    # Remove the job ID from the dictionary since it's been processed
                    del existing_items[job_project_id]
                else:
                    logger.debug(f"{job_project_id} not in existing_items")
                    # Create a new group item if necessary
                    if job_name not in job_groups:
                        group_item = QtWidgets.QTreeWidgetItem(self.list.ctrl)
                        group_item.setFlags(group_item.flags() & ~QtCore.Qt.ItemIsSelectable)
                        group_item.setText(0, job_name)
                        self.list.ctrl.addTopLevelItem(group_item)
                        job_groups[job_name] = group_item

                    group_item = job_groups[job_name]
                    sub_item = QtWidgets.QTreeWidgetItem(group_item, [job_type, "", job_project_id])
                    sub_item.setText(1, job.status)

                    if i == jobs.count() - 1 or self.get_tree_item_expanded(job_name, project_id, sess):
                        if not group_item.isExpanded():
                            group_item.setExpanded(True)
                        else:
                            logger.debug("Is already expanded")

            # Remove any items that no longer have corresponding job objects
            for job_id, sub_item in existing_items.items():
                parent_item = sub_item.parent()
                parent_item.removeChild(sub_item)

                if parent_item.childCount() == 0:
                    index = self.list.ctrl.indexOfTopLevelItem(parent_item)
                    self.list.ctrl.takeTopLevelItem(index)
                    job_name = parent_item.text(0)
                    del job_groups[job_name]
            # Automatically scroll to the bottom
            last_item = self.list.ctrl.topLevelItem(self.list.ctrl.topLevelItemCount() - 1)
            self.list.ctrl.scrollToItem(last_item, QtWidgets.QAbstractItemView.PositionAtBottom)

            #self.list.ctrl.expandAll()
            self.list.ctrl.setColumnCount(3)
            #self.list.ctrl.resizeColumnToContents(0)

        if gui_params['job_id'] is None:
            gui_params['job_id'] = self.get_active_job_id(sess)
            if gui_params['job_id']:
                gui_params['job_name'] = self.get_job_name_by_id(gui_params['job_id'], sess)
                gui_params['job_dir'] = self.get_job_dir(gui_params['job_name'])
                logger.debug(f"job_id is None, getting last active job from DB: {gui_params['job_id']}")
        else:
            self.set_job_active(gui_params['job_id'], sess)
            logger.debug(f"Setting {gui_params['job_id']} to active.")

        return gui_params


class Project(GUIVariables):
    """ Class for managing projects. 
    
    Attributes:
        db (object): Database object.
        db_table (str): Name of the database table.
        id (object): Variable object for the project id.
        jobs (object): Variable object for the jobs.
        name (object): Variable object for the project name.
        path (object): Variable object for the project path.
        list (object): Variable object for the project list.
        active (object): Variable object for the project active state."""
    def __init__(self):
        self.db = None
        self.db_table = 'project'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.jobs = Variable('jobs', None, db_relationship='Job')
        self.name = Variable('name', 'str', ctrl_type='lei')
        self.path = Variable('path', 'str', ctrl_type='lei')
        self.list = Variable('list', None, db=False, ctrl_type='cmb')
        self.active = Variable('active', 'bool', ctrl_type=None)

    def set_db(self, db: DBHelper) -> None:
        self.db = db

    def insert_project(self, data: list, sess: sqlalchemy.orm.session.Session) -> None:
        """Inserts a new project into the database."""
        assert isinstance(data, list)
        rows = [self.db.Project(**row) for row in data]
        for row in rows:
            sess.merge(row)
        sess.commit()

    def check_if_name_exists(self, project_name: str, sess: sqlalchemy.orm.session.Session) -> bool:
        """Checks if a project already exists in the database."""
        exists = False
        result = self.get_projects(sess)
        for row in result:
            if project_name == row.name:
                exists = True
        return exists
    
    def check_if_path_exists(self, project_path: str, sess: sqlalchemy.orm.session.Session) -> bool:
        """Checks if a project path already exists in the database."""
        exists = False
        result = self.get_projects(sess)
        for row in result:
            if project_path == row.path:
                exists = True
        return exists

    def delete_project(self, project_id: int, sess: sqlalchemy.orm.session.Session) -> None:
        """Deletes a project from the database."""
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

    def is_empty(self, sess: sqlalchemy.orm.session.Session) -> bool:
        """Checks if the project table is empty."""
        if sess.query(self.db.Project).all() == []:
            return True
        else:
            return False

    def get_project_by_id(self, project_id: int, sess: sqlalchemy.orm.session.Session) -> object:
        return sess.query(self.db.Project).get(project_id)

    def get_active_project(self, sess: sqlalchemy.orm.session.Session) -> Tuple[Union[str, None], Union[int, None]]:
        """Returns the name and id of the active project."""
        try:
            project_name, project_id = sess.query(self.db.Project.name, self.db.Project.id).filter_by(
                active=True).one()
        except NoResultFound:
            project_name, project_id = None, None
        return project_name, project_id

    def get_projects(self, sess: sqlalchemy.orm.session.Session) -> List[object]:
        result = sess.query(self.db.Project).all()
        return result

    def update_project(self, project_id: int, data: dict, sess: sqlalchemy.orm.session.Session) -> None:
        """Updates a project in the database."""
        assert isinstance(data, list)
        result = sess.query(self.db.Project).get(project_id)
        for k, v in data[0].items():
            setattr(result, k, v)
        sess.commit()

    def change_active_project(self, new_project: str, sess: sqlalchemy.orm.session.Session, new_active_id: int = None) -> int:
        """
        Change ative project in DB
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

    def get_path_by_project_id(self, project_id: int, sess: sqlalchemy.orm.session.Session) -> str:
        """Returns the path of a project."""
        result = sess.query(self.db.Project).filter_by(id=project_id).first()
        return result.path

    def init_gui(self, gui_params: dict, other: object = None, sess: Union[sqlalchemy.orm.session.Session, None] = None) -> dict:
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
            if name is None:
                for i, item in enumerate(projects):
                    if i == 0:
                        name = item.name
                        id = item.id
                        break
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
    """ Class for managing general settings."""
    def __init__(self) -> None:
        self.db = None
        self.db_table = 'settings'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.queue_submit = Variable('queue_submit', 'str', ctrl_type='lei')
        self.queue_cancel = Variable('queue_cancel', 'str', ctrl_type='lei')
        self.queue_account = Variable('queue_account', 'str', ctrl_type='lei')
        self.num_cpu = Variable('num_cpu', 'int', ctrl_type='sbo', cmd=True, db=False)
        self.min_cpus = Variable('min_cpus', 'int', ctrl_type='sbo')
        self.max_cpus = Variable('max_cpus', 'int', ctrl_type='sbo')
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
        self.uniref90_mmseqs_database_path = Variable('uniref90_mmseqs_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.uniref30_database_path = Variable('uniref30_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.uniref30_mmseqs_database_path = Variable('uniref30_mmseqs_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.colabfold_envdb_database_path = Variable('colabfold_envdb_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.mgnify_database_path = Variable('mgnify_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.bfd_database_path = Variable('bfd_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.small_bfd_database_path = Variable('small_bfd_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.uniprot_database_path = Variable('uniprot_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.uniprot_mmseqs_database_path = Variable('uniprot_mmseqs_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.pdb70_database_path = Variable('pdb70_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.pdb100_database_path = Variable('pdb100_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.pdb_seqres_database_path = Variable('pdb_seqres_database_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.template_mmcif_dir = Variable('template_mmcif_dir', 'str', ctrl_type='lei', cmd=True, required=True)
        self.obsolete_pdbs_path = Variable('obsolete_pdbs_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.custom_tempdir = Variable('custom_tempdir', 'str', ctrl_type='lei', cmd=True)
        self.use_gpu_relax = Variable('use_gpu_relax', 'bool', ctrl_type='chk', cmd=True)
        self.global_config = Variable('global_config', 'bool', ctrl_type='chk')

    def set_db(self, db):
        self.db = db

    def init_gui(self, gui_params: dict, other: object = None, sess: sqlalchemy.orm.Session = None):
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
            p = Popen("sacctmgr show User $(whoami) -p -n &> /dev/null", shell=True)
            stdout, _ = p.communicate()
            if not stdout is None:
                account_raw = stdout.decode()
                logger.debug(f'Output from sacctmgr: {account_raw}')
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
        self.prediction = 'alphafold'
        _, project_id = other.prj.get_active_project(other.sess)
        if project_id:
            self.precomputed_msas_path = other.prj.get_path_by_project_id(project_id, other.sess)
        settings = other.settings.get_from_db(other.sess)
        #update with defaults from advanced_settings
        advanced_settings_defaults = AdvancedSettingsDefaults()
        self.__dict__.update(advanced_settings_defaults.__dict__)
        logging.debug("Default values:")
        logging.debug(self.__dict__)
        if settings.queue_default:
            self.queue = True
