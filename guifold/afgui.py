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
import datetime
from shutil import copyfile
import time
from guifold.src import gui_threads
from guifold.src.gui_dialogs import LoadDialog, message_dlg, ProgressDialog
from guifold.src.gui_dlg_settings import SettingsDlg
from guifold.src.gui_dlg_about import AboutDlg
from guifold.src.gui_dlg_project import ProjectDlg
from guifold.src.gui_dlg_queue_submit import QueueSubmitDlg
from guifold.src.gui_dlg_advanced_params import AdvancedParamsDlg
from guifold.src.gui_dlg_first_n_seq import FirstNSeqDlg
from guifold.src.gui_dlg_split_sequence import SplitSeqDlg
import signal
import socket
from subprocess import Popen
import pkg_resources
import sys
import os
from glob import glob
from PyQt5 import QtWidgets, uic
from PyQt5.QtNetwork import QNetworkProxyFactory
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt5.QtGui import QIcon, QDesktopServices
import logging
from guifold.src.db_helper import DBHelper
import traceback
from guifold.src.gui_classes import Job, Settings, JobParams, Project, Evaluation, DefaultValues
from guifold.src.gui_dlg_advanced_params import DefaultValues as AdvancedParamsDefaultValues
import argparse
import configparser
import re



#Required for WebGL to work
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = "--ignore-gpu-blacklist"
#Prevent error when OpenGL not available
os.environ['QMLSCENE_DEVICE'] = "softwarecontext"
#Maybe makes Webview loading faster
QNetworkProxyFactory.setUseSystemConfiguration(False)

parser = argparse.ArgumentParser(description="GUI for running alphafold")
parser.add_argument('--debug', '-d',
                    help='Debug log.',
                    action='store_true')
parser.add_argument('--custom_db_path',
                    help='DB will be loaded from user specified location (default is home directory).')
args, unknown = parser.parse_known_args()

install_path = os.path.dirname(os.path.realpath(sys.argv[0]))
logger = logging.getLogger('guifold')
formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
if not args.debug:
    logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.DEBUG)
    log_file = 'guifold_debug.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class InputError(Exception):
    pass

class NoProjectSelected(Exception):
    pass

class DirectoryNotCreated(Exception):
    pass

class QueueSubmitError(Exception):
    pass

class JobSubmissionCancelledByUser(Exception):
    pass

class SequenceFormatError(Exception):
    pass

class MSAFolderExists(Exception):
    pass

class PrecomputedMSAConflict(Exception):
    pass

class MSAFolderNotExists(Exception):
    pass

class ProcessCustomTemplateError(Exception):
    pass

class PrepareCMDError(Exception):
    pass

class NoFeaturesExist(Exception):
    pass


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def main():
    shared_objects = [Settings(), Project(), Job(), JobParams(), Evaluation()]
    if not args.custom_db_path:
        db_path = os.path.join(os.path.expanduser("~"), '.guifold.db')
    else:
        db_path = args.custom_db_path

    db = DBHelper(shared_objects, db_path)
    db.upgrade_db()
    db.init_db()
    with db.session_scope() as sess:
        db.set_session(sess)
    for obj in shared_objects:
        obj.set_db(db)
    db.update_queue_jobid_regex(sess)
    db.migrate_to_pipeline_cmb(sess)
    db.update_job_type(sess)

    sys.excepthook = handle_exception
    app = QtWidgets.QApplication(sys.argv)
    #font = app.font()
    #font.setPointSize(10)
    #app.setFont(font)
    qss_path = pkg_resources.resource_filename('guifold.ui', 'gui_style_default.qss')
    with open(qss_path,"r") as qss:
        app.setStyleSheet(qss.read())
    #app.setStyle('Fusion')
    MainFrame(shared_objects, db, sess, install_path, app)
    app.exec()

        



def check_settings_locked():
    lock_settings = False
    config_file = pkg_resources.resource_filename('guifold.config', 'guifold.conf')
    if os.path.exists(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        if 'OTHER' in config:
            if 'lock_settings' in config['OTHER']:
                if config['OTHER']['lock_settings'] in ['True', 'true']:
                    lock_settings = True
            else:
                logger.debug("lock_settings not found in config.")
        else:
            logger.debug("\'OTHER\' section not found in config.")
    else:
        logger.debug("Config file not found")
    return lock_settings

def check_force_settings_update():
    force_update = False
    config_file = pkg_resources.resource_filename('guifold.config', 'guifold.conf')
    if os.path.exists(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        if 'OTHER' in config:
            if 'force_update' in config['OTHER']:
                if config['OTHER']['force_update'] in ['True', 'true']:
                    force_update = True
            else:
                logger.debug("lock_settings not found in config.")
        else:
            logger.debug("\'OTHER\' section not found in config.")
    else:
        logger.debug("Global config file not found")
    return force_update

def center_on_screen(obj):
    # Get the screen geometry
    screen_geometry = QtWidgets.QDesktopWidget().screenGeometry()

    # Calculate the center point
    center_x = screen_geometry.width() / 2
    center_y = screen_geometry.height() / 2

    # Calculate the new position for the window
    new_x = center_x - (obj.width() / 2)
    new_y = center_y - (obj.height() / 2)

    # Move the window to the center
    obj.move(int(new_x), int(new_y))

class MainFrame(QtWidgets.QMainWindow):
    def __init__(self, shared_objects, db, sess, install_path, app):
        super(MainFrame, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi(pkg_resources.resource_filename('guifold.ui', 'gui.ui'), self) # Load the .ui file
        center_on_screen(self)
        self.app = app
        self.install_path = install_path
        self.settings, self.prj, self.job, self.jobparams, self.evaluation = self.shared_objects = shared_objects


        self.gui_params = {'job_id': None,
                           'status': 'unknown',
                           'pid': None,
                           'queue_job_id': None,
                           'other_settings_changed': False,
                           'queue_account': None,
                           'job_project_id': None,
                           'project_id': None,
                           'log_file_lines': None,
                           'job_status_log_file': None,
                           'results_path': None,
                           'queue': None,
                           'errors': [],
                           'task_status': None,
                           'selected_combination_name': None,
                           'results_path_combination': None,
                           'pairwise_batch_prediction': False,
                           'prediction': 'alphafold'}
        self.gui_params['settings_locked'] = check_settings_locked()
        self.gui_params['force_settings_update'] = check_force_settings_update()
        self.files_selected_item = None
        self.db = db
        self.sess = sess
        self.screening_protocol_names = ['first_vs_all', 'all_vs_all', 'first_n_vs_rest', 'grouped_bait_vs_preys', 'grouped_all_vs_all']

        self.init_frame()
        #self.init_dialogs()

        self.init_menubar()
        self.init_toolbar()
        self.bind_event_handlers()
        logger.debug("GUI params")
        logger.debug(self.gui_params)
        self.init_settings()
        self.check_project_exists()
        self.default_values = DefaultValues(self)
        self.init_gui()
        logger.debug("GUI params")
        logger.debug(self.gui_params)

        logger.debug("GUI params")
        logger.debug(self.gui_params)

        msgs = self.validate_settings()




        self.currentDirectory = os.getcwd()
        self.threads = []
        self.thread_workers = []
        self.reconnect_jobs()
        self.show() # Show the GUI
        logger.debug("GUI params")
        logger.debug(self.gui_params)
        if len(msgs) > 0:
            msgs.append('You will not be able to run a default job!')
            message_dlg('warning', '\n'.join(msgs))
        if self.jobparams.queue.get_value():
            msgs = self.validate_settings(category='queue')
            if len(msgs) > 0:
                msgs.insert(0, 'Submit to Queue selected but queue submission not configured properly:')
                message_dlg('warning', '\n'.join(msgs))



    def init_frame(self):
        logger.debug("=== Initializing main frame ===")
        self.setWindowTitle("GUIFold")
        self.notebook = self.findChild(QtWidgets.QTabWidget, 'MainNotebook')
        self.notebook.setTabEnabled(2, False)
        self.panel = self.findChild(QtWidgets.QPushButton, 'InputPanel')
        #self.panel.SetScrollRate(20,20)
        self.log_panel = self.findChild(QtWidgets.QPushButton, 'LogPanel')
        self.btn_read_sequences = self.findChild(QtWidgets.QPushButton, 'btn_read_sequences')
        self.btn_split_sequences = self.findChild(QtWidgets.QPushButton, 'btn_split_sequences')
        #Only enable when batch_msas selected
        self.btn_split_sequences.setEnabled(False)
        self.btn_jobparams_advanced_settings = self.findChild(QtWidgets.QPushButton, 'btn_jobparams_advanced_settings')
        self.btn_evaluation_open_results_folder = self.findChild(QtWidgets.QPushButton, 'btn_evaluation_open_results_folder')
        self.btn_evaluation_open_pymol = self.findChild(QtWidgets.QPushButton, 'btn_evaluation_open_pymol')
        self.btn_open_browser = self.findChild(QtWidgets.QPushButton, 'btn_evaluation_open_browser')
        self.btn_prj_add = self.findChild(QtWidgets.QToolButton, 'btn_prj_add')
        self.btn_prj_add.setIcon(QIcon(pkg_resources.resource_filename('guifold.icons', 'gtk-add.png')))
        self.btn_prj_remove = self.findChild(QtWidgets.QToolButton, 'btn_prj_remove')
        self.btn_prj_remove.setIcon(QIcon(pkg_resources.resource_filename('guifold.icons', 'gtk-remove.png')))
        self.btn_prj_update = self.findChild(QtWidgets.QToolButton, 'btn_prj_update')
        self.btn_prj_update.setIcon(QIcon(pkg_resources.resource_filename('guifold.icons', 'gtk-edit.png')))
        self.btn_precomputed_msas_path = self.findChild(QtWidgets.QToolButton, 'btn_precomputed_msas_path')
        self.btn_multichain_template_path = self.findChild(QtWidgets.QToolButton, 'btn_multichain_template_path')
        self.lbl_status_1 = self.findChild(QtWidgets.QLabel, 'lbl_status_1')
        self.lbl_status_2 = self.findChild(QtWidgets.QLabel, 'lbl_status_2')
        self.lbl_status_3 = self.findChild(QtWidgets.QLabel, 'lbl_status_3')
        self.lbl_status_4 = self.findChild(QtWidgets.QLabel, 'lbl_status_4')
        self.lbl_status_5 = self.findChild(QtWidgets.QLabel, 'lbl_status_5')
        self.lbl_status_6 = self.findChild(QtWidgets.QLabel, 'lbl_status_6')
        self.lbl_status_7 = self.findChild(QtWidgets.QLabel, 'lbl_status_7')
        for obj in self.shared_objects:
            obj.set_controls(self, obj.db_table)


    def init_menubar(self):
        logger.debug("=== Initializing MenuBar ===")

        self.menubar = self.menuBar()
        self.file_menu = QtWidgets.QMenu("&File", self)
        self.menubar.addMenu(self.file_menu)
        self.project_menu = QtWidgets.QMenu("&Project", self)
        self.menubar.addMenu(self.project_menu)
        self.help_menu = QtWidgets.QMenu("&Help", self)
        self.menubar.addMenu(self.help_menu)

        self.exit_action = QtWidgets.QAction("Quit", self)
        self.add_prj_action = QtWidgets.QAction("Add Project", self)
        self.delete_prj_action = QtWidgets.QAction("Delete Project", self)
        self.change_prj_action = QtWidgets.QAction("Change Project", self)
        self.about_action = QtWidgets.QAction("About", self)
        self.wiki_action = QtWidgets.QAction("Wiki", self)

        self.exit_action.setShortcut('Ctrl+Q')
        self.exit_action.setStatusTip('Exit application')

        self.file_menu.addAction(self.exit_action)
        self.project_menu.addAction(self.add_prj_action)
        self.project_menu.addAction(self.delete_prj_action)
        self.project_menu.addAction(self.change_prj_action)
        self.help_menu.addAction(self.about_action)
        self.help_menu.addAction(self.wiki_action)


    def init_toolbar(self):
        logger.debug("=== Initializing ToolBar ===")   # Using a title
        self.tb = self.addToolBar("Toolbar")
        self.tb.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        self.tb_run = QtWidgets.QAction(QIcon(pkg_resources.resource_filename('guifold.icons', 'gtk-go-forward-ltr.png')),"Run",self)
        self.tb.addAction(self.tb_run)
        self.tb_run.setEnabled(False)

        self.tb_cancel = QtWidgets.QAction(QIcon(pkg_resources.resource_filename('guifold.icons', 'gtk-stop.png')),"Cancel",self)
        self.tb.addAction(self.tb_cancel)

        self.tb_settings = QtWidgets.QAction(QIcon(pkg_resources.resource_filename('guifold.icons', 'gtk-preferences.png')),"Settings",self)
        self.tb.addAction(self.tb_settings)

        self.tb_clear = QtWidgets.QAction(QIcon(pkg_resources.resource_filename('guifold.icons', 'gtk-clear.png')),"Clear",self)
        self.tb.addAction(self.tb_clear)





    def init_gui(self):
        logger.debug("=== Initializing GUI ===")

        for obj in self.shared_objects:
            logger.debug(f"Initializing  {obj}")
            self.gui_params = obj.init_gui(self.gui_params, other=self, sess=self.sess)
            logger.debug("GUI Params")
            logger.debug(self.gui_params)
        #Init DB Preset List
        db_preset_list = self.jobparams.db_preset_dict.values()
        for item in db_preset_list:
            self.jobparams.db_preset.ctrl.addItem(item)
        pipeline_list = self.jobparams.pipeline_dict.values()
        for item in pipeline_list:
            self.jobparams.pipeline.ctrl.addItem(item)
        prediction_list = self.jobparams.prediction_dict.values()
        for item in prediction_list:
            self.jobparams.prediction.ctrl.addItem(item) 
        self.default_values = DefaultValues(self)
        self.jobparams.update_from_default(self.default_values)
        advanced_params_default = AdvancedParamsDefaultValues()
        self.jobparams.update_from_default(advanced_params_default)


    def init_settings(self):
        logger.debug("=== Initializing Settings ===")
        if self.settings.add_blank_entry(self.sess):
            self.settings.update_from_config()
            slurm_account = self.settings.get_slurm_account()
            if not slurm_account is None:
                self.settings.set_slurm_account(slurm_account, self.sess)
            self.settings.update_settings(self.settings.get_dict_db_insert(), self.sess)
        elif self.gui_params['settings_locked'] or self.gui_params['force_settings_update']:
            logger.info("Settings are updated from global configuration file.")
            self.settings.update_from_config()
            self.settings.update_settings(self.settings.get_dict_db_insert(), self.sess)

    def create_monitor_thread(self, job_params):
        logger.debug(f"Creating monitor thread for {job_params['job_project_id']} {job_params['job_id']} {job_params['job_name']}")
        self.monitor_thread = QThread()
        self.monitor_worker = gui_threads.MonitorJob(self, job_params)
        self.monitor_worker.moveToThread(self.monitor_thread)
        self.monitor_thread.started.connect(self.monitor_worker.run)
        self.monitor_worker.finished.connect(self.monitor_thread.quit)
        self.monitor_worker.finished.connect(self.monitor_worker.deleteLater)
        self.monitor_thread.finished.connect(self.monitor_thread.deleteLater)
        self.monitor_worker.update_log.connect(self.OnUpdateLog)
        self.monitor_worker.clear_log.connect(self.OnClearLog)
        self.monitor_worker.job_status.connect(self.OnJobStatus)
        self.monitor_thread.start()
        self.threads.append(self.monitor_thread)
        self.thread_workers.append(self.monitor_worker)

    def check_project_exists(self):
        if self.prj.is_empty(self.sess):
            dlg = ProjectDlg(self, "add")
            if dlg.exec_():
                self.gui_params = self.prj.init_gui(self.gui_params, other=self, sess=self.sess)
        if self.prj.is_empty(self.sess):
            logger.error("Cannot start GUI without initial project.")
            raise SystemExit
        
    def validate_settings(self, category=None):
        msgs = []
        required_default = ['jackhmmer_binary_path',
                    'hhblits_binary_path',
                    'hhsearch_binary_path',
                    'hmmsearch_binary_path',
                    'hmmbuild_binary_path',
                    'hhalign_binary_path',
                    'kalign_binary_path',
                    'data_dir',
                    'uniref90_database_path',
                    'uniref30_database_path',
                    'bfd_database_path',
                    'uniprot_database_path',
                    'pdb70_database_path',
                    'pdb_seqres_database_path',
                    'template_mmcif_dir']
        required_queue = ['queue_submit',
                                    'queue_cancel',
                                    'min_cpus',
                                    'max_cpus',
                                    'max_gpu_mem',
                                    'min_ram',
                                    'max_ram',
                                    'queue_jobid_regex']
        required_colabfold = ['uniref90_mmseqs_database_path',
                                        'uniref30_mmseqs_database_path',
                                        'uniprot_mmseqs_database_path',
                                        'colabfold_envdb_database_path',
                                        'mmseqs_binary_path']
        if category is None:
            required = required_default
        elif category == 'queue':
            required = required_queue
        elif category == 'colabfold':
            required = required_colabfold
        settings = self.settings.get_from_db(self.sess)
        for var in vars(settings):
            if var in required:
                obj = getattr(settings, var)
                if obj is None or obj == '':
                    msgs.append(f'{var} needs to be defined in Settings.')
        return msgs

    def reconnect_jobs(self):
        jobs = self.job.reconnect_jobs(self.sess)
        logger.debug("Jobs to reconnect")
        logger.debug(jobs)
        if not jobs == []:
            for job_params in jobs:
                self.create_monitor_thread(job_params)

    def bind_event_handlers(self):
        logger.debug("=== Bind Event Handlers ===")
        #Menubar
        self.exit_action.triggered.connect(self.close)
        self.add_prj_action.triggered.connect(self.OnBtnPrjAdd)
        self.delete_prj_action.triggered.connect(self.OnBtnPrjRemove)
        self.change_prj_action.triggered.connect(self.OnBtnPrjUpdate)
        self.about_action.triggered.connect(self.OnAbout)
        self.wiki_action.triggered.connect(self.OnWiki)
        #Toolbar
        self.tb.actionTriggered[QtWidgets.QAction].connect(self.ToolbarSelected)
        #Buttons
        self.btn_prj_add.clicked.connect(self.OnBtnPrjAdd)
        self.btn_prj_update.clicked.connect(self.OnBtnPrjUpdate)
        self.btn_prj_remove.clicked.connect(self.OnBtnPrjRemove)
        self.btn_read_sequences.clicked.connect(self.OnBtnReadSequences)
        self.btn_split_sequences.clicked.connect(self.OnBtnSplitSequences)
        self.btn_evaluation_open_pymol.clicked.connect(lambda checked, x='pymol': self.OnOpenModelViewer(x))
        self.btn_open_browser.clicked.connect(self.OnOpenBrowser)
        self.btn_evaluation_open_results_folder.clicked.connect(self.OnOpenResultsFolder)
        #self.job.list.ctrl.cellClicked.connect(self.OnLstJobSelected)
        self.job.list.ctrl.itemClicked.connect(self.OnLstJobSelected)
        self.btn_jobparams_advanced_settings.clicked.connect(self.OnBtnAdvancedParams)
        self.btn_jobparams_advanced_settings.setEnabled(False)
        self.btn_precomputed_msas_path.clicked.connect(self.OnBtnPrecomputedMSAsPath)
        self.btn_multichain_template_path.clicked.connect(self.OnBtnMultichainTemplatePath)
        #Combos
        self.prj.list.ctrl.activated.connect(self.OnCmbProjects)
        self.job.list.ctrl.setContextMenuPolicy(Qt.CustomContextMenu)
        self.job.list.ctrl.customContextMenuRequested.connect(self.OnJobContextMenu)
        self.evaluation.pairwise_combinations_list.ctrl.activated.connect(self.OnCmbCombinations)
        self.jobparams.pipeline.ctrl.activated.connect(self.OnCmbPipeline)
        self.jobparams.prediction.ctrl.activated.connect(self.OnCmbPrediction)
        self.jobparams.db_preset.ctrl.activated.connect(self.OnCmbDbPreset)
        #TreeWidget
        self.job.list.ctrl.itemExpanded.connect(self.OnItemExpanded)
        self.job.list.ctrl.itemCollapsed.connect(self.OnItemCollapsed)

    def ToolbarSelected(self, s):
        selected = s.text()
        if selected == 'Run':
            self.OnBtnRun()
        elif selected == 'Cancel':
            self.OnBtnCancel()
        elif selected == 'Settings':
            self.OnBtnSettings()
        elif selected == 'Clear':
            self.OnBtnClear()

    def OnJobStatus(self, job_params):
        logger.debug("OnJobStatus")
        log_file = os.path.join(job_params['job_path'], job_params['log_file'])
        #Only change status in GUI if the job id from the thread matches the currently selected job.
        if int(self.gui_params['job_id']) == int(job_params['job_id']):
            logger.debug("Updating progress control:")
            logger.debug(job_params['task_status'])
            for item in [self.lbl_status_1,
                        self.lbl_status_2,
                        self.lbl_status_3,
                        self.lbl_status_4,
                        self.lbl_status_5,
                        self.lbl_status_6,
                        self.lbl_status_7]:
                item.setStyleSheet("color: gray;")
            try:
                if 'task_status' in job_params:
                    if job_params['task_status']['num_tasks_finished']:
                            if not 'num_jobs' in job_params:
                                job_params['num_jobs'] = '?'
                            self.lbl_status_1.setText(f"{job_params['task_status']['num_tasks_finished']}/{job_params['num_jobs']} tasks finished")
                    if job_params['task_status']['db_search_started']:
                        self.lbl_status_1.setStyleSheet("color: black;")
                    if job_params['task_status']['model_1_started']:
                        self.lbl_status_1.setStyleSheet("color: green;")
                        self.lbl_status_2.setStyleSheet("color: black;")
                    if job_params['task_status']['model_2_started']:
                        self.lbl_status_2.setStyleSheet("color: green;")
                        self.lbl_status_3.setStyleSheet("color: black;")
                    if job_params['task_status']['model_3_started']:
                        self.lbl_status_3.setStyleSheet("color: green;")
                        self.lbl_status_4.setStyleSheet("color: black;")
                    if job_params['task_status']['model_4_started']:
                        self.lbl_status_4.setStyleSheet("color: green;")
                        self.lbl_status_5.setStyleSheet("color: black;")
                    if job_params['task_status']['model_5_started']:
                        self.lbl_status_5.setStyleSheet("color: green;")
                        self.lbl_status_6.setStyleSheet("color: black;")
                    if job_params['task_status']['evaluation_started']:
                        self.lbl_status_6.setStyleSheet("color: green;")
                        self.lbl_status_7.setStyleSheet("color: black;")
                if job_params['status'] == 'finished':
                    self.lbl_status_7.setStyleSheet("color: green;")
            except KeyError:
                logger.debug("job status key(s) not found.")
                logger.debug(traceback.print_exc())
                pass
        else:
            logger.debug("Job ids do not match. not Updating progress controls.")  



        #self.job.update_job_status_params(self, job_params)

        if 'status' in job_params:
            logger.debug(f"Status found: {job_params['status']}")
            if job_params['status'] == "aborted":
                self.job.update_status("aborted", job_params['job_id'], self.sess)
            elif job_params['status'] == "waiting":
                self.job.update_status("waiting", job_params['job_id'], self.sess)
            elif job_params['status'] == "running":
                self.job.update_status("running", job_params['job_id'], self.sess)
            elif job_params['status'] == "starting":
                self.job.update_status("starting", job_params['job_id'], self.sess)
                #self.job.update_pid(job_params['pid'], job_params['job_id'], self.sess)
            
            #Submit second job if split_step option is selected
            if 'split_job' in job_params and 'queue' in job_params and 'split_job_step' in job_params and 'initial_submit' in job_params:
                logger.debug("split_job, queue and split_job_step found.")
                if all([job_params['queue'], job_params['split_job'], job_params['split_job_step'] == 'cpu',
                         job_params['status'] in ["waiting", "running"], job_params['initial_submit']]):
                        logger.debug(f"Submit second step with PID dependency {job_params['pid']}.")
                        if job_params['pid'] is None:
                            message_dlg("Error", "Could not get JobID from queue submission command."
                                                    " Second job cannot be submitted.")
                            logger.debug("Failed to submit gpu job because no queue_pid was found.")
                        else:
                            self.jobparams.pipeline.set_value("full")
                            self.jobparams.pipeline.set_cmb_by_text("full")
                            self.prepare_and_start_job(job_params['jobs_started'], split_job_step='gpu', queue_pid=job_params['pid'])
            if job_params['status'] == "finished":
                logger.debug(f"Status of {job_params['job_id']} is finished.")
                self.job.update_status("finished", job_params['job_id'], self.sess)
                job_params['pairwise_batch_prediction']  = self.jobparams.get_pairwise_batch_prediction(job_params['job_id'], self.sess)
                job_params['project_id'] = self.job.get_project_id_by_job_id(job_params['job_id'], self.sess)
                job_params['project_path'] = self.prj.get_path_by_project_id(job_params['project_id'], self.sess)
                evaluation = self.job.insert_evaluation(self.evaluation, job_params, self.sess)
                if evaluation:
                    if not self.gui_params['job_id'] is None:
                        if int(self.gui_params['job_id']) == int(job_params['job_id']):
                            self.notebook.setTabEnabled(2, True)
                            self.evaluation.init_gui(self.gui_params, self.sess)
                else:
                    self.notebook.setTabEnabled(2, False)
                    logger.debug("No evaluation report found!")
            if job_params['status'] == "error":
                logger.debug(f"Status of job_id {job_params['job_id']} is error")
                self.job.update_status("error", job_params['job_id'], self.sess)
            if job_params['status'] == "unknown":
                self.job.update_status("unknown", job_params['job_id'], self.sess)
                logger.debug(f"Status of job_id {job_params['job_id']} is unknown")
                
        else:
            job_params['status'] = 'unknown'
            logger.debug("Status not found in job_params")
        updated_status = self.job.get_status(job_params['job_id'], self.sess)
        logger.debug(f"Updated status for {job_params['job_id']} from the DB is {updated_status}")
        self.gui_params = self.job.init_gui(self.gui_params, other=self, sess=self.sess)
        #Only update log if the job id from the thread matches the currently selected job and the Log Tab is selected.
        
        self.job.update_log(log_file=log_file, job_id_active=int(self.gui_params['job_id']), job_id_thread=int(job_params['job_id']), append=False)

    def OnUpdateLog(self, log):
        lines, job_id = log
        page = self.notebook.currentWidget().objectName()
        logger.debug("Notebook page {} id {} {}".format(page, self.gui_params['job_id'], job_id))
        self.job.update_log(log_lines=lines, job_id_active=int(self.gui_params['job_id']), job_id_thread=job_id, notebook_page=page, append=True)

    def OnAbout(self):
        dlg = AboutDlg(self)
        dlg.exec_()

    def OnWiki(self):
        url = QUrl(f'https://github.com/fmi-basel/GUIFold/wiki')
        try:
            QDesktopServices.openUrl(url)
        except:
            error =  f"Could not open https://github.com/fmi-basel/GUIFold/wiki in an external browser. Maybe a default browser is not set."
            logger.error(error)
            message_dlg('Error', error)
            logger.debug(traceback.print_exc())

    def OnBtnReadSequences(self):
        error_msgs = self.jobparams.read_sequences()
        for msg in error_msgs:
            message_dlg('Error', msg)
        if len(error_msgs) == 0:
            self.btn_jobparams_advanced_settings.setEnabled(True)
            self.tb_run.setEnabled(True)
        else:
            self.btn_jobparams_advanced_settings.setEnabled(False)
            self.tb_run.setEnabled(False)

    def OnBtnSplitSequences(self):
        dlg = SplitSeqDlg(self)
        dlg.exec()
        self.jobparams.split_sequences()


    def start_thread(self, job_params, cmd):
        #Start  process thread
        self.process_thread = QThread()
        self.process_worker = gui_threads.RunProcessThread(self, job_params, cmd)
        self.process_worker.moveToThread(self.process_thread)
        self.process_thread.started.connect(self.process_worker.run)
        self.process_worker.finished.connect(self.process_thread.quit)
        self.process_worker.finished.connect(self.process_worker.deleteLater)
        self.process_thread.finished.connect(self.process_thread.deleteLater)
        self.process_worker.job_status.connect(self.OnJobStatus)
        self.process_worker.change_tab.connect(self.OnChangeTab)
        self.process_worker.error.connect(self.OnError)
        self.process_thread.start()
        self.threads.append(self.process_thread)
        self.thread_workers.append(self.process_worker)
        self.create_monitor_thread(self.job_params)

    def OnError(self, msgs):
        if len(msgs) > 0:
            for msg in msgs:
                message_dlg('Error', msg)

    def OnBtnRun(self) -> None:
        jobs_started = 0
        self.prepare_and_start_job(jobs_started, split_job_step=None, queue_pid=None)

    def prepare_and_start_job(self, jobs_started, split_job_step=None, queue_pid=None):
        #Make sure that not more than one additional job (split_job) can be started without pressing the run button.
        if not jobs_started > 2:
            try:
                # Prepare Job
                exec_messages = []
                settings = self.settings.get_from_db(self.sess)
                self.jobparams.update_from_gui()
                self.jobparams.update_from_sequence_table()
                #Set template date to today if not defined by user
                self.jobparams.set_max_template_date()
                #If db_preset is not defined by user set it to full_dbs
                self.jobparams.set_db_preset()

                #Define which param models to use:
                model_indices = []
                if self.jobparams.use_model_1.get_value():
                    model_indices.append('1')
                if self.jobparams.use_model_2.get_value():
                    model_indices.append('2')
                if self.jobparams.use_model_3.get_value():
                    model_indices.append('3')
                if self.jobparams.use_model_4.get_value():
                    model_indices.append('4')
                if self.jobparams.use_model_5.get_value():
                    model_indices.append('5')

                if len(model_indices) == 0:
                    msg = ("At least one Alphdafold model needs to be selected.")
                    message_dlg('error', msg)
                    raise InputError('msg')

                self.jobparams.model_list.set_value(','.join(model_indices))

                #exec_messages = self.settings.check_executables(self.sess)
                logger.debug("EXEC messages")
                logger.debug(exec_messages)
                if not exec_messages == []:
                    for message in exec_messages:
                        message_dlg('Error', message)
                else:
                    #Collect params from GUI and return as dict
                    job_params = self.jobparams.get_dict_run_job()
                    #Extend with status dict
                    job_params['task_status'] = self.job.get_status_dict()

                    #Prepare objects for DB insert


                    logger.debug("Params dict db")


                    self.job.set_next_job_project_id(self.gui_params['project_id'], self.sess)
                    self.job.set_timestamp()
                    job_params['time_started'] = self.job.timestamp.get_value()
                    self.job.set_host()
                    job_params['host'] = self.job.host.get_value()
                    self.job.set_status("starting")
                    job_params['job_name'] = self.jobparams.job_name.get_value()
                    job_params['min_cpus'] = settings.min_cpus
                    job_params['max_cpus'] = settings.max_cpus
                    job_params['split_job'] = settings.split_job
                    job_params['split_job_step'] = None
                    job_params['queue_jobid_regex'] = settings.queue_jobid_regex
                    job_params['submission_script_template_path'] = settings.submission_script_template_path
                    if job_params['split_job']:
                        if job_params['queue_jobid_regex'] is None or job_params['queue_jobid_regex'] == "":
                            message_dlg('Error', 'Split_job requested but no queue_jobid_regex defined!')


                    job_params['queue_pid'] = queue_pid
                    logger.debug(f"{self.jobparams.precomputed_msas_list.list_like_str_not_all_none()}")
                    logger.debug(f"{self.jobparams.precomputed_msas_path.is_set()}")
                    logger.debug(job_params['split_job'])
                    logger.debug(job_params['pipeline'])
                    logger.debug(split_job_step)

                    if job_params['pipeline'] == 'batch_msas':
                        self.jobparams.prediction.set_value('alphafold')
                        job_params['prediction'] = 'alphafold'
                    if job_params['pipeline'] in self.screening_protocol_names:
                        self.jobparams.pairwise_batch_prediction.set_value(True)
                        job_params['pairwise_batch_prediction'] = True
                    else:
                        self.jobparams.pairwise_batch_prediction.set_value(False)
                        job_params['pairwise_batch_prediction'] = False

                    #Adjust num cpus based on pipeline
                    if job_params['db_preset'] == 'colabfold_local':
                        if job_params['pipeline'] in ['full', 'only_features', 'batch_msas']:
                            job_params['num_cpu'] = job_params['max_cpus']
                            logger.debug(f"Switched CPUs to max {job_params['num_cpu']}")  
                        else:
                            job_params['num_cpu'] = job_params['min_cpus']   
                    else:
                        job_params['num_cpu'] = job_params['min_cpus']
                                      

                    #Prepare split job
                    if job_params['queue']:
                        if job_params['split_job'] and job_params['pipeline'] == 'full':
                            #Start with cpu step
                            if split_job_step is None:
                                split_job_step = job_params['split_job_step'] = 'cpu'
                                job_params['pipeline'] = 'only_features'
                                #self.jobparams.pipeline.set_cmb_by_text('only_features')
                                self.jobparams.pipeline.set_value('only_features')
                                job_params['job_name'] = f"{job_params['job_name']}"
                                self.jobparams.job_name.set_value(job_params['job_name'])
                            elif split_job_step == 'gpu':
                                logger.debug(f"queue pid {queue_pid}")
                                job_params['pipeline'] = 'continue_from_features'
                                #self.jobparams.pipeline.set_cmb_by_text('continue_from_features')
                                self.jobparams.pipeline.set_value('continue_from_features')
                                job_params['force_cpu'] = False
                                job_params['num_cpu'] = 1
                                job_params['split_job_step'] = 'gpu'
                                job_params['job_name'] = f"{job_params['job_name']}"
                                self.jobparams.job_name.set_value(job_params['job_name'])
                            else:
                                logger.debug('Unknown split_job_step.')
                        else:
                            job_params['split_job'] = False
                    else:
                        job_params['split_job'] = False
                    logger.debug(f"job name {job_params['job_name']}")

                    if self.gui_params['project_id'] is None:
                        msg = 'No Project selected!'
                        message_dlg('Error', msg)
                        raise InputError(msg)

                    type = job_params['type'] = self.job.get_type(job_params)
                    self.job.set_type(type)

                    job_params['job_project_id'] = self.job.job_project_id.get_value()
                    #Job folder name
                    job_params['job_dir'] = self.job.get_job_dir(job_params['job_name'])
                    #Project path
                    job_params['project_path'] = self.gui_params['project_path']
                    #Full path to job folder
                    job_params['output_dir'] = job_params['job_path'] = self.job.get_job_path(self.gui_params['project_path'],
                                                                   job_params['job_dir'])
                    job_params['log_file'] = self.job.build_log_file_path(job_params['job_name'], job_params['type'], job_params['prediction'], job_params['db_preset'], job_params['pipeline'], job_params['job_project_id'])
                    log_path = os.path.join(job_params['output_dir'], job_params['log_file'])
                    job_params['job_status_log_file'] = self.job.get_job_status_log_file(job_params['log_file'])
                    
                    if os.path.exists(log_path):
                        modification_time = os.path.getmtime(log_path)
                        modification_time_formatted = datetime.datetime.fromtimestamp(modification_time).strftime("%Y-%m-%d_%H-%M-%S")
                        log_path_bkp = log_path.replace('.log', f'_backup_{modification_time_formatted}.log')
                        copyfile(log_path, log_path_bkp)

                    #Check existing job dir
                    if os.path.exists(job_params['job_path']):
                        #Only show the warnings for the first step in case of split_job
                        # if not split_job_step == 'gpu':
                        #     if self.jobparams.precomputed_msas_list.list_like_str_not_all_none() or self.jobparams.precomputed_msas_path.is_set():
                        #         if not self.jobparams.precomputed_msas_list.get_value() is None:
                        #             precomputed_msas_list = self.jobparams.precomputed_msas_list.get_value().split(',')
                        #         else:
                        #             precomputed_msas_list = [None]
                        #         pc_msa_paths = precomputed_msas_list + [self.jobparams.precomputed_msas_path.get_value()]
                        #         pc_msa_paths = [x for x in pc_msa_paths if not x is None]
                        #         logger.debug(pc_msa_paths)
                        #         if any([re.match(job_params['output_dir'], item) for item in pc_msa_paths if not re.search('batch_msas', item)]):
                        #             message_dlg('error', 'One or more precomputed MSAs are from the current folder.'
                        #                                  ' Please select a new Job Name.')
                        #             raise PrecomputedMSAConflict("One or more precomputed MSAs are from the current folder.")
                        #     else:
                        if not split_job_step == 'gpu':
                            message = "Output directory already exists. Click \"Yes\" if you want to continue from existing MSAs or \"No\" " \
                                        "if existing MSAs should be recalculated."
                            ret = QtWidgets.QMessageBox.question(self, 'Warning', message,
                                                                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
                            if ret == QtWidgets.QMessageBox.Cancel:
                                raise JobSubmissionCancelledByUser
                            elif ret == QtWidgets.QMessageBox.Yes:
                                job_params['use_precomputed_msas'] = True
                            else:
                                job_params['use_precomputed_msas'] = False
                    else:
                        #Check if job name is too long
                        if len(job_params['job_name']) > 100:
                            msg = 'Job name is too long. Please choose a shorter one.'
                            message_dlg('error', msg)
                            raise InputError(msg)
                        else:
                            os.mkdir(job_params['job_path'])

                    #Process sequences
                    self.jobparams.set_fasta_paths(job_params['job_path'], job_params['job_name'])
                    job_params['fasta_path'] = self.jobparams.fasta_path.get_value()
                    self.jobparams.write_fasta()
                    logger.debug(f"Fasta path {self.jobparams.fasta_path.get_value()}")

                    job_params['features_path'] = os.path.join(job_params['job_path'], "features", job_params['db_preset'])
                    self.jobparams.features_dir.set_value(job_params['db_preset'])
                    sequences, seq_descriptions = self.jobparams.parse_fasta(self.jobparams.fasta_path.get_value())

                    #Check input for pairwise prediction protocols
                    if job_params['pipeline'] in self.screening_protocol_names:
                        batch_msas_path = job_params["features_path"]
                        dirs = []
                        missing_msa_list = []
                        if os.path.exists(batch_msas_path) and not self.jobparams.precomputed_msas_path.is_set():
                            dirs = [d for d in os.listdir(batch_msas_path) if os.path.isdir(os.path.join(batch_msas_path, d))]
                            logger.debug(f"batch msa folder found: {batch_msas_path} ")
                            self.jobparams.precomputed_msas_path.set_value(batch_msas_path)
                            self.jobparams.precomputed_msas_path.ctrl.setText(batch_msas_path)
                        else:
                            logger.debug(f"No batch msa folder found: {batch_msas_path}")
                        if not self.jobparams.precomputed_msas_path.is_set():
                            msg =  (f"Pairwise combinatorial prediction requires precomputed MSAs for all "
                                        f"sequences (from a batch_msas job). In the 'Precomputed MSAs path' choose " 
                                        f"a directory that contains subfolders with MSAs for all subunits and from the same feature pipeline. "
                                        f"e.g. /path/to/project/batch_msas_job_name/features/{job_params['db_preset']}")
                            message_dlg('error', msg) 
                            raise InputError(msg)
                        else:
                            dirs = [d for d in os.listdir(self.jobparams.precomputed_msas_path.get_value()) if os.path.isdir(os.path.join(self.jobparams.precomputed_msas_path.get_value(), d))]
                        for desc in seq_descriptions:
                            if not desc in dirs:
                                missing_msa_list.append(desc)
                        if len(missing_msa_list) > 0:
                            missing_msa_string = ', '.join(missing_msa_list)
                            msg = (f"MSA(s) for {missing_msa_string} not found in {self.jobparams.precomputed_msas_path.get_value()}\n\n"
                                        f"Pairwise combinatorial prediction requires precomputed MSAs for all "
                                        f"sequences (from a batch_msas job). In the 'Precomputed MSAs path' choose "
                                        f"a directory that contains subfolders with MSAs for all subunits and from the same feature pipeline. "
                                        f"e.g. /path/to/project/batch_msas_job_name/features/{job_params['db_preset']}")
                            message_dlg('error', msg) 
                            raise InputError(msg)      


                    params_hash = self.jobparams.get_params_hash()
                    job_params['params_hash'] = params_hash

                    #Full path to results folder created by AF inside job folder
                    if job_params['pairwise_batch_prediction']:
                        job_params['results_path'] = os.path.join(job_params['job_path'], "predictions", f"{job_params['prediction']}-{job_params['db_preset']}-{params_hash}")
                        self.jobparams.predictions_dir.set_value(f"{job_params['prediction']}-{job_params['db_preset']}-{params_hash}")
                        #Backward compatibility
                        if not os.path.exists(job_params['results_path']):
                            job_params['results_path'] = os.path.join(job_params['job_path'], "predictions", job_params['prediction'])
                            if not os.path.exists(job_params['results_path']):
                                job_params['results_path'] = job_params['job_path']
                    else:
                        job_params['results_path'] = os.path.join(job_params['job_path'], "predictions", f"{job_params['prediction']}-{job_params['db_preset']}-{params_hash}")
                        self.jobparams.predictions_dir.set_value(f"{job_params['prediction']}-{job_params['db_preset']}-{params_hash}")
                        #Backward compatibility
                        if not os.path.exists(job_params['results_path']):
                            job_params['results_path'] = os.path.join(job_params['job_path'], "predictions", job_params['prediction'])
                            if not os.path.exists(job_params['results_path']):
                                job_params['results_path'] = os.path.join(job_params['job_path'], job_params['job_dir'])

                    logger.debug(f"job path {job_params['job_path']}")
                    logger.debug(f"Log file {job_params['log_file']}")

                    #Check if mmseqs_api is selected and give warning notice
                    if job_params['db_preset'] == 'colabfold_web' and not split_job_step == 'gpu' and not job_params['pipeline'] == 'continue_from_features':
                        if job_params['pipeline'] in self.screening_protocol_names:
                            message = "You selected the colabfold_web preset. In case of missing MSAs, this will send your sequences to the MMseqs2 server (https://www.colabfold.com). Please confirm or cancel."
                        else:
                            message = "You selected the colabfold_web preset. This will send your sequences to the MMseqs2 server (https://www.colabfold.com). Please confirm or cancel." 
                        ret = QtWidgets.QMessageBox.question(self, 'Warning', message,
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
                        if ret == QtWidgets.QMessageBox.Cancel:
                            raise JobSubmissionCancelledByUser

                    #Check if new incompatible characters were introduced to the sequence
                    _, error_msgs = self.jobparams.sequence_params.sanitize_sequence_str(self.jobparams.sequences.get_value())
                    if len(error_msgs) > 0:
                        error_msgs_str = '\n'.join(error_msgs)
                        message_dlg('Error', error_msgs_str)
                        #Disable running a new job if sequence has incorrect format
                        self.btn_jobparams_advanced_settings.setEnabled(False)
                        self.tb_run.setEnabled(False)
                        raise SequenceFormatError(error_msgs_str)
                    self.job.log_file.set_value(job_params['log_file'])
                    self.job.job_status_log_file.set_value(job_params['job_status_log_file'])
                    self.job.path.set_value(job_params['job_path'])
                    self.job.name.set_value(job_params['job_name'])
                    self.jobparams.output_dir.set_value(job_params['job_path'])

                    #Get protein names from sequence names
                    protein_names = self.jobparams.seq_names.get_value().replace(',','_')
                    job_params['protein_names'] = protein_names

                    #Check if features.pkl exists when continue_from_features selected
                    if job_params['pipeline'] == 'continue_from_features' and not split_job_step == 'gpu':
                        feature_path_1 = os.path.join(job_params['features_path'], f'features_{protein_names}.pkl')
                        feature_path_2 = os.path.join(job_params['results_path'], 'features.pkl')
                        if not os.path.exists(feature_path_1) and not os.path.exists(feature_path_2):
                            message_dlg('error', 'continue_from_features requested but no features.pkl'
                                                 f' file found. Expected to find {feature_path_1} or {feature_path_2}. Either run a full or only_features job.')
                            raise NoFeaturesExist(f"No features.pkl found.")
                        

                    #Check if precomputed msas set
                    if self.jobparams.precomputed_msas_path.is_set() or self.jobparams.precomputed_msas_list.list_like_str_not_all_none():
                        job_params['use_precomputed_msas'] = True

                    #Process custom templates
                    if self.jobparams.custom_template_list.is_set():
                        new_custom_template_list, multichain_template_path, msgs = self.jobparams.process_custom_template_files(job_params['job_path'])
                        self.jobparams.custom_template_list.set_value(new_custom_template_list)
                        if multichain_template_path:
                            self.jobparams.multichain_template_path.set_value(multichain_template_path)
                        for msg in msgs:
                            logger.debug(msg)
                            message_dlg('Error', msg)
                            raise ProcessCustomTemplateError(msg)


                    #Change Log display for batch jobs
                    if job_params['pipeline'] in ['all_vs_all', 'first_vs_all', 'batch_msas', 'first_n_vs_rest', 'grouped_bait_vs_preys', 'grouped_all_vs_all']:
                        if job_params['pipeline'] == 'all_vs_all':
                            len_seqs = len(sequences)
                            num_jobs = (len_seqs * (len_seqs -1)) / 2 + len_seqs
                        elif job_params['pipeline'] == 'first_vs_all':
                            len_seqs = len(sequences)
                            num_jobs = len_seqs
                        elif job_params['pipeline'] == 'first_n_vs_rest':
                            len_seqs = len(sequences[job_params['first_n_seq'] + 1:])
                            num_jobs = len_seqs
                        elif self.jobparams.pipeline.get_value() == 'batch_msas':
                            num_jobs = len(sequences)
                        elif self.jobparams.pipeline.get_value() in ['grouped_bait_vs_preys', 'grouped_all_vs_all']:
                            #TODO: Implement proper calculation
                            num_jobs = len(sequences)
                        num_jobs = int(num_jobs)
                        job_params['num_jobs'] = num_jobs
                        if job_params['pipeline'] in self.screening_protocol_names:
                            placeholder = 'prediction'
                        else:
                            placeholder = 'feature'
                        message = f"This will start a batch {placeholder} job with {num_jobs} tasks. Continue?"
                        ret = QtWidgets.QMessageBox.question(self, 'Warning', message,
                                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                        if ret == QtWidgets.QMessageBox.No:
                            raise JobSubmissionCancelledByUser

                        for item in [self.lbl_status_2,
                                    self.lbl_status_3,
                                    self.lbl_status_4,
                                    self.lbl_status_5,
                                    self.lbl_status_6,
                                    self.lbl_status_7]:
                                    item.setHidden(True)
                        self.lbl_status_1.setText(f'0/{num_jobs} tasks finished')
                    else:
                        #Reset labels in case they were changed/hidden by a previous job
                        for item in [self.lbl_status_2,
                            self.lbl_status_3,
                            self.lbl_status_4,
                            self.lbl_status_5,
                            self.lbl_status_6,
                            self.lbl_status_7]:
                            item.setHidden(False)
                        self.lbl_status_1.setText('Feature generation')

                        

                    #Prepare DB objects
                    project_obj = self.prj.get_project_by_id(self.gui_params['project_id'], self.sess)
                    params_dict_db = self.jobparams.get_dict_db_insert()
                    params_obj_list = self.jobparams.generate_db_object(params_dict_db)
                    jobs_dict_db = self.job.get_dict_db_insert()
                    logger.debug(jobs_dict_db)
                    jobs_obj_list = self.job.generate_db_object(jobs_dict_db)
                    jobs_obj_list[0].jobparams_collection.append(params_obj_list[0])
                    project_obj.job_collection.append(jobs_obj_list[0])

                    #DB insert
                    self.sess.add(project_obj)
                    self.sess.commit()

                    #Get job params
                    job_id = jobs_obj_list[0].id

                    #Update job params
                    settings = self.settings.get_from_db(self.sess)
                    self.settings.update_from_db(settings)
                    job_params['id'] = job_id
                    job_params['job_id'] = job_id

                    #Check if number of sequences has changed
                    if len(self.jobparams.seq_names.get_value().split(',')) != len(sequences):
                        logger.debug(len(self.jobparams.seq_names.value.split(',')))
                        logger.debug(len(sequences))
                        error = "The number of sequences has changed. Re-read sequences before running the job."
                        message_dlg('Error', error)
                        raise SequenceFormatError(error)

                    #Setup queue related vars
                    if self.jobparams.queue.value:
                        job_params['queue'] = True
                        if settings.max_gpu_mem is None or settings.max_gpu_mem == '':
                            message_dlg("error", "Requested to submit to queue but found no GPU maximum memory value in settings.")
                            raise QueueSubmitError("Maximum GPU memory not specified")
                        else:
                            job_params['max_gpu_mem'] = settings.max_gpu_mem
                        if settings.max_ram is None or settings.max_ram == '':
                            message_dlg("error", "Requested to submit to queue but found no maximum RAM value in settings.")
                            raise QueueSubmitError("Maximum available RAM not specified")
                        else:
                            job_params['max_ram'] = settings.max_ram
                        if settings.min_ram is None or settings.min_ram == '':
                            message_dlg("error", "Requested to submit to queue but found no minimum RAM value in settings.")
                            raise QueueSubmitError("Minimum RAM not specified")
                        else:
                            job_params['min_ram'] = settings.min_ram
                        if settings.min_cpus is None or settings.min_cpus == '':
                            message_dlg("error", "Requested to submit to queue but found no minimum CPU number in settings.")
                            raise QueueSubmitError("Minimum CPU number not specified")
                        if settings.max_cpus is None or settings.max_cpus == '':
                            message_dlg("error", "Requested to submit to queue but found no maximum CPU number in settings.")
                            raise QueueSubmitError("Maximum CPU number not specified")
                        if settings.queue_account is None or settings.queue_account == '':
                            job_params['queue_account'] = None
                        else:
                            job_params['queue_account'] = settings.queue_account

                        if settings.queue_submit is None or settings.queue_submit == '':
                            message_dlg("error", "Requested to submit to queue but queue submit command not specified in settings.")
                            raise QueueSubmitError("Queue submit command not specified")
                        elif not self.job.check_queue_submit_cmd(settings.queue_submit):
                            message_dlg("error", f"Queue submit command {settings.queue_submit} is not available on this host.")
                            raise QueueSubmitError("Queue submit command not found")
                        else:
                            job_params['queue_submit'] = settings.queue_submit
                    batch_max_sequence_length = self.jobparams.batch_max_sequence_length.get_value()
                    if job_params['pipeline'] == 'first_vs_all':
                        seq_len_list = []
                        for i, seq in enumerate(sequences):
                            if i == 0:
                                len_seq1 = len(seq)
                                seq_len_list.append(len_seq1 + len_seq1)
                            else:
                                len_seq = len(seq)
                                seq_len_list.append(len_seq + len_seq1)
                        max_seq_len = max(seq_len_list)
                        if max_seq_len > batch_max_sequence_length:
                            job_params['total_seqlen'] = batch_max_sequence_length
                        else:
                            job_params['total_seqlen'] =  max_seq_len
                    elif job_params['pipeline'] in ['grouped_bait_vs_preys', 'grouped_all_vs_all']:
                        groups = {}
                        for i, desc in enumerate(seq_descriptions):
                            regex_bait = r'bait_([^_]+)(?:_split\d+)?'
                            regex_prey = r'prey_([^_]+)(?:_split\d+)?'
                            if re.search(regex_bait, desc):
                                id = re.search(regex_bait, desc).group(1)
                                if not id in groups:
                                    groups[id] = {'baits': [], 'preys': []}
                                groups[id]['baits'].append(sequences[i])
                            if re.search(regex_prey, desc):
                                id = re.search(regex_prey, desc).group(1)
                                if not id in groups:
                                    groups[id] = {'baits': [], 'preys': []}
                                groups[id]['preys'].append(sequences[i])
                        logger.info(groups)
                        if job_params['pipeline'] == 'grouped_bait_vs_preys':
                            max_seq_len_group_list = []
                            for id, group in groups.items():
                                total_bait_seq_len = sum([len(seq) for seq in group['baits']])
                                total_seq_len_list = []
                                for prey_seq in group['preys']:
                                    total_seq_len = total_bait_seq_len + len(prey_seq)
                                    total_seq_len_list.append(total_seq_len)
                                if len(total_seq_len_list) > 0:
                                    max_seq_len_group_list.append(max(total_seq_len_list))
                            if len(max_seq_len_group_list) > 0:
                                max_seq_len = max(max_seq_len_group_list)
                            logger.info(max_seq_len_group_list)
                            if max_seq_len > batch_max_sequence_length:
                                job_params['total_seqlen'] = batch_max_sequence_length
                            else:
                                job_params['total_seqlen'] =  max_seq_len
                        if job_params['pipeline'] == 'grouped_all_vs_all':
                            max_group_len = []
                            for id, group in groups.items():
                                all_seqs = group['baits'] + group['preys']
                                pair_len = []
                                for seq1 in all_seqs:
                                    for seq2 in all_seqs:
                                        pair_len.append(len(seq1) + len(seq2))
                                if len(pair_len) > 0:
                                    max_pair_len = max(pair_len)
                                max_group_len.append(max_pair_len)
                            if len(max_group_len) > 0:
                                max_seq_len = max(max_group_len) 
                            if max_seq_len > batch_max_sequence_length:
                                job_params['total_seqlen'] = batch_max_sequence_length
                            else:
                                job_params['total_seqlen'] =  max_seq_len
                    elif job_params['pipeline'] == 'first_n_vs_rest':
                        len_baits = []
                        len_seqs_list = []
                        for i, seq in enumerate(sequences):
                            if i < int(job_params['first_n_seq']):
                                len_baits.append(len(seq))
                            else:
                                len_seqs = sum(len_baits) + len(seq)
                                len_seqs_list.append(len_seqs)
                        max_seq_len = max(len_seqs_list)
                        if max_seq_len > batch_max_sequence_length:
                            job_params['total_seqlen'] = batch_max_sequence_length
                        else:
                            job_params['total_seqlen'] =  max_seq_len
                    elif job_params['pipeline'] == 'all_vs_all':
                        seq_len_list = []
                        for seq1 in sequences:
                            for seq2 in sequences:
                                seq_len_list.append(len(seq1) + len(seq2))
                        max_seq_len = max(seq_len_list)
                        if max_seq_len > batch_max_sequence_length:
                            job_params['total_seqlen'] = batch_max_sequence_length
                        else:
                            job_params['total_seqlen'] =  max_seq_len
                    elif job_params['pipeline'] == 'only_relax':
                        #Dummy sequence length to calculate memory
                        job_params['total_seqlen'] = 2000
                    else:
                        job_params['total_seqlen'] = sum([len(s) for s in sequences])
                    logger.debug(f"Number of amino acids: {job_params['total_seqlen']}")
                    job_params['multimer'] = True if len(sequences) > 1 else False
                    logger.debug(job_params)
                    logger.debug(self.gui_params)
                    self.gui_params['job_id'] = job_id



                    #Clear contents of file if it already exists
                    with open(os.path.join(job_params['job_path'], job_params['log_file']), 'w'):
                        pass


                    #Start thread
                    # if not os.path.exists(job_params['results_path']):
                    #     try:
                    #         os.mkdir(job_params['results_path'])
                    #     except:
                    #         message_dlg('Error', f"Could not create job directory in {job_params['job_path']}!")
                    #         logger.debug("Could not create job directory!")
                    #         raise DirectoryNotCreated


                    #thread.daemon = True
                    cmd_dict_jobparams = self.jobparams.get_dict_cmd()
                    cmd_dict_settings = self.settings.get_dict_cmd()
                    logger.debug("cmd dict settings")
                    logger.debug(cmd_dict_settings)
                    cmd_dict = {**cmd_dict_jobparams, **cmd_dict_settings}
                    #Backward compatibility
                    if 'num_cpu' in cmd_dict:
                        cmd_dict['num_cpu'] = job_params['num_cpu']
                    if 'use_precomputed_msas' in job_params:
                        if job_params['use_precomputed_msas']:
                            cmd_dict['use_precomputed_msas'] = ""
                    #Do not use precomputed MSAs in case of colabfold batch mode
                    if job_params['pipeline'] == 'batch_msas' and job_params['db_preset'] == 'colabfold_local':
                        if 'use_precomputed_msas' in cmd_dict:
                            del cmd_dict['use_precomputed_msas']
                        if 'precomputed_msas_path' in cmd_dict:
                            del cmd_dict['precomputed_msas_path']
                        if 'precomputed_msas_list' in cmd_dict:
                            del cmd_dict['precomputed_msas_list']
                    if job_params['force_cpu']:
                       del cmd_dict['use_gpu_relax']
                    if job_params['db_preset'] == 'full_dbs':
                        del cmd_dict['small_bfd_database_path']
                        del cmd_dict['uniref30_mmseqs_database_path']
                        del cmd_dict['colabfold_envdb_database_path']
                    if job_params['db_preset'] == 'reduced_dbs':
                        del cmd_dict['bfd_database_path']
                        del cmd_dict['uniref30_database_path']
                        del cmd_dict['uniref30_mmseqs_database_path']
                        del cmd_dict['colabfold_envdb_database_path']
                    if job_params['db_preset'] == 'colabfold_local':
                        del cmd_dict['small_bfd_database_path']
                        del cmd_dict['uniref30_database_path']
                    if job_params['db_preset'] == 'colabfold_web':
                        del cmd_dict['small_bfd_database_path']
                        del cmd_dict['uniref30_database_path']
                        del cmd_dict['uniref30_mmseqs_database_path']
                        del cmd_dict['colabfold_envdb_database_path']
                    #In case list-like strings contain only None remove them from the command
                    if not self.jobparams.no_msa_list.list_like_str_not_all_false() and 'no_msa_list' in cmd_dict:
                        del cmd_dict['no_msa_list']
                    if not self.jobparams.no_template_list.list_like_str_not_all_false() and 'no_template_list' in cmd_dict:
                        del cmd_dict['no_template_list']
                    if not self.jobparams.custom_template_list.list_like_str_not_all_none() and 'custom_template_list' in cmd_dict:
                        del cmd_dict['custom_template_list']
                    if not self.jobparams.precomputed_msas_list.list_like_str_not_all_none() and 'precomputed_msas_list' in cmd_dict:
                        del cmd_dict['precomputed_msas_list']

                    logger.debug(job_params['force_cpu'])
                    logger.debug(cmd_dict)
                    logger.debug("Job IDs before notebook: {}".format(self.gui_params['job_id']))


                    #Prepare command to run alphafold
                    self.job_params = job_params
                    if job_params['split_job'] and job_params['queue']:
                        cmd, error_msgs, warn_msgs, self.job_params['calculated_mem'] = self.job.prepare_cmd(self.job_params, cmd_dict, split_job_step=split_job_step)
                    else:
                        cmd, error_msgs, warn_msgs, self.job_params['calculated_mem'] = self.job.prepare_cmd(self.job_params, cmd_dict)

                    for msg in error_msgs:
                        logger.debug(msg)
                        message_dlg('Error', msg)
                        raise PrepareCMDError(msg)
                    
                    for msg in warn_msgs:
                        ret = QtWidgets.QMessageBox.question(self, 'Warning',
                                                             msg,
                                                             QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                        if ret == QtWidgets.QMessageBox.No:
                            raise JobSubmissionCancelledByUser

                    logger.debug(f"split queue: {job_params['split_job']} split_job_step {split_job_step}")
                    if job_params['queue']:
                        if job_params['split_job_step'] == 'cpu':
                            job_type_description = "First submission script for the feature generation step (The second submission script will be shown after pressing OK):"
                        elif job_params['split_job_step'] == 'gpu':
                            job_type_description = "Second submission script for the prediction step:"
                        else:
                            job_type_description = "Submission script:"
                        dlg = QueueSubmitDlg(self, job_type_description)
                        result = dlg.exec()
                        if not result:
                            raise JobSubmissionCancelledByUser

                    #Start the job and monitor threads
                    jobs_started += 1
                    job_params['jobs_started'] = jobs_started
                    self.start_thread(job_params, cmd)

            except Exception:
                logging.debug("Exception in start job")
                job_params['status'] = 'error'
                self.OnJobStatus(job_params)
                traceback.print_exc()
        else:
            logger.error("Too many jobs started without pressing run button.")

    def OnChangeTab(self):
        self.notebook.setCurrentIndex(1)
        self.gui_params = self.job.init_gui(self.gui_params, other=self, sess=self.sess)

    def OnBtnCancel(self):
        if self.gui_params['job_id'] is None:
            message_dlg('Error', 'No Job selected!', 'Error')
        elif self.gui_params['queue']:
            logger.debug("Cancel queue job")
            settings = self.settings.get_from_db(self.sess)
            queue_cancel = settings.queue_cancel
            host = self.job.get_host(self.gui_params['job_id'], self.sess)
            current_host = socket.gethostname()
            if host == current_host:
                if not queue_cancel == "" or not queue_cancel is None:
                    pid = self.job.get_pid(self.gui_params['job_id'], self.sess)
                    cmd = [queue_cancel, pid]
                    logger.debug(f"Queue pid {pid}")
                    try:
                        Popen(cmd)
                        self.job.update_status("aborted", self.gui_params['job_id'], self.sess)
                        self.job.init_gui(self.gui_params, other=self, sess=self.sess)
                        self.job.update_log(os.path.join(self.job_params['job_path'], self.gui_params['log_file']), int(self.gui_params['job_id']), append=False)
                    except:
                        cmd = ' '.join(cmd)
                        message_dlg('Error', f'Cannot cancel job. The command was {cmd}!')
                else:
                    message_dlg('Error', 'No cancel command for queue submission method defined!')
            else:
                message_dlg('Error', f'Cannot cancel this job because it was started on a different host ({host})'
                              f' and current host is {current_host}!')
        else:
            pid = self.job.get_pid(self.gui_params['job_id'], self.sess)
            host = self.job.get_host(self.gui_params['job_id'], self.sess)
            current_host = socket.gethostname()
            if host == current_host:
                try:
                    logger.debug("PID is {}".format(pid))
                    os.killpg(int(pid), signal.SIGINT)
                    os.killpg(int(pid), signal.SIGTERM)
                    self.job.update_status("aborted", self.gui_params['job_id'], self.sess)
                    self.job.init_gui(self.gui_params, other=self, sess=self.sess)
                    self.job.update_log(os.path.join(self.job_params['job_path'], self.gui_params['log_file']), int(self.gui_params['job_id']), append=False)
                except (TypeError, ProcessLookupError):
                    message_dlg('Error', 'No process ID found for this job!')
                except Exception as e:
                    logger.debug(e)
                    traceback.print_exc()
                    message_dlg('Error', 'Cannot cancel this job!')
            else:
                message_dlg('Error', f'Cannot cancel this job because it was started on a different host ({host})'
                              f' and current host is {current_host}!')

    # def OnBtnOutputDir(self):
    #     logger.debug(f"OnBtnChooseFolder")
    #     path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
    #     self.jobparams.output_dir.ctrl.setText(path)
    #     # if dlg.exec_():
    #     #     path = dlg.
    #     #     logger.debug(path)

    def job_init_gui_slot(self, gui_params, reset, other, sess, thread):
        logging.debug("job init")
        self.gui_params = self.job.init_gui(gui_params, reset=reset, other=other, sess=sess)
        thread.resume()

    def OnCmbProjects(self):
        project_name = str(self.prj.list.ctrl.currentText())
        progress_dlg = ProgressDialog()
        center_on_screen(progress_dlg)
        self.prj_load_thread = QThread()
        self.prj_load_worker = gui_threads.LoadProject(self, project_name)
        self.prj_load_worker.moveToThread(self.prj_load_thread)
        self.prj_load_thread.started.connect(self.prj_load_worker.run)
        self.prj_load_thread.finished.connect(self.prj_load_thread.quit)
        self.prj_load_worker.finished.connect(self.prj_load_thread.quit)
        self.prj_load_thread.finished.connect(self.prj_load_worker.deleteLater)
        self.prj_load_thread.finished.connect(self.prj_load_thread.deleteLater)
        self.prj_load_worker.job_init_gui_signal.connect(self.job_init_gui_slot)
        self.prj_load_worker.clear_signal.connect(self.OnBtnClear)
        self.prj_load_worker.progress_signal.connect(progress_dlg.update_progress_bar)
        self.prj_load_worker.error_signal.connect(progress_dlg.error)
        self.prj_load_thread.start()
        progress_dlg.accept_button.clicked.connect(progress_dlg.accept)
        progress_dlg.exec()


    # def OnCmbProjects(self):
    #     logger.debug("OnCmbProjects")
    #     project_name = str(self.prj.list.ctrl.currentText())
    #     if not project_name is None and not project_name == "":
    #         dlg = LoadDialog(text="Loading project...")
    #         close_button = dlg.button(QtWidgets.QMessageBox.Close)
    #         close_button.clicked.connect(dlg.accept)
    #         dlg.show()
    #         #self.app.processEvents()
    #         logger.debug(f"Current project {project_name}")
    #         new_project_id = self.prj.change_active_project(project_name, self.sess)
    #         logger.debug(f"New project ID {new_project_id}")
    #         self.gui_params['project_id'] = new_project_id
    #         self.gui_params['project_path'] = self.prj.get_path_by_project_id(new_project_id, self.sess)
    #         self.gui_params['other_settings_changed'] = False
    #         self.gui_params = self.job.init_gui(self.gui_params, reset=True, other=self, sess=self.sess)
    #         dlg.accept()
            #self.app.processEvents()

    def OnCmbCombinations(self):
        combination_name = str(self.evaluation.pairwise_combinations_list.ctrl.currentText())
        self.gui_params['selected_combination_name'] = combination_name
        logger.debug(f"Combination name: {self.gui_params['selected_combination_name']}")
        self.gui_params = self.evaluation.init_gui(self.gui_params, self.sess)
        self.evaluation.pairwise_combinations_list.ctrl.setCurrentText(combination_name)
        self.gui_params['results_path_combinaton'] = os.path.join(self.gui_params['results_path'], combination_name)

    def OnCmbPipeline(self):
        pipeline_name = self.jobparams.pipeline.ctrl.currentText()
        if pipeline_name in self.screening_protocol_names:
            self.jobparams.num_recycle.set_value('3')
        if pipeline_name in ['full', 'continue_from_features', 'first_vs_all', 'all_vs_all', 'first_n_vs_rest', 'grouped_bait_vs_preys', 'grouped_all_vs_all']:
            self.jobparams.force_cpu.set_value(False)
        if pipeline_name == 'first_n_vs_rest':
            self.jobparams.update_from_gui()
            dlg = FirstNSeqDlg(self)
            dlg.exec()
        if pipeline_name == 'batch_msas':
            self.btn_split_sequences.setEnabled(True)
        else:
            self.btn_split_sequences.setEnabled(False)
        if pipeline_name in self.screening_protocol_names:
            if 'features_path' in self.gui_params:
                batch_msas_path = self.gui_params["features_path"]
                if batch_msas_path:
                    if os.path.exists(batch_msas_path):
                        logger.debug(f"batch msa folder found: {batch_msas_path} ")
                        self.jobparams.precomputed_msas_path.set_value(batch_msas_path)
                        self.jobparams.precomputed_msas_path.ctrl.setText(batch_msas_path)
                    else:
                        logger.debug(f"{batch_msas_path} does not exist")
                else:
                    logger.debug("batch_msas_path not defined")
            else:
                logger.debug(f"features_path not in gui_params")

    def OnCmbPrediction(self):
        prediction = self.jobparams.prediction.ctrl.currentText()
        if prediction == 'alphafold':
            self.jobparams.num_gpu.set_value(1)

    def OnCmbDbPreset(self):
        logger.debug("OnCmbDbPreset")
        db_preset = self.jobparams.db_preset.ctrl.currentText()
        logger.debug(f"db_preset {db_preset}")
        if db_preset == 'colabfold_local':
            msgs = self.validate_settings(category='colabfold')
            if len(msgs) > 0:
                msgs.insert(0, 'Settings are not configured properly for colabfold_local pipeline:')
                message_dlg('warning', '\n'.join(msgs))


    def update_from_db_slot(self, params, thread):
        self.jobparams.update_from_db(params)
        thread.resume()
    
    def insert_evaluation_slot(self, evaluation, params, sess, thread):
        self.job.insert_evaluation(evaluation, params, sess)
        thread.resume()

    

    def OnLstJobSelected(self, item):
        self.job_load_threads = []
        self.job_load_workers = []
        for _ in range(len(self.job_load_threads)):
            thread = self.job_load_threads.pop()
            thread.quit()
            worker = self.job_load_workers.pop()
        logger.debug("Job load threads and workers:")
        logger.debug(self.job_load_threads)
        logger.debug(self.job_load_workers)

        logger.debug("OnLstJobSelected")
        job_project_id = item.text(2)
        progress_dlg = ProgressDialog()
        center_on_screen(progress_dlg)
        self.job_load_thread = QThread()
        self.job_load_worker = gui_threads.LoadJob(self, job_project_id)
        self.job_load_worker.moveToThread(self.job_load_thread)
        self.job_load_thread.started.connect(self.job_load_worker.run)
        self.job_load_thread.finished.connect(self.job_load_thread.quit)
        self.job_load_worker.finished.connect(self.job_load_thread.quit)
        self.job_load_thread.finished.connect(self.job_load_worker.deleteLater)
        self.job_load_thread.finished.connect(self.job_load_thread.deleteLater)
        #self.job_load_worker.update_signal.connect(progressDialog.updateProgressBar)
        self.job_load_worker.btn_jobparams_advanced_settings_set_enabled_signal.connect(self.btn_jobparams_advanced_settings.setEnabled)
        self.job_load_worker.tb_run_set_enabled_signal.connect(self.tb_run.setEnabled)
        self.job_load_worker.insert_evaluation_signal.connect(self.insert_evaluation_slot)
        self.job_load_worker.update_from_db_signal.connect(self.update_from_db_slot)
        self.job_load_worker.set_item_hidden_signal.connect(lambda item, state: item.setHidden(state))
        self.job_load_worker.notebook_tab_signal.connect(self.notebook.setTabEnabled)
        self.job_load_worker.progress_signal.connect(progress_dlg.update_progress_bar)
        self.job_load_worker.error_signal.connect(progress_dlg.error)
        self.job_load_worker.job_status_signal.connect(self.OnJobStatus)
        self.job_load_worker.evaluation_init_signal.connect(self.evaluation.init_gui)
        self.job_load_worker.lbl_status_signal.connect(self.lbl_status_1.setText)
        self.job_load_thread.start()
        self.job_load_threads.append(self.job_load_thread)
        self.job_load_workers.append(self.job_load_workers)
        progress_dlg.accept_button.clicked.connect(progress_dlg.accept)
        progress_dlg.exec()
        #self.job_load_thread.wait()
        #while True:
        #    if self.job_load_thread.isFinished():
        #        break

        #        self.job_load_thread.quit()


        #self.app.processEvents()

    def OnJobContextMenu(self, pos):
        item = self.job.list.ctrl.itemAt(pos)
        if not item is None:
            menu = QtWidgets.QMenu()
            job_project_id = item.text(2)
            job_name = item.text(1)
            job_id = self.job.get_job_id_by_job_project_id(job_project_id, self.gui_params['project_id'], self.sess)
            job_dir = self.job.get_job_dir(job_name)
            project_path = self.prj.get_path_by_project_id(self.gui_params['project_id'], self.sess)
            job_path = self.job.get_job_path(project_path, job_dir)
            logger.debug(f"job project id {job_project_id} job name {job_name} job id {job_id}")
            self.delete_job_action = QtWidgets.QAction("Delete Job", self)
            self.delete_jobfiles_action = QtWidgets.QAction("Delete Job+Files", self)
            self.set_finished_action = QtWidgets.QAction("Set to finished", self)
            self.set_running_action = QtWidgets.QAction("Set to running", self)
            menu.addAction(self.delete_job_action)
            #menu.addAction(self.delete_jobfiles_action)
            menu.addAction(self.set_finished_action)
            menu.addAction(self.set_running_action)
            self.delete_job_action.triggered.connect(lambda state, x=job_id: self.OnDeleteEntry(x))
            #self.delete_jobfiles_action.triggered.connect(lambda state, x=job_id, y=job_path: self.OnDeleteEntryFiles(x, y))
            self.set_finished_action.triggered.connect(lambda state, x=job_id: self.OnStatusFinished(x))
            self.set_running_action.triggered.connect(lambda state, x=job_id: self.OnStatusRunning(x))
            menu.exec_(self.job.list.ctrl.mapToGlobal(pos))

    def OnDeleteEntry(self, job_id):
        logger.debug(f"OnDeleteEntry. Job id {job_id}")
        self.job.delete_job(job_id, self.sess)
        self.gui_params = self.job.init_gui(self.gui_params, other=self, sess=self.sess)

    def OnDeleteEntryFiles(self, job_id, job_path):
        self.job.delete_job_files(job_id, job_path, self.sess)
        self.gui_params = self.job.init_gui(self.gui_params, other=self, sess=self.sess)

    def OnStatusRunning(self, job_id):
        self.job.update_status("running", job_id, self.sess)
        self.gui_params = self.job.init_gui(self.gui_params, other=self, sess=self.sess)

    def OnStatusFinished(self, job_id):
        self.job.update_status("finished", job_id, self.sess)
        self.gui_params = self.job.init_gui(self.gui_params, other=self, sess=self.sess)

    def OnBtnPrjAdd(self):
        dlg = ProjectDlg(self, "add")
        dlg.exec()
        self.gui_params = self.prj.init_gui(self.gui_params, sess=self.sess)
        self.OnBtnClear()
        logger.debug("PrjAdd button pressed")

    def OnBtnPrjRemove(self):
        logger.debug("PrjRemove button pressed")
        prj_name, prj_id = self.prj.get_active_project(self.sess)

        ret = QtWidgets.QMessageBox.question(self, 'Warning',
                                             f"Do you want to remove the project {prj_name} from the database (files will not be removed)?",
                                             QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if ret == QtWidgets.QMessageBox.Yes:
            self.prj.delete_project(prj_id, self.sess)
        self.gui_params = self.prj.init_gui(self.gui_params, sess=self.sess)

    def OnBtnPrjUpdate(self):
        dlg = ProjectDlg(self, "update")
        dlg.exec()
        self.gui_params = self.prj.init_gui(self.gui_params, sess=self.sess)
        logger.debug("PrjUpdate button pressed")

    def OnBtnAdvancedParams(self):
        self.jobparams.update_from_gui()
        dlg = AdvancedParamsDlg(self)
        dlg.exec()

    def OnBtnSettings(self):
        if not self.gui_params['settings_locked']:
            dlg = SettingsDlg(self)
            dlg.exec()
        else:
            message_dlg("Info", "Changing of settings is locked by the administrator.")

    def OnBtnPrecomputedMSAsPath(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.jobparams.precomputed_msas_path.set_value(path)
        self.jobparams.precomputed_msas_path.ctrl.setText(path)

    def OnBtnMultichainTemplatePath(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, 'Select CIF file', filter='CIF Files (*.cif)')
        self.jobparams.multichain_template_path.set_value(path[0])
        self.jobparams.multichain_template_path.ctrl.setText(path[0])

    def OnOpenModelViewer(self, model_viewer):
        if self.gui_params['results_path_combination']:
            results_path = self.gui_params['results_path_combination']
            logger.debug(f'Using results path for combination: {results_path}')
        elif 'results_path' in self.gui_params:
            results_path = self.gui_params['results_path']
            logger.debug(f'Using results path: {results_path}')
        else:
            logger.debug('results_path not in gui_params')
            return

        pdb_files = glob(os.path.join(results_path, "*.pdb"))
        pdb_files = ' '.join(pdb_files)
        logger.debug(f"PDB files {pdb_files}")
        pymol_pml = pkg_resources.resource_filename("guifold.templates", "pymol.pml")
        try:
            if model_viewer == 'pymol':
                Popen(f'{model_viewer} {pdb_files} {pymol_pml}', shell=True)
            else:
                Popen(f'{model_viewer} {pdb_files}', shell=True)
        except:
            error = f"Could not open {model_viewer}. Check if it is in your PATH."
            logger.error(error)
            message_dlg('Error', error)
            logger.debug(traceback.print_exc())
            

    def OnOpenBrowser(self):
        if 'results_html' in self.gui_params:
            if self.gui_params['pairwise_batch_prediction']:
                if re.search('results.html', self.gui_params['results_html']):
                    results_html = self.gui_params['results_html'].replace('results.html', 'results_model_viewer.html')
                else:
                    results_html = self.gui_params['results_html']
            else:
                results_html = self.gui_params['results_html'].replace('results.html', 'results_model_viewer.html')
            logger.debug(f"Opening {results_html} in web browser")
            if os.path.exists(results_html):
                url = QUrl(f'file://{results_html}')
                try:
                    QDesktopServices.openUrl(url)
                except:
                    error =  f"Could not open {results_html} in an external browser. Maybe a default browser is not set."
                    logger.error(error)
                    message_dlg('Error', error)
                    logger.debug(traceback.print_exc())
            else:
                logger.error(f'Results path {results_html} does not exist')
        else:
            logger.error('results_path not in gui_params')


    def OnOpenResultsFolder(self):
            logger.debug("open in file manager")
            if 'results_path' in self.gui_params:
                if os.path.exists(self.gui_params['results_path']):
                    results_path = self.gui_params['results_path']
                else:
                    results_path = None
                
                if results_path:
                    qfile = QUrl.fromLocalFile(
                        results_path)
                    logger.debug(qfile)
                    try:
                        QDesktopServices.openUrl(qfile)
                    except:
                        error = f"Could not open {results_path} in a file manager. Maybe a default file manager is not set"
                        logger.error(error)
                        message_dlg('Error', error)
                        logger.debug(traceback.print_exc())
                else:
                    logger.error(f"Results path {results_path} does not exist")

            else:
                logger.error('results_path not in gui_params')


    def OnClearLog(self):
        logger.debug("clear log")
        self.job.log.reset_ctrl()

    def OnBtnClear(self):
        logger.debug("Clear Btn pressed")
        self.jobparams.reset_ctrls()
        #self.jobparams.update_from_default(self.default_values)
        self.gui_params['job_id'] = None
        self.gui_params['job_project_id'] = None
        self.gui_params['other_settings_changed'] = False
        self.jobparams.init_gui(self.gui_params, self.sess)
        self.init_gui()


    def OnItemExpanded(self, item):
        # Item expanded
        fields = [item.text(column) for column in range(item.columnCount())]
        job_name = fields[0]
        logger.debug(f"Item expanded: {job_name}")
        self.job.update_tree_item_expanded(job_name, self.gui_params['project_id'], True, self.sess)

    def OnItemCollapsed(self, item):
        # Item collapsed
        fields = [item.text(column) for column in range(item.columnCount())]
        job_name = fields[0]
        logger.debug(f"Item collapsed: {job_name}")
        self.job.update_tree_item_expanded(job_name, self.gui_params['project_id'], False, self.sess)

if __name__ == "__main__":
    main()
