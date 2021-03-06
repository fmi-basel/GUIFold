#Copyright 2022 Georg Kempf, Friedrich Miescher Institute for Biomedical Research
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
from __future__ import absolute_import
from guifold.src import gui_threads
from guifold.src.gui_dialogs import message_dlg
from guifold.src.gui_dlg_settings import SettingsDlg
from guifold.src.gui_dlg_about import AboutDlg
from guifold.src.gui_dlg_project import ProjectDlg
from guifold.src.gui_dlg_queue_submit import QueueSubmitDlg
from guifold.src.gui_dlg_advanced_params import AdvancedParamsDlg
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
from sqlalchemy import create_engine
import traceback
from guifold.src.gui_classes import Job, Settings, JobParams, Project, Evaluation, DefaultValues
import argparse

#Required for WebGL to work
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = "--ignore-gpu-blacklist"
#Prevent error when OpenGL not available
os.environ['QMLSCENE_DEVICE'] = "softwarecontext"
#Maybe makes Webview loading faster
QNetworkProxyFactory.setUseSystemConfiguration(False)

parser = argparse.ArgumentParser(description="GUI for running alphafold")
parser.add_argument('--debug', '-r',
                    help='Debug log.',
                    action='store_true')
args, unknown = parser.parse_known_args()

install_path = os.path.dirname(os.path.realpath(sys.argv[0]))
logger = logging.getLogger('guifold')
formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
if not args.debug:
    logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

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


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def main():
    shared_objects = [JobParams(), Project(), Job(), Evaluation(), Settings()]
    db = DBHelper(shared_objects)
    upgrade_db()
    with db.session_scope() as sess:
        db.set_session(sess)
    for obj in shared_objects:
        obj.set_db(db)

    sys.excepthook = handle_exception
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet('Fusion')
    MainFrame(shared_objects, db, sess, install_path)
    app.exec()

def upgrade_db():
    logger.debug("Upgrading DB")
    db_path = os.path.join(os.path.expanduser("~"), '.guifold.db')
    engine = create_engine('sqlite:///{}'.format(db_path), connect_args={'check_same_thread': False})
    stmts = ['ALTER TABLE settings ADD queue_submit_dialog BOOLEAN DEFAULT FALSE']
    with engine.connect() as conn:
        for stmt in stmts:
            try:
                rs = conn.execute(stmt)
            except Exception as e:
                logger.debug(e)
                #traceback.print_exc()


class MainFrame(QtWidgets.QMainWindow):
    def __init__(self, shared_objects, db, sess, install_path):
        super(MainFrame, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi(pkg_resources.resource_filename('guifold.ui', 'gui.ui'), self) # Load the .ui file


        self.install_path = install_path
        self.jobparams, self.prj, self.job, self.evaluation, self.settings = self.shared_objects = shared_objects


        self.gui_params = {'job_id': None,
                           'status': 'unknown',
                           'pid': None,
                           'queue_job_id': None,
                           'other_settings_changed': False,
                           'queue_account': None,
                           'job_project_id': None,
                           'project_id': None,
                           'queue': None}
        self.files_selected_item = None
        self.db = db
        self.sess = sess

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


    # self.check_executables()
        self.currentDirectory = os.getcwd()
        self.threads = []

        self.reconnect_jobs()
        self.show() # Show the GUI
        logger.debug("GUI params")
        logger.debug(self.gui_params)


    def init_frame(self):
        logger.debug("=== Initializing main frame ===")
        self.setWindowTitle("GUIFold")
        self.notebook = self.findChild(QtWidgets.QTabWidget, 'MainNotebook')
        self.notebook.setTabEnabled(2, False)
        self.panel = self.findChild(QtWidgets.QPushButton, 'InputPanel')
        #self.panel.SetScrollRate(20,20)
        self.log_panel = self.findChild(QtWidgets.QPushButton, 'LogPanel')
        self.btn_read_sequences = self.findChild(QtWidgets.QPushButton, 'btn_read_sequences')
        self.btn_jobparams_advanced_settings = self.findChild(QtWidgets.QPushButton, 'btn_jobparams_advanced_settings')
        self.btn_evaluation_open_results_folder = self.findChild(QtWidgets.QPushButton, 'btn_evaluation_open_results_folder')
        self.btn_evaluation_open_pymol = self.findChild(QtWidgets.QPushButton, 'btn_evaluation_open_pymol')
        self.btn_evaluation_open_chimerax = self.findChild(QtWidgets.QPushButton, 'btn_evaluation_open_chimerax')
        self.btn_open_browser = self.findChild(QtWidgets.QPushButton, 'btn_evaluation_open_browser')
        self.btn_prj_add = self.findChild(QtWidgets.QToolButton, 'btn_prj_add')
        self.btn_prj_add.setIcon(QIcon(pkg_resources.resource_filename('guifold.icons', 'gtk-add.png')))
        self.btn_prj_remove = self.findChild(QtWidgets.QToolButton, 'btn_prj_remove')
        self.btn_prj_remove.setIcon(QIcon(pkg_resources.resource_filename('guifold.icons', 'gtk-remove.png')))
        self.btn_prj_update = self.findChild(QtWidgets.QToolButton, 'btn_prj_update')
        self.btn_prj_update.setIcon(QIcon(pkg_resources.resource_filename('guifold.icons', 'gtk-edit.png')))
        self.btn_precomputed_msas_path = self.findChild(QtWidgets.QToolButton, 'btn_precomputed_msas_path')
        self.lbl_status_db_search = self.findChild(QtWidgets.QLabel, 'lbl_status_db_search')
        self.lbl_status_model_1 = self.findChild(QtWidgets.QLabel, 'lbl_status_model_1')
        self.lbl_status_model_2 = self.findChild(QtWidgets.QLabel, 'lbl_status_model_2')
        self.lbl_status_model_3 = self.findChild(QtWidgets.QLabel, 'lbl_status_model_3')
        self.lbl_status_model_4 = self.findChild(QtWidgets.QLabel, 'lbl_status_model_4')
        self.lbl_status_model_5 = self.findChild(QtWidgets.QLabel, 'lbl_status_model_5')
        self.lbl_status_evaluation = self.findChild(QtWidgets.QLabel, 'lbl_status_evaluation')
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

        self.exit_action.setShortcut('Ctrl+Q')
        self.exit_action.setStatusTip('Exit application')

        self.file_menu.addAction(self.exit_action)
        self.project_menu.addAction(self.add_prj_action)
        self.project_menu.addAction(self.delete_prj_action)
        self.project_menu.addAction(self.change_prj_action)
        self.help_menu.addAction(self.about_action)



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
            self.gui_params = obj.init_gui(self.gui_params, self.sess)
            logger.debug("GUI Params")
            logger.debug(self.gui_params)
        self.jobparams.update_from_default(self.default_values)

    def init_settings(self):
        logger.debug("=== Initializing Settings ===")
        if self.settings.add_blank_entry(self.sess):
            slurm_account = self.settings.get_slurm_account()
            if not slurm_account is None:
                self.settings.set_slurm_account(slurm_account, self.sess)
            self.settings.update_from_global_config(self.sess)
            self.settings.update_settings(self.settings.get_dict_db_insert(), self.sess)

    def create_monitor_thread(self, job_params):
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

    def check_project_exists(self):
        if self.prj.is_empty(self.sess):
            dlg = ProjectDlg(self, "add")
            if dlg.exec_():
                self.gui_params = self.prj.init_gui(self.gui_params, self.sess)
        if self.prj.is_empty(self.sess):
            logger.error("Cannot start GUI without initial project.")
            raise SystemExit

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
        #Toolbar
        self.tb.actionTriggered[QtWidgets.QAction].connect(self.ToolbarSelected)
        #Buttons
        self.btn_prj_add.clicked.connect(self.OnBtnPrjAdd)
        self.btn_prj_update.clicked.connect(self.OnBtnPrjUpdate)
        self.btn_prj_remove.clicked.connect(self.OnBtnPrjRemove)
        self.btn_read_sequences.clicked.connect(self.OnBtnReadSequences)
        self.btn_evaluation_open_pymol.clicked.connect(lambda checked, x='pymol': self.OnOpenModelViewer(x))
        self.btn_evaluation_open_chimerax.clicked.connect(lambda checked, x='chimerax': self.OnOpenModelViewer(x))
        self.btn_open_browser.clicked.connect(self.OnOpenBrowser)
        self.btn_evaluation_open_results_folder.clicked.connect(self.OnOpenResultsFolder)
        self.job.list.ctrl.cellClicked.connect(self.OnLstJobSelected)
        self.btn_jobparams_advanced_settings.clicked.connect(self.OnBtnAdvancedParams)
        self.btn_jobparams_advanced_settings.setEnabled(False)
        self.btn_precomputed_msas_path.clicked.connect(self.OnBtnPrecomputedMSAsPath)
        #Combos
        self.prj.list.ctrl.activated.connect(self.OnCmbProjects)
        self.job.list.ctrl.setContextMenuPolicy(Qt.CustomContextMenu)
        self.job.list.ctrl.customContextMenuRequested.connect(self.OnJobContextMenu)

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


        for item in [self.lbl_status_db_search,
                     self.lbl_status_model_1,
                     self.lbl_status_model_2,
                     self.lbl_status_model_3,
                     self.lbl_status_model_4,
                     self.lbl_status_model_5,
                     self.lbl_status_evaluation]:
            item.setStyleSheet("color: gray;")
        try:
            if job_params['db_search_started']:
                self.lbl_status_db_search.setStyleSheet("color: black;")
            if job_params['model_1_started']:
                self.lbl_status_db_search.setStyleSheet("color: green;")
                self.lbl_status_model_1.setStyleSheet("color: black;")
            if job_params['model_2_started']:
                self.lbl_status_model_1.setStyleSheet("color: green;")
                self.lbl_status_model_2.setStyleSheet("color: black;")
            if job_params['model_3_started']:
                self.lbl_status_model_2.setStyleSheet("color: green;")
                self.lbl_status_model_3.setStyleSheet("color: black;")
            if job_params['model_4_started']:
                self.lbl_status_model_3.setStyleSheet("color: green;")
                self.lbl_status_model_4.setStyleSheet("color: black;")
            if job_params['model_5_started']:
                self.lbl_status_model_4.setStyleSheet("color: green;")
                self.lbl_status_model_5.setStyleSheet("color: black;")
            if job_params['evaluation_started']:
                self.lbl_status_model_5.setStyleSheet("color: green;")
                self.lbl_status_evaluation.setStyleSheet("color: black;")
            if job_params['finished']:
                self.lbl_status_evaluation.setStyleSheet("color: green;")
        except KeyError:
            logger.debug("job status key(s) not found.")
            pass

        if 'status' in job_params:
            if job_params['status'] == "aborted":
                self.job.update_status("aborted", job_params['job_id'], self.sess)
            elif job_params['status'] == "running":
                self.job.update_status("running", job_params['job_id'], self.sess)
                #self.job.update_pid(job_params['pid'], job_params['job_id'], self.sess)
            elif job_params['status'] == "finished":
                self.job.update_status("finished", job_params['job_id'], self.sess)
                evaluation = self.job.insert_evaluation(self.evaluation, job_params, self.sess)
                if evaluation:
                    if not self.gui_params['job_id'] is None:
                        if int(self.gui_params['job_id']) == int(job_params['job_id']):
                            self.notebook.setTabEnabled(2, True)
                            self.evaluation.init_gui(self.gui_params, self.sess)
                else:
                    self.notebook.setTabEnabled(2, False)
                    logger.debug("No evaluation report found!")
            elif job_params['status'] == "error":
                logger.debug("Status is error")
                self.job.update_status("error", job_params['job_id'], self.sess)
        else:
            job_params['status'] = 'unknown'
        self.gui_params = self.job.init_gui(self.gui_params, self.sess)
        self.job.update_log(job_params)

    def OnUpdateLog(self, log):
        lines, job_id = log
        page = self.notebook.currentWidget().objectName()
        logger.debug("Notebook page {} id {} {}".format(page, self.gui_params['job_id'], job_id))
        if not self.gui_params['job_id'] is None:
            if int(self.gui_params['job_id']) == int(job_id) and page == "LogTab":
                for line in lines:
                    logger.debug(line)
                    self.job.log.ctrl.appendPlainText(line)
            else:
                logger.debug("job id or page not matching")
        else:
            logger.debug("no job selected")

    def OnAbout(self):
        dlg = AboutDlg(self)
        dlg.exec_()

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


    def OnBtnRun(self):
        try:
            # Prepare Job
            exec_messages = []
            self.jobparams.update_from_gui()
            self.jobparams.update_from_sequence_table()
            #Set template date to today if not defined by user
            self.jobparams.set_max_template_date()
            #If db_preset is not defined by user set it to full_dbs
            self.jobparams.set_db_preset()


            #exec_messages = self.settings.check_executables(self.sess)
            logger.debug("EXEC messages")
            logger.debug(exec_messages)
            if not exec_messages == []:
                for message in exec_messages:
                    message_dlg('Error', message)
            else:
                #Collect params from GUI and return as dict
                job_params = self.jobparams.get_dict_run_job()

                #Prepare objects for DB insert


                logger.debug("Params dict db")


                self.job.set_next_job_project_id(self.gui_params['project_id'], self.sess)
                self.job.set_timestamp()
                self.job.set_host()
                job_params['host'] = self.job.host.get_value()
                self.job.set_status("starting")
                job_params['job_name'] = self.jobparams.job_name.get_value()
                logger.debug(f"job name {job_params['job_name']}")

                if self.gui_params['project_id'] is None:
                    message_dlg('Error', 'No Project selected!')
                    raise NoProjectSelected("No project selected by user!")

                job_params['job_project_id'] = self.job.job_project_id.get_value()
                #Job folder name
                job_params['job_dir'] = self.job.get_job_dir(job_params['job_name'])
                #Full path to job folder
                job_params['output_dir'] = job_params['job_path'] = self.job.get_job_path(self.gui_params['project_path'],
                                                               job_params['job_dir'])
                job_params['log_file'] = self.job.get_log_file(self.gui_params['project_path'],
                                                               job_params['job_name'])
                #Full path to results folder created by AF inside job folder
                job_params['results_path'] = os.path.join(job_params['job_path'], job_params['job_name'])
                logger.debug(f"job path {job_params['job_path']}")
                logger.debug(f"Log file {job_params['log_file']}")
                if os.path.exists(job_params['job_path']):
                    ret = QtWidgets.QMessageBox.question(self, 'Warning',
                                                         "Output directory already exists. Confirm to override files (If \"Use Precomputed MSAs\" is activated these files will not be overriden).",
                                                         QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                    if ret == QtWidgets.QMessageBox.No:
                        raise JobSubmissionCancelledByUser
                else:
                    os.mkdir(job_params['job_path'])
                #Process custom templates
                if self.jobparams.custom_template_list.is_set():
                    new_custom_template_list, msgs = self.jobparams.process_custom_template_files(job_params['job_path'])
                    self.jobparams.custom_template_list.set_value(new_custom_template_list)
                    for msg in msgs:
                        logger.debug(msg)
                        message_dlg('Error', msg)
                        raise ProcessCustomTemplateError(msg)
                precomputed_msas_path = self.jobparams.precomputed_msas_path.get_value()
                if not precomputed_msas_path in [None, ""]:
                    if not os.path.exists(precomputed_msas_path):
                        error = "The given precomputed MSAs folder does not exist!"
                        message_dlg('Error', error)
                        raise MSAFolderNotExists(error)
                    #This directory would be also created by AF pipeline
                    if not os.path.exists(job_params['results_path']):
                        os.mkdir(job_params['results_path'])
                    if not os.path.exists(os.path.join(job_params['results_path'], 'msas')):
                        os.symlink(precomputed_msas_path, os.path.join(job_params['results_path'], 'msas'))
                    else:
                        error = "A job folder with the same name already exists and contains a MSAs folder." \
                                " Cannot override. Choose a different job name."
                        message_dlg('Error', error)
                        raise MSAFolderExists(error)



                self.jobparams.set_fasta_paths(job_params['job_path'], job_params['job_name'])
                job_params['fasta_path'] = self.jobparams.fasta_path.get_value()
                self.jobparams.write_fasta()
                logger.debug(f"Fasta path {self.jobparams.fasta_path.get_value()}")
                sequences = self.jobparams.parse_fasta(self.jobparams.fasta_path.get_value())
                #Check if new incompatible characters were introduced to the sequence
                _, error_msgs = self.jobparams.sequence_params.sanitize_sequence_str(self.jobparams.sequences.get_value())
                for msg in error_msgs:
                    message_dlg('Error', msg)
                    #Disable running a new job if sequence has incorrect format
                    self.btn_jobparams_advanced_settings.setEnabled(False)
                    self.tb_run.setEnabled(False)
                    raise SequenceFormatError(msg)
                self.job.log_file.set_value(job_params['log_file'])
                self.job.path.set_value(job_params['job_path'])
                self.job.name.set_value(job_params['job_name'])
                self.jobparams.output_dir.set_value(job_params['job_path'])

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
                logger.debug(f'gpu mem list {settings.gpu_mem_list}')
                logger.debug(f'gpu name list {settings.gpu_name_list}')

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
                    if settings.gpu_mem_list is None or settings.gpu_mem_list == '':
                        message_dlg("error", "Requested to submit to queue but found no GPU memory in settings.")
                        raise QueueSubmitError("GPU memory not specified")
                    else:
                        job_params['gpu_mem_list'] = settings.gpu_mem_list.split(',')
                    if settings.gpu_name_list is None or settings.gpu_name_list == '':
                        message_dlg("error", "Requested to submit to queue but found no GPU model in settings.")
                        raise QueueSubmitError("GPU name not specified")
                    else:
                        job_params['gpu_name_list'] = settings.gpu_name_list.split(',')
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
                job_params['total_seqlen'] = sum([len(s) for s in sequences])
                logger.debug(f"Number of sequences: {job_params['total_seqlen']}")
                job_params['multimer'] = True if len(sequences) > 1 else False
                logger.debug(job_params)
                logger.debug(self.gui_params)
                self.gui_params['job_id'] = job_id



                #Clear contents of file if it already exists
                with open(job_params['log_file'], 'w'):
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
                print("cmd dict settings")
                print(cmd_dict_settings)
                cmd_dict = {**cmd_dict_jobparams, **cmd_dict_settings}
                print(cmd_dict)
                if job_params['force_cpu']:
                   del cmd_dict['use_gpu_relax']
                if job_params['multimer'] is True:
                    del cmd_dict['pdb70_database_path']
                else:
                    del cmd_dict['pdb_seqres_database_path']
                    del cmd_dict['uniprot_database_path']
                if job_params['db_preset'] == 'full_dbs':
                    del cmd_dict['small_bfd_database_path']
                if job_params['db_preset'] == 'reduced_dbs':
                    del cmd_dict['bfd_database_path']
                    del cmd_dict['uniclust30_database_path']
                logger.debug("Job IDs before notebook: {}".format(self.gui_params['job_id']))
                self.job_params = job_params

                #Prepare command to run alphafold
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

                if job_params['queue'] and self.settings.queue_submit_dialog.get_value():
                    dlg = QueueSubmitDlg(self)
                    result = dlg.exec()
                    if not result:
                        raise JobSubmissionCancelledByUser


                #Start  process thread
                self.process_thread = QThread()
                self.process_worker = gui_threads.RunProcessThread(self, self.job_params, cmd)
                self.process_worker.moveToThread(self.process_thread)
                self.process_thread.started.connect(self.process_worker.run)
                self.process_worker.finished.connect(self.process_thread.quit)
                self.process_worker.finished.connect(self.process_worker.deleteLater)
                self.process_thread.finished.connect(self.process_thread.deleteLater)
                self.process_worker.job_status.connect(self.OnJobStatus)
                self.process_worker.change_tab.connect(self.OnChangeTab)
                self.process_thread.start()
                self.threads.append(self.process_thread)

                self.create_monitor_thread(self.job_params)

                # else:
                #     message_dlg('Error', 'Directory with the same name already exists!')
                #     logger.debug("Job directory already exists.")
        except Exception:
            job_params['status'] = 'error'
            self.OnJobStatus(job_params)
            traceback.print_exc()

    def OnChangeTab(self):
        self.notebook.setCurrentIndex(1)
        self.gui_params = self.job.init_gui(self.gui_params, self.sess)

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
                        self.job.init_gui(self.gui_params, self.sess)
                        self.job.update_log(self.gui_params)
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
                    self.job.init_gui(self.gui_params, self.sess)
                    self.job.update_log(self.gui_params)
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

    def OnCmbProjects(self):
        logger.debug("OnCmbProjects")
        project_name = str(self.prj.list.ctrl.currentText())
        if not project_name is None and not project_name == "":
            logger.debug(f"Current project {project_name}")
            new_project_id = self.prj.change_active_project(project_name, self.sess)
            logger.debug(f"New project ID {new_project_id}")
            self.gui_params['project_id'] = new_project_id
            self.gui_params['project_path'] = self.prj.get_path_by_project_id(new_project_id, self.sess)
            self.gui_params['other_settings_changed'] = False
            self.gui_params = self.job.init_gui(self.gui_params, self.sess)

    def OnLstJobSelected(self):
        self.btn_jobparams_advanced_settings.setEnabled(True)
        self.tb_run.setEnabled(True)
        index = int(self.job.list.ctrl.currentRow())
        logger.debug(f"OnLstJobSelected index {index}")
        logger.debug(self.job.list.ctrl.item(index, 0).text())
        self.gui_params['job_project_id'] = self.job.list.ctrl.item(index, 0).text()
        logger.debug(f"Job project id {self.gui_params['job_project_id']} project id {self.gui_params['project_id']}")
        self.gui_params['job_id'] = self.job.get_job_id_by_job_project_id(self.gui_params['job_project_id'],
                                                                          self.gui_params['project_id'],
                                                                          self.sess)
        logger.debug(f"{self.gui_params['job_id']}")
        self.gui_params['job_name'] = self.jobparams.get_name_by_job_id(self.gui_params['job_id'],
                                                                              self.sess)
        self.gui_params['job_dir'] = self.job.get_job_dir(self.gui_params['job_name'])
        self.gui_params['job_path'] = self.job.get_job_path(self.gui_params['project_path'],
                                                            self.gui_params['job_dir'])
        self.gui_params['log_file'] = self.job.get_log_file(self.gui_params['project_path'],
                                                            self.gui_params['job_name'])
        self.gui_params['results_path'] = os.path.join(self.gui_params['job_path'], self.gui_params['job_name'])
        self.gui_params['other_settings_changed'] = True
        result = self.jobparams.get_params_by_job_id(self.gui_params['job_id'], self.sess)
        self.jobparams.update_from_db(result)

        exit_code, status_dict = self.job.get_job_status(self.gui_params['log_file'])
        self.gui_params.update(status_dict)
        self.gui_params['status'] = self.job.get_status(self.gui_params['job_id'], self.sess)
        self.gui_params['pid'] = self.job.get_pid(self.gui_params['job_id'], self.sess)
        self.gui_params['queue'] = self.jobparams.queue.get_value()
        self.OnJobStatus(self.gui_params)
        self.job.update_log(self.gui_params)
        self.job.insert_evaluation(self.evaluation, self.gui_params, self.sess)
        if self.evaluation.check_exists(self.gui_params['job_id'], self.sess):
            self.notebook.setTabEnabled(2, True)
            self.evaluation.init_gui(self.gui_params, self.sess)
        else:
            self.notebook.setTabEnabled(2, False)
            logger.debug("No evaluation found for this job")

    def OnJobContextMenu(self, pos):
        item = self.job.list.ctrl.itemAt(pos)
        if not item is None:
            menu = QtWidgets.QMenu()
            job_project_id = self.job.list.ctrl.item(item.row(), 0).text()
            job_name = self.job.list.ctrl.item(item.row(), 1).text()
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
            menu.addAction(self.delete_jobfiles_action)
            menu.addAction(self.set_finished_action)
            menu.addAction(self.set_running_action)
            self.delete_job_action.triggered.connect(lambda state, x=job_id: self.OnDeleteEntry(x))
            self.delete_jobfiles_action.triggered.connect(lambda state, x=job_id, y=job_path: self.OnDeleteEntryFiles(x, y))
            self.set_finished_action.triggered.connect(lambda state, x=job_id: self.OnStatusFinished(x))
            self.set_running_action.triggered.connect(lambda state, x=job_id: self.OnStatusRunning(x))
            menu.exec_(self.job.list.ctrl.mapToGlobal(pos))

    def OnDeleteEntry(self, job_id):
        logger.debug(f"OnDeleteEntry. Job id {job_id}")
        self.job.delete_job(job_id, self.sess)
        self.gui_params = self.job.init_gui(self.gui_params, self.sess)

    def OnDeleteEntryFiles(self, job_id, job_path):
        self.job.delete_job_files(job_id, job_path, self.sess)
        self.gui_params = self.job.init_gui(self.gui_params, self.sess)

    def OnStatusRunning(self, job_id):
        self.job.update_status("running", job_id, self.sess)
        self.gui_params = self.job.init_gui(self.gui_params, self.sess)

    def OnStatusFinished(self, job_id):
        self.job.update_status("finished", job_id, self.sess)
        self.gui_params = self.job.init_gui(self.gui_params, self.sess)

    def OnBtnPrjAdd(self):
        dlg = ProjectDlg(self, "add")
        dlg.exec()
        self.gui_params = self.prj.init_gui(self.gui_params, self.sess)
        logger.debug("PrjAdd button pressed")

    def OnBtnPrjRemove(self):
        logger.debug("PrjRemove button pressed")
        prj_name, prj_id = self.prj.get_active_project(self.sess)

        ret = QtWidgets.QMessageBox.question(self, 'Warning',
                                             f"Do you want to remove the project {prj_name} from the database (files will not be removed)?",
                                             QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if ret == QtWidgets.QMessageBox.Yes:
            self.prj.delete_project(prj_id, self.sess)
        self.gui_params = self.prj.init_gui(self.gui_params, self.sess)

    def OnBtnPrjUpdate(self):
        dlg = ProjectDlg(self, "update")
        dlg.exec()
        self.gui_params = self.prj.init_gui(self.gui_params, self.sess)
        logger.debug("PrjUpdate button pressed")

    def OnBtnAdvancedParams(self):
        self.jobparams.update_from_gui()
        dlg = AdvancedParamsDlg(self)
        dlg.exec()

    def OnBtnSettings(self):
        dlg = SettingsDlg(self)
        dlg.exec()

    def OnBtnPrecomputedMSAsPath(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.jobparams.precomputed_msas_path.set_value(path)
        self.jobparams.precomputed_msas_path.ctrl.setText(path)
        self.jobparams.use_precomputed_msas.set_value(True)
        self.jobparams.use_precomputed_msas.ctrl.setChecked(True)

    def OnOpenModelViewer(self, model_viewer):
        if 'results_path' in self.gui_params:
            pdb_files = glob(os.path.join(self.gui_params['results_path'], "*.pdb"))
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
                traceback.print_exc()
        else:
            logger.debug('results_path not in gui_params')

    def OnOpenBrowser(self):
        if 'results_path' in self.gui_params:
            results_html = os.path.join(self.gui_params['results_path'], "results_model_viewer.html")
            if os.path.exists(results_html):
                url = QUrl(f'file://{results_html}')
                QDesktopServices.openUrl(url)
            else:
                logger.debug('Results path does not exist')
        else:
            logger.debug('results_path not in gui_params')


    def OnOpenResultsFolder(self):
            logger.debug("open in file manager")
            if 'results_path' in self.gui_params:
                if os.path.exists(self.gui_params['results_path']):
                    qfile = QUrl.fromLocalFile(
                        self.gui_params['results_path'])
                    logger.debug(qfile)
                    QDesktopServices.openUrl(qfile)
                else:
                    logger.debug('Results path does not exist')

            else:
                logger.debug('results_path not in gui_params')


    def OnClearLog(self):
        logger.debug("clear log")
        self.job.log.reset_ctrl()

    def OnBtnClear(self):
        logger.debug("Clear Btn pressed")
        self.jobparams.reset_ctrls()
        self.jobparams.update_from_default(self.default_values)
        self.gui_params['job_id'] = None
        self.gui_params['job_project_id'] = None
        self.gui_params['other_settings_changed'] = False
        self.jobparams.init_gui(self.gui_params)


if __name__ == "__main__":
    main()
