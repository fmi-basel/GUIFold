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

import pkg_resources
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIcon
import logging
from guifold.src.gui_dialogs import message_dlg

logger = logging.getLogger("guifold")

class SettingsDlg(QtWidgets.QDialog):
    def __init__(self, _parent):
        super(SettingsDlg, self).__init__()
        self.sess = _parent.sess
        self.settings = _parent.settings
        uic.loadUi(pkg_resources.resource_filename('guifold.ui', 'settings.ui'), self)
        #Second element determines whether folder (True) or File (False)
        self.btn_choosefolder_names = [('jackhmmer_binary_path', False),
                                       ('hhblits_binary_path', False),
                                       ('hhsearch_binary_path', False),
                                       ('hmmsearch_binary_path', False),
                                       ('hmmbuild_binary_path', False),
                                       ('hhalign_binary_path', False),
                                       ('mmseqs_binary_path', False),
                                       ('kalign_binary_path', False),
                                       ('uniref90_database_path', False),
                                       ('uniref90_mmseqs_database_path', False),
                                       ('uniref30_database_path', False),
                                       ('uniref30_mmseqs_database_path', False),
                                       ('colabfold_envdb_database_path', False),
                                       ('mgnify_database_path', False),
                                       ('bfd_database_path', False),
                                       ('small_bfd_database_path', False),
                                       ('uniprot_database_path', False),
                                       ('uniprot_mmseqs_database_path', True),
                                       ('pdb70_database_path', True),
                                       ('pdb_seqres_database_path', False),
                                       ('template_mmcif_database_path', True),
                                       ('obsolete_pdbs_path', False),
                                       ('custom_tempdir', False),
                                       ('data_dir', True)]

        for item, folder in self.btn_choosefolder_names:
            setattr(self, f'btn_settings_{item}', self.findChild(QtWidgets.QToolButton, f'btn_settings_{item}'))
            logger.debug(f'btn_settings_{item}')
            logger.debug(vars(self))
            var = getattr(self, f'btn_settings_{item}')
            logger.debug(f"{item} {var} is_folder {folder}")
            var.setIcon(QIcon(pkg_resources.resource_filename('guifold.icons', 'gtk-open.png')))

        self.settings.set_controls(self, self.settings.db_table)
        self.btn_settings_load_global_settings = self.findChild(QtWidgets.QPushButton, "btn_settings_load_global_settings")
        self.btn_settings_load_local_settings = self.findChild(QtWidgets.QPushButton, "btn_settings_load_local_settings")
        self.btn_settings_submission_script_template_path = self.findChild(QtWidgets.QToolButton, "btn_settings_submission_script_template_path")
        self.button_box = self.findChild(QtWidgets.QDialogButtonBox, "btn_settings_button_box")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.bind_event_handlers()
        self.init()

    def bind_event_handlers(self):
        #Bind all choose folder buttons at once
        for item, folder in self.btn_choosefolder_names:
            logger.debug(f'getattr(self, f"btn_settings_{item}").clicked.connect(lambda: self.OnBtnChooseFolder({item}))')
            getattr(self, f"btn_settings_{item}").clicked.connect(lambda checked, a=item, b=folder: self.OnBtnChooseFolder(a, b))
            logger.debug(f"{item}")
        #self.btn_ok.clicked.connect(self.OnBtnOk)
        self.btn_settings_load_global_settings.clicked.connect(self.OnBtnLoadGlobalSettings)
        self.btn_settings_load_local_settings.clicked.connect(self.OnBtnLoadLocalSettings)
        self.btn_settings_submission_script_template_path.clicked.connect(self.OnBtnChooseFolderSubmissionTemplate)

    def init(self):
        settings = self.settings.get_from_db(self.sess)
        self.settings.update_from_db(settings)

    def OnBtnChooseFolder(self, name, folder=False):
        logger.debug(f"OnBtnChooseFolder {name}")
        dlg = QtWidgets.QFileDialog()
        if dlg.exec_():
            if folder:
                path = dlg.getExistingDirectory(self, 'Select Folder')
            else:
                path = dlg.selectedFiles()[0]
            logger.debug(path)
            var = getattr(self.settings, name)
            var.set_value(path)
            var.ctrl.setText(path)

    def OnBtnLoadGlobalSettings(self):
        msgs = self.settings.update_from_config()
        if len(msgs) == 0:
            self.settings.update_from_self()
        else:
            message_dlg("error", '\n'.join(msgs))   

    def OnBtnLoadLocalSettings(self):
        dlg = QtWidgets.QFileDialog()
        if dlg.exec_():
            path = dlg.selectedFiles()[0]
            logger.debug(path)
            msgs = self.settings.update_from_config(custom_path=path)
            if len(msgs) == 0:
                self.settings.update_from_self()
            else:
                message_dlg("error", '\n'.join(msgs))

    def OnBtnChooseFolderSubmissionTemplate(self):
        logger.debug("OnBtnChooseFolderQueueTemplate")
        dlg = QtWidgets.QFileDialog()
        if dlg.exec_():
            path = dlg.selectedFiles()[0]
            logger.debug(path)
        self.settings.submission_script_template_path.set_value(path)
        self.settings.submission_script_template_path.ctrl.setText(path)

    def accept(self):
        self.settings.update_from_gui()
        self.settings.update_settings(self.settings.get_dict_db_insert(), self.sess)
        self.settings.unset_controls(self, self.settings.db_table)
        super().accept()

    def reject(self):
        self.settings.unset_controls(self, self.settings.db_table)
        super().reject()


