import logging
import pkg_resources
from PyQt5 import QtWidgets, uic
import os

logger = logging.getLogger("guifold")

class QueueSubmitDlg(QtWidgets.QDialog):
    def __init__(self, _parent):
        super(QueueSubmitDlg, self).__init__()
        self.sess = _parent.sess
        self.settings = _parent.settings
        self.job_params = _parent.job_params
        uic.loadUi(pkg_resources.resource_filename('guifold.ui', 'queue_submit.ui'), self)
        self.submit_script_field = self.findChild(QtWidgets.QPlainTextEdit, 'pte_job_queue_submit')
        self.queue_submit_dialog = self.settings.queue_submit_dialog
        self.queue_submit_dialog.ctrl = self.findChild(QtWidgets.QCheckBox, 'chk_settings_queue_submit_dialog')
        self.submit_script_path = os.path.join(self.job_params['job_path'], f'submit_script_{self.job_params["type"]}.run')
        with open(self.submit_script_path, 'r') as f:
            text = f.read()
        self.submit_script_field.setPlainText(text)

    def accept(self):
        text = self.submit_script_field.toPlainText()
        with open(self.submit_script_path, 'w') as f:
            f.write(text)
        if self.queue_submit_dialog.ctrl.isChecked():
            self.settings.update_queue_submit(False, self.sess)
        self.queue_submit_dialog.unset_control()
        super().accept()

    def reject(self):
        if self.queue_submit_dialog.ctrl.isChecked():
            self.settings.update_queue_submit(False, self.sess)
        self.queue_submit_dialog.unset_control()
        super().reject()
