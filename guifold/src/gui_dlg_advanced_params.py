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
import logging
from datetime import date

logger = logging.getLogger("guifold")

class DefaultValues:
    def __init__(self):
        self.benchmark = False
        self.random_seed = None
        self.max_template_date = str(date.today())
        self.model_preset = 'automatic'
        self.force_cpu = False
        self.num_recycle = 20
        self.num_multimer_predictions_per_model = 1
        self.num_gpu = 1
        self.use_model_1 = True
        self.use_model_2 = True
        self.use_model_3 = True
        self.use_model_4 = True
        self.use_model_5 = True
        self.first_n_seq = 2
        self.batch_max_sequence_length = 5000
        self.msa_pairing = 'paired'



class AdvancedParamsDlg(QtWidgets.QDialog):
    def __init__(self, _parent):
        super(AdvancedParamsDlg, self).__init__()
        self.sess = _parent.sess
        self.jobparams = _parent.jobparams
        self.gui_params = _parent.gui_params
        uic.loadUi(pkg_resources.resource_filename('guifold.ui', 'advanced_params.ui'), self)
        self.jobparams.set_controls(self, self.jobparams.db_table)
        self.button_box = self.findChild(QtWidgets.QDialogButtonBox, "btn_jobparams_button_box")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.init()

    def __del__(self):
        self.jobparams.delete_controls(DefaultValues())

    def init(self):
        default_values = DefaultValues()
        for item in self.jobparams.model_preset_dict.values():
            self.jobparams.model_preset.ctrl.addItem(item)
        for item in self.jobparams.msa_pairing_dict.values():
            self.jobparams.msa_pairing.ctrl.addItem(item)
        if not self.gui_params['other_settings_changed']:
            if self.gui_params['job_id'] is None:
                logger.debug("Other params not changed and job_id is None. Update from default values.")
                self.jobparams.update_from_default(default_values)
            else:
                logger.debug("Other params not changed. Getting params from DB.")
                params = self.jobparams.get_params_by_job_id(self.gui_params['job_id'], self.sess)
                logger.debug(params)
                self.jobparams.update_from_db(params, default_values)
        else:
            logger.debug("Other params changed")
            self.jobparams.update_from_self()

    def accept(self):
        self.jobparams.update_from_gui()
        self.gui_params['other_settings_changed'] = True
        super().accept()

    def reject(self):
        super().reject()