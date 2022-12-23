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

import logging
import copy
import pkg_resources
from PyQt5 import QtWidgets, uic

logger = logging.getLogger('guifold')

def message_dlg(title, text):
    dlg = QtWidgets.QMessageBox()
    dlg.setIcon(QtWidgets.QMessageBox.Information)
    dlg.setText(text)
    dlg.setWindowTitle(title)
    return_value = dlg.exec()

    return return_value