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
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtCore import Qt

logger = logging.getLogger('guifold')

def message_dlg(title, text):
    dlg = QtWidgets.QMessageBox()
    dlg.setIcon(QtWidgets.QMessageBox.Information)
    dlg.setText(text)
    dlg.setWindowTitle(title)
    return_value = dlg.exec()

    return return_value

class ProgressDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.init()

    def init(self):
        self.setWindowTitle('Progress')
        self.setGeometry(100, 100, 300, 100)
        self.layout = QtWidgets.QVBoxLayout()
        self.progress_bar = QtWidgets.QProgressBar()
        self.label = QtWidgets.QLabel() 
        self.label.setStyleSheet("color: red")
        self.accept_button = QtWidgets.QPushButton('Accept')
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.accept_button) 
        self.accept_button.setHidden(True)
        self.label.setHidden(True)
        self.setLayout(self.layout)


    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)
        if value == 100:
            self.progress_bar.setFormat("Task Complete")
            self.accept()

    def error(self, text):
        self.accept_button.setHidden(False)
        self.label.setHidden(False)
        text = f"The following errors occured:\n{text}"
        self.label.setText(text)

        


class LoadDialog(QtWidgets.QMessageBox):
    def __init__(self, parent=None, text=None):
        super(LoadDialog, self).__init__(parent)
        self.setWindowTitle("Loading...")
        self.setText(text)
        self.setStandardButtons(QtWidgets.QMessageBox.Close)

#From https://stackoverflow.com/a/64340482
def open_files_and_dirs_dlg(parent=None, caption='', directory=None, 
                        filter=None, initialFilter=None, options=None):
    def updateText():
        rows = view.selectionModel().selectedRows()
        selected = []
        for index in rows:
            selected.append('"{}"'.format(index.data()))
        lineEdit.setText(' '.join(selected))

    dialog = QtWidgets.QFileDialog(parent, windowTitle=caption)
    dialog.setFileMode(dialog.ExistingFiles)
    if options:
        dialog.setOptions(options)
    dialog.setOption(dialog.DontUseNativeDialog, True)
    if directory:
        dialog.setDirectory(directory)
    if filter:
        dialog.setNameFilter(filter)
        if initialFilter:
            dialog.selectNameFilter(initialFilter)


    dialog.accept = lambda: QtWidgets.QDialog.accept(dialog)

    #recursively searching for the listview results in segfault
    view = dialog.findChild(QtWidgets.QSplitter, "splitter").findChild(QtWidgets.QFrame, "frame").findChild(QtWidgets.QStackedWidget, "stackedWidget").findChild(QtWidgets.QWidget, "page").findChild(QtWidgets.QListView, "listView")

    if dialog.exec_():
        view.selectionModel().selectionChanged.connect(updateText)
        lineEdit = dialog.findChild(QtWidgets.QLineEdit)
        dialog.directoryEntered.connect(lambda: lineEdit.setText(''))
        files = dialog.selectedFiles()
    
        return files