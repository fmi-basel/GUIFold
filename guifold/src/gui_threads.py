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
from subprocess import Popen, PIPE, STDOUT
import logging
import os
import shlex
import time
from PyQt5.QtCore import QObject, pyqtSignal, QFileSystemWatcher, pyqtSlot

logger = logging.getLogger('guifold')

class MonitorJob(QObject):
    finished = pyqtSignal()
    update_log = pyqtSignal(tuple)
    clear_log = pyqtSignal(int)
    job_status = pyqtSignal(dict)

    def __init__(self, parent, job_params):
        super(MonitorJob, self).__init__()
        self._parent = parent
        self.job_params = job_params.copy()
        self.exit_code = None
        self.status_dict = None
        self.pointer = None
        self.current_job_id = None
        self.job_params['pid'] = None

        with self._parent.db.session_scope() as sess:
            self.sess = sess

    def make_hash(self, d):
        check = ''
        for key in d:
            check += str(d[key])
        return hash(check)

    def log_changed(self):
        logger.debug("Log file changed. updating")
        with open(self.job_params['log_file'], 'r') as log:
            if not self.pointer is None:
                log.seek(self.pointer)
            lines = log.readlines()
            if not lines == []:
                self.update_log.emit((lines, self.current_job_id))
                pass
            if self.pointer is None:
                log.seek(0, 2)
            self.pointer = log.tell()
        if self.job_params['queue']:
            self._parent.job.get_queue_pid(self.job_params['log_file'], self.job_params['job_id'], self.sess)
        self.exit_code, self.status_dict = self._parent.job.get_job_status(self.job_params['log_file'])


    def changed_test(self):
        logger.debug("file changed test")

    def run(self):
        self.current_job_id = self.job_params['job_id']
        logger.debug(f"In monitor thread for job id {self.current_job_id}")
        logger.debug("Job params are:")
        logger.debug(self.job_params)
        log_file_found = False
        pid_found = False
        pid = None

        self.clear_log.emit(0)

        i = 0
        while i < 10 or self.job_params['queue']:
            if os.path.exists(self.job_params['log_file']):
                log_file_found = True
                logger.debug("Log file found")
                break
            else:
                time.sleep(1)
                logger.debug("No log file found. Trying again in 1 sec.")
                i += 1
        else:
            logger.error("No log file found after 10 retries.")
            self.job_params['status'] = "error"
            self._parent.job.update_status("error", self.job_params['job_id'], self.sess)


        checksum_prev_job_params = self.make_hash(self.job_params)

        if log_file_found:
            #QFileSystemWatcher does not work with connect signal for strange reasons. Falling back to os.stat to check
            #if log file has changed before opening it
            #log_watcher = QFileSystemWatcher([self.job_params['log_file']])

            previous_stamp = os.stat(self.job_params['log_file']).st_mtime

            while True:
                stamp = os.stat(self.job_params['log_file']).st_mtime
                logger.debug("setting log_watcher")
                if stamp != previous_stamp:
                    self.log_changed()


                pid = self._parent.job.get_pid(self.current_job_id, self.sess)
                host_started = self._parent.job.get_host(self.current_job_id, self.sess)
                current_host = self._parent.job.hostname()
                if not host_started == current_host:
                    logger.debug("Not the same host. Resetting PID to None")
                    pid = None
                if not pid in [None, "None", ""]:
                    pid_found = True
                    self.job_params['pid'] = pid
                    self.job_params['status'] = self.job_status
                else:
                    if self.job_params['queue']:
                        logger.debug("Getting pid for queue job.")
                        while pid is None:
                            pid = self._parent.job.get_queue_pid(self.job_params['log_file'], self.current_job_id, self.sess)
                            time.sleep(5)
                        self._parent.job.update_pid(pid, self.job_params['job_id'], self.sess)
                    else:
                        logger.debug("No pid found and is not a queue job.")

                if not self.status_dict is None:
                    self.job_params.update(self.status_dict.copy())

                if not self.job_status == 'aborted':
                    if self.exit_code is None:
                        logger.debug("No exit code found so far")
                        if pid_found and not self._parent.job.check_pid(pid) and not self.job_params['queue']:
                            logger.debug(f"no exit code but pid does not exist. assuming there is an error.")
                            self.job_params['status'] = "error"
                            self._parent.job.update_status("error", self.job_params['job_id'], self.sess)
                            break
                        else:
                            logger.debug(f"pid {pid} found for job id or is queue job.")
                    else:
                        logger.debug("exit code found")
                        if self.exit_code == 0:
                            self.job_params['status'] = "finished"
                            #self._parent.job.update_status("finished", self.job_params['job_id'])
                        elif self.exit_code == 1:
                            self.job_params['status'] = "error"
                            #self._parent.job.update_status("error", self.job_params['job_id'])

                        elif self.exit_code == 2:
                            self.job_params['status'] = "aborted"
                            #self._parent.job.update_status("aborted", self.job_params['job_id'])
                        break
                else:
                    self.job_params['status'] = "aborted"
                    logger.debug("Job status is aborted")
                    break

                if checksum_prev_job_params != self.make_hash(self.job_params):
                    logger.debug('Dictionary changed')
                    self.job_status.emit(self.job_params)
                checksum_prev_job_params = self.make_hash(self.job_params)
                previous_stamp = stamp
                time.sleep(5)
        self.job_status.emit(self.job_params)
        logger.debug("Job finished.")



class RunProcessThread(QObject):
    finished = pyqtSignal()
    job_status = pyqtSignal(dict)
    change_tab = pyqtSignal()

    def __init__(self, parent, job_params, cmd):
        super(RunProcessThread, self).__init__()
        self._parent = parent
        self.job_params = job_params.copy()
        self.cmd = cmd

        with self._parent.db.session_scope() as sess:
            self.sess = sess

    def run(self):
        #self._parent.validation_panel.Enable(False)
        #self._parent.validation_panel.Hide()

        try:
            #basedir = os.getcwd()
            os.chdir(self.job_params['job_path'])
        except:
            logger.error('Job directory not accessible!')
        logger.debug("Starting job")
        logger.debug(self.job_params)

        #pid = os.fork()
        #if pid == 0:

        #cpid = os.getpid()
        #ppid = os.getppid()
        cmd = ' '.join(self.cmd)
        logger.debug("Starting with command\n{}".format(cmd))
        with open("run.sh", 'w') as f:
            f.write(cmd)
        #cmd = shlex.split(cmd, posix=False)
        logger.debug("Starting with command\n{}".format(cmd))
        logger.debug("job params")
        logger.debug(cmd)

        with Popen(cmd, preexec_fn=os.setsid, shell=True) as p, open(self.job_params['log_file'], 'w') as f:
            cpid = p.pid
            logger.debug("PID of child is {}".format(cpid))
            self.job_params['status'] = "running"
            self.job_params['pid'] = cpid
            if not self.job_params['queue']:
                self._parent.job.update_pid(cpid, self.job_params['job_id'], self.sess)
            logger.debug("Job started. emitting signals from run process thread")
            self.job_status.emit(self.job_params)
            self.change_tab.emit()
            f.write("############################################################\n\n")
            f.write(f"Job command:\n\n{cmd}\n\n")
            f.write(f"Estimated GPU memory: {self.job_params['calculated_mem']} GB\n")
            if not self.job_params['queue']:
                f.write(f"Job PID: {cpid}\n")
                f.write(f"Host: {self.job_params['host']}\n\n")
            f.write("############################################################\n\n")