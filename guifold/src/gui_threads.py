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

import copy
from subprocess import Popen, PIPE
import logging
import os
import time
from typing import Union
from PyQt5.QtCore import QObject, pyqtSignal, QFileSystemWatcher, pyqtSlot, QThread
import traceback
import re
import datetime

logger = logging.getLogger('guifold')

class MonitorJob(QObject):
    """A thread to monitor a job. If the PID is known it checks if the job is running,
    otherwise it tries to get the queue id from the logfile.
    It checks the status of the job and updates the log if the log file has changed.
        """
    finished = pyqtSignal()
    update_log = pyqtSignal(tuple)
    clear_log = pyqtSignal(int)
    job_status = pyqtSignal(dict)

    def __init__(self, parent, job_params):
        super(MonitorJob, self).__init__()
        self._parent = parent
        self.job_params = copy.deepcopy(job_params)
        self.job_params['exit_code'] = None
        self.job_params['initial_submit'] = False
        self.status_dict = None
        self.pointer = None
        self.current_job_id = None
        self.job_params['pid'] = None
        self.log_file = os.path.join(self.job_params['job_path'], self.job_params['log_file'])
        logger.debug(f"Log file for monitor thread is {self.log_file}")

        with self._parent.db.session_scope() as sess:
            self.sess = sess

    def make_hash(self, d):
        check = ''
        for key in d:
            check += str(d[key])
        return hash(check)

    def log_changed(self) -> None:
        logger.debug(f"Log file {self.log_file} changed. updating")
        logger.debug(f"Reading from {self.log_file}")
        with open(self.log_file, 'r') as log:
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
            self._parent.job.get_queue_pid(self.log_file, self.job_params['job_id'], self.sess)
        #Update job_params status parameters exit_code, status, and task_status
        self.job_params = self._parent.job.get_job_status_from_log(self.job_params)
        time.sleep(10)

    def changed_test(self) -> None:
        logger.debug("file changed test")

    def check_runtime_exceeded(self, job_started: Union[str, datetime.datetime], runtime_days: int = 10) -> bool:
        if isinstance(job_started, str):
            job_started = datetime.datetime.strptime(job_started, '%Y-%m-%d %H:%M:%S.%f')
        delta = datetime.datetime.now() - job_started
        if delta > datetime.timedelta(days=runtime_days):
            return True
        else:
            return False

    def run(self) -> None:
        self.current_job_id = self.job_params['job_id']
        thread_info = f"JobID: {self.job_params['job_project_id']} Job Name: {self.job_params['job_name']}"
        logger.debug(f"{thread_info} Starting thread")
        checksum_prev_job_params = self.make_hash(self.job_params)
        log_file_found = False
        pid_found = False
        pid = None
        time_counter = 0
        runtime_exceeded = False

        self.clear_log.emit(0)

        #Try find the log file. If it is not found after 10 s or the runtime is longer than 10 days, set status to error and exit thread.
        i = 0
        while i < 10 or self.job_params['queue']:
            if self.check_runtime_exceeded(self.job_params['time_started'], runtime_days=10):
                logger.warning(f"Maximum runtime exceeded and no logfile found. Not monitoring this job any longer (JobID {self.current_job_id}).")
                self.job_params['status'] = "unknown"
                runtime_exceeded = True
                break
            if os.path.exists(self.log_file):
                log_file_found = True
                logger.debug(f"{thread_info} Log file found")
                break
            else:
                time.sleep(1)
                logger.debug(f"{thread_info} No log file found. Trying again in 1 sec.")
                i += 1
        else:
            logger.error(f"{thread_info} No log file found after 10 retries.")
            self.job_params['status'] = "error"
            self._parent.job.update_status("error", self.job_params['job_id'], self.sess)

        if log_file_found and not runtime_exceeded:
            #QFileSystemWatcher does not work with connect signal for strange reasons. Falling back to os.stat to check
            #if log file has changed before opening it
            #log_watcher = QFileSystemWatcher([self.job_params['log_file']])

            #Get time stamp of log file
            previous_stamp = os.stat(self.log_file).st_mtime
            #Execute log_changed once to get the current status
            self.log_changed()
            while True:
                #Check if the job has been running for more than 10 days. If so, stop monitoring it.
                if self.check_runtime_exceeded(self.job_params['time_started']):
                    logger.warning(f"{thread_info} Maximum runtime exceeded. Not monitoring this job any longer.")
                    self.job_params['status'] = "unknown"
                    break

                #If time stamp as has changed, assume the log file has changed. Also check every 12*5 seconds if the log file has changed.
                stamp = os.stat(self.log_file).st_mtime
                logger.debug(f"{thread_info} setting log_watcher")
                if stamp != previous_stamp or time_counter > 12:
                    self.log_changed()
                    if time_counter > 12:
                        time_counter = 0

                #Check if the job has started on the same host, otherwise the PID is not valid
                host_started = self._parent.job.get_host(self.current_job_id, self.sess)
                current_host = self._parent.job.hostname()
                if not host_started == current_host:
                    logger.debug(f"{thread_info} Not the same host. Resetting PID to None")
                    pid = None

                #Try to get process or queue id.
                pid = self._parent.job.get_pid(self.current_job_id, self.sess)
                if not pid in [None, "None", ""]:
                    pid_found = True
                    self.job_params['pid'] = pid
                else:
                    if self.job_params['queue']:
                        logger.debug(f"{thread_info} Getting pid for queue job from log file.")
                        while pid is None:
                            pid = self._parent.job.get_queue_pid(self.log_file, self.current_job_id, self.sess)
                            time.sleep(5)
                        self._parent.job.update_pid(pid, self.job_params['job_id'], self.sess)
                    else:
                        logger.debug(f"{thread_info} No pid found and is not a queue job.")

                #Check for exit_code and set job_status accordingly. Exit the loop if the job is aborted or no PID can be found for non-queue job.
                if not self.job_params['status'] == 'aborted':
                    if self.job_params['exit_code'] is None:
                        logger.debug(f"{thread_info} No exit code found so far")
                        #If no PID is found and the job is not a queue job, it is assumed that the job has crashed
                        if pid_found and not self._parent.job.check_pid(pid) and not self.job_params['queue']:
                            logger.debug(f"{thread_info} No exit code but pid does not exist. assuming there is an error.")
                            self.job_params['status'] = "error"
                            self._parent.job.update_status("error", self.job_params['job_id'], self.sess)
                            break
                        else:
                            logger.debug(f"{thread_info} pid {pid} found for job id or is queue job.")
                    else:
                        logger.debug(f"{thread_info} exit code {self.job_params['exit_code']} found")
                        if self.job_params['exit_code'] == 0:
                            self.job_params['status'] = "finished"
                            #self._parent.job.update_status("finished", self.job_params['job_id'])
                        elif self.job_params['exit_code'] == 1:
                            self.job_params['status'] = "error"
                            #self._parent.job.update_status("error", self.job_params['job_id'])
                        elif self.job_params['exit_code'] == 2:
                            self.job_params['status'] = "aborted"
                            #self._parent.job.update_status("aborted", self.job_params['job_id'])
                        break
                else:
                    self.job_params['status'] = "aborted"
                    logger.debug(f"{thread_info} Job status is aborted")
                    break

                #Check if the job parameters have changed and emit signal.
                if checksum_prev_job_params != self.make_hash(self.job_params):
                    logger.debug(f"{thread_info} Job status dictionary changed")
                    self.job_status.emit(self.job_params)
                else:
                    logger.debug(f"{thread_info} Job status dictionary did not change.")
                    #self.job_status.emit(self.job_params)
                #else:
                #    logger.debug(f"{thread_info} Dictionary did not change, waited for 60 sec.")
                #    self.job_status.emit(self.job_params)
                checksum_prev_job_params = self.make_hash(self.job_params)

                #Set previous time stamp to current time stamp
                previous_stamp = stamp
                time_counter += 1
                time.sleep(5)
        #Emit signal at the end of the thread.
        logger.debug(f"{thread_info} Status is {self.job_params['status']}.")
        self.job_status.emit(self.job_params)
        logger.debug(f"{thread_info} Job finished.")

class LogThread(QThread):
    """Thread to read log file."""
    log_updated = pyqtSignal(list)

    def __init__(self, log_file: str = None, log_lines: str = None):
        super().__init__()
        self.log_file = log_file
        self.log_lines = log_lines

    def run(self):
        logger.debug("Starting LogReadThread")
        if self.log_file:
            logger.debug("Reading from logfile")
            if os.path.exists(self.log_file):
                logger.debug(f"Reading from {self.log_file}")
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
            else:
                logger.error(f"Log file {self.log_file} does not exist!")
                lines = []
        elif self.log_lines:
            logger.debug("Found log lines")
            lines = self.log_lines
        self.log_updated.emit(lines)
        

class RunProcessThread(QObject):
    """Thread to run a process in the background."""
    finished = pyqtSignal()
    job_status = pyqtSignal(dict)
    change_tab = pyqtSignal()
    error = pyqtSignal(list)

    def __init__(self, parent, job_params, cmd):
        super(RunProcessThread, self).__init__()
        self._parent = parent
        self.job_params = copy.deepcopy(job_params)
        self.cmd = cmd
        self.log_file = os.path.join(self.job_params['job_path'], self.job_params['log_file'])
        logger.debug(f"Log file for process thread is {self.log_file}")

        with self._parent.db.session_scope() as sess:
            self.sess = sess

    def run(self):
        #self._parent.validation_panel.Enable(False)
        #self._parent.validation_panel.Hide()
        error_msgs = []
        try:
            #basedir = os.getcwd()
            os.chdir(self.job_params['job_path'])
            logger.debug(f"Changing into directory {self.job_params['job_path']}")
        except:
            logger.error('Job directory not accessible!')
        logger.debug(f"Starting job in {self.job_params['job_path']}")
        logger.debug(self.job_params)

        #pid = os.fork()
        #if pid == 0:

        #cpid = os.getpid()
        #ppid = os.getppid()
        cmd = ' '.join(self.cmd)
        logger.debug("Starting with command\n{}".format(cmd))
        with open(os.path.join(self.job_params['job_path'], "run.sh"), 'w') as f:
            f.write(cmd)
        #cmd = shlex.split(cmd, posix=False)
        logger.debug("Starting with command\n{}".format(cmd))
        logger.debug("job params")
        logger.debug(cmd)

        error_found = False
        logger.debug(f"Writing to file {self.log_file}")
        with Popen(cmd, preexec_fn=os.setsid, shell=True, stdout=PIPE, stderr=PIPE) as p, open(self.log_file, 'w') as f:
            try:
                cpid = p.pid
                logger.debug("PID of child is {}".format(cpid))
                logger.debug("Starting process")
                #Get job id of queue job and check if queue submission was successful
                #without queue submission communicate() would block until process is finished
                if self.job_params['queue']:
                    output, error = p.communicate()
                    if not output is None:
                        output = output.decode()
                    if not error is None:
                        error = error.decode()
                        if len(error) > 0:
                            error_msgs.append(f"Failed to launch job because of the following error:\n{error}")
                            error_found = True
                    elif re.search('error', output):
                        error_msgs.append(f"Failed to launch job because of the following error:\n{output}")
                        error_found = True
                    
                    logger.debug(output)
                    regex = self.job_params["queue_jobid_regex"].replace('\\\\', '\\')
                    regex = rf'{regex}'
                    if re.match(regex, output):
                        queue_job_id = re.match(regex, output).group(1)
                        logger.debug(f"pid from queue {queue_job_id}")
                        self.job_params['status'] = "waiting"
                    else:
                        #Only add this error message if no other upstream error was found.
                        if not error_found:
                            msg = f"Could not match queue_job_id pattern to queue submit output. The output was {output} and the pattern is {regex}. Check the pattern defined in Settings."
                        logger.error(msg)
                        error_msgs.append(msg)
                        queue_job_id = None
                        self.job_params['status'] = "error"
                    self.job_params['pid'] = queue_job_id
                else:
                    self.job_params['pid'] = cpid
                    self.job_params['status'] = "running"
                if not self.job_params['pid'] is None:
                    self._parent.job.update_pid(self.job_params['pid'], self.job_params['job_id'], self.sess)
                    self._parent.gui_params['pid'] = self.job_params['pid']
                logger.debug(f"Job started. emitting signals from run process thread. pid is {self.job_params['pid']} and status is {self.job_params['status']}")
            except:
                logger.debug("Could not start the job.")
                queue_job_id = None
                self.job_params['status'] = "error"
                traceback.print_exc()
            f.write("############################################################\n\n")
            f.write(f"Job command:\n\n{cmd}\n\n")
            if not self.job_params['pipeline'] in ['batch_msas', 'only_features']:
                f.write(f"Estimated GPU memory: {self.job_params['calculated_mem']} GB\n\n")
            if self.job_params['status'] in ["running", "waiting"]:
                if not self.job_params['queue']:
                    f.write(f"Job PID: {self.job_params['pid']}\n")
                    f.write(f"Host: {self.job_params['host']}\n\n")
                else:
                    if self.job_params['split_job_step'] == 'gpu':
                        f.write(f"Job submitted to the queue with a dependency on the 'Feature' job and will start as soon as the 'Feature' job is finished.\n")
                    else:
                        f.write(f"Job submitted to the queue and waiting to start. This can take an indefinite amount of time depending on the load.\n")
            else:
                for msg in error_msgs:
                    f.write(f"{msg}\n\n")
            f.write("############################################################\n\n")
            self.job_params['initial_submit'] = True
            self.job_status.emit(self.job_params)
            self.change_tab.emit()
            self.error.emit(error_msgs)