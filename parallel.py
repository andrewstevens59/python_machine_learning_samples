# NOTICE: THIS PROGRAM CONSISTS OF TRADE SECRETS THAT ARE THE PROPERTY OF TRUEMOTION, INC. AND ARE PROPRIETARY MATERIAL OF TRUEMOTION, INC.

import multiprocessing
import pickle
import traceback
import uuid
import os
import tempfile
import shutil
import gzip
from tqdm import tqdm

class WorkItem(object):
    def __init__(self, fn, args, kwargs):
        """
        fn: a pickle-able function
        args: an iterable sequence of arguments
        kwargs: a dict of keyword-arguments
        """
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.outfile = None
        self.pid = None
        self.traceback = None

    def run(self):
        return self.fn(*self.args, **self.kwargs)

class ProcessPoolExecutor(object):
    def __init__(self, max_workers=None, abort_on_job_error=True):
        """Initializes a new ProcessPoolExecutor instance.

        Args:
            max_workers: The maximum number of processes that can be used to
                execute the given calls. If None or not given then as many
                worker processes will be created as the machine has processors.
        """

        if max_workers is None:
            self._max_workers = multiprocessing.cpu_count()
        else:
            self._max_workers = max_workers

        self.abort_on_job_error = abort_on_job_error

    def _start_job(self, tmpdir, job):
        worker_output_file = os.path.join(
            tmpdir,
            'worker_{}_output_file.pkl'.format(str(uuid.uuid4()))
        )
        job.outfile = worker_output_file
        
        # create a new task
        pid = os.fork()
        if pid==0:
            try:
                output = job.run()
                with open(job.outfile, 'wb') as f:
                    pickle.dump(output, f)
            except:
                message = traceback.format_exc()
                print message
                with open(job.outfile, 'wb') as f:
                    pickle.dump(message, f)
            os._exit(0)
        else:
            job.pid = pid

        return job

    def _wait_for_worker_and_collect_result(self, pending_jobs):
        result = None
        found_result = False
        while found_result == False:
            try:
                pid, _ = os.wait()
            except OSError:
                pass
            if pid not in pending_jobs:
                continue

            found_result = True
            job = pending_jobs.pop(pid)

            with open(job.outfile, 'rb') as f:
                result = pickle.load(f)
            os.remove(job.outfile)

            if isinstance(result, unicode) and "Traceback" in result:
                fname = 'failed_job_{}.pkl'.format(pid)
                print('Failed to load output from {}, dumping to {}'.format(job.outfile, fname))
                with gzip.open(fname, 'wb') as f:
                    job.traceback = result
                    pickle.dump(job,f)
                result = None
                if self.abort_on_job_error:
                    raise e

        return result
    
    def run(self, jobs, f_on_complete=None, show_progress_bar=True):
        """
        Processes the WorkItems in child processes. Doesn't return until all
        jobs are finished
        
        jobs: iterable of WorkItem
            an iterable of WorkItem objects
            
        f_on_complete: function
            a function to call as results are available.
            
        return None if f_on_complete is provided, else returns list of results
        """
        # Goal here is to distribute the workload (each worker)
        # among the available worker processes. Do this by iterating
        # over users and peanut-butter smoothing them among the 
        # workers
        tmpdir = tempfile.mkdtemp()
        num_available_workers = self._max_workers
        pending_jobs = dict()
        output = []

        if show_progress_bar == True:
            prog_func = tqdm
        else:
            prog_func = lambda x: x

        for job in prog_func(jobs):
            if num_available_workers>0:
                job = self._start_job(tmpdir, job)
                pending_jobs[job.pid] = job
                num_available_workers -= 1
            else:
                result = self._wait_for_worker_and_collect_result(pending_jobs)
                if f_on_complete is None:
                    output.append(result)
                else:
                    f_on_complete(result)

                job = self._start_job(tmpdir, job)
                pending_jobs[job.pid] = job
    
        # wait for all pending tasks to complete
        while len(pending_jobs)>0:
            result = self._wait_for_worker_and_collect_result(pending_jobs)
            if f_on_complete is None:
                output.append(result)
            else:
                f_on_complete(result)

        shutil.rmtree(tmpdir)
        
        if f_on_complete is None:
            return output

# NOTICE: THIS PROGRAM CONSISTS OF TRADE SECRETS THAT ARE THE PROPERTY OF TRUEMOTION, INC. AND ARE PROPRIETARY MATERIAL OF TRUEMOTION, INC.
