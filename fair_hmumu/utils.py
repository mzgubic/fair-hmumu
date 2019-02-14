import os
import time
import pickle
import subprocess as sp


def timeit(f):

    def timed(*args, **kwargs):
        print('-------------')
        t0 = time.time()
        ret = f(*args, **kwargs)
        print('-> Took {:2.2f}s'.format(time.time()-t0))
        return ret

    return timed


def makedir(d):
    """
    Creates a new directory if it doesn't exist.
    """
    if not os.path.exists(d):
        os.makedirs(d)

    return d


def write_job(commands, job_path):

    # make dir if needed
    job_dir = makedir(os.path.split(job_path)[0])
    src_dir = os.getenv('SRC')

    # default contents
    contents = [
                '#!/bin/sh',
                'source {}/setup_env.sh'.format(src_dir),
                'cd {}'.format(job_dir),
                ]

    # add the commands requested
    if isinstance(commands, str):
        contents.append(commands)

    if isinstance(commands, list):
        contents += commands

    # and write them to a file
    with open(job_path, 'w') as f:
        for line in contents:
            f.write(line+'\n')


def send_job(job_path):

    job_dir = os.path.dirname(job_path)
    condor_submit = os.path.splitext(job_path)[0] + ".submit"

    with open(condor_submit, 'w') as f:
        f.write('executable     = {}\n'.format(job_path))
        f.write('arguments      = $(ClusterID)\n')
        f.write('output         = {}/$(ClusterId).out\n'.format(job_dir))
        f.write('error          = {}/$(ClusterId).err\n'.format(job_dir))
        f.write('log            = {}/$(ClusterId).log\n'.format(job_dir))
        f.write('request_cpus   = 10\n')
        f.write('request_memory = 16 GB\n')
        f.write('stream_output  = True\n')
        f.write('stream_error   = True\n')
        f.write('queue\n')

    # call the job submitter
    sp.check_output(['condor_submit', condor_submit])
    print("submitted '" + condor_submit + "'")


class Saveable:

    def __init__(self, name):
        self.name = name

    @classmethod
    def classname(cls):
        return cls.__name__

    @classmethod
    def from_file(cls, path):
        print('--- Reading {} from {}' .format(cls.classname(), path))
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path):
        print('--- Saving {} as {}' .format(self.classname(), path))
        with open(path, 'wb') as f:
            pickle.dump(self, f)

