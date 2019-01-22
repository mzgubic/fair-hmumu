import os
import pickle
import subprocess as sp


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

    # default contents
    contents = [
                '#!/bin/sh',
                'cd {}'.format(os.getenv('SRC')),
                'source setup_env.sh',
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
        f.write('executable = {}\n'.format(job_path))
        f.write('arguments  = $(ClusterID)\n')
        f.write('output     = {}/$(ClusterId).out\n'.format(job_dir))
        f.write('error      = {}/$(ClusterId).err\n'.format(job_dir))
        f.write('log        = {}/$(ClusterId).log\n'.format(job_dir))
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

