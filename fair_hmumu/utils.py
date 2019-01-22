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

    # default contents
    contents = [
                '#!/bin/sh',
                'cd {}'.format(os.getenv('SRC')),
                'source setup_env.sh',
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

    job_base, _ = os.path.splitext(job_path)
    job_dir = os.path.dirname(job_path)
    submit_file_path = job_base + ".submit"

    with open(submit_file_path, 'w') as submit_file:
        submit_file.write("executable = " + job_path + "\n")
        submit_file.write("universe = vanilla\n")
        submit_file.write("output = " + os.path.join(job_dir, "output.$(Process)\n"))
        submit_file.write("error = " + os.path.join(job_dir, "error.$(Process)\n"))
        submit_file.write("log = " + os.path.join(job_dir, "log.$(Process)\n"))
        submit_file.write("notification = never\n")
        submit_file.write("should_transfer_files = Yes\n")
        submit_file.write("when_to_transfer_output = ON_EXIT\n")
        submit_file.write("queue 1")

    # call the job submitter
    sp.check_output(['condor_submit', submit_file_path])
    print("submitted '" + submit_file_path + "'")


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

