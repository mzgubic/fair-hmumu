import os
import ast
import itertools
import configparser
import pandas as pd
from fair_hmumu import utils


class Configuration:

    def __init__(self, path):

        # path
        self.path = os.path.abspath(path)
        self.name = os.path.basename(self.path)
        self.loc = os.path.dirname(self.path)

        # parser
        self.config = configparser.ConfigParser()

        # read
        self.read()

    @classmethod
    def from_dict(cls, d, path):

        # create the configuration
        conf = Configuration(path)

        # add sections and options
        for section in d.keys():

            # add section
            try:
                conf.config.add_section(section)
            except configparser.DuplicateSectionError:
                pass

            # add option
            for option in d[section]:
                conf.set(section, option, d[section][option])

        return conf

    def as_dict(self, which='all'):
        """
        Get the section settings as a dict, with types already converted.
        """

        # check input
        assert which in ['all', 'fixed', 'sweep'], "which much be in ['all', 'fixed', 'sweep']"

        # build dictionary
        d = {section:{} for section in self.config.sections()}

        for section in self.config.sections():
            for option in self.config[section]:

                value = self.get(section, option)
                fixed = not isinstance(value, list)

                if which == 'all':
                    d[section][option] = value

                if which == 'fixed' and fixed:
                    d[section][option] = value

                if which == 'sweep' and not fixed:
                    d[section][option] = value

        return d

    def get(self, section, option=None):
        """
        Handle types other than strings.
        """

        # if only section is provided return a dict
        if option is None:
            d = {}
            for opt in self.config.options(section):
                d[opt] = self.get(section, opt)
            return d

        # if section and option are provided return the option value
        else:
            try:
                return ast.literal_eval(self.config.get(section, option))
            except (ValueError, SyntaxError) as e:
                return self.config.get(section, option)

    def set(self, section, option, value):
        """
        Handle types other than strings.
        """
        try:
            self.config.set(section, option, value)
        except TypeError:
            self.config.set(section, option, str(value))

    def read(self):

        self.config.read(self.path)

    def write(self):

        with open(self.path, 'w') as f:
            self.config.write(f)

    def __str__(self):

        ret = '\n'
        for section in self.config.sections():

            # write a section
            ret += '[{}]\n'.format(section)
            for option in self.config.options(section):
                ret += '{} = {}\n'.format(option, self.config[section][option])
            ret += '\n'

        return ret[:-1]

    def make_new_runconf_dir(self):

        for i in itertools.count():

            # does this run dir already exist?
            trial_path = os.path.join(self.loc, 'points', 'run{:04d}'.format(i))

            # try next one if it exists, otherwise return the path
            if os.path.exists(trial_path):
                continue
            else:
                return utils.makedir(trial_path)

    def __iter__(self):
        """
        If it is a sweep conf, iterate over run confs.
        """

        # get a dict of all fixed and variable parameters
        fixed = self.as_dict('fixed')
        sweep = self.as_dict('sweep')

        # make all the combinations
        sw_sections = [section for section in sweep if not sweep[section] == {}]
        desc = [(section, option) for section in sw_sections for option in sweep[section]]
        lists = [sweep[section][option] for section in sw_sections for option in sweep[section]]

        combinations = list(itertools.product(*lists))

        # loop over combinations and make run configs
        for comb in combinations:

            # construct the run conf dictionary
            par_dict = fixed
            for (section, option), value in zip(desc, comb):
                par_dict[section][option] = value

            # get location and write the run conf
            loc = self.make_new_runconf_dir()
            run_conf = Configuration.from_dict(par_dict, os.path.join(loc, 'run_conf.ini'))
            run_conf.write()

            yield run_conf

def read_results(sweepname):

    # get the location
    sweep_loc = os.path.join(os.getenv('RUN'), sweepname)
    points_loc = os.path.join(sweep_loc, 'points')

    # construct the dataframe
    sweep_conf = Configuration(os.path.join(sweep_loc, 'sweep_conf.ini'))
    sweep_dict = sweep_conf.as_dict()
    options = ['{}__{}'.format(section, option) for section in sweep_dict for option in sweep_dict[section]]
    scores = [fname.split('.')[0] for fname in os.listdir(os.path.join(points_loc, 'run0')) if fname.startswith('metric')]
    metrics = list(set([score.split('__')[-1] for score in scores]))
    results = pd.DataFrame(columns=options+scores)

    # and fill it up
    for run in os.listdir(points_loc):
        run_dict = Configuration(os.path.join(points_loc, run, 'run_conf.ini')).as_dict()
        point_dict = {'{}__{}'.format(section, option):run_dict[section][option] for section in run_dict for option in run_dict[section]}

        # see if the particular run has finished
        try:
            for score in scores:
                with open(os.path.join(points_loc, run, '{}.txt'.format(score)), 'r') as f:
                    point_dict[score.replace('metric__', '')] = float(f.read())

        # ignore if not
        except FileNotFoundError:
            continue

        results = results.append(point_dict, ignore_index=True)

    return results, options, metrics




