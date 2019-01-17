import os
import ast
import itertools
import configparser


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
            conf.config.add_section(section)
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
                fixed = type(value) in [str, float, int]

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
        if option == None:
            d = {}
            for option in self.config.options(section):
                d[option] = self.get(section, option)
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

    def __iter__(self):
        """
        If it is a sweep conf, iterate over run confs.
        """

        # get a dict of all fixed and variable parameters
        fixed = self.as_dict('fixed')
        sweep = self.as_dict('sweep')

        # check whether it is indeed a sweep conf
        is_sweep = bool([1 for section in sweep if not sweep[section] == {}])

        # yield self if not sweep conf
        if not is_sweep:
            yield self

        # yield run confs if sweep conf
        if is_sweep:

            # make all the combinations
            sw_sections = [section for section in sweep if not sweep[section] == {}]
            desc = [(section, option) for section in sw_sections for option in sweep[section]]
            lists = [sweep[section][option] for section in sw_sections for option in sweep[section]]
    
            combinations = list(itertools.product(*lists))
    
            # loop over combinations and make run configs
            for comb in combinations:
                
                par_dict = fixed
                for (section, option), value in zip(desc, comb):
                    par_dict[section][option] = value
    
                run_conf = Configuration.from_dict(par_dict, 'name.ini')
                run_conf.write()
    
                yield run_conf


