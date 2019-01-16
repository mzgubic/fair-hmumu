import os
import ast
import configparser


class Configuration:

    def __init__(self, path):

        # path
        self.path = os.path.abspath(path)
        self.name = os.path.basename(self.path)
        self.loc = os.path.dirname(self.path)

        # parser
        self.config = configparser.ConfigParser()

    def get_dict(self, section):
        """
        Get the section settings as a dict, with types already converted.
        """

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
            except ValueError:
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
        for section in self.config:
            
            # skip default
            if section == 'DEFAULT': continue

            # write a section
            ret += '[{}]\n'.format(section)
            for option in self.config.options(section):
                ret += '{} = {}\n'.format(option, self.config[section][option])
            ret += '\n'

        return ret[:-1]


conf = Configuration('../examples/one_run_conf.ini.example')
conf.read()
print(conf.get('Adversary', 'type'))
conf.set('Adversary', 'type', 'GaussMixNLL')
print(conf.get('Adversary', 'type'))
conf.write()
print(conf.get('Adversary'))
