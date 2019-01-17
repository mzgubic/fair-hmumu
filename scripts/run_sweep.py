import os
import argparse
import fair_hmumu.utils as utils
import fair_hmumu.configuration as configuration


def main():

    # parse args
    parser = argparse.ArgumentParser(description='ArgParser')
    parser.add_argument('-s', '--sweep',
                        default='testsweep',
                        help='Location of the sweep home dir.')
    args = parser.parse_args()

    print('Running a sweep!')

    # get the location and configuration file
    loc = utils.makedir(os.path.join(os.getenv('RUN'), args.sweep))
    sweep_conf = configuration.Configuration(os.path.join(loc, 'sweep_conf.ini'))

    print(sweep_conf)

if __name__ == '__main__':
    main()
