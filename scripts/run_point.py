import os
import argparse
import fair_hmumu.utils as utils
import fair_hmumu.configuration as configuration


def main():

    # parse args
    parser = argparse.ArgumentParser(description='ArgParser')
    parser.add_argument('-p', '--point',
                        default=None,
                        help='Path of the point config')
    args = parser.parse_args()

    # get the location and configuration file
    run_conf = configuration.Configuration(args.point)

    print(run_conf.loc)
    print(run_conf.name)



if __name__ == '__main__':
    main()
