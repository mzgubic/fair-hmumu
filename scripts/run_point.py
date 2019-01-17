import os
import argparse
import fair_hmumu.utils as utils
import fair_hmumu.models as models
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

    # get the individual configs
    clf_conf = run_conf.get('Classifier')
    adv_conf = run_conf.get('Adversary')
    opt_conf = run_conf.get('Optimiser')
    trn_conf = run_conf.get('Training')

    clf = models.Classifier('Classifier', clf_conf)
    print(clf)
    


if __name__ == '__main__':
    main()
