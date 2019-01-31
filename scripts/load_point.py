import os
import argparse
import fair_hmumu.trainer
import fair_hmumu.configuration


def main():

    # parse args
    parser = argparse.ArgumentParser(description='ArgParser')
    parser.add_argument('-p', '--point',
                        default=None,
                        help='Path of the point config')
    args = parser.parse_args()

    # get the location and configuration file
    run_conf = fair_hmumu.configuration.Configuration(args.point)

    # make the coach
    predictor = fair_hmumu.trainer.Predictor(run_conf)

    # load the tf model
    predictor.load_model()

    # predict on some data


if __name__ == '__main__':
    main()
