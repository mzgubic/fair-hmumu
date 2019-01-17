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
    trainer = fair_hmumu.trainer.Trainer(run_conf)

    # pretraining
    trainer.pretrain()

    # training
    trainer.train()


if __name__ == '__main__':
    main()
