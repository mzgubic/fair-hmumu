import os
import itertools
import argparse
import fair_hmumu.trainer
import fair_hmumu.configuration
from fair_hmumu import defs


def main():

    # parse args
    parser = argparse.ArgumentParser(description='ArgParser')
    parser.add_argument('-m', '--model-path',
                        default='/data/atlassmallfiles/users/zgubic/hmumu/fair-hmumu/models/test1',
                        help='Path to the models')
    parser.add_argument('--input',
                        help='Input file')
    parser.add_argument('--output',
                        help='Output file')
    parser.add_argument('--ndiv',
                        default=2,
                        help='Number of folds')
    args = parser.parse_args()

    # load models
    models = {njet:{} for njet in defs.channels}
    run_confs = {njet:{} for njet in defs.channels}

    for njet, rmd in itertools.product(defs.channels, range(args.ndiv)):

        # get the location and configuration file
        loc = os.path.join(args.model_path, njet, 'ndiv{}_nrmd{}'.format(args.ndiv, rmd), 'run_conf.ini')
        run_conf = fair_hmumu.configuration.Configuration(loc)
        run_confs[njet][rmd] = run_conf

        # load the predictive model
        model = fair_hmumu.trainer.Predictor(run_conf)
        model.load_model()
        models[njet][rmd] = model

    # loop over the tree and predict on individual events
    print('looping over the tree!')


if __name__ == '__main__':
    main()
