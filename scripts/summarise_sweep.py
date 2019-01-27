import os
import argparse
import itertools
from fair_hmumu import configuration
from fair_hmumu import plot
from fair_hmumu import utils


def main():

    # parse args
    parser = argparse.ArgumentParser(description='ArgParser')
    parser.add_argument('-s', '--sweep',
                        default='testsweep',
                        help='Name of the sweep.')
    args = parser.parse_args()

    print('--- Summarising {}'.format(args.sweep))

    # read the results as a dataframe
    results, options, metrics = configuration.read_results(args.sweep)

    # plot the results summary
    loc = utils.makedir(os.path.join(os.getenv('RUN'), args.sweep, 'plots'))
    for metric, option in itertools.product(metrics, options):
        plot.metric_vs_parameter(metric, option, results, loc)

    for option in options:
        plot.metric2d('sig_eff', 'roc_auc', option, results, loc)
        plot.metric2d('ks_metric', 'roc_auc', option, results, loc)
        plot.metric2d('ks_metric', 'sig_eff', option, results, loc)

if __name__ == '__main__':
    main()
