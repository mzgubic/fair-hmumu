import os
import argparse
from fair_hmumu import configuration
from fair_hmumu import plot


def main():

    # parse args
    parser = argparse.ArgumentParser(description='ArgParser')
    parser.add_argument('-s', '--sweep',
                        default='testsweep',
                        help='Name of the sweep.')
    args = parser.parse_args()

    print('--- Summarising {}'.format(args.sweep))

    # read the results as a dataframe
    results, options, scores = configuration.read_results(args.sweep)

    # plot the results summary
    loc = os.path.join(os.getenv('RUN'), args.sweep)
    plot.metric_vs_parameter('roc_auc_bcm', 'Benchmark__eta', results, loc)
    

if __name__ == '__main__':
    main()
