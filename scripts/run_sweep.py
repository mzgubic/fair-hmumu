import os
import argparse
import fair_hmumu.utils as utils
import fair_hmumu.configuration as configuration


def main():

    # parse args
    parser = argparse.ArgumentParser(description='ArgParser')
    parser.add_argument('-s', '--sweep',
                        default='testsweep',
                        help='Name of the sweep.')
    parser.add_argument('--batch',
                        default=False,
                        action='store_true')
    args = parser.parse_args()

    # get the location and configuration file
    loc = utils.makedir(os.path.join(os.getenv('RUN'), args.sweep))
    sweep_conf = configuration.Configuration(os.path.join(loc, 'sweep_conf.ini'))

    print('--- Running {}'.format(args.sweep))
    print(sweep_conf)

    # submit the jobs for all the run configs
    for run_conf in sweep_conf:

        print()
        print('--- Running point')
        print()

        # make the command
        e = os.path.join(os.getenv('SRC'), 'scripts', 'run_point.py')
        p = run_conf.path
        command = 'python3 {} -p {}'.format(e, p)

        # run the command (submit)
        if args.batch:
            path = os.path.join(os.path.split(run_conf.path)[0], 'training.sh')
            print(path)
            utils.write_job(command, path)
            utils.send_job(path)
        else:
            os.system(command)
            break

if __name__ == '__main__':
    main()
