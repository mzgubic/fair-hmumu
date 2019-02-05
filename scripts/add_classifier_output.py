import os
import itertools
import argparse
import ROOT as r
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
    parser.add_argument('--tree',
                        default='DiMuonSelection',
                        help='Name of the tree')
    args = parser.parse_args()

    # load models for all njet channels and folds
    models = {njet:{} for njet in defs.channels}
    run_confs = {njet:{} for njet in defs.channels}

    for njet in defs.channels:

        for fold_dir in os.listdir(os.path.join(args.model_path, njet)):

            # load the run configuration
            loc = os.path.join(args.model_path, njet, fold_dir, 'run_conf.ini')
            run_conf = fair_hmumu.configuration.Configuration(loc)
            n_rmd = run_conf.get('Training', 'n_rmd')
            run_confs[njet][n_rmd] = run_conf
        
            # load the predictive model
            model = fair_hmumu.trainer.Predictor(run_conf)
            model.load_model()
            models[njet][n_rmd] = model

    # prepare input and output files
    in_file = r.TFile(args.input)
    in_tree = in_file.Get(args.tree)

    out_file = r.TFile(args.output, 'recreate')
    out_tree = in_tree.CloneTree(0)

    # add new branch
    var = array.array('F', [0])
    branch = tree.Branch('DNN', var, 'DNN/F')

    # loop over the tree



if __name__ == '__main__':
    main()
