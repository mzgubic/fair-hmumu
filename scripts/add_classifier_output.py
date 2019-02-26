import os
import itertools
import argparse
import array
import numpy as np
import ROOT as r
from fair_hmumu import trainer
from fair_hmumu import configuration
from fair_hmumu import dataset
from fair_hmumu import defs


def main():

    # parse args
    parser = argparse.ArgumentParser(description='ArgParser')
    parser.add_argument('-m', '--model-path',
                        default='/data/atlassmallfiles/users/zgubic/hmumu/fair-hmumu/models/test3',
                        help='Path to the models')
    parser.add_argument('--input',
                        default='/data/atlassmallfiles/users/zgubic/hmumu/fair-hmumu/postprocessing_test/xgb/mc16a.345097/group.phys-higgs.15906089._000001.di_muon_ntuple__all_events.root',
                        help='Input file')
    parser.add_argument('--output',
                        default='/data/atlassmallfiles/users/zgubic/hmumu/fair-hmumu/postprocessing_test/test_tree.root',
                        help='Output file')
    parser.add_argument('--tree',
                        default='DiMuonNtuple',
                        help='Name of the tree')
    args = parser.parse_args()

    ##################
    # load models for all njet channels and folds
    ##################

    models = {njet:{} for njet in defs.channels}
    run_confs = {njet:{} for njet in defs.channels}
    n_divs = []

    for njet in defs.channels:

        for fold_dir in os.listdir(os.path.join(args.model_path, njet)):

            # load the run configuration
            loc = os.path.join(args.model_path, njet, fold_dir, 'run_conf.ini')
            run_conf = configuration.Configuration(loc)
            n_rmd = run_conf.get('Training', 'n_rmd')
            run_confs[njet][n_rmd] = run_conf

            # store ndiv (as a check)
            n_divs.append(run_conf.get('Training', 'n_div'))
        
            # load the predictive model
            model = trainer.Predictor(run_conf)
            model.load_model()
            models[njet][n_rmd] = model

    # check ndivs are all the same
    assert len(set(n_divs)) == 1, 'n_div must be the same for all models'
    n_div = n_divs[0]

    # prepare input and output files
    in_file = r.TFile(args.input)
    in_tree = in_file.Get(args.tree)

    out_file = r.TFile(args.output, 'recreate')
    out_tree = in_tree.CloneTree(0)

    # add new branch
    dnn_var = array.array('f', [0])
    dnn_branch = out_tree.Branch('ClassOut_DNN', dnn_var, 'ClassOut_DNN/F')

    ##################
    # loop over the tree
    ##################

    for index, entry in enumerate(in_tree):

        # determine the number of jets and event number
        event_number = entry.EventInfo_EventNumber
        n_rmd = event_number % n_div
        jet_multip = entry.Jets_jetMultip
        if jet_multip == 0:
            njet = 'jet0'
        elif jet_multip == 1:
            njet = 'jet1'
        else:
            njet = 'jet2'

        # construct the feature array X
        collection = run_confs[njet][n_rmd].get('Training', 'clf_features')
        features = dataset.feature_list(collection, njet)
        X = np.zeros(shape=(1, len(features)))

        for i, feature in enumerate(features):
            
            try:
                X[0, i] = getattr(entry, feature)

            except AttributeError:

                if feature == 'Muons_PTM_Lead':
                    value = entry.Muons_PT_Lead / entry.Muons_Minv_MuMu
                elif feature == 'Muons_PTM_Sub':
                    value = entry.Muons_PT_Sub / entry.Muons_Minv_MuMu
                elif feature == 'Z_PTM':
                    value = entry.Z_PT / entry.Muons_Minv_MuMu
                else:
                    raise AttributeError

                X[0, i] = value

        # predict the value of the DNN using the right model
        model = models[njet][n_rmd]
        dnn_output = model.predict(X)[0, 0]
        dnn_var[0] = dnn_output

        # fill the tree
        out_tree.Fill()

        # report progress
        if index % 10000 == 0:
            print('{}/{}'.format(index, in_tree.GetEntries()))

    # close the file
    out_file.Write()
    out_file.Close()


if __name__ == '__main__':
    main()
