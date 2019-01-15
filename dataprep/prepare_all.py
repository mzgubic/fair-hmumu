import os
import fair_hmumu.utils as utils

def main():

    # configurable
    prod_name = 'default'

    # other settings
    step1_name = 'step1'

    #####################
    # Step 1: Mass cut
    #####################

    # prepare directories
    loc = {}
    loc['sig']     = utils.makedir('/data/atlassmallfiles/users/zgubic/hmumu/v17/hadd')
    loc['data']    = utils.makedir('/data/atlassmallfiles/users/zgubic/hmumu/v17/hadd')
    loc['ss_z']    = utils.makedir('/data/atlassmallfiles/users/zgubic/hmumu/spurious_signal/user.gartoni.Hmumu.SpuriousSignalStudies.MC15_Hmm_PowhegInclZ_TruthSamples.v01_di_muon_ntuple.root')
    loc['ss_vbf']  = utils.makedir('/data/atlassmallfiles/users/zgubic/hmumu/spurious_signal/user.gartoni.Hmumu.SpuriousSignalStudies.MC15_Hmm_VBFZjNp2_TruthSamples.v01_di_muon_ntuple.root')

    loc['out']     = utils.makedir('/data/atlassmallfiles/users/zgubic/hmumu/tf_ready/{}'.format(prod_name))

    # get the file names
    fnames = {}
    fnames['sig'] = ['mc16a.345097.root',
                    'mc16d.345097.root',
                    'mc16a.345106.root',
                    'mc16d.345106.root',
                    ]
    fnames['data'] = ['data15.allYear.sideband.root',
                     'data16.allYear.sideband.root',
                     'data17.allYear.sideband.root',
                     ]
    fnames['ss_z'] = os.listdir(loc['ss_z'])
    fnames['ss_vbf'] = os.listdir(loc['ss_vbf'])


    # loop over full analysis ntuples:
    for dataset in ['sig', 'data', 'ss_vbf', 'ss_z']:
        for fname in fnames[dataset]:

            # input file
            in_file = os.path.join(loc[dataset], fname)

            # output file
            step1_outdir = utils.makedir(os.path.join(loc['out'], step1_name))
            out_file = os.path.join(step1_outdir, fname)

            # run the selection
            full_selection = 1 if dataset in ['sig', 'bkg', 'data'] else 0
            is_signal = 1 if dataset in ['sig'] else 0
            command = "root -l -q 'selection.cxx(\"{i}\", \"{o}\", {sel}, {sig})'".format(i=in_file, o=out_file, sel=full_selection, sig=is_signal)
            os.system(command)
            break
        break

if __name__ == '__main__':
    main()
