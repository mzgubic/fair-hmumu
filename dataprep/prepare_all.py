import os
from datetime import datetime
import root_pandas as rpd
import fair_hmumu.utils as utils
import fair_hmumu.defs as defs

def main():

    #####################
    # Les configurables
    #####################

    prod_name = 'default'
    prod_name = '{}_{}'.format(datetime.strftime(datetime.today(), '%Y%m%d'), prod_name)

    # other settings
    do_steps = [1, 2, 3, 4]
    step1 = 'step1'
    step2 = 'step2'
    step3 = 'step3'
    out = 'out'

    #####################
    # Prepare directories and files
    #####################

    loc = {}
    loc[defs.sig]     = utils.makedir('/data/atlassmallfiles/users/zgubic/hmumu/v17/hadd')
    loc[defs.data]    = utils.makedir('/data/atlassmallfiles/users/zgubic/hmumu/v17/hadd')
    loc[defs.ss]      = utils.makedir('/data/atlassmallfiles/users/zgubic/hmumu/spurious_signal')

    loc[out]     = utils.makedir(os.path.join(os.getenv('DATA'), prod_name))

    # get the file names
    fnames = {}
    fnames[defs.sig] = ['mc16a.345097.root',
                    'mc16d.345097.root',
                    'mc16a.345106.root',
                    'mc16d.345106.root',
                    ]
    fnames[defs.data] = ['data15.allYear.sideband.root',
                     'data16.allYear.sideband.root',
                     'data17.allYear.sideband.root',
                     ]
    fnames[defs.ss] = os.listdir(loc[defs.ss])

    #####################
    # Step 1: Mass cut + selection
    #####################

    if 1 in do_steps:
        loc[step1]   = utils.makedir(os.path.join(loc[out], step1))
        for dataset in defs.datasets:
            for fname in fnames[dataset]:
    
                # input file
                in_file = os.path.join(loc[dataset], fname)
    
                # output file
                out_file = os.path.join(loc[step1], fname)
    
                # run the selection
                full_selection = 1 if dataset in [defs.sig, defs.data] else 0
                is_signal = 1 if dataset in [defs.sig] else 0

                command = "root -l -q 'selection.cxx(\"{i}\", \"{o}\", {sel}, {sig})'".format(i=in_file, o=out_file, sel=full_selection, sig=is_signal)
                os.system(command)

    #####################
    # Step 2: Hadd
    #####################

    if 2 in do_steps:
        loc[step2]   = utils.makedir(os.path.join(loc[out], step2))
        for dataset in defs.datasets:

            # collect input files
            in_files = [os.path.join(loc[step1], fname) for fname in fnames[dataset]]

            # hadded file
            out_file = os.path.join(loc[step2], dataset) + '.root'

            # hadd them
            command = 'hadd -f {} {}'.format(out_file, ' '.join(in_files))
            os.system(command)

    #####################
    # Step 3: Split in jet categories
    #####################

    if 3 in do_steps:
        loc[step3]   = utils.makedir(os.path.join(loc[out], step3))
        for dataset in defs.datasets:

            # hadded file
            hadd_file = os.path.join(loc[step2], '{}.root'.format(dataset))

            # split in 0, 1, 2+ jet datasets (separate trees inside)
            split_file = os.path.join(loc[step3], '{}.root'.format(dataset))

            # run the macro to make new trees
            command = "root -l -q 'njet_split.cxx(\"{i}\", \"{o}\")'".format(i=hadd_file, o=split_file)
            os.system(command)

    #####################
    # Step 4: Shuffle the entries
    #####################

    if 4 in do_steps:
        for dataset in defs.datasets:
            for tree in defs.channels:
            
                # read the dataframes
                in_file = os.path.join(loc[step3], '{}.root'.format(dataset))
                df = rpd.read_root(in_file, key=tree)

                # shuffle the tree
                df = df.sample(frac=1).reset_index(drop=True)

                # and write it
                out_file = os.path.join(loc[out], '{}.root'.format(dataset))
                mode = 'w' if tree == defs.channels[0] else 'a'
                df.to_root(out_file, key=tree, mode=mode)


if __name__ == '__main__':
    main()
