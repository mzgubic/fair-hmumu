import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import fair_hmumu.defs as defs

mpl.rcParams['font.size'] = 15

def roc_curve(clf_scores, labels, colours, styles, loc, unique_id):

    # plot
    fig, ax = plt.subplots(figsize=(7,7))
    for i, score in enumerate(clf_scores):
        fprs, tprs = score.roc_curve
        ax.plot(1-fprs, tprs, color=colours[i], linestyle=styles[i], label=labels[i])
    ax.legend(loc='best')
    ax.set_xlabel('Background rejection')
    ax.set_ylabel('Signal efficiency')
    
    # save
    path = os.path.join(loc, 'roc_curve_{}.pdf'.format(unique_id))
    plt.savefig(path)
    print(path)

def clf_output(clf_scores, labels, colours, styles, loc, unique_id):

    # compute the centres of the bins
    edges = np.linspace(0, 1, defs.bins+1)
    lows = edges[:-1]
    highs = edges[1:]
    centres = (lows+highs)*0.5

    # plot
    fig, ax = plt.subplots(figsize=(7,7))

    # loop over classifiers
    for i, score in enumerate(clf_scores):

        # get clf outputs
        clf_hists = score.clf_hist

        # plot them
        kwargs = {'bins':defs.bins, 'color':colours[i], 'histtype':'step', 'density':True, 'label':labels[i]}
        for mass_range, lstyle in zip(sorted(clf_hists.keys()), ['--', ':', '-']):
            ax.hist(centres, weights=clf_hists[mass_range], linestyle=lstyle, **kwargs)

    ax.legend(loc='best')
    ax.set_xlabel('Classifier output')
    ax.set_ylabel('Normalised events')

    # save
    path = os.path.join(loc, 'clf_output_{}.pdf'.format(unique_id))
    plt.savefig(path)
    print(path)

        
