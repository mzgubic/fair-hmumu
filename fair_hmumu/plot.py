import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import fair_hmumu.defs as defs

mpl.rcParams['font.size'] = 15

def losses(losses, labels, colours, styles, loc, unique_id, trn_conf, plt_conf):

    # determine the training steps at which the losses were recorded
    n_pre = trn_conf['n_pre']
    n_epochs = trn_conf['n_epochs']
    n_skip = plt_conf['n_skip']
    steps_pre = list(range(1, 2*n_pre+1))
    steps_tr = [1 + 2*n_pre + i for i in range(n_epochs) if i%n_skip==0 or i==n_epochs-1]
    steps = steps_pre + steps_tr
 
    # plot
    fig, ax = plt.subplots(3, 1, figsize=(7,7), sharex=True)
    fig.subptitle('Losses')

    for i, nn in enumerate(losses):
        loss = losses[nn]
        ax[i].plot(steps[:len(loss)], loss, color=colours[i], linestyle=styles[i], label=labels[i])
        # plot the vertical lines showing pretraining
        ax[i].set_ylim(ax[i].get_ylim())
        ax[i].set_xlim(np.min(steps), 2*n_pre+n_epochs)
        ax[i].plot([n_pre, n_pre], list(ax[i].get_ylim()), 'k:')
        ax[i].plot([2*n_pre, 2*n_pre], list(ax[i].get_ylim()), 'k:')
        ax[i].legend(loc='best')

    ax[-1].set_xlabel('Training step')

    # save
    path = os.path.join(loc, 'losses_{}.pdf'.format(unique_id))
    plt.savefig(path)
    plt.close(fig)
    print(path)


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
    plt.close(fig)
    print(path)


def clf_output(clf_scores, labels, colours, styles, loc, unique_id):

    # compute the centres of the bins
    edges = np.linspace(0, 1, defs.bins+1)
    lows = edges[:-1]
    highs = edges[1:]
    centres = (lows+highs)*0.5

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(7,7), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    fig.suptitle('Background-only MC')

    # loop over classifiers
    for i, score in enumerate(clf_scores):

        # get clf outputs
        clf_hists = score.clf_hists

        # common keyword arguments
        kwargs = {'bins':defs.bins, 'color':colours[i], 'histtype':'step', 'range':(0,1)}

        # plot all mass ranges
        for mass_range, lstyle in zip(['low', 'medium', 'high'], [':', '-', '--']):

            # top panel
            mass_comment = {'low':'< 120 GeV', 'medium':'120 < 130 GeV', 'high':'> 130 GeV'}
            label = '{} ({})'.format(labels[i], mass_comment[mass_range])
            ax[0].hist(centres, weights=clf_hists[mass_range], linestyle=lstyle, label=label, **kwargs)

            # bottom panel
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = clf_hists[mass_range]/clf_hists['medium']
            ratio[ratio == np.inf] = 0
            ratio = np.nan_to_num(ratio)
            ax[1].hist(centres, weights=ratio, linestyle=lstyle, **kwargs)

    ax[0].legend(loc='best', fontsize=10)
    ax[0].set_xlim(0,1)
    ax[0].set_ylabel('Normalised events')
    ax[1].set_xlim(0,1)
    ax[1].set_ylim(0,2)
    ax[1].set_xlabel('Classifier output')

    # save
    path = os.path.join(loc, 'clf_output_{}.pdf'.format(unique_id))
    plt.savefig(path)
    plt.close(fig)
    print(path)


def mass_shape(clf_scores, perc, labels, colours, styles, loc, unique_id):

    # compute the centres of the bins
    edges = np.linspace(defs.mlow, defs.mhigh, defs.bins+1)
    lows = edges[:-1]
    highs = edges[1:]
    centres = (lows+highs)*0.5

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(7,7), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    fig.suptitle('Background-only MC')

    # loop over classifiers
    for i, score in enumerate(clf_scores):

        # get clf outputs
        sel_hist, full_hist = score.mass_hists[perc]

        # common keyword arguments
        kwargs = {'bins':defs.bins, 'color':colours[i], 'histtype':'step', 'range':(defs.mlow, defs.mhigh)}

        # top panel
        label = '{} (best {}%)'.format(labels[i], perc)
        ax[0].hist(centres, weights=sel_hist, linestyle=styles[i], label=label, **kwargs)

        # bottom panel
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = sel_hist/full_hist
        ratio[ratio == np.inf] = 0
        ratio = np.nan_to_num(ratio)
        ax[1].hist(centres, weights=ratio, linestyle=styles[i], **kwargs)

    ax[1].plot([defs.mlow, defs.mhigh], [perc/100., perc/100.], 'k:')
    ax[0].legend(loc='best', fontsize=10)
    ax[0].set_xlim(defs.mlow, defs.mhigh)
    ax[0].set_ylabel('Events')
    ax[1].set_xlim(defs.mlow, defs.mhigh)
    ax[1].set_xlabel('Invariant mass [GeV]')

    # save
    path = os.path.join(loc, 'mass_shape_{}p_{}.pdf'.format(perc, unique_id))
    plt.savefig(path)
    plt.close(fig)
    print(path)

   










