import os
import numpy as np
import matplotlib as mpl
mpl.use('agg') # order of magnitude faster than tkagg
import matplotlib.pyplot as plt
from fair_hmumu import defs

mpl.rcParams['font.size'] = 15


def losses(losses, loc, unique_id, trn_conf, plt_conf):

    # determine the training steps at which the losses were recorded
    n_pre = trn_conf['n_pre']
    n_epochs = trn_conf['n_epochs']
    steps = list(range(1, 1 + 2*n_pre + n_epochs))

    # setup
    colours = {'C':'r', 'A':'b', 'CA':'g', 'BCM':'k'}
    lstyles = {'test':'-', 'train':':'}
    def get_label(tt, ltype):
        labels = {'C':'Classifier',
                  'A':'Adversary',
                  'CA':r'Clf. + $ \lambda \times$ Adv.',
                  'BCM':'Benchmark'.format(tt)}
        return '{} ({})'.format(labels[ltype], tt)

    # plot
    fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    fig.suptitle('Losses')

    # plot benchmark separately
    for tt in losses:
        xs = [1, len(steps)]
        ys = [losses[tt]['BCM'], losses[tt]['BCM']]
        ax[0].plot(xs, ys, color=colours['BCM'], linestyle=lstyles[tt], label=get_label(tt, 'BCM'), alpha=0.5)

    # plot them
    for i, ltype in enumerate(['C', 'A', 'CA']):
        for tt in losses:

            # plot the loss
            loss = losses[tt][ltype]
            ax[i].plot(steps[:len(loss)], loss, color=colours[ltype], linestyle=lstyles[tt], label=get_label(tt, ltype))

        # set the x limits to the full number of steps
        ax[i].set_ylim(ax[i].get_ylim()) # fix it
        ax[i].set_xlim(np.min(steps), 2*n_pre+n_epochs)

        # plot the vertical lines showing pretraining steps
        ax[i].plot([n_pre, n_pre], list(ax[i].get_ylim()), 'k:')
        ax[i].plot([2*n_pre, 2*n_pre], list(ax[i].get_ylim()), 'k:')

        # scale and legends
        ax[i].set_xscale('log')
        ax[i].legend(loc='best', fontsize=10)

    # styling
    ax[-1].set_xlabel('Training step')

    # save
    path = os.path.join(loc, 'losses_{}.pdf'.format(unique_id))
    plt.savefig(path)
    plt.close(fig)
    print(path)


def roc_curve(plot_setups, loc, unique_id):

    # plot
    fig, ax = plt.subplots(figsize=(7, 7))

    for setup in plot_setups:
        fprs, tprs = setup['score'].roc_curve
        ax.plot(1-fprs, tprs, color=setup['colour'], linestyle=setup['style'], label=setup['label'])

    ax.legend(loc='best', fontsize=10)
    ax.set_xlabel('Background rejection')
    ax.set_ylabel('Signal efficiency')

    # save
    path = os.path.join(loc, 'roc_curve_{}.pdf'.format(unique_id))
    plt.savefig(path)
    plt.close(fig)
    print(path)


def clf_output(plot_setups, loc, unique_id):

    # compute the centres of the bins
    edges = np.linspace(0, 1, defs.bins+1)
    lows = edges[:-1]
    highs = edges[1:]
    centres = (lows+highs)*0.5

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw={'height_ratios':[3, 1]})
    fig.suptitle('Background-only MC')

    # loop over classifiers
    for setup in plot_setups:

        # get clf outputs
        clf_hists = setup['score'].clf_hists

        # common keyword arguments
        kwargs = {'bins':defs.bins, 'color':setup['colour'], 'histtype':'step', 'range':(0, 1)}

        # plot all mass ranges
        for mass_range, lstyle in zip(['low', 'medium', 'high'], [':', '-', '--']):

            # top panel
            mass_comment = {'low':'< 120 GeV', 'medium':'120 < 130 GeV', 'high':'> 130 GeV'}
            label = '{} ({})'.format(setup['label'], mass_comment[mass_range])
            ax[0].hist(centres, weights=clf_hists[mass_range], linestyle=lstyle, label=label, **kwargs)

            # bottom panel
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = clf_hists[mass_range]/clf_hists['medium']
            ratio[ratio == np.inf] = 0
            ratio = np.nan_to_num(ratio)
            ax[1].hist(centres, weights=ratio, linestyle=lstyle, **kwargs)

    ax[0].legend(loc='best', fontsize=10)
    ax[0].set_xlim(0, 1)
    ax[0].set_ylabel('Normalised events')
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 2)
    ax[1].set_xlabel('Classifier output')

    # save
    path = os.path.join(loc, 'clf_output_{}.pdf'.format(unique_id))
    plt.savefig(path)
    plt.close(fig)
    print(path)


def mass_shape(plot_setups, perc, loc, unique_id):

    # compute the centres of the bins
    edges = np.linspace(defs.mlow, defs.mhigh, defs.bins+1)
    lows = edges[:-1]
    highs = edges[1:]
    centres = (lows+highs)*0.5

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw={'height_ratios':[3, 1]})
    fig.suptitle('Background-only MC')

    # loop over classifiers
    for setup in plot_setups:

        # get clf outputs
        sel_hist, full_hist = setup['score'].mass_hists[perc]

        # common keyword arguments
        kwargs = {'bins':defs.bins, 'color':setup['colour'], 'histtype':'step', 'range':(defs.mlow, defs.mhigh)}

        # top panel
        label = '{} (best {}%)'.format(setup['label'], perc)
        ax[0].hist(centres, weights=sel_hist, linestyle=setup['style'], label=label, **kwargs)

        # bottom panel
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = sel_hist/full_hist
        ratio[ratio == np.inf] = 0
        ratio = np.nan_to_num(ratio)
        ax[1].hist(centres, weights=ratio, linestyle=setup['style'], **kwargs)

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












