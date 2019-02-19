import os
import numpy as np
import itertools
import matplotlib as mpl
mpl.use('agg') # order of magnitude faster than tkagg
import matplotlib.pyplot as plt
from fair_hmumu import defs

mpl.rcParams['font.size'] = 13


def losses(losses, run_conf, loc, unique_id):

    # determine the training steps at which the losses were recorded
    run_conf = run_conf.as_dict()
    n_pre = run_conf['Training']['n_pre']
    n_epochs = run_conf['Training']['n_epochs']
    steps = list(range(1, 1 + 2*n_pre + n_epochs))

    # setup
    colours = {'C':'r', 'A':'b', 'CA':'g', 'BCM':'k'}
    lstyles = {'test':'-', 'train':':'}
    def get_label(tt, ltype):
        labels = {'C':'Classifier',
                  'A':'Adversary',
                  'CA':r'Clf. - $ \lambda \times$ Adv.',
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
        ax[i].legend(loc='upper right', fontsize=10)

    # styling
    write_conf_info(ax[0], {s:run_conf[s] for s in run_conf if s in ['Classifier', 'Benchmark']}, isloss=True)
    write_conf_info(ax[1], {s:run_conf[s] for s in run_conf if s in ['Adversary', 'Optimiser']}, isloss=True)
    write_conf_info(ax[2], {s:run_conf[s] for s in run_conf if s in ['Training']}, isloss=True)
    ax[-1].set_xlabel('Training step')
    ax[0].set_ylim(ax[0].get_ylim()[0], 0.80)

    # save
    path = os.path.join(loc, 'losses_{}.pdf'.format(unique_id))
    plt.savefig(path)
    plt.close(fig)
    print(path)

def metrics(metric_vals, bcm_score, run_conf, loc, unique_id):

    # get data
    run_conf = run_conf.as_dict()
    metrics = sorted(list(metric_vals.keys()))
    n_metrics = len(metrics)

    # plot
    fig, ax = plt.subplots(n_metrics, 1, figsize=(7, 7), sharex=True)
    plt.subplots_adjust(left=0.15)
    fig.suptitle('Metrics')

    # helpers
    lstyles = {'test':'-', 'train':':'}

    # plot values
    for i, metric in enumerate(metric_vals):

        # add classifier info
        for tt in metric_vals[metric]:
            
            # dont bother making an empty plot
            vals = metric_vals[metric][tt]
            if len(vals) == 0:
                return

            # add curves to the plot
            ax[i].plot(range(len(vals)), vals, c=defs.blue, linestyle=lstyles[tt], label='DNN ({})'.format(tt))

        # two loops for order in the legend
        for tt in metric_vals[metric]:
            bcm_val = bcm_score[tt][metric]
            ax[i].plot([0, len(vals)], [bcm_val, bcm_val], c=defs.dark_blue, linestyle=lstyles[tt], label='XGB ({})'.format(tt))

        # each subplot once
        ax[i].set_xlim(1, run_conf['Training']['n_epochs'])
        ax[i].set_ylabel(metric)
        ax[i].set_xscale('log')
        ax[i].legend(loc='lower right', fontsize=10)

    # cosmetics
    write_conf_info(ax[0], {s:run_conf[s] for s in run_conf if s in ['Classifier', 'Benchmark']}, isloss=True)
    write_conf_info(ax[1], {s:run_conf[s] for s in run_conf if s in ['Adversary', 'Optimiser']}, isloss=True)
    write_conf_info(ax[2], {s:run_conf[s] for s in run_conf if s in ['Training']}, isloss=True)
    ax[-1].set_xlabel('Training step')

    # save
    path = os.path.join(loc, 'metric_{}.pdf'.format(unique_id))
    plt.savefig(path)
    plt.close(fig)
    print(path)

def gy(y1, dy):
    for i in range(1000):
        yield y1 - i*dy

def write_conf_info(ax, run_conf, isloss=False):

    # params
    fontsize = 8

    # get data
    if not isinstance(run_conf, dict):
        run_conf = run_conf.as_dict()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # three equal panels in loss, no space in one
    dy = (y1-y0)/35
    dx = 0.05*(x1-x0)
    if isloss:
        dy = 2.5*dy
        dx = np.log(dx)

    # y position generator
    y = gy(y1, dy)
    next(y) # skip one

    # loop over sections
    for section in run_conf:

        if section == 'Plotting':
            continue

        # loop over options
        ax.text(x0, next(y), section, fontsize=fontsize)
        for option in run_conf[section]:
            value = run_conf[section][option]
            ax.text(x0+dx, next(y), '{}: {}'.format(option, value), fontsize=fontsize)


def roc_curve(plot_setups, run_conf, loc, unique_id, zoom=True):

    # plot
    fig, ax = plt.subplots(figsize=(7, 7))

    # zoom
    if zoom:
        ax.set_xlim(0.95, 1.0)
        ax.set_ylim(0.0, 0.05)

    y0 = 0.025 if zoom else 0.5
    dy = 0.002 if zoom else 0.05
    ax.text(0.5, y0, 'ROC AUC:')
    for i, setup in enumerate(plot_setups):

        # add roc curve
        fprs, tprs = setup['score'].roc_curve
        ax.plot(1-fprs, tprs, color=setup['colour'], linestyle=setup['style'], label=setup['label'])

        # add ROC AUC score
        x = 0.975 if zoom else 0.5
        ax.text(x, y0-dy*(i+1), '{}: {:2.3f}'.format(setup['label'], setup['score'].roc_auc))

    # add run conf text
    write_conf_info(ax, run_conf)

    # settings
    ax.legend(loc='best', fontsize=10)
    ax.set_xlabel('Background rejection')
    ax.set_ylabel('Signal efficiency')

    # save
    path = os.path.join(loc, 'roc_curve_zoom{}_{}.pdf'.format(zoom, unique_id))
    plt.savefig(path)
    plt.close(fig)
    print(path)


def clf_output(plot_setups, run_conf, loc, unique_id):

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

    ax[0].set_xlim(0, 1)
    write_conf_info(ax[0], run_conf)
    ax[0].legend(loc='best', fontsize=10)
    ax[0].set_ylabel('Normalised events')
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 2)
    ax[1].set_xlabel('Classifier output')

    # save
    path = os.path.join(loc, 'clf_output_{}.pdf'.format(unique_id))
    plt.savefig(path)
    plt.close(fig)
    print(path)


def mass_shape(plot_setups, perc, run_conf, loc, unique_id):

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

    # line
    ax[1].plot([defs.mlow, defs.mhigh], [perc/100., perc/100.], 'k:')
    # settings
    ax[0].set_xlim(defs.mlow, defs.mhigh)
    write_conf_info(ax[0], run_conf)
    ax[0].legend(loc='best', fontsize=10)
    ax[0].set_ylabel('Events')
    ax[1].set_xlim(defs.mlow, defs.mhigh)
    ax[1].set_xlabel('Invariant mass [GeV]')

    # save
    path = os.path.join(loc, 'mass_shape_{}p_{}.pdf'.format(perc, unique_id))
    plt.savefig(path)
    plt.close(fig)
    print(path)


def KS_test(plot_setups, run_conf, loc, unique_id):

    # compute the centres of the bins
    edges = np.linspace(defs.mlow, defs.mhigh, defs.bins+1)
    lows = edges[:-1]
    highs = edges[1:]
    centres = (lows+highs)*0.5

    # figure
    fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw={'height_ratios':[3, 1]})
    fig.suptitle('Background-only MC')

    # plotting kwargs
    kwargs = {'bins':defs.bins, 'histtype':'step', 'range':(defs.mlow, defs.mhigh)}

    y1 = 0.5
    dy = 0.05
    y = gy(y1, dy)
    ax[0].text(140, next(y), 'KS metric:')

    # get the full mass histo
    for setup in plot_setups:

        # get the score
        score = setup['score']
        ks_vals = []
        for perc in score.mass_hists:

            # get the full and the selected mass histogram
            sel, full = score.mass_hists[perc]

            # make their cumulative (normalise)
            cum_full = np.cumsum(full) / np.sum(full)
            cum_sel = np.cumsum(sel) / np.sum(sel)

            # compute KS metric
            ks_metric = np.max(np.abs(cum_sel-cum_full))
            ks_vals.append(ks_metric)

            # prepare for plotting
            label = '{}, best {}%'.format(setup['label'], perc)
            alpha = (perc+100)/200.

            # and plot them
            ax[0].hist(centres, weights=cum_sel, color=setup['colour'], label=label, alpha=alpha, **kwargs)
            ax[0].text(140, next(y), '{}: {:2.2f}'.format(label, ks_metric), fontsize=10)

            # and bottom panel
            ax[1].hist(centres, weights=cum_full-cum_sel, color=setup['colour'], alpha=alpha, **kwargs)

        # choose the largest one as the metric
        score.ks_metric = max(ks_vals)

    # plot the full spectrum as well
    ax[0].hist(centres, weights=cum_full, color='r', label='No cut', **kwargs)

    # settings
    ax[0].set_xlim(defs.mlow, defs.mhigh)
    write_conf_info(ax[0], run_conf)
    ax[0].legend(loc='best', fontsize=10)
    ax[0].set_ylabel('Normalised cumulative events')
    ax[1].set_ylabel('Full - Selected')
    ax[1].set_xlim(defs.mlow, defs.mhigh)
    ax[1].set_xlabel('Invariant mass [GeV]')

    # save
    path = os.path.join(loc, 'KS_test_{}.pdf'.format(unique_id))
    plt.savefig(path)
    plt.close(fig)
    print(path)


def metric_vs_parameter(metric, parameter, results, loc):

    # check parameter is not something confusing
    if type(results[parameter][0]) in [list, tuple]:
        return None

    # check there are different values of the parameter
    if len(results[parameter].unique()) == 1:
        return None

    # plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.15)
    colours = {'clf':defs.blue, 'bcm':defs.dark_blue}
    markers = {'clf':{'test':'X', 'train':'x'}, 'bcm':{'test':'o', 'train':'.'}}
    alphas = {'test':1, 'train':0.2}
    for m, tt in itertools.product(['clf', 'bcm'], ['test', 'train']):
        if m == 'bcm' and tt == 'train':
            continue
        xs = results[parameter]
        ys = results['{}__{}__{}'.format(m, tt, metric)]
        m_labels = {'clf':'DNN', 'bcm':'XGB'}
        label = '{} ({})'.format(m_labels[m], tt)
        ax.scatter(xs, ys, color=colours[m], marker=markers[m][tt], alpha=alphas[tt], label=label)

    # final touches
    try:
        if max(xs)/min(xs) > 100:
            ax.set_xscale('log')
            ax.set_xlim(0.8*min(xs), max(xs)*1.2)
    except TypeError: # if string
        pass
    ax.set_xlabel(parameter)
    ax.set_ylabel(metric)
    ax.legend(loc='best', fontsize=10)

    # save
    path = os.path.join(loc, '{}_vs_{}.pdf'.format(metric, parameter))
    plt.savefig(path)
    plt.close(fig)
    print(path)


def metric2d(metric_x, metric_y, parameter, results, loc):

    # check parameter is not something confusing
    if type(results[parameter][0]) in [list, tuple]:
        return None

    # check there are different values of the parameter
    if len(results[parameter].unique()) == 1:
        return None

    # check whether parameter is a string or a number
    is_string = isinstance(results[parameter][0], str)

    # plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.15)
    markers = {'clf':{'test':'X', 'train':'x'}, 'bcm':{'test':'o', 'train':'.'}}
    alphas = {'test':1, 'train':0.2}
    for m, tt in itertools.product(['clf', 'bcm'], ['train', 'test']):
        if m == 'bcm' and tt == 'train':
            continue
        xs = results['{}__{}__{}'.format(m, tt, metric_x)]
        ys = results['{}__{}__{}'.format(m, tt, metric_y)]
        zs = results[parameter]
        m_labels = {'clf':'DNN', 'bcm':'XGB'}
        label = '{} ({})'.format(m_labels[m], tt)
        if is_string:
            digits_map = {par:i for i, par in enumerate(zs.unique())}
            digits = [digits_map[par] for par in zs]
            cm = plt.cm.get_cmap('cool', len(zs.unique()))
            sc = ax.scatter(xs, ys, marker=markers[m][tt], label=label, c=digits, cmap=cm, alpha=alphas[tt], vmin=-0.5, vmax=len(digits_map)-0.5)
        else:
            cm = plt.cm.get_cmap('cool')
            sc = ax.scatter(xs, ys, marker=markers[m][tt], label=label, c=zs, cmap=cm, alpha=alphas[tt])

    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)
    leg = ax.legend(loc='best', fontsize=10)
    for marker in leg.legendHandles:
        marker.set_color('k')
    if is_string:
        cbar = plt.colorbar(sc)
        cbar.set_ticks(range(len(zs.unique())))
        cbar.set_ticklabels(zs.unique())
        cbar.set_label(parameter)
    else:
        cbar = plt.colorbar(sc)
        cbar.set_label(parameter)

    # save
    path = os.path.join(loc, '{}_{}_vs_{}.pdf'.format(metric_x, metric_y, parameter))
    plt.savefig(path)
    plt.close(fig)
    print(path)



    






