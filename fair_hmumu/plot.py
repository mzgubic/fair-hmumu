import os
import matplotlib.pyplot as plt
import matplotlib as mpl

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


