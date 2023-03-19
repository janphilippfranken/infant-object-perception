import matplotlib.pyplot as plt
import matplotlib
from collections import Counter

def plot_bitmask(data, t, figsizes=(50, 10), obj=0, world=0, perm=0):
    fig, axs = plt.subplots(1, t, figsize=figsizes)
    cmap = matplotlib.colors.ListedColormap(['grey', 'black'])
    for i, ax in enumerate(axs):
        ax.imshow(data[i].detach().numpy(), cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(w_pad=8)
    plt.savefig("fig/imaginary" + str(obj) + "_" + str(perm) + "_" + str(world) + ".pdf")


def plot_ll(Masks, mcmc_res, likelihoods):

    maps = []
    lls = []

    regularities = {k: [] for k, _ in enumerate(Masks.codes)}
    for res in mcmc_res:
        for i, _ in enumerate(Masks.codes):
            regularities[i].append(str(res[i]._methods))

    for i, _ in enumerate(Masks.codes):
        counts = Counter(regularities[i])
        map_k = max(counts, key=counts.get)
        map_v = counts[map_k]
        maps.append({map_k: map_v})
        lls.append(likelihoods[regularities[i].index(map_k)][:, i])

    fig, ax = plt.subplots(1, len(lls), figsize=(5 * len(lls), 5))

    for i, data in enumerate(lls):
        ax[i].bar(range(len(data)), -data, color="grey")
        ax[i].set_xlabel("Permutations")
        if i == 0:
            ax[i].set_ylabel("-LogLikelihood")
        ax[i].set_xticks(range(len(Masks.permutations)))
        ax[i].set_xticklabels(Masks.permutations, rotation=45)
        ax[i].set_title("Object {}".format(i))
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ymin, ymax = ax[i].get_ylim()
        ax[i].set_ylim(ymin - ymax * .025, ymax)
        fig.suptitle("Observed Objects", y=1.0, fontsize=24, ha="center")
        fig.tight_layout()
        fig.show()
        # plt.savefig('fig/lls_world' + str(world) + '.pdf')        
    
    
plot_params = {'legend.fontsize': 'large',
               'axes.labelsize': 'large',
               'axes.titlesize':'20',
               'axes.labelsize':'20',
               'xtick.labelsize':'20',
               'font.family': 'Arial',
               'xtick.color':'grey',
               'ytick.color':'grey',
               'ytick.labelsize':'20',
               'axes.linewidth': '3'}