import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

cmap = plt.cm.jet  # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist[0] = (.5, .5, .5, 1.0)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
bounds = np.linspace(0, 20, 21)
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

def plot_codes(codes, n_img: int=10, h: int=64, w: int=64):
    cmap = plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (.5, .5, .5, 1.0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0, 20, 21)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    fig, axs = plt.subplots(1, n_img, figsize=(30,30))
    for i in range(n_img):
        axs[i].imshow(np.array(codes.view(n_img, h, w).permute(1, 2, 0).detach().numpy())[:, :, i], cmap=cmap, norm=norm)
        axs[i].set_title("codebook  " + str(i))  
    plt.tight_layout()
    plt.show()