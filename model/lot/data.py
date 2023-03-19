import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from typing import TypeVar

Tensor = TypeVar(torch.tensor)


class DataObject:
    
    def __init__(self, 
                 root_dir: str="data_directory", 
                 timesteps: int=10, 
                 h: int=16, 
                 w: int=16,
                 codebooks=None,
                ): 
        if codebooks is None:
            self.codebooks = self._get_codebooks(root_dir, timesteps, h, w)
        else:
            self.codebooks = codebooks
        self.codes = torch.unique(self.codebooks)
    
    def _get_codebooks(self, 
                       root_dir: str, 
                       timesteps: int, 
                       h: int,
                       w: int,
                      ) -> Tensor:
        codebooks = [torch.load(root_dir + str(t) + '.pt') for t in range(timesteps)]
        return torch.stack(codebooks)
    
    def _plot_codebooks(self, 
                        t, 
                        figsizes=(30, 8),
                        ):
        
        fig = plt.figure(figsize=figsizes)
        
        for i in range(t):
            ax = fig.add_subplot(1, t, i + 1)
            ax.imshow(self.codebooks[i].detach().numpy(), cmap='Greys')
            ax.xaxis.labelpad=10
            ax.yaxis.labelpad=10
            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
        fig.tight_layout(pad=2.0, w_pad=4.0)
        
        return fig