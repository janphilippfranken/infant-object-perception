import os 
import cv2
from torch.utils.data import Dataset
import torch

class ObjectData(Dataset):
    def __init__(
        self, 
        root_dir: str,
        n_img: int=10,
        h: int=64,
        w: int=64,
        n_channels: int=3,
        is_tensor: bool=False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.data_path = os.path.join(self.root_dir)
        self.data = torch.zeros(n_img, n_channels, h, w)
        if is_tensor:
            a = torch.load(root_dir).float()
            self.data = torch.load(root_dir).float()[0][None,:].permute(1, 0, 2, 3)
            self.data[self.data != self.data] = float(0)
        else:
            for img in range(n_img):
                self.data[img] = self.__getitem__(img)
        assert os.path.exists(self.root_dir), f"Path {self.root_dir} does not exist"
        
    def __len__(self):
        return len(self.img_nums)
    
    def __getitem__(self, index: int):
        img_path = os.path.join(self.data_path, str(index) + '.png')
        img = cv2.imread(img_path)
        img = torch.FloatTensor(img).permute(2, 0, 1)
        img -= img.min(1, keepdim=True)[0] # normalise to [0, 1]
        img /= img.max(1, keepdim=True)[0]
        img[img != img] = float(0) # get rid of nan
        return img