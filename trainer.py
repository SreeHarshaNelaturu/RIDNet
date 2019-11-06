import os
import math
from decimal import Decimal

import utility
import scipy.misc as misc
import torch
from torch.autograd import Variable
from tqdm import tqdm
import common
import numpy as np

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(_l) for _l in l]

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

class Trainer():
    def __init__(self, noise_g, my_model, my_loss, precision='single'):
        #self.image = image
        self.noise_g = noise_g
        self.precision = precision
        self.model = my_model
        self.loss = my_loss
        self.device_mode = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_mode)

    def test(self, image):

        self.model.to(self.device).eval()

        timer_test = utility.timer()
        with torch.no_grad():
            lr = image
            lr = self.prepare([lr]).to(self.device)
            
            sr = self.model(lr, 1)
            sr = utility.quantize(sr, 255)
            #save_list = [sr]
            sr = sr.squeeze(0)
            normalized = sr.data.mul(255 / 255)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()    
            #misc.imsave('{}{}.png'.format("out", "_lol"), ndarr)
        return ndarr

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.device else 'cuda')
        ltensor = set_channel(l, 3)[0]
        img_tensor = np2Tensor([ltensor], 255)[0]
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor.to(self.device)

        return img_tensor

    def terminate(self):
        self.test()
        return True

