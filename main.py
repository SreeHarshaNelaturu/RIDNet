import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import cv2
import scipy.misc as misc
torch.manual_seed(args.seed)
#checkpoint = utility.checkpoint(args)

 
image = misc.imread("../LR/LRBI/RNI15/X1/Dog.png")
model = model.Model("../experiment/ridnet.pt", 1, 'single')


t = Trainer(image, 1, model, None, 'single')
t_out = t.test()
misc.imsave('{}{}.png'.format("out", "boom"), t_out)


