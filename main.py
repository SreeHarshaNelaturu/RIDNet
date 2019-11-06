import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import cv2
torch.manual_seed(args.seed)
#checkpoint = utility.checkpoint(args)

 
image = cv2.imread("Figs/Net.PNG")
model = model.Model("../experiment/ridnet.pt", 1, 'single')

t = Trainer(1, model, None, 'single')
t_out = t.test(image)
misc.imsave('{}{}.png'.format("out", "boom"), t_out)


