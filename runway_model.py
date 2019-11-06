import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
torch.manual_seed(1)
import runway
from runway.data_types import *


@runway.setup(options={"pre_trained" : file(extension=".pt")})
def setup(opts):
    pre_trained = opts["pre_trained"]

    model_ckpt = model.Model(pre_trained, 1, 'single')
    t = Trainer(1, model_ckpt, None, 'single')
    return t

command_inputs = {"input_image" : image}
command_outputs = {"output_image" : image}

@runway.command("denoise_image", inputs=command_inputs, outputs=command_outputs, description="Denoise Image")
def denoise_image(t, inputs):
    img = np.array(inputs["input_image"])
    
    t_out = t.test(img)

    return {"output_image" : t_out}

if __name__ == "__main__":
    runway.run(model_options={"pre_trained": "experiment/ridnet.pt"})

