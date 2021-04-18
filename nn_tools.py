import sys
sys.path.insert(1, './PraNet/')

from torch import nn 
from lib.PraNet_Res2Net import PraNet
import torch
import numpy as np

def get_network():
  net = PraNet().float().eval()
  net.load_state_dict(torch.load("./models/model_weights_0.5000_0.9126.save", map_location='cpu'))
  return net


def pass_img(network, img):
    img_copy = np.vstack([img, np.zeros((72,800,3))])  # for the sake of nn model  (read comment in the end of the file)
    img_torch = torch.from_numpy(img_copy.transpose(2,0,1))[None]/255
    img_torch = img_torch.type(torch.FloatTensor)

    return handle_by_crops(img_torch, network.eval(), 672, 224,'cpu')[0, :-72] 


def handle_by_crops(images, network, dsr_size, margin_size, device='cuda'):
    
    height = images.shape[-2]
    width = images.shape[-1]
    lines = []
    network.to(device=device)
    x_prev = 0
    sigmoid = nn.Sigmoid()
    for x in range(dsr_size, height+dsr_size - 2 * margin_size - 1, dsr_size - 2 * margin_size):
        x = min(x, height)
        line = []
        y_prev = 0
        for y in range(dsr_size, width+dsr_size - 2 * margin_size - 1, dsr_size - 2*margin_size):
            y = min(y, width)
            img_cropped = images[:,:,x-dsr_size:x, y - dsr_size: y]
            img_cropped = img_cropped.to(device=device)
            
            
            pred = network(img_cropped)
            
            mask = sigmoid(pred[0] + pred[1] + pred[2] + pred[3])  
            mask = mask[:,0].detach().cpu().numpy()
            if y - dsr_size != 0:
                mask = mask[:,:,margin_size:]

            if y != width:
                mask = mask[:,:,:-margin_size]
            else:
                mask = mask[:,:,y_prev-(y-dsr_size + margin_size):]


            if x != height:
                mask = mask[:,:-margin_size]
            
            line.append(mask)
            y_prev = y - margin_size

        lines.append(np.concatenate(line,axis=2))
        x_prev =  x - margin_size
        
    result_mask = np.concatenate(lines, axis=1)
    return result_mask



# The used acrhitecture of neural network is able to handle image of size multiple of 224. So that wee need to make 600 to 672 pixel
# The height (800 pixel) will be handled in two steps:
# - From 0 to 671 pixel
# - From 127 to 799
# - And then two part will be stitched
# - The whole magic is going inside of @handle_by_crops function