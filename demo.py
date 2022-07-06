import os
import numpy as np

import torch
import cv2
from utils import *

from model import MatchingNet
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

mean_sar = np.array([0.33247536, 0.33247536, 0.33247536],dtype=np.float32).reshape(3,1,1)
std_sar = np.array([0.16769384, 0.16769384, 0.16769384],dtype=np.float32).reshape(3,1,1)
mean_opt = np.array([0.31578836, 0.31578836, 0.31578836],dtype=np.float32).reshape(3,1,1)
std_opt = np.array([0.1530546, 0.1530546 ,0.1530546],dtype=np.float32).reshape(3,1,1)



def trans(s_img, o_img):
    s_img = s_img.transpose(2, 0, 1)
    s_img = ((s_img / 255.0) - mean_sar ) / std_sar

    o_img = o_img.transpose(2, 0, 1)
    o_img = ((o_img / 255.0) - mean_opt ) / std_opt

    return s_img, o_img


if __name__ == "__main__":

    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MatchingNet(
            256, 128, 4, 1, 8, 'resnet50', 'dual_softmax', 0.2, 5, 1, 100
            )
    model = model.to(DEV)
    state_dict = torch.load("best.pt",  map_location='cpu')
    model.load_state_dict(state_dict['model'])

    model.eval()
    s_img = cv2.imread("sar.jpg")
    o_img = cv2.imread("opt.jpg")
    
    s_img = cv2.resize(s_img, (256, 256))
    o_img = cv2.resize(o_img, (256, 256))

    s_disp_img = s_img.copy()
    o_disp_img = o_img.copy()
    s_norm_img, o_norm_img = trans(s_img, o_img)

    preds = model(torch.from_numpy(s_norm_img).unsqueeze(0).cuda().float(), 
                  torch.from_numpy(o_norm_img).unsqueeze(0).cuda().float())

    disp_img = draw_match_image(preds['mkpts0'][:, 1:],
                                preds['mkpts1'],
                                s_disp_img, 
                                o_disp_img)
    #cv2.imshow("xxxx", disp_img)
    #cv2.waitKey()
    cv2.imwrite("demo.jpg", disp_img)


