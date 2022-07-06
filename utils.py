import os
import numpy as np
import cv2
import torch
from einops.einops import rearrange, repeat

def draw_match_image(mkpts0, mkpts1, query,refer, homo_filter=True, patch_size=8, th=1e3):
    out_img = np.concatenate([query,refer],axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    refer_pts[:, 0] = refer_pts[:, 0] + wq

    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,
                                    cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    return out_img

def batch_get_mkpts(matches, query, refer, patch_size=8):
    wq = query.shape[2]
    wr = refer.shape[2]
    qcols = wq // patch_size
    rcols = wr // patch_size
    x0 = patch_size/2 + (matches[:, 1] % qcols) * patch_size
    y0 = patch_size/2 + torch.div(matches[:, 1], qcols) * patch_size
    x1 = patch_size/2 + (matches[:, 2] % qcols) * patch_size
    y1 = patch_size/2 + torch.div(matches[:, 2], qcols) * patch_size
    query_pts = torch.cat((matches[:, 0].unsqueeze(1), x0.unsqueeze(1), y0.unsqueeze(1)), 1)
    refer_pts = torch.cat((matches[:, 0].unsqueeze(1), x1.unsqueeze(1), y1.unsqueeze(1)), 1)
    if len(query_pts) < 0:
        query_pts = torch.Tensor(0, 2).cuda()
        refer_pts = torch.Tensor(0, 2).cuda()

    return query_pts, refer_pts
