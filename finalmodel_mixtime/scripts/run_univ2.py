import os
os.environ['HF_HOME'] = '/home/tl688/scratch/'
os.environ['HF_HUB_CACHE'] = '/home/tl688/scratch/'
import torch

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
# pretrained=True needed to load UNI2-h weights (and download weights for the first time)
timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
model_uni = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
transform_uni = create_transform(**resolve_data_config(model_uni.pretrained_cfg, model=model_uni))
model_uni.eval()

import glob

filenames = glob.glob("/home/tl688/pitl688/PathGen-1.6M/slide_report/*.png")

from os import listdir
from os.path import isfile, join
mypath = '/home/tl688/pitl688/PathGen-1.6M/slide_report/set1/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

image_file_all = []
for idx in range(1,6):
    for item in glob.glob(f"/home/tl688/pitl688/PathGen-1.6M/slide_report/set{idx}/*.png"):
        image_file_all.append(item)

from PIL import Image
import numpy as np

model_uni.cuda()

from tqdm import tqdm
emb_1 = []
for item in tqdm(image_file_all):
    with torch.inference_mode():
        img = Image.open(item).convert('RGB')
        outinfo = transform_uni(img)
        outf = model_uni(outinfo.unsqueeze(dim=0).cuda()).cpu()
        torch.save(outf, "../report_embedding/"+item.split('/')[-1].replace('.png','_univ2.pkl'))