import os
os.environ['HF_HOME'] = '/home/tl688/scratch/'
os.environ['HF_HUB_CACHE'] = '/home/tl688/scratch/'
import torch

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
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

from huggingface_hub import login
from transformers import AutoModel 

# login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
conch, eval_transform = titan.return_conch()

# conch.cuda()

# # model_uni.cuda()
# with torch.no_grad():
#     output = tile_encoder(sample_input).squeeze()
model = conch.cuda()

from tqdm import tqdm
emb_1 = []
for item in tqdm(image_file_all):
    with torch.inference_mode():
        img = Image.open(item).convert('RGB')
        outinfo = eval_transform(img)
        outf = model(outinfo.unsqueeze(dim=0).cuda()).cpu()
        torch.save(outf, "../report_embedding/"+item.split('/')[-1].replace('.png','_conch.pkl'))




