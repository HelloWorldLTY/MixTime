import os
import torch
os.environ['HF_HOME'] = '/home/tl688/scratch/'
os.environ['HF_HUB_CACHE'] = '/home/tl688/scratch/'
os.environ['HF_TOKEN'] = ''

import timm
from PIL import Image
from torchvision import transforms
import torch

# Older versions of timm have compatibility issues. Please ensure that you use a newer version by running the following command: pip install timm>=1.0.3.
tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
val_transform = transform
transform_uni = transform

# model = convert_model(model,outdim=3)
tile_encoder.cuda()
tile_encoder.eval()
# with torch.no_grad():
#     output = tile_encoder(sample_input).squeeze()

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

len(image_file_all)

from tqdm import tqdm
emb_1 = []
for item in tqdm(image_file_all):
    with torch.inference_mode():
        img = Image.open(item).convert('RGB')
        outinfo = transform_uni(img)
        outf = tile_encoder(outinfo.unsqueeze(dim=0).cuda()).cpu()
        torch.save(outf, "/home/tl688/zhao_project/GigaTIME/report_embedding/"+item.split('/')[-1].replace('.png','_gigapath.pkl'))

import gzip
import io
from PIL import Image
import scanpy as sc

slide_info = ['GSM8797975_S11_SpT',
 'GSM8797978_S4_SpT',
 'GSM8797974_S10_SpT',
 'GSM8797980_S6_SpT',
 'GSM8797979_S5_SpT',
 'GSM8797977_S3_SpT',
 'GSM8797983_S9_SpT',
 'GSM8797973_S1_SpT',
 'GSM8797982_S8_SpT',
 'GSM8797981_S7_SpT',
 'GSM8797976_S15_SpT']

from tqdm import tqdm

for i in slide_info:
    adata = sc.read_h5ad(f"/home/tl688/zhao_project/GigaTIME/cscc_info_mixtime/gsminfo_out/{i}_out.h5ad")
    image_info = Image.open(f"/home/tl688/zhao_project/GigaTIME/cscc_info_mixtime/gsminfo_out/{i}_tissue_hires_image.png").convert('RGB')
    file_list = []
    for loc in tqdm(adata.obsm['X_loc']):
        image_new = image_info.crop((int(loc[0]) - 256, int(loc[1]) - 256, int(loc[0]) + 256, int(loc[1]) + 256))
        with torch.inference_mode():
            outinfo = transform_uni(image_new)
            outf = tile_encoder(outinfo.unsqueeze(dim=0).cuda()).cpu()
            file_list.append(outf)
    outdata = torch.concat(file_list)
    torch.save(outdata, f"/home/tl688/zhao_project/GigaTIME/allemb_cscc/{i}_gigapath.pkl")
