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




