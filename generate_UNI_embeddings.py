import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download
from tqdm import tqdm

"""
screen -dmS generate_embeddings sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name jol_job1  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/generate_UNI_embeddings.py; exec bash'
"""

"""
screen -dmS wsi_sync sh -c 'rsync -av --progress --exclude=*.ndpi --exclude=*.pt --exclude=*.log --exclude=*.csv --exclude=*plots/ -e 'ssh -p 30044' r4:/data/pamly/storage/ /sybig/projects/FedL/data/pamly/storage'
"""

if __name__ == '__main__':
    torch.hub.set_dir('tmp/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    login(add_to_git_credential=False)  # login with your User Access Token, found at https://huggingface.co/settings/tokens

    local_dir = "./tmp/vit_large_patch16_256.dinov2.uni_mass100k"
    os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    # model.load_state_dict(torch.load(os.path.join(local_dir, "model.pth"), map_location="cpu"), strict=True)
    torch.save(model.state_dict(), os.path.join(local_dir, "model.pth"))
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.to(device)
    model.eval()
    for slide in tqdm(os.listdir("/data/pamly/sampled_tiles")):
        if not os.path.exists(f"/data/pamly/embeddings/{slide}"):
            os.makedirs(f"/data/pamly/embeddings/{slide}", exist_ok=True)

        for patch in tqdm(os.listdir(f"/data/pamly/sampled_tiles/{slide}")):
            image, label = torch.load(f"/data/pamly/sampled_tiles/{slide}/{patch}")
            image = transform(image.div(255)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(image).squeeze()
            torch.save((embedding, label), f"/data/pamly/embeddings/{slide}/{patch}")



