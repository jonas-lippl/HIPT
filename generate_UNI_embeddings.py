import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download
from tqdm import tqdm

"""
screen -dmS generate_embeddings_normalized_512px_all_tiles sh -c 'docker run --shm-size=400gb --gpus \"device=1\" --name jol_job2  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/generate_UNI_embeddings.py; exec bash'
"""

"""
screen -dmS wsi_sync sh -c 'rsync -av --progress --exclude=*.ndpi --exclude=*.pt --exclude=*.log --exclude=*.csv --exclude=*plots/ -e 'ssh -p 30044' r4:/data/pamly/storage/ /sybig/projects/FedL/data/pamly/storage'
"""


def load_uni_model(device):
    # login(add_to_git_credential=False)  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    local_dir = "./tmp/vit_large_patch16_256.dinov2.uni_mass100k"
    # os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    # hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5,
                              num_classes=0, dynamic_img_size=True)
    # model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    model.load_state_dict(torch.load(os.path.join(local_dir, "model.pth"), map_location="cpu"), strict=True)
    # torch.save(model.state_dict(), os.path.join(local_dir, "model.pth"))
    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    torch.hub.set_dir('tmp/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_uni_model(device)

    embedding_dir = "/data/pamly/embeddings_512px_normalized_patches"
    tile_dir = "/data/pamly/sampled_tiles"

    for slide in tqdm(os.listdir(tile_dir)):
        mean, std = torch.load(f"/data/pamly/slide_means_stds_labeled_only/{slide}/mean_std.pt")
        if not os.path.exists(f"{embedding_dir}/{slide}"):
            os.makedirs(f"{embedding_dir}/{slide}", exist_ok=True)
        transform = transforms.Compose([
            # transforms.Resize((224, 224), antialias=True),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            transforms.Normalize(mean, std)
        ])
        for patch in tqdm(os.listdir(f"{tile_dir}/{slide}")):
            image, label = torch.load(f"{tile_dir}/{slide}/{patch}")
            image = transform(image.div(255)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(image).squeeze()
            torch.save((embedding, label), f"{embedding_dir}/{slide}/{patch}")
