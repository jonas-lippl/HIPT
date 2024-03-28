import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download

"""
screen -dmS supervised_classification sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name jol_job1  -it --rm -u `id -u $USER` -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/supervised_classification.py; exec bash'
"""

if __name__ == '__main__':
    torch.hub.set_dir('tmp/')

    # login(add_to_git_credential=False)  # login with your User Access Token, found at https://huggingface.co/settings/tokens

    local_dir = "./tmp/vit_large_patch16_224.dinov2.uni_mass100k"
    # os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    # hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    # model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    model.load_state_dict(torch.load(os.path.join(local_dir, "model.pth"), map_location="cpu"), strict=True)
    # torch.save(model.state_dict(), os.path.join(local_dir, "model.pth"))
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()


