import argparse
import gc
import math
import os
import time

from torch import nn
from torch.backends import cudnn
from torch.distributed import init_process_group, destroy_process_group

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DistributedSampler, DataLoader
from torchvision import transforms
import vision_transformer as vits

from main_dino_attn_guidance import PathDataset

"""
screen -dmS hipt_finetune_attn_guidance sh -c 'docker run --shm-size=200gb --gpus all  -it --rm -u `id -u $USER` -v /sybig/projects/camelyon17/patches:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt torchrun --standalone --nproc_per_node=8 /mnt/finetune_pretrained_models_attn_guidance.py --batch_size=32; exec bash'
"""

parser = argparse.ArgumentParser(description='HIPT finetuning on lymphoma images')
parser.add_argument('--epochs', type=int, default=40, metavar='N', help='total epochs for training')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size')
parser.add_argument('--save_folder', type=str, default='hipt_4k_finetune_with_supcon_loss',
                    metavar='N', help='save folder')
parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='warmup epochs')
parser.add_argument('--save_every', type=int, default=10, metavar='N', help='save every x epochs')


class AttentionGuidanceLoss(nn.Module):
    def __init__(self):
        super(AttentionGuidanceLoss, self).__init__()

    @staticmethod
    def forward(attns, masks):
        attentions = attns[:, :, 1:, 1:].mean(dim=1)
        # print(f"Student attentions: {student_attentions.shape}")
        # print(f"Masks: {masks.shape}")
        attention_guidance_loss = ((attentions - masks) ** 2).mean()
        return attention_guidance_loss


def get_vit256(pretrained_weights, arch='vit_small', device=torch.device('cuda:0'), requires_grad=True):
    """
    Builds ViT-256 Model.

    Args:
    - pretrained_weights (str): Path to ViT-256 Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.

    Returns:
    - model256 (torch.nn): Initialized model.
    """

    checkpoint_key = 'teacher'
    model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
    for p in model256.parameters():
        p.requires_grad = requires_grad
    model256.eval()

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model256.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

    return model256


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def dump_args(args):
    """
    Dump all arguments to a text file.
    """
    with open(f"/experiments/{args.save_folder}/args.txt", 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")


def general_setup(seed: int = 1, benchmark=True, hub_dir: str = 'tmp/'):
    """
    General setup for training.
    """
    # en- or disable cudnn auto-tuner to find the best algorithm for current hardware
    cudnn.benchmark = benchmark
    torch.hub.set_dir(hub_dir)
    torch.manual_seed(seed)


class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, epochs, lr, batch_size, save_path,
                 save_every, loss_fn):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = DDP(model.to(self.gpu_id), device_ids=[self.gpu_id])
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.save_path = f"experiments/{save_path}"
        self.loss_fn = loss_fn
        self.save_every = save_every

    def _run_batch(self, X: torch.Tensor, masks: torch.Tensor):
        self.optimizer.zero_grad()
        attns = self.model.module.get_last_selfattention(X)
        loss = self.loss_fn(attns, masks)
        loss.backward()
        if self.gpu_id == 0:
            print(f"Loss: {loss.item()}")
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Free up memory
        # del prob, pred, loss
        # torch.cuda.empty_cache()

    def _run_epoch(self, epoch):
        batch_count = 0
        self.train_loader.sampler.set_epoch(epoch)
        self.model.train()
        for X, _, masks in self.train_loader:
            print(
                f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.batch_size} | "
                f"Batch {batch_count * 8 + self.gpu_id}/{len(self.train_loader) * 8}")
            X = X.to(self.gpu_id)
            masks = masks.to(self.gpu_id)
            self._run_batch(X, masks)
            batch_count += 1

        torch.cuda.empty_cache()

    def train(self):
        for epoch in range(self.epochs):
            start = time.time()
            self._run_epoch(epoch)
            end = time.time()
            self.scheduler.step()
            if self.gpu_id == 0:
                print(f"Epoch {epoch} took {round(end - start, 2)} seconds")
            if epoch % self.save_every == 0 or epoch == self.epochs - 1:
                self.save_data()
            gc.collect()

    def save_data(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.model.module.state_dict(), f"{self.save_path}/model.pt")


def main():
    args = parser.parse_args()
    general_setup()
    ddp_setup()
    path_to_data = '/data/'
    annotation_patches = []
    for center in os.listdir(path_to_data):
        if '256_lvl_1_with_annotation_mask' in center:
            for patient in os.listdir(path_to_data + center):
                for patch in os.listdir(path_to_data + center + '/' + patient):
                    annotation_patches.append(path_to_data + center + '/' + patient + '/' + patch)
    transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    dataset = PathDataset(paths=annotation_patches, transform=transform)
    model = get_vit256(pretrained_weights='ckpts/pretrain_40_epochs_64_bs')
    model.train()
    train_loader = DataLoader(dataset, sampler=DistributedSampler(dataset, shuffle=True), batch_size=args.batch_size,
                              num_workers=4, pin_memory=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_epochs,
                                                num_training_steps=args.epochs)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        save_path=args.save_folder,
        save_every=args.save_every,
        loss_fn=AttentionGuidanceLoss(),
    )
    trainer.train()

    destroy_process_group()


if __name__ == '__main__':
    main()
