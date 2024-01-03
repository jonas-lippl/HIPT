import argparse
import gc
import math
import os
import time

import numpy as np
import pandas as pd
from torch.backends import cudnn
from torch.distributed import init_process_group, destroy_process_group

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from HIPT_4K.hipt_4k import HIPT_4K
from utils.load_data import load_lymphoma_data_single_patches
from utils.losses import SupConLoss

"""
screen -dmS hipt_finetune sh -c 'docker run --shm-size=200gb --gpus all  -it --rm -u `id -u $USER` -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt torchrun --standalone --nproc_per_node=8 /mnt/finetune_pretrained_models.py --batch_size=8; exec bash'
"""

parser = argparse.ArgumentParser(description='HIPT finetuning on lymphoma images')
parser.add_argument('--epochs', type=int, default=40, metavar='N', help='total epochs for training')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size')
parser.add_argument('--save_folder', type=str, default='hipt_4k_finetune_with_supcon_loss',
                    metavar='N', help='save folder')
parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='warmup epochs')
parser.add_argument('--save_every', type=int, default=5, metavar='N', help='save every x epochs')
parser.add_argument('--test_every', type=int, default=50, metavar='N', help='test every x epochs')


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
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, epochs, lr, batch_size, save_path,
                 save_every, test_every, loss_fn):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = DDP(model, device_ids=[self.gpu_id], find_unused_parameters=True)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.save_every = save_every
        self.test_every = test_every
        self.save_path = f"experiments/{save_path}"
        self.intermediate_losses = []
        self.statistics = {'epoch': [], 'train_loss': [], 'train_acc': []}  # , 'val_acc': []}
        self.training_corrects = 0
        self.validation_corrects = 0
        self.loss_fn = loss_fn

    def _run_batch(self, X: torch.Tensor, y: torch.Tensor):
        self.optimizer.zero_grad()
        out = self.model.forward(X)
        # out = out.unsqueeze(1)  # Add dimension for contrastive loss
        # prob, pred = self.model.forward(X)
        # self.training_corrects += torch.sum(pred == y)
        loss = self.loss_fn(out.unsqueeze(1), y)
        loss.backward()
        self.intermediate_losses.append(loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Free up memory
        # del prob, pred, loss
        # torch.cuda.empty_cache()

    def _run_validation_batch(self, X: torch.Tensor, y: torch.Tensor):
        with torch.no_grad():
            prob, pred = self.model.forward(X)
            self.validation_corrects += torch.sum(pred == y)

        # Free up memory
        # del prob, pred
        # torch.cuda.empty_cache()

    def _run_epoch(self, epoch):
        batch_count = 0
        self.training_corrects, self.validation_corrects = 0, 0
        total_len_train_data, total_len_validation_data = 0, 0
        self.train_loader.sampler.set_epoch(epoch)
        self.statistics['epoch'].append(epoch)
        self.model.train()
        for X, y in self.train_loader:
            total_len_train_data += len(y)
            print(
                f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.batch_size} | "
                f"Batch {batch_count * 8 + self.gpu_id}/{len(self.train_loader) * 8}")
            X = X.to(self.gpu_id)
            y = y.to(self.gpu_id)
            self._run_batch(X, y)
            batch_count += 1
        self.statistics["train_loss"].append(round(np.mean(self.intermediate_losses), 4))
        self.intermediate_losses = []
        train_acc = round(float(self.training_corrects) / total_len_train_data, 4)
        print(f"[GPU{self.gpu_id}] Training accuracy: {train_acc}")
        self.statistics['train_acc'].append(train_acc)

        # self.model.eval()
        # if epoch % self.test_every == 0:
        #     self.val_loader.sampler.set_epoch(epoch)
        #     with torch.no_grad():
        #         for X, y in self.val_loader:
        #             total_len_validation_data += len(y)
        #             X = X.to(self.gpu_id)
        #             y = y.to(self.gpu_id)
        #             self._run_validation_batch(X, y)
        #     val_acc = round(float(self.validation_corrects) / total_len_validation_data, 4)
        #     print(f"[GPU{self.gpu_id}] Validation accuracy: {val_acc}")
        #     self.statistics['val_acc'].append(val_acc)
        # else:
        #     self.statistics['val_acc'].append(float('nan'))

        torch.cuda.empty_cache()

    def train(self):
        for epoch in range(self.epochs):
            start = time.time()
            self._run_epoch(epoch)
            end = time.time()
            self.scheduler.step()
            if self.gpu_id == 0:
                print(f"Epoch {epoch} took {round(end - start, 2)} seconds")
            if epoch % self.save_every == 0:
                self.save_data()
            gc.collect()

    def save_data(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.model.module.state_dict(), f"{self.save_path}/model.pt")
        # Save building block from model separately
        torch.save(self.model.module.model256.state_dict(), f"{self.save_path}/model256.pt")
        torch.save(self.model.module.model4k.state_dict(), f"{self.save_path}/model4k.pt")
        # torch.save(self.model.module.fc.state_dict(), f"{self.save_path}/classifier.pt")
        df = pd.DataFrame(self.statistics)
        df.to_csv(f"{self.save_path}/statistics_gpu_{self.gpu_id}.csv", index=False)


def main():
    args = parser.parse_args()
    general_setup()
    ddp_setup()
    train_loader = load_lymphoma_data_single_patches(args.batch_size, mode='train')
    # test_loader = load_lymphoma_data_single_patches(args.batch_size, mode='test')
    model = HIPT_4K(device256=int(os.environ["LOCAL_RANK"]), device4k=int(os.environ["LOCAL_RANK"]))
    model.train()
    # classifier = ClassificationHead().to(int(os.environ["LOCAL_RANK"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_epochs,
                                                num_training_steps=args.epochs)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=None,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        save_path=args.save_folder,
        save_every=args.save_every,
        test_every=args.test_every,
        # loss_fn=torch.nn.CrossEntropyLoss(),
        loss_fn=SupConLoss(),
    )
    trainer.train()

    destroy_process_group()


if __name__ == '__main__':
    main()
