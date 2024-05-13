import os
import random
import time

import matplotlib.pyplot as plt
import pamly
import timm
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.distributed import init_process_group
from torch.nn import SyncBatchNorm
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, resnet18
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

"""
screen -dmS binary_tissue_classification_full_uni_model sh -c 'docker run --shm-size=400gb --gpus \"device=0,1,2,3,4,5,6,7\" --name jol_job1  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt torchrun --standalone --nproc_per_node=8 /mnt/tissue_classification_full_model.py; exec bash'
screen -dmS binary_tissue_classification_lr_decay_resnet50 sh -c 'docker run --shm-size=400gb --gpus \"device=0,1,2,3,4,5,6,7\" --name jol_job2  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt torchrun --standalone --nproc_per_node=8 /mnt/tissue_classification_full_model.py; exec bash'
screen -dmS binary_tissue_classification_lr_decay_resnet18 sh -c 'docker run --shm-size=400gb --gpus \"device=0,1,2,3,4,5,6,7\" --name jol_job2  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt torchrun --standalone --nproc_per_node=8 /mnt/tissue_classification_full_model.py; exec bash'
screen -dmS binary_tissue_classification_lr_decay_patch_based_train_test_split_full_model sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name jol_job1  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt torchrun --standalone --nproc_per_node=1 /mnt/tissue_classification_full_model.py; exec bash'
"""

BINARY_LABEL_MAPPING = {
    1: 0,
    4: 0,
    16: 1,
    17: 1,
    18: 1,
    19: 0,
    20: 0,
    21: 0,
    22: 0,
    23: 0,
    24: 0,
    27: 0,
    28: 0,
    30: 0,
    31: 0,
    32: 0,
    33: 0,
    34: 0,
    36: 0,
    37: 0,
}


class CustomLRScheduler(LambdaLR):
    def __init__(self, optimizer, total_epochs, decay_rate, last_epoch=-1):
        self.total_epochs = total_epochs
        self.decay_rate = decay_rate
        super().__init__(optimizer, lr_lambda=self.custom_lr_lambda, last_epoch=last_epoch)

    def custom_lr_lambda(self, epoch):
        return (1 + 10 * epoch / self.total_epochs) ** (-self.decay_rate)


class PathDataset(Dataset):
    def __init__(self, paths, slide_means_stds):
        self.paths = paths
        self.slide_means_stds = slide_means_stds

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data, label = torch.load(self.paths[idx], map_location="cpu")
        slide = self.paths[idx].split("/")[-2]
        mean, std = self.slide_means_stds[slide]
        return transforms.Normalize(mean, std)(data.div(255)), torch.tensor(BINARY_LABEL_MAPPING[label.item()])


class UNI_classifier(torch.nn.Module):
    def __init__(self, n_classes=20):
        super().__init__()
        self.fc1 = torch.nn.Linear(1000, 1)
        # self.batch_norm = torch.nn.BatchNorm1d(512)
        # self.fc2 = torch.nn.Linear(512, n_classes)
        # self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x):
        x = self.fc1(x)
        # x = torch.nn.functional.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        return F.sigmoid(x)


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def main():
    ddp_setup()
    torch.hub.set_dir("tmp/")
    torch.manual_seed(0)
    gpu_id = int(os.environ["LOCAL_RANK"])

    classifier = UNI_classifier(n_classes=2)
    classifier = DDP(classifier.to(gpu_id), device_ids=[gpu_id])
    # model = timm.create_model(
    #     "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    # )
    # local_dir = "./tmp/vit_large_patch16_256.dinov2.uni_mass100k"
    # model.load_state_dict(torch.load(os.path.join(local_dir, "model.pth"), map_location="cpu"), strict=True)
    model = resnet18(weights=None)
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model.to(gpu_id), device_ids=[gpu_id])

    effective_batch_size = 64
    epochs = 40
    # lr = 0.001
    train_losses = []
    test_losses = []

    optimizer_G = torch.optim.Adam(model.parameters())
    # optimizer_G = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    optimizer_C = torch.optim.Adam(classifier.parameters())
    # optimizer_C = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.0001)

    # Define schedulers
    warmup_epochs = 7
    total_epochs = 40
    step_size = 7
    initial_lr = 0.00001  # Both SGD and ADAM
    #final_lr = 0.0001 # ADAM
    final_lr = 0.001  # SGD

    scheduler_G = torch.optim.lr_scheduler.OneCycleLR(optimizer_G,
                                                      max_lr=final_lr,
                                                      total_steps=total_epochs,
                                                      pct_start=warmup_epochs / total_epochs,
                                                      anneal_strategy='linear',
                                                      div_factor=(final_lr / initial_lr),
                                                      final_div_factor=1.0)
    scheduler_C = torch.optim.lr_scheduler.OneCycleLR(optimizer_C,
                                                      max_lr=final_lr,
                                                      total_steps=total_epochs,
                                                      pct_start=warmup_epochs / total_epochs,
                                                      anneal_strategy='linear',
                                                      div_factor=(final_lr / initial_lr),
                                                      final_div_factor=1.0)
    # scheduler_G = CustomLRScheduler(optimizer_G, total_epochs=epochs, decay_rate=0.75)
    # scheduler_C = CustomLRScheduler(optimizer_C, total_epochs=epochs, decay_rate=0.75)
    slides = os.listdir("/data/pamly/sampled_tiles")
    # all_paths = []
    # for folder in folders:
    #     for patch in os.listdir(f"/data/pamly/sampled_tiles/{folder}"):
    #         all_paths.append(f"/data/pamly/sampled_tiles/{folder}/{patch}")
    # random.shuffle(all_paths)
    # train_paths = all_paths[:int(0.8 * len(all_paths))]
    # test_paths = all_paths[int(0.8 * len(all_paths)):]
    train_slides = slides[:int(0.8 * len(slides))]
    test_slides = slides[int(0.8 * len(slides)):]
    train_paths = []
    train_slide_means_stds = {}
    for slide in train_slides:
        mean, std = torch.load(f"/data/pamly/slide_means_stds/{slide}/mean_std.pt")
        train_slide_means_stds[slide] = (mean, std)
        for patch in os.listdir(f"/data/pamly/sampled_tiles/{slide}"):
            train_paths.append(f"/data/pamly/sampled_tiles/{slide}/{patch}")
    test_paths = []
    test_slide_means_stds = {}
    for slide in test_slides:
        mean, std = torch.load(f"/data/pamly/slide_means_stds/{slide}/mean_std.pt")
        test_slide_means_stds[slide] = (mean, std)
        for patch in os.listdir(f"/data/pamly/sampled_tiles/{slide}"):
            test_paths.append(f"/data/pamly/sampled_tiles/{slide}/{patch}")
    if gpu_id == 0:
        print(f"Train samples: {len(train_paths)} Test samples: {len(test_paths)}")

    train_set = PathDataset(train_paths, train_slide_means_stds)
    train_loader = DataLoader(train_set, batch_size=int(effective_batch_size / dist.get_world_size()),
                              sampler=DistributedSampler(train_set), num_workers=4, prefetch_factor=2)
    test_set = PathDataset(test_paths, test_slide_means_stds)
    test_loader = DataLoader(test_set, batch_size=int(effective_batch_size / dist.get_world_size()),
                             sampler=DistributedSampler(test_set), num_workers=4, prefetch_factor=2)

    for epoch in range(epochs):
        start = time.time()
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        if gpu_id == 0:
            print(f"Epoch: {epoch}")
        train_loss = 0
        model.train()
        classifier.train()
        for (data, label) in tqdm(train_loader, "Training"):
            data = data.to(gpu_id)
            label = label.float().to(gpu_id)
            optimizer_G.zero_grad()
            optimizer_C.zero_grad()
            embedding = model(data)
            output = classifier(embedding).squeeze()
            loss = F.binary_cross_entropy(output, label)
            train_loss += loss.item()
            loss.backward()
            optimizer_C.step()
            optimizer_G.step()

        train_loss = torch.tensor(train_loss).to(gpu_id)
        dist.barrier()
        dist.all_reduce(train_loss)
        train_loss = train_loss / dist.get_world_size()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        classifier.eval()
        correct = 0
        total = 0
        test_loss = 0
        per_class_correct = {i: 0 for i in range(2)}
        # per_class_correct = {i: 0 for i in range(20)}
        per_class_total = {i: 0 for i in range(2)}
        # per_class_total = {i: 0 for i in range(20)}
        count = 0
        for (data, label) in tqdm(test_loader, "Testing"):
            with torch.no_grad():
                data = data.to(gpu_id)
                label = label.float().to(gpu_id)
                embedding = model(data)
                output = classifier(embedding).squeeze()
                # if count < 5 and gpu_id == 0:
                #     print(f"Output: {output}")
                #     count += 1
                loss = F.binary_cross_entropy(output, label)
                test_loss += loss.item()
                pred = output.round()
                # pred = output.argmax(dim=1)
                # print("Output: ", output)
                correct += torch.sum(pred == label).item()
                total += len(label)
                for p, l in zip(pred, label):
                    per_class_total[l.item()] += 1
                    if p == l:
                        per_class_correct[l.item()] += 1

        test_loss = torch.tensor(test_loss).to(gpu_id)
        accuracy = torch.tensor(correct / total).to(gpu_id)
        dist.barrier()
        dist.all_reduce(test_loss)
        dist.all_reduce(accuracy)
        test_loss = test_loss / dist.get_world_size()
        accuracy = accuracy / dist.get_world_size()
        test_losses.append(test_loss / len(test_loader))
        if gpu_id == 0:
            print(f"Accuracy: {round(accuracy.item(), 4)}")
        for i in range(2):
            # for i in range(20):
            print(
                # f"Class {pamly.TileLabel.from_int(REVERSE_LABEL_MAPPING[i]).to_string()} ({i}): "
                f"Class {i}: "
                f"{per_class_correct[i]}/{per_class_total[i]} = "
                f"{round(per_class_correct[i] / per_class_total[i], 4) if per_class_total[i] != 0 else 0}")

        scheduler_G.step()
        scheduler_C.step()
        stop = time.time()
        if gpu_id == 0:
            print(f"Epoch took {stop - start} seconds")

    if not os.path.exists("./experiments/tissue_classifier_resnet18_normalized_patches"):
        os.makedirs("./experiments/tissue_classifier_resnet18_normalized_patches", exist_ok=True)
    torch.save(classifier.state_dict(), "./experiments/tissue_classifier_resnet18_normalized_patches/classifier.pt")
    torch.save(model.state_dict(), "./experiments/tissue_classifier_resnet18_normalized_patches/feature_extractor.pt")

    plt.plot([loss.cpu() for loss in train_losses], label="Train Loss")
    plt.plot([loss.cpu() for loss in test_losses], label="Test Loss")
    plt.legend()
    plt.savefig("./plots/tissue_classification_loss_resnet18_normalized_patches.png")


if __name__ == '__main__':
    main()
