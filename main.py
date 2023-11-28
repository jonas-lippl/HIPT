import os

import torch

from HIPT_4K.hipt_4k import HIPT_4K
from utils.load_data import load_lymphoma_data
import torch.nn.functional as F


"""
screen -dmS hipt_test sh -c 'docker run --gpus \"device=0\"  -it --rm -u `id -u $USER` -v /sybig/home/ftoelkes/preprocessed_data/sandbox_data:/sandbox_data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/main.py; exec bash'
"""

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HIPT_4K(device256=device, device4k=device)
    train_loader = load_lymphoma_data(1, mode='train')
    test_loader = load_lymphoma_data(1, mode='test')
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(20):
        model.train()
        for X, y in train_loader:
            optim.zero_grad()
            out = model.forward(X)
            loss = F.cross_entropy(out, y)
            loss.backward()
            print(loss.item())
            optim.step()

        model.eval()
        correct = 0
        total = 0
        for X, y in test_loader:
            out = model.forward(X)
            pred = torch.argmax(out, dim=1)
            correct += torch.sum(pred == y)
            total += len(y)
        print(f"Accuracy: {correct / total}")

    save_folder = "experiments/hipt_4k"
    if not os.exists(save_folder):
        os.mkdir(save_folder)
    torch.save(model.state_dict(), "experiments/hipt_4k/model.pt")




if __name__ == '__main__':
    main()
