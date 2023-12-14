# HIPT for Lymphoma Classification

## 1-Hierarchical Pretraining

Pretrain the Vit256 model on the lymphoma patches with 256px resolution and 128µm size. The data is stored as `.pt` files in the folder `data/single_256_px_128mu`. 
The pretraining is done with the script `main_dino.py`.
To be able to use the `.pt` patches the transformations for Dino had to be modified, since the original code works with .png files.

```bash
screen -dmS hipt_256_pretraining sh -c 'docker run --shm-size=200gb --gpus all  -it --rm -u `id -u $USER` -v /sybig/home/jol/Code/blobyfire/data/single_256_px_128mu:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt torchrun --standalone --nproc_per_node=8 /mnt/main_dino.py --arch vit_small --data_path /data/ --output_dir /mnt/ckpts/pretrain_40_epochs_64_bs/ --epochs 40 --batch_size_per_gpu 64; exec bash'
```

Now the Vit256 model is used to generate patch embeddings for all 4k patches. Each 4k patch is split into 256 patches of size 256x256px.
Each of those patches is then embedded with the pretrained Vit256 model. This results in a 256x384 tensor for each 4k patch.
The script `prepare_256_embedding_tokens.py` is used to generate those embeddings. The embeddings are stored in the folder `data/256x384_embedding_tokens`.
These embeddings are used to train the Vit4k model.

```bash
screen -dmS hipt_4k_pretraining sh -c 'docker run --shm-size=200gb --gpus all  -it --rm -u `id -u $USER` -v /sybig/home/jol/Code/blobyfire/data/256x384_embedding_tokens:/data -v /sybig/home/jol/Code/HIPT/1-Hierarchical-Pretraining:/mnt jol_hipt torchrun --standalone --nproc_per_node=8 /mnt/main_dino4k.py --arch vit4k_xs --data_path /data/ --output_dir /mnt/ckpts/pretrain4k_100_epochs_64_bs/ --epochs 100 --batch_size_per_gpu 64; exec bash'
```

## 2-Weakly-Supervised-Subtyping

The Vit4k model is used to generate patch embeddings for all 4k patches of each individual WSI. The Vit4k model produces a 192 dimensional embedding for each patch.
The script `prepare_4k_embedding_tokens.py` is used to generate those embeddings. The embeddings are stored in the folder `data/WSI_patches_4096px_2048mu_4k_embeddings`.
Regarding of how many patches are generated for each WSI, the embeddings are stored in a 2D tensor of shape (n_patches, 192).
The embeddings are then used to train the final VitWSI model.

```bash
screen -dmS hipt_WSI_finetuning sh -c 'docker run --shm-size=200gb --gpus \"device=0\" -it --rm -u `id -u $USER` -v /sybig/home/jol/Code/blobyfire/data:/data -v /sybig/home/jol/Code/HIPT/2-Weakly-Supervised-Subtyping:/mnt jol_hipt python3 /mnt/main.py; exec bash'
```


## Classification using 4k patches, pretrained Vit256 and Vit4k and a fully connected layer.

