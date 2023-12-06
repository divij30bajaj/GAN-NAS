# GAN-NAS
Inspired by [Off-Policy Reinforcement Learning for Efficient and Effective GAN Architecture Search, ECCV 2020](https://arxiv.org/pdf/2007.09180.pdf), our work proposes four changes to the original work, which has used SAC to search the best Generator network while keeping the Discriminator constant. Our contributions include:
1. **Replacing SAC with DDPG** (Updated *search.py*, added *ddpg.py, ddpg_model.py* in `DDPG/search` directory)
2. **Replacing SAC with DQN** (Updated *search.py, added *dqn.py, dqn_model.py* in `DQN/search` directory)
3. **Updating the reward function to incorporate stability of the GAN** (Updated *search.py* and *functions.py* in `StabilityReward/search` directory)
4. **Adding another SAC agent to learn the discriminator using a separate action space** (Updated *search.py, functions.py* in `DiscriminatorSearch/search`, added *shared_gen_dis.py, dis_blocks_search.py* in `DiscriminatorSearch/search/models_search` and `DiscriminatorSearch/eval/models_search`, updated *train_derived.py, functions.py* in `DiscriminatorSearch/eval`)
   
### Dependencies ###
```pip install pytorch==1.4.0 torchvision==0.5.0
python3 -m pip install imageio
python3 -m pip install scipy
python3 -m pip install six
python3 -m pip install numpy==1.18.1
python3 -m pip install python-dateutil==2.7.3
python3 -m pip install tensorboardX==1.6
python3 -m pip install tensorflow-gpu==1.13.1
python3 -m pip install tqdm==4.29.1
```

Go to the folder of the experiment you want to perform. Then run following commands:

### Searching an architecture ###
```cd search
python -u search.py -gen_bs 128 -dis_bs 64 --dataset cifar10 --bottom_width 4 --img_size 32 --gen_model shared_gan --dis_model shared_gan --controller controller --latent_dim 128 --gf_dim 128 --df_dim 64 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 20 --ctrl_sample_batch 1 --shared_epoch 15 --grow_step1 15 --grow_step2 35 --max_search_iter 65 --ctrl_step 30 --random_seed 12345 --exp_name e2gan_search | tee search.log
```

### Training the best architecture ###
```cd eval
python3 train_derived.py -gen_bs 128 -dis_bs 64 --dataset cifar10 --bottom_width 4 --img_size 32 --max_iter 150000 --gen_model shared_gan --dis_model shared_gan --latent_dim 128 --gf_dim 256 --df_dim 128 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 20 --arch 0 2 2 1 1 2 2 1 1 0 2 2 1 3 --exp_name cifar
```
Note: In the above python command, replace the --arch value witht the best architecture found in the previous step. You can find the reward score of each epiode in the output of the search step. Each step of very episode mentions the architecture of that step. It will be an array of size 4 for layer 0 and an array of size 5 for the other two layers. Combine these 14 values and mention against the --arch argument in the above command.

### Testing the trained model ###
Load the checkpoint in eval/checkpoints and mention the architecture of the generator against the --arch argument. Then run the following commands:
```cd eval
python3 test.py --dataset cifar10 --img_size 32 --bottom_width 4 --gen_model shared_gan --latent_dim 128 --gf_dim 256 --g_spectral_norm False --load_path checkpoints/discriminator.pth --arch 1 1 1 1 0 0 0 1 1 1 1 1 1 2 --exp_name test_cifar
```
