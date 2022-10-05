CUDA_VISIBLE_DEVICES=0 python train_GGDC.py --gpu_ids 0 --dataroot ./datasets/males --name males_model --batchSize 1 --verbose --which_epoch latest --checkpoints_dir ./GGDC --save_epoch_freq 10
