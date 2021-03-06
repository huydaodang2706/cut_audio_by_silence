import argparse
import os
from collections import OrderedDict

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from agent import get_agent
from common import PHASE_TESTING, PHASE_TRAINING, get_config
from dataset import get_dataloader
from utils import cycle


# Use multiple GPUs
# OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=0,1 python3 train.py
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 python3 train.py
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train.py

# Visualization:
# In project directory run: tensorboard --logdir train_log
# On server run: python3 -m tensorboard.main --logdir train_log --port 10086 --host 127.0.0.1
# On google cloud server run: tensorboard --logdir train_log --port 10086 --host 127.0.0.1
# Multiple experiments:
# 1. ln -s train_log/log joint_experiments/exp1_log
# 2. ln -s ...
# 3. tensorboard --logdir joint_experiments --port 10086 --host 127.0.0.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue', dest='cont', action='store_true', help="continue training from checkpoint")
    parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")
    parser.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
    args = parser.parse_args()

    # create experiment config
    config = get_config(args)
    print(config)

    # create network and training agent
    tr_agent = get_agent(config)
    print(tr_agent.net)

    # load from checkpoint if provided
    if args.cont:
        tr_agent.load_ckpt(load_path=args.ckpt)

    writer = SummaryWriter()
    # create dataloader
    train_data = '/home3/huydd/vad_dataset/vad_silence_data/train_8khz_aicc_6h_voice_6h_unvoice.csv'
    val_data = '/home3/huydd/vad_dataset/vad_silence_data/train_8khz_aicc_6h_voice_6h_unvoice.csv'
    val_data_2 = '/home3/huydd/vad_dataset/vad_data_team_data_gan_17_2_2022/vad_data.csv'
    # train_data = '/home3/huydd/cut_audio_by_silence/Speech-Denoise/model_1_silent_interval_detection/train.csv'
    # val_data = '/home3/huydd/cut_audio_by_silence/Speech-Denoise/model_1_silent_interval_detection/val.csv'
    
    train_loader = get_dataloader(PHASE_TRAINING, sample_rate=8000, batch_size=config.batch_size, num_workers=config.num_workers, csv_file=train_data)
    val_loader = get_dataloader(PHASE_TESTING, sample_rate=8000, batch_size=config.batch_size, num_workers=config.num_workers, csv_file=val_data)
    val_loader_2 = get_dataloader(PHASE_TESTING, sample_rate=8000, batch_size=config.batch_size, num_workers=config.num_workers, csv_file=val_data_2)
    # val_loader = cycle(val_loader)
    # val_loader = get_dataloader(PHASE_TESTING, batch_size=config.batch_size, num_workers=config.num_workers, csv_file='/home3/huydd/cut_by_mean/GLDNN_EOU_detection/val_silence_6.csv')
    # val_loader_step = get_dataloader(PHASE_TESTING, batch_size=config.batch_size, num_workers=config.num_workers, csv_file='/home3/huydd/cut_by_mean/GLDNN_EOU_detection/val_silence_6.csv')
    # val_loader_step = cycle(val_loader_step)

    # start training
    clock = tr_agent.clock
    max_epoch_acc = 0
    for e in range(clock.epoch, config.nr_epochs):
        n = 0
        # begin iteration
        train_losses_sum = 0
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            n += 1
            outputs, train_losses = tr_agent.train_func(data)
            
            train_losses_sum += train_losses['bce'].item()
            
            # visualize
            # if args.vis and clock.step % config.visualize_frequency == 0:
            #     tr_agent.visualize_batch(data, "train", outputs)

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in train_losses.items()}))

            clock.tick()
        train_losses_sum =  train_losses_sum / n
        print("\nResult Epoch {} Train Loss {} ".format(e, train_losses_sum))
        writer.add_scalar('Loss/train',train_losses_sum, e)
    
        # Calculate val loss
        # val_loss = tr_agent.eval(val_loader)
    
        # save the best accuracy
        epoch_acc, val_loss = tr_agent.evaluate(val_loader)

        print("\nVal loss all data epoch {}:{}".format(e,val_loss))
        writer.add_scalar('Val loss data all:',val_loss, e)

        print("Epoch {} - accuracy all data {}".format(e, epoch_acc))
        writer.add_scalar('Val data all accuracy',epoch_acc, e)

        epoch_acc_2, val_loss_2 = tr_agent.evaluate(val_loader_2)

        print("\nVal loss aicc 6h voice epoch {}:{}".format(e,val_loss_2))
        writer.add_scalar('Val loss aicc 6h voice:',val_loss_2, e)

        print("Epoch {} - accuracy aicc 6h voice {}".format(e, epoch_acc_2))
        writer.add_scalar('Val aicc 6h voice accuracy',epoch_acc_2, e)

        if epoch_acc_2 > max_epoch_acc:
            tr_agent.save_ckpt('best_acc')
            max_epoch_acc = epoch_acc_2

        tr_agent.update_learning_rate()
        clock.tock()

        if clock.epoch % config.save_frequency == 0:
            tr_agent.save_ckpt()
        tr_agent.save_ckpt('latest')
    
    writer.close()


if __name__ == '__main__':
    main()
