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
        tr_agent.load_ckpt(args.ckpt)

    writer = SummaryWriter()
    # create dataloader
    # train_data = '/home3/huydd/cut_by_mean/GLDNN_EOU_detection/data/train_youtube_huy_gan.csv'
    # val_data = '/home3/huydd/cut_by_mean/GLDNN_EOU_detection/data/val_youtube_huy_gan.csv'
    train_data = '/home3/huydd/cut_audio_by_silence/Speech-Denoise/model_1_silent_interval_detection/train_first_data.csv'
    val_data = '/home3/huydd/cut_audio_by_silence/Speech-Denoise/model_1_silent_interval_detection/val_first_data.csv'
    
    train_loader = get_dataloader(PHASE_TRAINING, batch_size=config.batch_size, num_workers=config.num_workers, csv_file=train_data,n_fft=254, win_length=200, sample_rate=8000)
    val_loader = get_dataloader(PHASE_TESTING, batch_size=config.batch_size, num_workers=config.num_workers, csv_file=val_data, n_fft=254, win_length=200, sample_rate=8000)
    val_loader_step = get_dataloader(PHASE_TESTING, batch_size=config.batch_size, num_workers=config.num_workers, csv_file=val_data, n_fft=254, win_length=200, sample_rate=8000)
    val_loader_step = cycle(val_loader_step)
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

            # print("\nTrain Loss {}".format(train_losses))
            # validation step
            if clock.step % config.val_frequency == 0:
                data = next(val_loader_step)
                outputs, losses = tr_agent.val_func(data)
                # print("Val Loss {}".format(losses))


                # visualize
                # if args.vis and clock.step % config.visualize_frequency == 0:
                #     tr_agent.visualize_batch(data, "validation", outputs)

            clock.tick()
        train_losses_sum =  train_losses_sum / n
        print("\nResult Epoch {} Train Loss {} ".format(e, train_losses_sum))
        writer.add_scalar('Loss/train',train_losses_sum, e)
    
        # Calculate val loss
        # val_loss = tr_agent.eval(val_loader)
    
        # save the best accuracy
        epoch_acc, val_loss = tr_agent.evaluate(val_loader)

        print("\nVal loss epoch {}:{}".format(e,val_loss))
        writer.add_scalar('Val loss:',val_loss, e)

        print("Epoch {} - accuracy {}".format(e, epoch_acc))
        writer.add_scalar('Val accuracy',epoch_acc, e)

        if epoch_acc > max_epoch_acc:
            tr_agent.save_ckpt('best_acc')
            max_epoch_acc = epoch_acc

        tr_agent.update_learning_rate()
        clock.tock()

        if clock.epoch % config.save_frequency == 0:
            tr_agent.save_ckpt()
        tr_agent.save_ckpt('latest')
    
    writer.close()


if __name__ == '__main__':
    main()
