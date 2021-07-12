import argparse
import os
from collections import OrderedDict

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from agent import get_agent
from common import PHASE_TESTING, PHASE_TRAINING, get_config
from dataset import get_dataloader
from utils import cycle


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

    # writer = SummaryWriter()
    # create dataloader
    # train_loader = get_dataloader(PHASE_TRAINING, batch_size=config.batch_size, num_workers=2, dataset_json="/home/huydd/train_noise/result_json/result.json")
    val_loader = get_dataloader(PHASE_TESTING, batch_size=config.batch_size, num_workers=2, dataset_json="/home/huydd/other_done/result_json/result.json")
    val_loader_step = get_dataloader(PHASE_TESTING, batch_size=config.batch_size, num_workers=2, dataset_json="/home/huydd/other_done/result_json/result.json")
    val_loader_step = cycle(val_loader_step)

    epoch_acc = tr_agent.evaluate(val_loader)
    print(epoch_acc)


if __name__ == '__main__':
    main()

    # Evaluation based on best acc in music_done : 0.9448766530044075 
    # Evaluation based on best acc in other_done : 0.9293496286682528
