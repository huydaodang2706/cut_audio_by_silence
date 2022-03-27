import torch
import torch.optim as optim
from networks import get_network
import torch.nn as nn
from dataset import get_dataloader
from tqdm import tqdm
import argparse
import os
import numpy as np
# CUDA_VISIBLE_DEVICES=0,1 python3 train.py
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 python3 train.py
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train.py

# data_csv='/home/huydd/NLP/ASR/SentenceSplit/EOU_detection/EOU/dataset.csv'
# trainloader = get_dataloader(data_csv, batch_size=2, num_workers=1)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def calculate_pr_rc(ground_truth_label, frame_predictions):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(ground_truth_label)):
        if ground_truth_label[i] == frame_predictions[i]:
            if ground_truth_label[i] == 0:
                tp += 1
            else:
                tn += 1
        else:
            if ground_truth_label[i] == 0:
                fn += 1
            else:
                fp += 1

    return (tp,tn,fp,fn)

def main():
    # --ckpt example 
    # ckpt_1, ckpt_2, ckpt_3
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
    parser.add_argument('--gpu', type=str2bool, default=False, help="GPU or not")
    
    args = parser.parse_args()

    # EXP_DIR = './exp'
    #Default present_epoch = 0 (training without pretrain)
    present_epoch = 0

    # number_epochs = 40
    batch_size = 1
    num_workers = 2
    # lr = 1e-3
    # lr_step_size = 20

    if args.gpu:
        net = get_network().cuda()
    else:
        net = get_network()
    
    # print(net)
    # print("number_epochs:", number_epochs)
    # print("batch_size:", batch_size)
    # print("learning_rate:",lr)
    # print("lr_step_size:", lr_step_size)

    # Loss function for binary classification
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, lr_step_size)

    # if args.cont:
        # args.ckpt: path to the model 
    check_point = torch.load(args.ckpt)
    net.load_state_dict(check_point['model_state_dict'])
        # optimizer.load_state_dict(check_point['optimizer_state_dict'])
        # scheduler.load_state_dict(check_point['scheduler_state_dict'])
   

    val_data_csv='/home3/huydd/data_silence_youtube/val_youtube_huy_gan.csv'
    # val_data_csv = '/Users/huydd/Hedspi/Hedspi-05/code/cut_audio_by_silence/Speech-Denoise/model_1_silent_interval_detection/val.csv'
    PHASE_TESTING = 'testing'
    val_loader = get_dataloader(PHASE_TESTING, batch_size=batch_size, num_workers=num_workers, csv_file=val_data_csv)
    
    # net = GLDNN().cuda()
    
    
    # Evaluation stage
    # Accuracy
    epoch_acc = []
    val_losses = []

    net.eval()
    pbar = tqdm(val_loader)
    
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for i, data in enumerate(pbar):
            if args.gpu:
                inputs = data['audio'].float().cuda()
                label = data['label'].float().cuda()
            else:
                inputs = data['audio'].float()
                label = data['label'].float()

            # pbar.set_description("EVALUATION")
            outputs = net(inputs)

            # outputs = torch.squeeze(outputs, -1)
            # label = torch.squeeze(label, -1)
            # print("Outputs shape:",outputs.shape)
            loss = criterion(outputs,label)

            val_losses.append(loss.item())
            # print('outputs:', outputs)
            # print('losses:', losses)
            # metrics.update(losses["bce"].item())
            pred_labels = (torch.sigmoid(outputs).detach().cpu().numpy() >= 0.5).astype(float)
            # print('pred_labels:', pred_labels)
            labels = data['label'].numpy()  
            # print('labels:', labels)
            # print(pred_labels)
            tp,tn,fp,fn = calculate_pr_rc(labels[0], pred_labels[0])
            true_positive += tp
            true_negative += tn
            false_positive += fp
            false_negative += fn
            acc = np.mean(pred_labels == labels)
            # print('acc:', acc)
            epoch_acc.append(acc)
            pbar.set_description("Val Accuracy, batch[{}]".format(i))
    
    avg_acc = 0

    print("True_positive:",true_positive)
    print("True_negative:",true_negative)
    print("False_positive:",false_positive)
    print("False_negative:",false_negative)

    precision = (true_positive/(true_positive + false_positive))
    recall = (true_positive/(true_positive + false_negative))
    f1_score = 2*precision*recall/(precision + recall)
    print("Precision:", precision )
    print("Recall:", recall)
    print("F1 score:", f1_score)

    for x in epoch_acc:
        avg_acc += x
    
    avg_acc = avg_acc / len(epoch_acc)
    print("Accuracy at epoch : {}".format(avg_acc))
    
    avg_loss = 0
    for x in val_losses:
        avg_loss += x
    print("Number of Val examples:",len(val_losses))
    val_loss = avg_loss / (len(val_losses))
    print("Val Loss: {}".format(val_loss))
    
    # print(val_losses)

if __name__ == '__main__':
    main()
