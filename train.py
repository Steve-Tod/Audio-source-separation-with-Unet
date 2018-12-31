from dataset import STFTData, TestTransformerDb, TrainTransformerDb
from models.unet.unet_model import UNet
from validation import evaluate, eval_train

import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np

import os, argparse, datetime, random
import matplotlib.pyplot as plt
import scipy
import librosa.display

plt.switch_backend('agg')
SEG_LENGTH = 130816  # about 3 seconds
WINDOW_SIZE = 1022
HOP_LENGTH = 256
instrument_dict = {
    'flute': 5,
    'acoustic_guitar': 1,
    'accordion': 6,
    'xylophone': 3,
    'saxophone': 2,
    'cello': 7,
    'violin': 4,
    'trumpet': 0
}
instrument_array = [
    'trumpet', 'acoustic_guitar', 'saxophone', 'xylophone', 'violin', 'flute',
    'accordion', 'cello'
]


def dt():
    return datetime.datetime.now().strftime('%D %H:%M:%S')


def get_sdr_dict(sdr_list, instrument_array):
    assert len(sdr_list) == len(instrument_array)
    mean_sdr_dict = {}
    std_sdr_dict = {}
    for i, l in enumerate(sdr_list):
        a = np.array(l)
        mean_sdr_dict[instrument_array[i]] = np.mean(a)
        std_sdr_dict[instrument_array[i]] = np.std(a)
    return mean_sdr_dict, std_sdr_dict


start_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
parser = argparse.ArgumentParser(description='Unet source separation')
parser.add_argument('--trainlist', default='./info/train_list.json', type=str)
parser.add_argument('--testlist', default='./info/test_list.json', type=str)
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--bs', default=4, type=int, help='batch size')
parser.add_argument(
    '--output_dir', default='./experiment', type=str, help='output dir')
parser.add_argument(
    '--test_name', default=start_time, type=str, help='output dir')
parser.add_argument('--epoch', default=50, type=int, help='Number of epoch')
parser.add_argument(
    '--log_interval', default=100, type=int, help='Steps between two log')
parser.add_argument(
    '--load_checkpoint', default='', type=str, help='Path to checkpoint model')
args = parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()

stft_trainset = STFTData(args.trainlist, transform=TrainTransformerDb())
dl = data.DataLoader(
    stft_trainset, batch_size=args.bs, shuffle=True, num_workers=6)

EPOCH = args.epoch
PRINT_INTERVAL = args.log_interval
log_dir = os.path.join(args.output_dir, args.test_name, "log")
model_dir = os.path.join(args.output_dir, args.test_name, "model")
os.makedirs(log_dir)
os.makedirs(model_dir)
print("Log to " + log_dir)
print("Save models to " + model_dir)
writer = SummaryWriter(log_dir=log_dir, comment="Unet, 100 epochs, lr=0.001")

model = UNet(1, 8).float().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[8], gamma=0.1)

if args.load_checkpoint != '':
    trained = torch.load(args.load_checkpoint)
    model.load_state_dict(trained)

sdr_list = [[] for x in range(len(instrument_array))]
model.train()
global_step = 0
hist_step = 0

# train
print('Start training! Time: ' + dt())
for epoch in range(EPOCH):
    #scheduler.step()
    running_loss = 0.0
    for i, sample in enumerate(dl, 0):

        optimizer.zero_grad()

        input_stft = sample["stft"][2].cuda()
        mask_list = model(input_stft)
        ref = torch.arange(0, args.bs)
        mask1 = mask_list[ref, sample['instrument'][0]]
        mask2 = mask_list[ref, sample['instrument'][1]]
        mask1 = torch.unsqueeze(mask1, 1)
        mask2 = torch.unsqueeze(mask2, 1)
        output1 = mask1 * input_stft
        output2 = mask2 * input_stft

        loss = criterion(output1, sample["stft"][0].cuda())
        loss += criterion(output2, sample["stft"][1].cuda())

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        writer.add_scalar('result/loss', loss.item(), global_step)

        if i % PRINT_INTERVAL == (PRINT_INTERVAL - 1):
            # save stft figure
            fig = plt.figure()
            ax = plt.subplot(2, 3, 1)
            librosa.display.specshow(
                sample["stft"][0][3, 0, :, :].numpy(),
                sr=44100,
                hop_length=256)
            ax.set_title(instrument_array[sample['instrument'][0][3].item()])
            ax = plt.subplot(2, 3, 2)
            librosa.display.specshow(
                sample["stft"][1][3, 0, :, :].numpy(),
                sr=44100,
                hop_length=256)
            ax.set_title(instrument_array[sample['instrument'][1][3].item()])
            plt.subplot(2, 3, 3)
            librosa.display.specshow(
                input_stft.cpu().numpy()[3, 0, :, :], sr=44100, hop_length=256)
            plt.subplot(2, 3, 4)
            librosa.display.specshow(
                output1.cpu().detach().numpy()[3, 0, :, :],
                sr=44100,
                hop_length=256)
            plt.subplot(2, 3, 5)
            librosa.display.specshow(
                output2.cpu().detach().numpy()[3, 0, :, :],
                sr=44100,
                hop_length=256)
            writer.add_figure('STFT_compare', fig, hist_step)
            eval_train(output1, output2, sample, sdr_list)

            writer.add_histogram('mask1',
                                 mask1.cpu().detach().numpy(), hist_step)
            writer.add_histogram('mask2',
                                 mask2.cpu().detach().numpy(), hist_step)

            # save sdr mean and std
            mean_dict, std_dict = get_sdr_dict(sdr_list, instrument_array)
            writer.add_scalars(
                'result/sdr_mean', mean_dict, global_step=hist_step)
            writer.add_scalars(
                'result/sdr_std', std_dict, global_step=hist_step)

            running_loss = 0.0
            hist_step += 1
        global_step += 1
    torch.save(model.state_dict(),
               os.path.join(model_dir, 'model%d.pth' % (epoch)))
print('End training! Time: ' + dt())
writer.close()

print('Start evaluate')
stft_testset = STFTData(args.testlist, transform=TestTransformerDb())
test_dl = data.DataLoader(
    stft_testset, batch_size=16, shuffle=True, num_workers=16)
model = model.eval()
sdr_list = evaluate(model, test_dl, instrument_dict, bs=16)
res_str = []
for i, l in enumerate(sdr_list):
    a = np.array(l)
    res_str.append('%s\tmean:%.4f\tstd:%.4f' % (instrument_array[i],
                                                np.mean(a), np.std(a)))
res_str = '\n'.join(res_str)
print('Evaluate result:')
print(res_str)

log_dir = os.path.join(args.output_dir, args.test_name, "log")
with open(os.path.join(log_dir, 'eval_res.txt'), 'w') as f:
    f.write(res_str)
