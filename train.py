import argparse
import torch
from torch.autograd import Variable
from prednet import PredNet
from dataset import VideosDataset, VideoDataset
import os
import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil

# copied directly from the original source
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-epoch", "--num_epochs", help="number of epochs", default=150, type=int)
    parser.add_argument("-bs", "--batch_size", help="batch size", default=16)
    parser.add_argument("-lr", "--lr", help="learning rate", default=0.001)
    parser.add_argument("-nt", "--nt", help="number of time steps", default=10)
    parser.add_argument("-i", "--input_path", help="path where pre-processed .hkl inputs are present", default="/ssd_scratch/cvit/bullu/preprocessed_inputs/", required=True)
    parser.add_argument("-o", "--output_path", help="path where results should be kept", default="./results/", required=True)
    parser.add_argument("-ld", "--log_dir", help="path where results should be kept", default="./logs/", required=True)
    return parser.parse_args()

def lr_scheduler(args, optimizer, epoch):
    if epoch >= args.num_epochs // 2:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr / 10
    return optimizer

if __name__ == "__main__":
    args = parse_args()
    shutil.rmtree(args.output_path, ignore_errors=True); shutil.rmtree(args.log_dir, ignore_errors=True);
    os.makedirs(args.output_path); os.makedirs(args.log_dir);
    writer = SummaryWriter(log_dir=args.log_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]).to(device))
    time_loss_weights = 1./(args.nt - 1) * torch.ones(args.nt, 1).to(device)
    time_loss_weights[0] = 0
    time_loss_weights = Variable(time_loss_weights)

    model = PredNet(R_channels, A_channels, output_mode='error').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dataset_1 = VideosDataset(args.input_path)
    dataloader_1 = torch.utils.data.DataLoader(dataset_1)

    for epoch in range(args.num_epochs):
        print(f"---- Epoch {epoch} ----")
        optimizer = lr_scheduler(args, optimizer, epoch)
        for hkl_file in tqdm.tqdm(dataloader_1):
            hkl_file = hkl_file[0]
            dataset_2 = VideoDataset(hkl_file, args.nt)
            dataloader_2 = torch.utils.data.DataLoader(dataset_2, batch_size = args.batch_size)

            for i, inputs in enumerate(dataloader_2):
                inputs = Variable(inputs.permute(0, 1, 4, 2, 3).to(device)) # batch x time_steps x channel x width x height
                errors = model(inputs) # batch x n_layers x nt
                loc_batch = errors.size(0)
                errors = torch.mm(errors.view(-1, args.nt), time_loss_weights) # batch*n_layers x 1
                errors = torch.mm(errors.view(loc_batch, -1), layer_loss_weights)
                errors = torch.mean(errors)

                optimizer.zero_grad()
                errors.backward()
                optimizer.step()

                # if (i+1)%10 == 0:
                #     print('\nEpoch: {}/{}, step: {}/{}, errors: {}'.format(epoch, args.num_epochs, i, len(dataset_2)//args.batch_size, errors.item()))
                
            writer.add_scalar(f"Error/train/{os.path.basename(hkl_file)}", errors.item(), epoch)
    
    writer.flush()
    writer.close()
    torch.save(model.state_dict(), os.path.join(args.output_path, 'training.pt'))