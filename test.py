import os, shutil, argparse, torch, tqdm, torchvision, numpy as np
from prednet import PredNet
from dataset import VideosDataset, VideoDataset
from torch.autograd import Variable
from PIL import Image

# copied directly from the original source
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)

def save_images(tensor, output_path):
    for i in range(tensor.shape[0]):
        im = Image.fromarray(tensor[i])
        im.save(output_path + "{0:02d}.jpg".format(i+1))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", help="batch size", default=16)
    parser.add_argument("-nt", "--nt", help="number of time steps", default=10)
    parser.add_argument("-i", "--input_path", help="path where pre-processed .hkl inputs are present", default="/ssd_scratch/cvit/bullu/preprocessed_inputs/", required=True)
    parser.add_argument("-o", "--output_path", help="path where results should be kept", default="./results/", required=True)
    parser.add_argument("-wts", "--weights_path", help="path to the trained weights .pt file", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    shutil.rmtree(args.output_path, ignore_errors=True)
    os.makedirs(os.path.join(args.output_path, 'origin')); os.makedirs(os.path.join(args.output_path, 'pred'));

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = PredNet(R_channels, A_channels, output_mode='prediction').to(device)
    model.load_state_dict(torch.load(args.weights_path))

    dataset_1 = VideosDataset(args.input_path)
    dataloader_1 = torch.utils.data.DataLoader(dataset_1)

    for hkl_file in tqdm.tqdm(dataloader_1):
        hkl_file = hkl_file[0]
        dataset_2 = VideoDataset(hkl_file, args.nt)
        dataloader_2 = torch.utils.data.DataLoader(dataset_2, batch_size = args.batch_size)

        output_origin_path = os.path.join(args.output_path, "origin", os.path.splitext(os.path.basename(hkl_file))[0])
        output_pred_path = os.path.join(args.output_path, "pred", os.path.splitext(os.path.basename(hkl_file))[0])

        for i, inputs in enumerate(dataloader_2):
            inputs = Variable(inputs.permute(0, 1, 4, 2, 3).to(device)) # batch x time_steps x channel x width x height
            
            origin = inputs.data.cpu().byte()[:, args.nt-1]
            origin = np.moveaxis(origin.numpy(), 1, -1)
            
            pred = model(inputs).data.cpu().byte()
            pred = np.moveaxis(pred.numpy(), 1, -1)

            save_images(origin, output_origin_path + f"_b{i+1}_")
            save_images(pred, output_pred_path + f"_b{i+1}_")