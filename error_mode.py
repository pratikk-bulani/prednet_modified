import os, shutil, argparse, torch, tqdm, numpy as np
from prednet import PredNet
from dataset import VideosDataset, VideoDataset
from torch.autograd import Variable
# from PIL import Image
import cv2

# copied directly from the original source
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)

def save_images(tensor, output_path):
    os.makedirs(output_path)
    for i in range(tensor.shape[0]):
        cv2.imwrite(os.path.join(output_path, "{0:04d}.png".format(i + 1)), (tensor[i]*255.).astype(np.uint8))
        # im = Image.fromarray(tensor[i]).convert('RGB')
        # im.save(os.path.join(output_path, "{0:04d}.png".format(i + 1)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", help="batch size", default=128)
    parser.add_argument("-nt", "--nt", help="number of time steps", default=10)
    parser.add_argument("-i", "--input_path", help="path where pre-processed .hkl inputs are present", default="/ssd_scratch/cvit/bullu/preprocessed_inputs/", required=True)
    parser.add_argument("-o", "--output_path", help="path where results should be kept", default="./results/", required=True)
    parser.add_argument("-wts", "--weights_path", help="path to the trained weights .pt file", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    shutil.rmtree(args.output_path, ignore_errors=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = PredNet(R_channels, A_channels, output_mode='error_pixel_wise').to(device)
    model.load_state_dict(torch.load(args.weights_path))
    model.eval()

    dataset_1 = VideosDataset(args.input_path)
    dataloader_1 = torch.utils.data.DataLoader(dataset_1)

    result = dict() # key = hkl file, value = list of len equals no. of frames and each element holds the errors

    for hkl_file in tqdm.tqdm(dataloader_1):
        hkl_file = hkl_file[0]
        dataset_2 = VideoDataset(hkl_file, args.nt)
        dataset_2.possible_starts = list(range(0, dataset_2.X.shape[0]-dataset_2.nt+1)) # Need to predict all the frames
        dataloader_2 = torch.utils.data.DataLoader(dataset_2, batch_size = args.batch_size)

        hkl_file_basename = os.path.basename(hkl_file)
        result[hkl_file_basename] = [[] for i in range(dataset_2.X.shape[0])]

        with torch.no_grad():
            frame_no = 0
            for i, inputs in enumerate(dataloader_2):
                inputs = Variable(inputs.permute(0, 1, 4, 2, 3).to(device)) # batch x time_steps x channel x width x height
                errors = model(inputs)
                for j in range(errors[0].shape[0]):
                    for k in range(args.nt):
                        result[hkl_file_basename][frame_no+k].append(errors[k][j])
                    frame_no += 1
            result[hkl_file_basename] = torch.stack([torch.stack(i, dim = 0).mean(dim = 0) for i in result[hkl_file_basename]], dim = 0)
            max_element = result[hkl_file_basename].max()
            result[hkl_file_basename] = result[hkl_file_basename] / max_element # normalizing
            save_images(result[hkl_file_basename].cpu().numpy(), os.path.join(args.output_path, os.path.splitext(hkl_file_basename)[0]))
            del result[hkl_file_basename]