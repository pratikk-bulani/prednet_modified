import argparse
import hickle as hkl
from imageio import mimread
import os
import numpy as np
from scipy.misc import imresize
import shutil

desired_im_sz = (128, 160)

def process_video(video_path, output_path):
    video_data = np.array(mimread(video_path, memtest=False))
    X = np.zeros((video_data.shape[0],) + desired_im_sz + (3,), np.uint8)
    for i in range(X.shape[0]):
        img = video_data[i]
        X[i] = process_im(img, desired_im_sz)
    hkl.dump(X, os.path.join(output_path, os.path.splitext(os.path.basename(video_path))[0] + '.hkl'))

def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--videos_path", help="path where all the video files are present", required=True)
    parser.add_argument("-o", "--output_path", help="path where all the output files should be kept", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    shutil.rmtree(args.output_path, ignore_errors=True)
    os.makedirs(args.output_path)
    for path in os.listdir(args.videos_path):
        process_video(os.path.join(args.videos_path, path), args.output_path)