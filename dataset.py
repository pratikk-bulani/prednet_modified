import hickle as hkl
import torch
import os

class VideosDataset(torch.utils.data.Dataset):
    def __init__(self, videos_path):
        self.videos_path = videos_path
        self.video_files = sorted(os.listdir(self.videos_path))
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, index):
        return os.path.join(self.videos_path, self.video_files[index])

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, hkl_file, nt):
        self.hkl_file = hkl_file
        self.X = hkl.load(self.hkl_file)
        self.nt = nt
        self.possible_starts = list(range(0, self.X.shape[0] - self.nt + 1, self.nt))

    def __getitem__(self, index):
        loc = self.possible_starts[index]
        return self.X[loc:loc+self.nt]

    def __len__(self):
        return len(self.possible_starts)

if __name__ == "__main__":
    dataset_1 = VideosDataset('/ssd_scratch/cvit/bullu/preprocessed_inputs/')
    dataloader_1 = torch.utils.data.DataLoader(dataset_1)
    for hkl_file in dataloader_1:
        hkl_file = hkl_file[0]
        dataset_2 = VideoDataset(hkl_file, 10)
        dataloader_2 = torch.utils.data.DataLoader(dataset_2, batch_size = 10)

        for inp in dataloader_2:
            print(inp.shape)