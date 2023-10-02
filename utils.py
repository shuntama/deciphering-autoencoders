import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_dir, npz_path, transform=None):
        self.data_dir = data_dir
        self.npz_path = npz_path
        self.transform = transform

        self.image_list = []
        files = os.listdir(self.data_dir)
        files = sorted(files)
        for image_name in files:
            image_path = os.path.join(self.data_dir, image_name)
            self.image_list.append(image_path)

        npz_file = np.load(self.npz_path, allow_pickle=True)
        self.masks = [npz_file['arr_{}'.format(i)] for i in range(len(self.image_list))]
        npz_file.close()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        mask = self.masks[idx]
        mask = [torch.from_numpy(mask[i].astype(np.float32)) for i in range(mask.shape[0])]
        return image, mask


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = MyDataset(args.data_dir, args.npz_path, transform=transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )
    return dataloader


def save_images(images, path):  # images: [-1, 1]
    images = images[:256] if images.size()[0] > 256 else images
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    images = torchvision.utils.make_grid(images, nrow=8, padding=2, pad_value=1)
    images = torchvision.transforms.ToPILImage()(images)
    images.save(path)


def setup_log(run_name):
    os.makedirs(os.path.join('logs', 'runs', run_name), exist_ok=True)
    os.makedirs(os.path.join('logs', 'weights', run_name), exist_ok=True)
    os.makedirs(os.path.join('logs', 'images', run_name), exist_ok=True)


def save_config(args, path):
    with open(path, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('\n')
