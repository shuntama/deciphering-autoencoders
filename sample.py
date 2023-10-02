import os
import random
import argparse
import numpy as np
import torch
import torchvision

from model import Net


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(args):
    model = Net(in_channels=3, latent_dim=args.latent_dim, dims=args.dims).to(device)
    model.load_state_dict(torch.load(args.weight_path, map_location=torch.device(device)))
    #model.eval()

    b, c, size = args.batch_size, 3, args.image_size
    torch.manual_seed(2)

    for i in range(args.n_samples // args.batch_size):
        with torch.no_grad():
            z = torch.randn((b, c, size, size)).to(device)

            dims = [128, 256, 512]
            n_actives = [1, 4, 16]
            m = []
            for j in range(len(dims)):
                mask = torch.zeros((b, dims[j]), dtype=torch.bool)
                for k in range(b):
                    true_positions = np.random.choice(dims[j], n_actives[j], replace=False)
                    mask[k][true_positions] = True
                m.append(mask)

            r = torch.ones(b).to(device) * 50
            y = model(z, m, r, train=False)

        y = (y + 1) / 2
        y = torch.clamp(y, 0, 1)
        for j, img in enumerate(y):
            torchvision.utils.save_image(img, f'{args.save_dir}/{i:04d}_{j:04d}.png')


def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.save_dir = './samples'
    args.weight_path = './logs/weights/nf128_na1-4-16_nc32/1000_ema.pth'
    args.n_samples = 50000
    args.batch_size = 250
    args.image_size = 32
    args.dims = [128, 256, 512, 1024]
    args.latent_dim = 128

    os.makedirs(args.save_dir, exist_ok=True)
    sample(args)


if __name__ == '__main__':
    launch()
