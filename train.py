import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ema_pytorch import EMA
from lpips import LPIPS

from model import Net
from utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def warmup_weight_decay(optimizer, warmup_steps, target_weight_decay):
    current_step = 0

    def warmup():
        nonlocal current_step
        weight_decay = target_weight_decay * current_step / warmup_steps
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = weight_decay
        current_step += 1

    return warmup


def spatial_shift(x, shift_range, unit_size=1):
    b = x.size()[0]
    n_unit = b // unit_size

    r = np.random.randint(-shift_range, shift_range + 1, size=n_unit)
    x_new = torch.zeros_like(x)
    for i in range(n_unit):
        x_tmp = x[unit_size * i:unit_size * (i + 1)]

        r_abs = np.abs(int(r[i]))
        if r[i] > 0:
            x_tmp = F.pad(x_tmp, (r_abs, 0, 0, 0), 'reflect')
            x_tmp = x_tmp[:, :, :, :-r_abs]
        elif r[i] < 0:
            x_tmp = F.pad(x_tmp, (0, r_abs, 0, 0), 'reflect')
            x_tmp = x_tmp[:, :, :, r_abs:]

        x_new[unit_size * i:unit_size * (i + 1)] = x_tmp

    r = 100 * (r + shift_range) / (shift_range * 2)  # [0, 100]
    r = torch.from_numpy(r.astype(np.float32))
    r = r.repeat_interleave(unit_size)
    return x_new.to(device), r.to(device)


def train(args):
    logger = SummaryWriter(f'./logs/runs/{args.run_name}')

    model = Net(in_channels=3, latent_dim=args.latent_dim, dims=args.dims).to(device)
    if args.weight_path:
        model.load_state_dict(torch.load(args.weight_path))

    ema_model = EMA(
        model,
        beta=0.99995,
        update_after_step=100,
        update_every=10,
        include_online_model=False
    ).to(device)

    dataloader = get_data(args)

    optimizer = torch.optim.AdamW([
        {'params': model.mlp.parameters(), 'lr': args.lr / 10},
        {'params': [p for n, p in model.named_parameters() if 'mlp' not in n], 'lr': args.lr}
    ], weight_decay=0.0 if args.warmup else args.weight_decay)

    warmup_steps = int(args.epochs * 0.4)
    target_weight_decay = args.weight_decay
    if args.warmup:
        warmup_fn = warmup_weight_decay(optimizer, warmup_steps, target_weight_decay)

    lpips = LPIPS(net='alex').to(device)
    upsample = torch.nn.Upsample(size=128, mode='bilinear').to(device)

    b, c, size = args.batch_size, 3, args.image_size
    step = 0
    for epoch in range(args.epochs):
        print('Epoch:', epoch + 1)
        model.train()

        if args.warmup and epoch < warmup_steps:
            warmup_fn()
            print("Current weight decay:", optimizer.param_groups[0]['weight_decay'])

        pbar = tqdm(dataloader)
        for _, (x, m) in enumerate(pbar):
            step += 1

            if epoch % 2 == 1:
                x = torch.flip(x, [3])
                m[2] = torch.flip(m[2], [1])

            if args.shift_range > 0:
                x, r = spatial_shift(x, args.shift_range)
            else:
                x = x.to(device)
                r = torch.ones(b).to(device) * 50

            z = torch.randn(b, c, size, size).to(device)
            y = model(z, m, r)

            loss = lpips(upsample(x), upsample(y)).mean() if size < 128 else lpips(x, y).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema_model.update()

            pbar.set_postfix(loss=loss.item())
            logger.add_scalar('loss', loss.item(), global_step=step)

        if (epoch + 1) % 100 == 0:
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
                ye = model(z, m, r, train=False)
                ye_ema = ema_model(z, m, r, train=False)

        if (epoch + 1) % 100 == 0:  # save images and weights
            save_images(x, f'./logs/images/{args.run_name}/{epoch+1:04d}_x.png')
            save_images(y, f'./logs/images/{args.run_name}/{epoch+1:04d}_y.png')
            save_images(ye, f'./logs/images/{args.run_name}/{epoch+1:04d}_ye.png')
            save_images(ye_ema, f'./logs/images/{args.run_name}/{epoch+1:04d}_ye_ema.png')
            torch.save(
                model.state_dict(),
                f'./logs/weights/{args.run_name}/{epoch+1:04d}.pth'
            )
            torch.save(
                ema_model.ema_model.state_dict(),
                f'./logs/weights/{args.run_name}/{epoch+1:04d}_ema.pth'
            )


def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = 'nf128_na1-4-16_nc32'
    args.data_dir = './datasets/cifar_train'
    args.npz_path = './datasets/masks_N50000_nf128_na1-4-16_nc32.npz'
    args.epochs = 1000
    args.lr = 2e-3
    args.weight_decay = 0.08
    args.warmup = True
    args.batch_size = 256
    args.image_size = 32
    args.shift_range = 8
    args.dims = [128, 256, 512, 1024]
    args.latent_dim = 128
    args.weight_path = ''  # not load if None

    setup_log(args.run_name)
    save_config(args, f'./logs/runs/{args.run_name}/config.txt')
    train(args)


if __name__ == '__main__':
    launch()
