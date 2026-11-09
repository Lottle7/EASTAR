import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from EASTAR_Model import EASTARBase
from parse_args import args


def _mob_loss(s_embeddings, t_embeddings, mob):
    inner_prod = torch.mm(s_embeddings, t_embeddings.T)
    phat = F.softmax(inner_prod, dim=-1)
    loss = torch.sum(-mob * torch.log(phat + 1e-4))
    inner_prod = torch.mm(t_embeddings, s_embeddings.T)
    phat = F.softmax(inner_prod, dim=-1)
    loss += torch.sum(-mob.T * torch.log(phat + 1e-4))
    return loss


def _general_loss(embeddings, adj):
    sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    return F.mse_loss(sim, adj)


def _channel_losses(z_sem, z_free, sem_dist, sem_mask):
    # z_sem:[N,K,d], z_free:[N,M,d], sem_dist:[N,K], sem_mask:[N,K]
    N, K, _ = z_sem.shape
    M = z_free.shape[1]

    mask_region = (sem_mask.sum(dim=1) == K)

    if mask_region.any():
        z_valid = z_sem[mask_region]
        s_valid = sem_dist[mask_region]
        scores = torch.norm(z_valid, p=2, dim=-1)  # [Nv,K]
        p = torch.softmax(scores, dim=1)
        s = s_valid / (s_valid.sum(dim=1, keepdim=True) + 1e-8)
        l_align = torch.sum(s * (torch.log(s + 1e-8) - torch.log(p + 1e-8)), dim=1).mean()
    else:
        l_align = torch.zeros((), device=z_sem.device)

    zf = F.normalize(z_free, p=2, dim=-1)
    zs = F.normalize(z_sem, p=2, dim=-1)

    cos_ff = torch.einsum('nmd,ned->nme', zf, zf)
    eye = torch.eye(M, device=z_sem.device).unsqueeze(0)
    l_div = ((cos_ff * (1.0 - eye)) ** 2).sum(dim=(1, 2)).mean()

    cos_fs = torch.einsum('nmd,nkd->nmk', zf, zs)
    l_cross = cos_fs.mean()

    return l_align, l_div, l_cross


class ModelLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        out_s,
        out_t,
        mob_adj,
        out_p,
        poi_sim,
        out_l,
        land_sim,
        z_sem,
        z_free,
        sem_dist,
        sem_mask,
    ):
        l_mob = _mob_loss(out_s, out_t, mob_adj)
        l_poi = _general_loss(out_p, poi_sim)
        l_land = _general_loss(out_l, land_sim)

        l_align, l_div, l_cross = _channel_losses(z_sem, z_free, sem_dist, sem_mask)

        loss = l_mob + l_poi + l_land
        loss = loss + args.lambda_align * l_align + args.lambda_div * l_div + args.lambda_cross * l_cross

        return {
            'total': loss,
            'mob': l_mob,
            'poi': l_poi,
            'land': l_land,
            'align': l_align,
            'div': l_div,
            'cross': l_cross,
        }


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features, mob_adj, poi_sim, land_sim, profiles, distribution, mask = utils.load_data()
    expected = (args.region_num, args.K, args.text_dim)
    if tuple(profiles.shape) != expected:
        raise ValueError(f'Event profiles have shape {tuple(profiles.shape)}, expected {expected}.')

    model = EASTARBase(
        poi_dim=args.POI_dim,
        landUse_dim=args.landUse_dim,
        text_dim=args.text_dim,
        region_num=args.region_num,
        input_dim=args.embedding_size,
        output_dim=args.embedding_size,
        d_prime=args.d_prime,
        d_m=args.d_m,
        c=args.c,
        input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout,
    ).to(device)
    criterion = ModelLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f'Stage 1 | city={args.city} | device={device} | seed={args.seed} | epochs={args.epochs}')
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        out_s, out_t, out_p, out_l, _, _, z_sem, z_free, _, _ = model(
            features, profiles, mask, mob_adj
        )
        losses = criterion(
            out_s, out_t, mob_adj, out_p, poi_sim, out_l, land_sim,
            z_sem, z_free, distribution, mask
        )
        losses['total'].backward()
        optimizer.step()
        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} | loss={losses['total'].item():.6f}", flush=True)

    # Export the final shared channels with dropout disabled.
    model.eval()
    with torch.inference_mode():
        model(features, profiles, mask, mob_adj)
        channels = model.out_channel_set().cpu().numpy()
    if not np.isfinite(channels).all():
        raise FloatingPointError('Stage 1 produced non-finite channels.')

    output_dir = args.output_dir / args.city
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'stage1_channels.npy'
    np.save(path, channels)
    print(f'Saved {path} | shape={channels.shape}')


if __name__ == '__main__':
    main()
