import os

from parse_args import args
import numpy as np
import torch

def load_data():
    data_path = args.data_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    landUse_feature = np.load(data_path + args.landUse_dist)
    landUse_feature = landUse_feature[np.newaxis]
    landUse_feature = torch.Tensor(landUse_feature).to(device)

    POI_feature = np.load(data_path + args.POI_dist)
    POI_feature = POI_feature[np.newaxis]
    POI_feature = torch.Tensor(POI_feature).to(device)

    mob_feature = np.load(data_path + args.mobility_dist)
    mob_feature = mob_feature[np.newaxis]
    mob_feature = torch.Tensor(mob_feature).to(device)

    mob_adj = np.load(data_path + args.mobility_adj)
    mob_adj = mob_adj/np.mean(mob_adj)
    mob_adj = torch.Tensor(mob_adj).to(device)

    poi_sim = np.load(data_path + args.POI_simi)
    poi_sim = torch.Tensor(poi_sim).to(device)

    land_sim = np.load(data_path + args.landUse_simi)
    land_sim = torch.Tensor(land_sim).to(device)

    # Load semantic channel tensors
    sem_dir = data_path + args.semantic_dir
    sem_profiles_path = os.path.join(sem_dir, args.semantic_profiles)
    sem_dist_path = os.path.join(sem_dir, args.semantic_distribution)
    sem_mask_path = os.path.join(sem_dir, args.semantic_mask)

    semantic_profiles = np.load(sem_profiles_path)  # [N, K, d]
    semantic_distribution = np.load(sem_dist_path)  # [N, K]
    semantic_mask = np.load(sem_mask_path)  # [N, K]

    semantic_profiles = torch.Tensor(semantic_profiles).to(device)
    semantic_distribution = torch.Tensor(semantic_distribution).to(device)
    semantic_mask = torch.Tensor(semantic_mask).to(device)

    features = [POI_feature, landUse_feature, mob_feature]

    return (
        features,
        mob_adj,
        poi_sim,
        land_sim,
        semantic_profiles,
        semantic_distribution,
        semantic_mask,
    )
