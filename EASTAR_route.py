import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch import optim


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_pred, y_test):
    y_pred[y_pred < 0] = 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, np.sqrt(mse), r2


def task_to_file(task: str):
    if task == 'crime':
        return 'crime_counts.npy', 'prompt_embs_llama_crime.npy'
    if task == 'checkIn':
        return 'check_counts.npy', 'prompt_embs_llama_checkIn.npy'
    if task == 'serviceCall':
        return 'serviceCall_counts.npy', 'prompt_embs_llama_serviceCall.npy'
    if task == 'pop':
        return 'population_counts.npy', 'prompt_embs_llama_pop.npy'
    raise ValueError('Unknown task')


def city_kfold_protocol(city: str):
    # 10-fold K-fold protocol shared across cities
    return {'n_splits': 10, 'shuffle': True, 'random_state': 2024}


def prior_file_for_task(task: str):
    mapping = {
        'crime': 'crime_prior.json',
        'checkIn': 'checkIn_prior.json',
        'serviceCall': 'serviceCall_prior.json',
        'pop': 'pop_prior.json',
    }
    return mapping[task]


def load_prior_weights(prior_dir: str, task: str, channel_num: int):
    prior_path = os.path.join(prior_dir, prior_file_for_task(task))
    with open(prior_path, 'r', encoding='utf-8') as f:
        obj = json.load(f)

    weights = np.array(obj['weights'], dtype=np.float32)
    if weights.shape[0] != channel_num:
        raise ValueError(
            f'Prior channel count mismatch for task={task}: expected {channel_num}, got {weights.shape[0]}'
        )

    weights = np.clip(weights, 1e-6, None)
    weights = weights / weights.sum()
    return weights, obj


def build_ridge_init(region_train, labels_train, alpha=1.0):
    reg = Ridge(alpha=alpha)
    reg.fit(region_train, labels_train)
    weight = torch.tensor(reg.coef_, dtype=torch.float32)
    bias = torch.tensor(np.atleast_1d(reg.intercept_), dtype=torch.float32)
    return weight, bias


class TaskAdaptiveRouter(nn.Module):
    """
    Task-adaptive channel router: an LLM-prior-initialized channel weight
    is refined by downstream supervision, with a ridge-initialized region
    head and a residual text-conditioned refinement head.
    """

    def __init__(self, channel_num=11, region_dim=144, text_dim=4096, hidden_dim=128, init_w_logits=None, tau=1.0):
        super(TaskAdaptiveRouter, self).__init__()
        self.channel_num = channel_num
        self.region_dim = region_dim
        self.tau = tau

        if init_w_logits is None:
            init_w_logits = torch.zeros(channel_num, dtype=torch.float32)
        self.w_t = nn.Parameter(init_w_logits.clone().detach())

        self.text_bottleneck = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.q_proj = nn.Linear(region_dim, hidden_dim)

        self.region_head = nn.Linear(region_dim, 1)

        self.delta_predictor = nn.Sequential(
            nn.Linear(region_dim + hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.delta_predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.delta_predictor[-1].weight)
        nn.init.zeros_(self.delta_predictor[-1].bias)
        nn.init.xavier_normal_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_normal_(self.text_bottleneck[0].weight)
        nn.init.zeros_(self.text_bottleneck[0].bias)

    def init_region_head(self, weight, bias):
        with torch.no_grad():
            self.region_head.weight.copy_(weight.view(1, -1))
            self.region_head.bias.copy_(bias.view(-1))

    def channel_alpha(self):
        return torch.softmax(self.w_t / self.tau, dim=0)

    def fuse_region_emb(self, r_struct):
        # r_struct: (N, C, d)
        alpha = self.channel_alpha().view(1, -1, 1)
        return torch.sum(r_struct * alpha, dim=1)

    def forward(self, r_struct, t_emb):
        r_emb = self.fuse_region_emb(r_struct)

        t_feat = self.text_bottleneck(t_emb)
        q = self.q_proj(r_emb)
        scores = torch.matmul(q, t_feat.transpose(0, 1)) / (q.size(-1) ** 0.5)
        attn_matrix = torch.softmax(scores, dim=-1)
        routed_text = torch.matmul(attn_matrix, t_feat)

        concat_feat = torch.cat([r_emb, routed_text], dim=1)
        base_pred = self.region_head(r_emb)
        delta_pred = self.delta_predictor(concat_feat)
        return base_pred + delta_pred


def train_fold(region_train, prompt_train, labels_train, prior_logits, args, device):
    model = TaskAdaptiveRouter(
        channel_num=region_train.shape[1],
        region_dim=region_train.shape[2],
        text_dim=prompt_train.shape[1],
        hidden_dim=args.hidden_dim,
        init_w_logits=prior_logits,
        tau=args.tau,
    ).to(device)
    model.text_bottleneck[2].p = args.dropout
    with torch.no_grad():
        alpha = model.channel_alpha().cpu().numpy()
    ridge_input = np.sum(region_train * alpha[None, :, None], axis=1)
    ridge_weight, ridge_bias = build_ridge_init(ridge_input, labels_train)
    model.init_region_head(ridge_weight.to(device), ridge_bias.to(device))

    rs = torch.as_tensor(region_train, dtype=torch.float32, device=device)
    text = torch.as_tensor(prompt_train, dtype=torch.float32, device=device)
    target = torch.as_tensor(labels_train, dtype=torch.float32, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(rs, text).flatten(), target)
        loss.backward()
        optimizer.step()
    return model


def main():
    root = Path(__file__).resolve().parent
    defaults = json.loads((root / 'configs' / 'default.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', choices=tuple(defaults), default='NY')
    parser.add_argument('--task', choices=('crime', 'checkIn', 'serviceCall', 'pop'), default='crime')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--prior_dir', type=Path, default=root / 'task_channel_priors')
    parser.add_argument('--output_dir', type=Path, default=root / 'runs')
    selection, _ = parser.parse_known_args()
    parser.set_defaults(**defaults[selection.city]['stage2'][selection.task])
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = root / f'data_{args.city}'
    output_dir = args.output_dir / args.city
    channel_path = output_dir / 'stage1_channels.npy'
    if not channel_path.is_file():
        raise FileNotFoundError(f'Missing {channel_path}. Run Stage 1 with the same city and output directory.')
    channels = np.load(channel_path)
    label_file, prompt_file = task_to_file(args.task)
    prompts = np.load(data_dir / prompt_file)
    labels = np.load(data_dir / label_file).reshape(-1)
    if channels.ndim != 3 or prompts.ndim != 2 or len(channels) != len(labels) or len(prompts) != len(labels):
        raise ValueError('Channels, prompt embeddings and labels must have the same region ordering and count.')

    prior_weights, _ = load_prior_weights(args.prior_dir, args.task, channels.shape[1])
    prior_logits = torch.from_numpy(np.log(prior_weights + 1e-6))

    # Keep the original positive-label cohort for sparse check-in prediction.
    region_ids = np.arange(len(labels))
    if args.task == 'checkIn':
        region_ids = region_ids[labels > 0]
    protocol = city_kfold_protocol(args.city)
    splitter = KFold(**protocol)
    predictions = np.empty(len(region_ids), dtype=np.float32)
    fold_ids = np.empty(len(region_ids), dtype=np.int32)
    fold_alphas = []

    print(f'Stage 2 | city={args.city} | task={args.task} | device={device} | '
          f'seed={args.seed} | epochs per fold={args.epochs}')
    for fold, (train_index, test_index) in enumerate(splitter.split(region_ids), start=1):
        train_ids = region_ids[train_index]
        test_ids = region_ids[test_index]
        model = train_fold(channels[train_ids], prompts[train_ids], labels[train_ids], prior_logits, args, device)
        model.eval()
        with torch.inference_mode():
            rs_test = torch.as_tensor(channels[test_ids], dtype=torch.float32, device=device)
            text_test = torch.as_tensor(prompts[test_ids], dtype=torch.float32, device=device)
            predictions[test_index] = model(rs_test, text_test).flatten().cpu().numpy()
            fold_alphas.append(model.channel_alpha().cpu().numpy())
        fold_ids[test_index] = fold
        print(f'Fold {fold}/{protocol["n_splits"]} complete', flush=True)

    if not np.isfinite(predictions).all():
        raise FloatingPointError('Stage 2 produced non-finite predictions.')
    mae, rmse, r2 = compute_metrics(predictions, labels[region_ids])
    np.savez(
        output_dir / f'{args.task}_predictions.npz',
        region_ids=region_ids, fold_ids=fold_ids, predictions=predictions,
        labels=labels[region_ids], channel_weights=np.stack(fold_alphas),
    )
    metrics = {
        'city': args.city,
        'task': args.task,
        'seed': args.seed,
        'epochs_per_fold': args.epochs,
        'num_regions': len(region_ids),
        'split': protocol,
        'optimizer': {
            key: getattr(args, key) for key in ('lr', 'weight_decay', 'hidden_dim', 'dropout', 'tau')
        },
        'metrics': {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)},
    }
    path = output_dir / f'{args.task}_metrics.json'
    path.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    print(f'MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}')
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
