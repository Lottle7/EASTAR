import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()

# ----------------------- File -----------------------
parser.add_argument('--city', choices=('NY', 'Chi'), default='NY')
parser.add_argument('--output_dir', type=Path, default=ROOT / 'runs')

parser.add_argument('--mobility_dist',     default='/mob_dist.npy')
parser.add_argument('--POI_dist',          default='/poi_dist.npy')
parser.add_argument('--landUse_dist',      default='/landUse_dist.npy')
parser.add_argument('--mobility_adj',      default='/mob-adj.npy')
parser.add_argument('--POI_simi',          default='/poi_simi.npy')
parser.add_argument('--landUse_simi',      default='/landUse_simi.npy')

# ----------------------- Model architecture (design constants) -----------------------
parser.add_argument('--embedding_size', type=int, default=256)
parser.add_argument('--epochs',         type=int, default=3000)
# Text embedding
parser.add_argument('--text_dim', type=int, default=384,
                    help='Dimension of Stage 1 event semantic profiles.')
# Channel design
parser.add_argument('--K',             type=int, default=8,    help='Number of anchored semantic channels.')
parser.add_argument('--M',             type=int, default=2,    help='Number of free channels.')
# Loss weights (design constants)
parser.add_argument('--lambda_mech',    type=float, default=0.8, help='Weight for the mechanism loss.')
parser.add_argument('--lambda_align',   type=float, default=0.4, help='Weight for the alignment loss.')
parser.add_argument('--lambda_div',     type=float, default=0.1, help='Weight for free-channel diversity loss.')
parser.add_argument('--lambda_cross',   type=float, default=0.1, help='Weight for free-vs-semantic decorrelation loss.')

# Optimizer defaults are loaded from configs/default.json for the selected city.
parser.add_argument('--learning_rate',  type=float, default=None, help='optimizer learning rate.')
parser.add_argument('--weight_decay',   type=float, default=None, help='optimizer weight decay.')
parser.add_argument('--input_dropout',  type=float, default=None, help='input dropout probability.')
parser.add_argument('--hidden_dropout', type=float, default=None, help='hidden dropout probability.')
parser.add_argument('--seed', type=int, default=42)

# ----------------------- City (per-city defaults — design constants) -----------------------
parser.add_argument('--data_path',                    default=None)
parser.add_argument('--POI_dim',         type=int,    default=None)
parser.add_argument('--landUse_dim',     type=int,    default=None)
parser.add_argument('--region_num',      type=int,    default=None)
parser.add_argument('--n_struct_blocks', type=int,    default=None)
parser.add_argument('--n_cross_blocks',  type=int,    default=None)
parser.add_argument('--n_global_blocks', type=int,    default=None)
parser.add_argument('--NO_head',         type=int,    default=None)
parser.add_argument('--d_prime',         type=int,    default=None)
parser.add_argument('--d_m',             type=int,    default=None)
parser.add_argument('--c',               type=int,    default=None)
parser.add_argument('--semantic_dir',    default=None)
parser.add_argument('--semantic_profiles',      default=None)
parser.add_argument('--semantic_distribution',  default=None)
parser.add_argument('--semantic_mask',          default=None)

pre_args, _ = parser.parse_known_args()

if pre_args.city == 'NY':
    parser.set_defaults(
        data_path=str(ROOT / 'data_NY'),
        POI_dim=26,
        landUse_dim=11,
        region_num=180,
        n_struct_blocks=3,
        n_cross_blocks=3,
        n_global_blocks=3,
        NO_head=4,
        d_prime=64,
        d_m=72,
        c=32,
        semantic_dir='/semantic',
        semantic_profiles='Manh_k8_semantic_profiles.npy',
        semantic_distribution='Manh_k8_semantic_distribution.npy',
        semantic_mask='Manh_k8_semantic_mask.npy',
    )
elif pre_args.city == 'Chi':
    parser.set_defaults(
        data_path=str(ROOT / 'data_Chi'),
        POI_dim=26,
        landUse_dim=12,
        region_num=77,
        n_struct_blocks=1,
        n_cross_blocks=2,
        n_global_blocks=3,
        NO_head=1,
        d_prime=32,
        d_m=36,
        c=32,
        semantic_dir='/semantic',
        semantic_profiles='CHI_k8_semantic_profiles.npy',
        semantic_distribution='CHI_k8_semantic_distribution.npy',
        semantic_mask='CHI_k8_semantic_mask.npy',
    )
else:
    raise ValueError(f'Unsupported city: {pre_args.city}. Supported: NY, Chi.')

defaults = json.loads((ROOT / 'configs' / 'default.json').read_text(encoding='utf-8'))
parser.set_defaults(**defaults[pre_args.city]['stage1'])
args = parser.parse_args()
if args.epochs < 1:
    parser.error('--epochs must be positive')
