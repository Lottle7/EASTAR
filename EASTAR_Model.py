import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from parse_args import args


class DeepFc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepFc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class HighOrderDependencyBlock(nn.Module):
    def __init__(self, input_dim, nhead, dropout, dim_feedforward=2048):
        super(HighOrderDependencyBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout, batch_first=True, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class IntraModalSpatialEncoderBlock(nn.Module):
    def __init__(self, input_dim, nhead, c, dropout, dim_feedforward=2048):
        super(IntraModalSpatialEncoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout, batch_first=True, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.expand = nn.Conv2d(1, c, kernel_size=1)
        self.pooling = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.proj = nn.Linear(c, input_dim)
        self.activation = F.relu

    def forward(self, src):
        src2, attnScore = self.self_attn(src, src, src)
        attnScore = attnScore[:, np.newaxis]
        edge_emb = self.expand(attnScore)
        w = edge_emb.softmax(dim=-1)
        w = (w * edge_emb).sum(-1).transpose(-1, -2)
        w = self.proj(w)
        src2 = src2 + w
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class IntraModalSpatialEncoder(nn.Module):
    def __init__(self, input_dim, c, dropout=0.1):
        super(IntraModalSpatialEncoder, self).__init__()
        self.num_block = args.n_struct_blocks
        self.blocks = nn.ModuleList([
            IntraModalSpatialEncoderBlock(input_dim=input_dim, nhead=args.NO_head, c=c, dropout=dropout)
            for _ in range(self.num_block)
        ])
        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        if out.dim() == 3 and out.size(0) == 1:
            out = out.squeeze(0)
        elif out.dim() == 3:
            out = out.squeeze()
        return self.fc(out)


class HighOrderDependencyLayer(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super(HighOrderDependencyLayer, self).__init__()
        self.num_block = args.n_global_blocks
        self.blocks = nn.ModuleList([
            HighOrderDependencyBlock(input_dim=input_dim, nhead=args.NO_head, dropout=dropout)
            for _ in range(self.num_block)
        ])
        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        if len(out.shape) == 3:
            out = out.squeeze()
        return self.fc(out)


class CrossModalPrototypeMemoryBlock(nn.Module):
    def __init__(self, d_model, S):
        super(CrossModalPrototypeMemoryBlock, self).__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)
        attn = self.softmax(attn)
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        return self.mv(attn)


class CrossModalPrototypeMemory(nn.Module):
    def __init__(self, input_dim, d_m):
        super(CrossModalPrototypeMemory, self).__init__()
        self.blocks = nn.ModuleList([
            CrossModalPrototypeMemoryBlock(input_dim, d_m) for _ in range(args.n_cross_blocks)
        ])
        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        if len(out.shape) == 3:
            out = out.squeeze()
        return self.fc(out)


class SemanticAwareModalityRouter(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super(SemanticAwareModalityRouter, self).__init__()
        self.W = nn.Conv1d(emb_dim, out_dim, kernel_size=1, bias=False)
        self.f1 = nn.Conv1d(out_dim, 1, kernel_size=1)
        self.f2 = nn.Conv1d(out_dim, 1, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, src):
        seq_fts = self.W(src)
        f_1 = self.f1(seq_fts)
        f_2 = self.f2(seq_fts)
        logits = f_1 + f_2.transpose(1, 2)
        coefs = torch.mean(self.act(logits), dim=-1)
        coefs = torch.mean(coefs, dim=0)
        return F.softmax(coefs, dim=-1)


class FlowGuidedMobilityUpdater(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.q = nn.Linear(input_dim, input_dim)
        self.k = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, input_dim)

    def forward(self, h_mob, sem_tokens):
        # h_mob: [N,d], sem_tokens:[N,K,d]
        q = self.q(h_mob)
        k = self.k(sem_tokens)
        v = self.v(sem_tokens)
        attn = torch.einsum('nd,nkd->nk', q, k) / (h_mob.shape[-1] ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        upd = torch.einsum('nk,nkd->nd', attn, v)
        return 0.5 * h_mob + 0.5 * upd


class SemanticHardCompletion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.q = nn.Linear(input_dim, input_dim, bias=False)
        self.k = nn.Linear(input_dim, input_dim, bias=False)
        self.v = nn.Linear(input_dim, input_dim, bias=False)
        self.flow_bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, E, mask_region, mob_adj):
        # E:[N,K,d], mask_region:[N]
        N, K, d = E.shape
        assert mob_adj.shape == (N, N)
        E_t = E.transpose(0, 1)  # [K,N,d]
        q = self.q(E_t)
        k = self.k(E_t)
        v = self.v(E_t)

        # Flow-aware graph attention: weights positively correlated with flow strength.
        base_scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)  # [K,N,N]
        flow_strength = torch.log1p(torch.clamp(mob_adj, min=0.0))  # [N,N]
        flow_bias = self.flow_bias_scale * flow_strength.unsqueeze(0)  # [1,N,N]

        # Keep only flow-connected edges (plus self-loops) for propagation.
        eye = torch.eye(N, device=E.device)
        flow_mask = ((mob_adj > 0).float() + eye).clamp(max=1.0)  # [N,N]
        scores = base_scores + flow_bias
        scores = scores.masked_fill(flow_mask.unsqueeze(0) == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        E_propagated = torch.matmul(attn, v).transpose(0, 1)  # [N,K,d]

        mask_expand = mask_region.view(N, 1, 1)
        return torch.where(mask_expand == 1, E, E_propagated)


class BroadcastAttention(nn.Module):
    def __init__(self, d, nhead):
        super().__init__()
        self.d = d
        self.nhead = nhead
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

    def forward(self, q, k, v):
        # q:[B,d], k/v:[B,N,d]
        B, N, d = k.shape
        H = self.nhead
        Dh = d // H
        qh = self.q_proj(q).view(B, H, Dh)
        kh = self.k_proj(k).view(B, N, H, Dh).transpose(1, 2)
        vh = self.v_proj(v).view(B, N, H, Dh).transpose(1, 2)
        scores = torch.einsum('bhd,bhnd->bhn', qh, kh) / (Dh ** 0.5)
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum('bhn,bhnd->bhd', probs, vh).reshape(B, d)
        return self.o_proj(out)


class EASTARBase(nn.Module):
    def __init__(self, poi_dim, landUse_dim, text_dim, region_num, input_dim, output_dim,
                 d_prime, d_m, c, input_dropout=0.1, hidden_dropout=0.3):
        super(EASTARBase, self).__init__()
        self.K = args.K
        self.M = args.M

        self.densePOI2 = nn.Linear(poi_dim, input_dim)
        self.denseLandUse3 = nn.Linear(landUse_dim, input_dim)
        self.denseMob = nn.Linear(region_num, input_dim)
        self.semanticProjector = nn.Linear(text_dim, input_dim)

        self.input_drop_layer = nn.Dropout(input_dropout)

        # Structural branch: per-modality spatial encoder for POI / land-use / mobility
        self.encoderPOI = IntraModalSpatialEncoder(input_dim, c, dropout=hidden_dropout)
        self.encoderLandUse = IntraModalSpatialEncoder(input_dim, c, dropout=hidden_dropout)
        self.encoderMob = IntraModalSpatialEncoder(input_dim, c, dropout=hidden_dropout)
        self.mobility_updater = FlowGuidedMobilityUpdater(input_dim)
        self.semantic_completion = SemanticHardCompletion(input_dim)
        self.decompose_attn = BroadcastAttention(input_dim, max(1, args.NO_head))
        self.q_free = nn.Parameter(torch.randn(self.M, input_dim) * 0.02)

        # Global consensus branch: cross-modal fusion over structural views
        self.fusion_gate = nn.Parameter(torch.zeros(1, input_dim))
        self.highOrderDependencyLayer = HighOrderDependencyLayer(input_dim, dropout=hidden_dropout)
        self.crossModalPrototypeMemory = CrossModalPrototypeMemory(input_dim, d_m)
        self.fc = DeepFc(input_dim, output_dim)
        self.para1 = nn.Parameter(torch.tensor(0.1))
        self.para2 = nn.Parameter(torch.tensor(0.9))
        self.semanticAwareModalityRouter = SemanticAwareModalityRouter(input_dim, d_prime)
        self.activation = F.relu

        # Per-channel task heads: one probe head per output view used in training losses
        self.decoder_s = nn.Linear(output_dim, output_dim)
        self.decoder_t = nn.Linear(output_dim, output_dim)
        self.decoder_p = nn.Linear(output_dim, output_dim)
        self.decoder_l = nn.Linear(output_dim, output_dim)
        self.channel_weight = nn.Parameter(torch.zeros(self.K + self.M + 1))

        self.feature = None
        self.channel_set = None

    def forward(self, x, semantic_profiles, semantic_mask, mob_adj):
        poi_emb, landUse_emb, mob_emb = x

        # base preprocessing
        poi_emb = self.input_drop_layer(self.activation(self.densePOI2(poi_emb)))
        landUse_emb = self.input_drop_layer(self.activation(self.denseLandUse3(landUse_emb)))
        mob_emb = self.input_drop_layer(self.activation(self.denseMob(mob_emb)))

        # structural branch encoders
        poi_emb = self.encoderPOI(poi_emb)
        landUse_emb = self.encoderLandUse(landUse_emb)
        h_flow = self.encoderMob(mob_emb)

        # semantic completion + flow-guided update
        N, K, d_text = semantic_profiles.shape
        sem_proj = self.semanticProjector(semantic_profiles.reshape(N * K, d_text)).view(N, K, -1)
        mask_region = (semantic_mask.sum(dim=1) == K).float()
        E_complete = self.semantic_completion(sem_proj, mask_region, mob_adj)
        mob_upd = self.mobility_updater(h_flow, E_complete)

        # global consensus on {poi, land, mob_upd}
        out = torch.stack([poi_emb, landUse_emb, mob_upd])
        intra_view_embs = out
        out = out.transpose(0, 1)
        out = self.crossModalPrototypeMemory(out)
        out = out.transpose(0, 1)

        p1 = self.para1 / (self.para1 + self.para2)
        p2 = self.para2 / (self.para1 + self.para2)
        out = out * p2 + intra_view_embs * p1

        out1 = out.permute(1, 2, 0)
        coef = self.semanticAwareModalityRouter(out1)
        temp_out = torch.sum(coef.view(-1, 1, 1) * out, dim=0)
        temp_out = temp_out[np.newaxis]
        G = self.highOrderDependencyLayer(temp_out)
        G = self.fc(G)  # [N,d]

        # semantic decomposition branch
        d = h_flow.shape[-1]
        Q = E_complete.reshape(N * K, d)
        K_full = h_flow.unsqueeze(0).expand(N * K, N, d)
        V_full = h_flow.unsqueeze(0).expand(N * K, N, d)
        Z_sem = self.decompose_attn(Q, K_full, V_full).view(N, K, d)

        Q_free = self.q_free.unsqueeze(0).expand(N, self.M, d).reshape(N * self.M, d)
        K_free = h_flow.unsqueeze(0).expand(N * self.M, N, d)
        V_free = h_flow.unsqueeze(0).expand(N * self.M, N, d)
        Z_free = self.decompose_attn(Q_free, K_free, V_free).view(N, self.M, d)

        Z_all = torch.cat([Z_sem, Z_free, G.unsqueeze(1)], dim=1)
        alpha = torch.softmax(self.channel_weight, dim=0)
        R = (alpha.view(1, -1, 1) * Z_all).sum(dim=1)

        self.feature = R
        self.channel_set = Z_all

        out_s = self.decoder_s(R)
        out_t = self.decoder_t(R)
        out_p = self.decoder_p(R)
        out_l = self.decoder_l(R)
        return out_s, out_t, out_p, out_l, coef, mob_upd, Z_sem, Z_free, G, alpha

    def out_feature(self):
        return self.feature

    def out_channel_set(self):
        return self.channel_set
