"""Dual-stream cross-stage PPA predictor: macro-net graph stream + spatial CNN stream + design context.

Input is the RAW macro placement Y (B,N,2) of one design (before hard projection); targets are the
post-GRT costs after projection + LP repair + OpenROAD. No BatchNorm anywhere (guidance runs with B=1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import features as feat


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def segment_mean(x, seg, n_seg):
    """x: (B,K,H), seg: (K,) -> (B,n_seg,H)."""
    B, _, H = x.shape
    out = torch.zeros(B, n_seg, H, device=x.device, dtype=x.dtype).index_add_(1, seg, x)
    cnt = torch.zeros(n_seg, device=x.device, dtype=x.dtype).index_add_(0, seg, torch.ones_like(seg, dtype=x.dtype))
    return out / cnt.clamp_min(1.0).view(1, -1, 1)


class MacroNetLayer(nn.Module):
    """macro -> pin -> net (mean) -> pin -> macro (mean), residual + LayerNorm."""

    def __init__(self, h, pin_dim=4, net_dim=4):
        super().__init__()
        self.pin_up = mlp(h + pin_dim, h, h)
        self.net_upd = mlp(h + net_dim, h, h)
        self.pin_down = mlp(2 * h + pin_dim, h, h)
        self.norm = nn.LayerNorm(h)

    def forward(self, hm, pin_x, net_x, design):
        M = design.net_degree.shape[0]
        N = hm.shape[1]
        msg = self.pin_up(torch.cat([hm[:, design.pin_macro], pin_x], -1))           # (B,P,H)
        hn = self.net_upd(torch.cat([segment_mean(msg, design.pin_net, M), net_x], -1))  # (B,M,H)
        back = self.pin_down(torch.cat([hn[:, design.pin_net], msg, pin_x], -1))     # (B,P,H)
        return self.norm(hm + segment_mean(back, design.pin_macro, N))


class SpatialCNN(nn.Module):
    def __init__(self, cin=7, width=32, out=256):
        super().__init__()
        chans = [cin, width, 2 * width, 4 * width, 4 * width]
        self.blocks = nn.ModuleList()
        for a, b in zip(chans[:-1], chans[1:]):
            self.blocks.append(nn.Sequential(nn.Conv2d(a, b, 3, stride=2, padding=1), nn.GroupNorm(8, b), nn.SiLU(),
                                             nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU()))
        self.proj = nn.Linear(2 * chans[-1], out)

    def forward(self, x):
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        g = torch.cat([x.mean((2, 3)), x.amax((2, 3))], -1)
        return self.proj(g), feats


class MapDecoder(nn.Module):
    """Predict (cong_H, cong_V, cell_density) 64x64 maps from CNN pyramid features (16x16 level)."""

    def __init__(self, c16, c8, out_ch=3):
        super().__init__()
        self.up8 = nn.Conv2d(c8, c16, 1)
        self.body = nn.Sequential(nn.Conv2d(c16, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
                                  nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                  nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
                                  nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                  nn.Conv2d(32, out_ch, 3, padding=1))

    def forward(self, f16, f8):
        x = f16 + F.interpolate(self.up8(f8), size=f16.shape[-2:], mode="bilinear", align_corners=False)
        return self.body(x)


class PPAPredictor(nn.Module):
    def __init__(self, n_targets, hidden=128, layers=3, cnn_width=32, ctx_dim=8, node_dim=12,
                 use_graph=True, use_cnn=True, map_head=True, dropout=0.1):
        super().__init__()
        self.cfg = dict(n_targets=n_targets, hidden=hidden, layers=layers, cnn_width=cnn_width, ctx_dim=ctx_dim,
                        node_dim=node_dim, use_graph=use_graph, use_cnn=use_cnn, map_head=map_head, dropout=dropout)
        self.use_graph, self.use_cnn, self.map_head = use_graph, use_cnn, map_head
        fused = 64
        if use_graph:
            self.node_in = mlp(node_dim, hidden, hidden)
            self.layers = nn.ModuleList([MacroNetLayer(hidden) for _ in range(layers)])
            self.graph_out = nn.Linear(2 * hidden, 256)
            fused += 256
        if use_cnn:
            self.cnn = SpatialCNN(7, cnn_width, 256)
            fused += 256
            if map_head:
                self.decoder = MapDecoder(2 * cnn_width, 4 * cnn_width)
        self.ctx = mlp(ctx_dim, 64, 64)
        self.fuse = nn.Sequential(nn.Linear(fused, 256), nn.SiLU(), nn.Dropout(dropout), nn.Linear(256, 256), nn.SiLU())
        self.head = nn.Linear(256, n_targets)

    def net_features(self, Y, design):
        lo, hi = feat.net_bboxes(Y, design)
        deg = torch.log(design.net_degree).view(1, -1, 1).expand(Y.shape[0], -1, 1) / 5.0
        ext = hi - lo
        return torch.cat([ext, (ext[..., :1] + ext[..., 1:]), deg], -1)            # (B,M,4)

    def forward(self, Y, design, want_maps=False):
        B = Y.shape[0]
        parts = [self.ctx(design.ctx.view(1, -1).expand(B, -1))]
        if self.use_graph:
            hm = self.node_in(feat.node_features(Y, design))
            pin_x = feat.pin_features(Y, design)
            net_x = self.net_features(Y, design)
            for layer in self.layers:
                hm = layer(hm, pin_x, net_x, design)
            parts.append(self.graph_out(torch.cat([hm.mean(1), hm.amax(1)], -1)))
        maps = None
        if self.use_cnn:
            g, pyr = self.cnn(feat.spatial_maps(Y, design))
            parts.append(g)
            if self.map_head and want_maps:
                maps = self.decoder(pyr[1], pyr[2])                                  # 16x16 and 8x8 levels -> 64x64
        out = self.head(self.fuse(torch.cat(parts, -1)))
        return out, maps
