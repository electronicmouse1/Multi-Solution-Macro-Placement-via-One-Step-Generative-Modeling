"""Selection and Refine predictors. Input = the LEGAL macro layout (final_xy, after V0 + LP repair).

SelectionPredictor (accuracy first; used to pick among candidates)
  May use what one OpenROAD floorplan + place run reveals (~25% of the flow time up to GRT): the placed
  std-cell density and macro density maps, and the placement-stage scalars (data.PLACE_FEATS: global /
  detailed placement HPWL, placement-stage WNS/TNS before/after repair_design, buffering, DRV counts,
  power, ...), all z-normalized per design, plus the geometry streams. A linear skip from the
  placement-stage scalars to the heads is initialized at "each head = its own placement-stage estimate"
  (dp_hpwl -> wl, pl_wns -> wns, pl_tns -> tns, pl_power -> pwr) so training starts from the best single proxy.

RefinePredictor (a smooth function of the macro centers only; used for gradient refinement)
  Geometry + differentiable std-cell proxy (stdcell.py: quadratic placement + spreading with macros as
  density blockages). Stage A predicts what OpenROAD's placement would show: std-cell density and
  congestion maps (decoder) and the placement-stage scalars. Stage B predicts the post-GRT costs from the
  shared embedding + stage-A outputs. Trained jointly; stage A is supervised by the placement-stage
  observations, which are dense and much less noisy than the post-GRT labels.

RidgePredictor: closed-form ridge on the per-design z-scored placement-stage scalars (Selection baseline).

MacroSelectionPredictor (selection from the macro layout ALONE, no OpenROAD run per candidate)
  The Selection predictor without its placement-stage inputs: geometry streams (+ the netlist std-cell proxy
  unless use_proxy=False) -> heads, non-smooth pooling. Placement-stage scalars / maps are auxiliary training
  targets only (place_head, decoder), never inputs.

All models return {"out": (B,T) predicted per-design z-scored costs, ...}. No BatchNorm (B = 1 when refining).
"""
import torch
import torch.nn as nn

from . import features as feat
from .data import PLACE_NAMES, PLACE_PRIOR, PLACE_PROXY_OF
from .model import MacroNetLayer, MapDecoder, SpatialCNN, mlp
from .stdcell import StdCellProxy

N_GEOM_MAPS = 7
TIMING_HEADS = ("wns", "tns", "setup_vio")


def smooth_max(x, dim, tau=0.1):
    """tau * logsumexp(x / tau) - tau * log(n): a C-infinity stand-in for max pooling (refinement gradients)."""
    return tau * torch.logsumexp(x / tau, dim) - tau * torch.log(torch.tensor(float(x.shape[dim]), device=x.device))


class SmoothSpatialCNN(SpatialCNN):
    """SpatialCNN with mean + smooth-max pooling instead of mean + max."""

    def forward(self, x):
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        g = torch.cat([x.mean((2, 3)), smooth_max(x.flatten(2), 2)], -1)
        return self.proj(g), feats


class GeometryEncoder(nn.Module):
    """Macro-net graph stream + spatial CNN stream (+ std-cell proxy maps / macro features / scalars) + context."""

    def __init__(self, hidden=128, layers=3, cnn_width=32, ctx_dim=8, node_dim=12, use_proxy=True, extra_maps=0,
                 use_graph=True, smooth=False):
        super().__init__()
        self.use_proxy, self.use_graph, self.smooth = use_proxy, use_graph, smooth
        nd = node_dim + (StdCellProxy.N_MACRO if use_proxy else 0)
        cin = N_GEOM_MAPS + (StdCellProxy.N_MAPS if use_proxy else 0) + extra_maps
        self.out_dim = 64 + 256
        if use_graph:
            self.node_in = mlp(nd, hidden, hidden)
            self.layers = nn.ModuleList([MacroNetLayer(hidden) for _ in range(layers)])
            self.graph_out = nn.Linear(2 * hidden, 256)
            self.out_dim += 256
        self.cnn = (SmoothSpatialCNN if smooth else SpatialCNN)(cin, cnn_width, 256)
        self.ctx = mlp(ctx_dim, 64, 64)
        if use_proxy:
            self.pscal = mlp(StdCellProxy.N_SCALARS, 64, 64)
            self.out_dim += 64

    def net_features(self, Y, design):
        lo, hi = feat.net_bboxes(Y, design)
        deg = torch.log(design.net_degree).view(1, -1, 1).expand(Y.shape[0], -1, 1) / 5.0
        ext = hi - lo
        return torch.cat([ext, ext[..., :1] + ext[..., 1:], deg], -1)

    def forward(self, Y, design, extra_maps=None):
        B, N, _ = Y.shape
        node = feat.node_features(Y, design)
        maps = feat.spatial_maps(Y, design, smooth=self.smooth)
        parts = [self.ctx(design.ctx.view(1, -1).expand(B, -1))]
        if self.use_proxy:
            if design.proxy is not None:
                pm, pn, ps = design.proxy.features(Y)
            else:                                   # design without a proxy file: neutral inputs
                G = maps.shape[-1]
                pm = Y.new_zeros(B, StdCellProxy.N_MAPS, G, G)
                pn = Y.new_zeros(B, N, StdCellProxy.N_MACRO)
                ps = Y.new_zeros(B, StdCellProxy.N_SCALARS)
            node = torch.cat([node, pn], -1)
            maps = torch.cat([maps, pm], 1)
            parts.append(self.pscal(ps))
            self.last_proxy_scalars = ps            # read by MacroSelectionPredictor(prior_fixed=True)
        if extra_maps is not None:
            maps = torch.cat([maps, extra_maps], 1)
        if self.use_graph:
            hm = self.node_in(node)
            pin_x = feat.pin_features(Y, design)
            net_x = self.net_features(Y, design)
            for layer in self.layers:
                hm = layer(hm, pin_x, net_x, design)
            mx = smooth_max(hm, 1) if self.smooth else hm.amax(1)
            parts.append(self.graph_out(torch.cat([hm.mean(1), mx], -1)))
        g, pyr = self.cnn(maps)
        parts.append(g)
        return torch.cat(parts, -1), pyr


class MapSummary(nn.Module):
    """Small CNN over predicted maps -> vector (stage B reads stage A's maps)."""

    def __init__(self, cin=3, out=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
                                 nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
                                 nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU())
        self.proj = nn.Linear(64, out)

    def forward(self, x):
        x = self.net(x)
        return self.proj(torch.cat([x.mean((2, 3)), smooth_max(x.flatten(2), 2)], -1))


class RefinePredictor(nn.Module):
    kind = "refine"

    def __init__(self, n_targets, n_place=len(PLACE_NAMES), hidden=128, layers=3, cnn_width=32, ctx_dim=8,
                 node_dim=12, use_proxy=True, use_graph=True, dropout=0.1, smooth=True, **_):
        super().__init__()
        self.cfg = dict(n_targets=n_targets, n_place=n_place, hidden=hidden, layers=layers, cnn_width=cnn_width,
                        ctx_dim=ctx_dim, node_dim=node_dim, use_proxy=use_proxy, use_graph=use_graph, dropout=dropout,
                        smooth=smooth)
        self.enc = GeometryEncoder(hidden, layers, cnn_width, ctx_dim, node_dim, use_proxy, 0, use_graph, smooth)
        self.fuse = nn.Sequential(nn.Linear(self.enc.out_dim, 256), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(256, 256), nn.SiLU())
        self.decoder = MapDecoder(2 * cnn_width, 4 * cnn_width, out_ch=3)     # stage A: cong H, cong V, cell density
        self.place_head = nn.Linear(256, n_place)                               # stage A: placement-stage scalars
        self.map_summary = MapSummary(3, 64)
        self.stage_b = nn.Sequential(nn.Linear(256 + n_place + 64, 256), nn.SiLU(), nn.Dropout(dropout),
                                     nn.Linear(256, n_targets))

    def forward(self, Y, design, obs=None, want_maps=False):
        h, pyr = self.enc(Y, design)
        f = self.fuse(h)
        place = self.place_head(f)
        maps = self.decoder(pyr[1], pyr[2])
        out = self.stage_b(torch.cat([f, place, self.map_summary(maps)], -1))
        return {"out": out, "place": place, "maps": maps}


class SelectionPredictor(nn.Module):
    kind = "selection"

    def __init__(self, n_targets, n_place=len(PLACE_NAMES), hidden=128, layers=3, cnn_width=32, ctx_dim=8,
                 node_dim=12, use_proxy=True, use_graph=True, dropout=0.1, targets=None, map_head=True,
                 prior_fixed=False, **_):
        super().__init__()
        self.cfg = dict(n_targets=n_targets, n_place=n_place, hidden=hidden, layers=layers, cnn_width=cnn_width,
                        ctx_dim=ctx_dim, node_dim=node_dim, use_proxy=use_proxy, use_graph=use_graph,
                        dropout=dropout, targets=targets, map_head=map_head, prior_fixed=prior_fixed)
        self.prior_fixed = prior_fixed
        self.enc = GeometryEncoder(hidden, layers, cnn_width, ctx_dim, node_dim, use_proxy, N_OBS_MAPS, use_graph)
        self.place_mlp = mlp(2 * n_place, 128, 64)
        self.fuse = nn.Sequential(nn.Linear(self.enc.out_dim + 64, 256), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(256, 256), nn.SiLU())
        self.head = nn.Linear(256, n_targets)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.skip = nn.Linear(n_place, n_targets, bias=False)
        nn.init.zeros_(self.skip.weight)
        prior = torch.zeros(n_targets, n_place)
        if targets is not None and n_place == len(PLACE_NAMES):
            for j, t in enumerate(targets):
                if t in PLACE_PRIOR:
                    prior[j, PLACE_NAMES.index(PLACE_PRIOR[t])] = 1.0
        if prior_fixed:
            # out = fixed physical prior (head z = its placement-stage observation's z) + learned residual from 0:
            # the model starts exactly at the single-signal baseline and can only move away where data supports it
            self.register_buffer("prior", prior)
            self.skip.weight.requires_grad_(False)
        else:
            with torch.no_grad():
                self.skip.weight.copy_(0.5 * prior * (torch.tensor([t in PLACE_PROXY_OF for t in (targets or [])]
                                                                   or [False] * n_targets).float().unsqueeze(1)))
        self.decoder = MapDecoder(2 * cnn_width, 4 * cnn_width, out_ch=3) if map_head else None

    def forward(self, Y, design, obs=None, want_maps=False):
        assert obs is not None, "SelectionPredictor needs placement-stage observations"
        h, pyr = self.enc(Y, design, extra_maps=obs["maps"])
        pz = obs["place"] * obs["mask"]
        h = torch.cat([h, self.place_mlp(torch.cat([pz, obs["mask"]], -1))], -1)
        out = self.head(self.fuse(h)) + (pz @ self.prior.t() if self.prior_fixed else self.skip(pz))
        maps = self.decoder(pyr[1], pyr[2]) if (want_maps and self.decoder is not None) else None
        return {"out": out, "maps": maps}


class RidgePredictor(nn.Module):
    """y_z = [place_z, 1] @ W, fitted in closed form on pooled training samples (per-target sample masks)."""
    kind = "ridge"

    def __init__(self, n_targets, n_place=len(PLACE_NAMES), **_):
        super().__init__()
        self.cfg = dict(n_targets=n_targets, n_place=n_place)
        self.register_buffer("W", torch.zeros(n_place + 1, n_targets))

    def forward(self, Y, design, obs=None, want_maps=False):
        x = torch.cat([obs["place"] * obs["mask"], torch.ones_like(obs["place"][:, :1])], -1)
        return {"out": x @ self.W}

    @torch.no_grad()
    def fit(self, X, Z, M, lam=0.1, z_clip=4.0):
        """X (S,F), Z (S,T), M (S,T) bool usable labels. Per-sample normalized normal equations
        (lam is relative to the mean squared feature), targets clipped to +-z_clip (robust z has heavy tails)."""
        Xa = torch.cat([X, torch.ones_like(X[:, :1])], -1).double()
        Zc = Z.double().clamp(-z_clip, z_clip)
        reg = lam * torch.eye(Xa.shape[1], dtype=torch.float64)
        reg[-1, -1] = 0.0
        for j in range(Z.shape[1]):
            m = M[:, j]
            if m.sum() < 10:
                continue
            n = float(m.sum())
            A = Xa[m].T @ Xa[m] / n + reg
            self.W[:, j] = torch.linalg.solve(A, Xa[m].T @ Zc[m, j] / n).float().to(self.W.device)


class MacroSelectionPredictor(nn.Module):
    """prior_fixed=True (needs use_proxy): out = z(proxy full-netlist HPWL) for every head + learned residual from 0
    (zero-initialized head). z = the proxy's per-design label-free robust normalization over the design's candidate
    pool (StdCellProxy.calibrate), so the model starts exactly at the std-cell proxy single-signal scorer.
    prior_crit=<variant> (wc1 | wc4 | wc90): the timing heads (TIMING_HEADS) take the criticality-weighted proxy HPWL
    z (StdCellProxy.crit_scalar) as their prior instead; designs without crit weights fall back to hpwl_all."""
    kind = "macro"

    def __init__(self, n_targets, n_place=len(PLACE_NAMES), hidden=128, layers=3, cnn_width=32, ctx_dim=8,
                 node_dim=12, use_proxy=True, use_graph=True, dropout=0.1, prior_fixed=False, prior_crit="",
                 targets=None, **_):
        super().__init__()
        assert use_proxy or not prior_fixed, "prior_fixed needs the std-cell proxy"
        assert not prior_crit or (prior_fixed and targets), "prior_crit needs prior_fixed and the target names"
        self.cfg = dict(n_targets=n_targets, n_place=n_place, hidden=hidden, layers=layers, cnn_width=cnn_width,
                        ctx_dim=ctx_dim, node_dim=node_dim, use_proxy=use_proxy, use_graph=use_graph, dropout=dropout,
                        prior_fixed=prior_fixed, prior_crit=prior_crit, targets=targets)
        self.prior_fixed, self.prior_crit = prior_fixed, prior_crit
        # timing heads take the criticality-weighted proxy HPWL as prior, the other heads the plain proxy HPWL
        self.register_buffer("crit_heads", torch.tensor([float(prior_crit != "" and t in TIMING_HEADS)
                                                         for t in (targets or [])] or [0.0] * n_targets))
        self.enc = GeometryEncoder(hidden, layers, cnn_width, ctx_dim, node_dim, use_proxy, 0, use_graph)
        self.fuse = nn.Sequential(nn.Linear(self.enc.out_dim, 256), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(256, 256), nn.SiLU())
        self.head = nn.Linear(256, n_targets)
        if prior_fixed:
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.place_head = nn.Linear(256, n_place)                               # auxiliary target only
        self.decoder = MapDecoder(2 * cnn_width, 4 * cnn_width, out_ch=3)       # auxiliary target only

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        k = prefix + "crit_heads"
        if k not in state_dict:                      # checkpoints saved before prior_crit existed
            state_dict[k] = self.crit_heads.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, Y, design, obs=None, want_maps=False):
        h, pyr = self.enc(Y, design)
        f = self.fuse(h)
        resid = self.head(f)
        out = resid
        if self.prior_fixed:
            prior = self.enc.last_proxy_scalars[:, :1]                          # hpwl_all, (B,1)
            if self.prior_crit and design.proxy is not None and self.prior_crit in design.proxy.crit:
                m = self.crit_heads.view(1, -1)
                prior = prior * (1 - m) + design.proxy.crit_scalar(self.prior_crit) * m
            out = resid + prior.clamp(-6.0, 6.0)
        maps = self.decoder(pyr[1], pyr[2]) if want_maps else None
        return {"out": out, "resid": resid if self.prior_fixed else None, "place": self.place_head(f), "maps": maps}


class StudentPredictor(nn.Module):
    """B2 Refine student: inference input = legal macro layout + the (calibrated) std-cell proxy, exactly like
    RefinePredictor (smooth in the macro centers, no OpenROAD run); training adds privileged supervision from the
    REAL std-cell placements (stdcell_real.py, train2 --real):
      stage A  placement-stage scalars (as Refine), maps (cong H, cong V, cell density, + real cell-level RUDY),
               and the 'oracle' proxy scalars = proxy features evaluated at the real cluster centroids;
      corr     (default on) a per-cluster correction moves the proxy's clusters toward where OpenROAD places them
               (supervised by the real centroids, bounded by corr_max, zero at init); the proxy features are
               re-evaluated at the corrected positions (features_at): corrected full-netlist HPWL / RUDY / macro-to-
               cell distances feed stage B and are supervised by their real-placement values.
    forward returns out, place, maps, oracle (B,7) and, with corr, xc (B,K,2) / macro (B,N,3) / scal (B,7) of the
    corrected clusters; hpwl_hat (B,) = the student's full-netlist HPWL estimate (a single-signal scorer)."""
    kind = "student"

    def __init__(self, n_targets, n_place=len(PLACE_NAMES), hidden=128, layers=3, cnn_width=32, ctx_dim=8,
                 node_dim=12, use_proxy=True, use_graph=True, dropout=0.1, smooth=True, corr=True, corr_max=0.25, **_):
        super().__init__()
        assert use_proxy, "the student needs the std-cell proxy"
        self.cfg = dict(n_targets=n_targets, n_place=n_place, hidden=hidden, layers=layers, cnn_width=cnn_width,
                        ctx_dim=ctx_dim, node_dim=node_dim, use_proxy=True, use_graph=use_graph, dropout=dropout,
                        smooth=smooth, corr=corr, corr_max=corr_max)
        ns = StdCellProxy.N_SCALARS
        self.corr, self.corr_max = corr, corr_max
        self.enc = GeometryEncoder(hidden, layers, cnn_width, ctx_dim, node_dim, True, 0, use_graph, smooth)
        self.fuse = nn.Sequential(nn.Linear(self.enc.out_dim, 256), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(256, 256), nn.SiLU())
        self.decoder = MapDecoder(2 * cnn_width, 4 * cnn_width, out_ch=4)   # cong H, cong V, cell density, real RUDY
        self.place_head = nn.Linear(256, n_place)
        self.oracle_head = nn.Linear(256, ns)
        self.map_summary = MapSummary(4, 64)
        extra = 0
        if corr:
            cin = cnn_width + 2 * cnn_width + 3 + 6       # CNN features at the cluster (2 levels), static, positions
            self.corr_net = nn.Sequential(nn.Linear(cin, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                                          nn.Linear(128, 2))
            nn.init.zeros_(self.corr_net[-1].weight); nn.init.zeros_(self.corr_net[-1].bias)
            self.corr_scal = mlp(ns, 64, 64)
            self.corr_maps = MapSummary(StdCellProxy.N_MAPS, 64)
            extra = 128
        self.stage_b = nn.Sequential(nn.Linear(256 + n_place + 64 + ns + extra, 256), nn.SiLU(), nn.Dropout(dropout),
                                     nn.Linear(256, n_targets))

    @staticmethod
    def _sample(fmap, xy):
        """CNN feature map (B,C,H,W) [y][x] at normalized points xy (B,K,2) -> (B,K,C); bicubic = C1 in xy."""
        g = torch.nn.functional.grid_sample(fmap, xy.unsqueeze(1), mode="bicubic", padding_mode="border",
                                            align_corners=False)
        return g[:, :, 0].transpose(1, 2)

    def _cluster_static(self, p):
        a = p.area
        an = torch.log(a / a.mean().clamp_min(1e-12) + 1e-6)
        seq = p.seq_area / a.clamp_min(1e-12)
        bw = p.Bk.sum(1)
        bw = torch.log1p(bw / bw[bw > 0].median().clamp_min(1e-12)) if bool((bw > 0).any()) else bw
        return torch.stack([an / 3.0, seq, bw], -1)

    def forward(self, Y, design, obs=None, want_maps=False):
        p = design.proxy
        assert p is not None, f"{design.name}: the student needs the std-cell proxy"
        import contextlib
        ctx = p.cached() if getattr(p, "_cache", None) is None else contextlib.nullcontext()
        with ctx:                                                    # proxy spread computed once per Y
            h, pyr = self.enc(Y, design)
            f = self.fuse(h)
            place = self.place_head(f)
            maps = self.decoder(pyr[1], pyr[2])
            oracle = self.oracle_head(f)
            parts = [f, place, self.map_summary(maps), oracle]
            res = {"place": place, "maps": maps, "oracle": oracle, "hpwl_hat": oracle[:, 0]}
            if self.corr:
                Xc, X0, _, _ = p.spread(Y, return_all=True)
                B, K, _ = Xc.shape
                st = self._cluster_static(p).unsqueeze(0).expand(B, -1, -1)
                z = torch.cat([self._sample(pyr[0], Xc), self._sample(pyr[1], Xc), st, Xc, X0, Xc - X0], -1)
                delta = self.corr_max * torch.tanh(self.corr_net(z))
                Xc2 = (Xc + delta).clamp(-0.999, 0.999)
                pm2, pn2, ps2 = p.features_at(Y, Xc2)
                parts += [self.corr_scal(ps2), self.corr_maps(pm2)]
                res.update({"xc": Xc2, "xc_proxy": Xc, "macro": pn2, "scal": ps2, "hpwl_hat": ps2[:, 0]})
            res["out"] = self.stage_b(torch.cat(parts, -1))
        return res


MODELS = {"refine": RefinePredictor, "selection": SelectionPredictor, "ridge": RidgePredictor,
          "macro": MacroSelectionPredictor, "student": StudentPredictor}


def build_model(kind, model_cfg):
    return MODELS[kind](**model_cfg)


N_OBS_MAPS = 3   # placed std-cell density, macro density, availability flag


def make_obs(d, idx, device):
    """Placement-stage observations of samples idx of design d for the Selection / ridge models.
    place: (B,F) z-scores; mask: (B,F) 1 where the entry was observed and the feature is usable for the design;
    maps: (B,3,G,G) = placed std-cell density, macro density, availability (1 if this sample has the maps)."""
    it = torch.as_tensor(idx, dtype=torch.long)
    obs = {"place": d.place_z[it.to(d.place_z.device)].to(device).float()}
    valid = (d.place_valid if d.place_valid is not None else d.place_ok.unsqueeze(1))[it.to(d.place_z.device)]
    obs["mask"] = (d.place_mask.to(device).unsqueeze(0) & valid.to(device)).float()
    G = feat.GRID
    if d.extra.get("obs") is not None:
        m = d.extra["obs"][it.to(d.extra["obs"].device)].to(device).float()
        ok = d.extra["obs_ok"][it.to(d.extra["obs_ok"].device)].to(device).float()
    else:
        m = torch.zeros(len(idx), 2, G, G, device=device)
        ok = torch.zeros(len(idx), device=device)
    obs["maps"] = torch.cat([m * ok.view(-1, 1, 1, 1), ok.view(-1, 1, 1, 1).expand(-1, 1, G, G)], 1)
    return obs
