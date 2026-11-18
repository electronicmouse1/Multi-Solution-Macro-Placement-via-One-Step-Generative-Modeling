import copy
import random
from flow_matching import schedulers
import pos_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tfd
import numpy as np
import networks
from omegaconf import open_dict
import guidance
import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from flow_matching.hard_constraint_legalizer import Legalizer

class FlowMatchingModel(nn.Module):
    backbones = {
        "att_gnn": networks.AttGNN,
    }
    time_encoders = {
        "sinusoid": pos_encoding.SinusoidContEncoding,
    }
    def __init__(
            self, 
            backbone, 
            backbone_params,
            input_shape, 
            t_encoding_type, 
            t_encoding_dim, 
            legality_guidance_weight=0.0,
            hpwl_guidance_weight=0.0,
            grad_descent_steps=0,
            grad_descent_rate=0.0,
            alpha_init=0.0,  # for opt guidance
            alpha_lr=0.0,  # for opt guidance
            alpha_critical_factor=0.0,  # for opt guidance
            legality_potential_target=0.0,  # for opt guidance
            use_adam=False,  # for opt guidance
            max_diffusion_steps=1000,  # default eval timesteps
            guidance_mode="none",  # none | sgd | opt
            guidance_step = 1000,
            noise_schedule="linear", 
            mask_key=None, 
            use_mask_as_input=False, 
            device="cpu", 
            legality_softmax_factor_min=10.0,
            legality_softmax_factor_max=10.0,
            legality_softmax_critical_factor=0,  # fraction of time at which legality factor reaches max
            sampling_method="euler",  # euler | heun
            prior_noise_dist = 'normal',
            use_hard_constraint_in_sampling = False,
            hard_constraint_w_reg = 0.0,
            hard_constraint_w_dist = 0.0,
            **kwargs
    ):
        super().__init__()
        with open_dict(backbone_params):
            backbone_params.update({
                "in_node_features": input_shape[1],
                "out_node_features": input_shape[1],
                "t_encoding_dim": t_encoding_dim,
                "device": device,
                "mask_key": mask_key if use_mask_as_input else None,
            })

        print(f"Backbone name: {backbone}, params: {backbone_params}")
        if t_encoding_dim > 0:
            self.t_encoder = FlowMatchingModel.time_encoders[t_encoding_type](t_encoding_dim)
        self.mask_key = mask_key
        self.t_encoding_dim = t_encoding_dim
        self._reverse_model = FlowMatchingModel.backbones[backbone](**backbone_params)
        self.input_shape = input_shape
        self.max_diffusion_steps = max_diffusion_steps
        if noise_schedule == "linear":
            print("Using linear noise schedule for flow matching")
            self._scheduler = schedulers.LinearTimeScheduler()
        elif noise_schedule == "cosine":
            print("Using cosine noise schedule for flow matching. Note: only linear schedule is supported for flow matching Now.")
            self._scheduler = schedulers.CosineScheduler()
        else:
            raise NotImplementedError("Only linear schedule is supported for flow matching")
        
        self._lossfn = nn.MSELoss(reduction="mean")

        # cache some variables
        if prior_noise_dist == 'normal':
            print("Using normal noise distribution for flow matching")
            self._noise_dist = tfd.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        elif prior_noise_dist == 'uniform':
            print("Using uniform noise distribution for flow matching")
            self._noise_dist = tfd.Uniform(torch.tensor([-1.0], device=device), torch.tensor([1.0], device=device))
        elif prior_noise_dist.startswith('normal_sigma'):
            # e.g., "normal_sigma_0.5"
            sigma = float(prior_noise_dist.split('_')[-1])
            print(f"Using normal distribution with σ={sigma} for flow matching")
            self._noise_dist = tfd.Normal(loc=torch.tensor([0.0], device=device),
                                        scale=torch.tensor([sigma], device=device))

        elif prior_noise_dist == 'truncated_normal':
            print("Using truncated normal distribution for flow matching")

            base_dist = tfd.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))

            class TruncatedNormalDist:
                def sample(self, shape):
                    x = base_dist.sample(shape)
                    return torch.clamp(x, -1.0, 1.0)  # clip to layout region
            self._noise_dist = TruncatedNormalDist()


        self._t_dist = tfd.Uniform(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        self.device = device
        
        # guidance parameters
        self.is_guided_sampling = (
            (legality_guidance_weight > 0.0 or hpwl_guidance_weight > 0.0 or alpha_lr > 0.0 or alpha_init > 0.0)
            and (grad_descent_steps > 0)
            and (grad_descent_rate > 0.0)
            and (guidance_mode != "none")
        )
        self.guidance_mode = guidance_mode
        self.legality_guidance_weight = legality_guidance_weight
        self.hpwl_guidance_weight = hpwl_guidance_weight
        self.grad_descent_steps = grad_descent_steps
        self.grad_descent_rate = grad_descent_rate
        self.alpha_init = alpha_init
        self.alpha_lr = alpha_lr
        self.alpha_critical_factor = alpha_critical_factor
        self.legality_potential_target = legality_potential_target
        self.use_adam = use_adam
        self.legality_softmax_factor_min = legality_softmax_factor_min
        self.legality_softmax_factor_max = legality_softmax_factor_max
        self.legality_softmax_critical_factor = legality_softmax_critical_factor
        self.sampling_method = sampling_method

        self.use_hard_constraint_in_sampling = use_hard_constraint_in_sampling
        self.hard_constraint_scores = { 'w_reg': hard_constraint_w_reg, 'w_dist': hard_constraint_w_dist }

        if self.use_hard_constraint_in_sampling:
            print(f"Using hard constraint in sampling.")


    def __call__(self, x, cond, t):
        # input: x=(B, V, F), cond=PyG Data (V nodes, E edges), t=(B)
        # output: vector filed: v_t=(B, V, F)
        t_embed = self.t_encoder(t)
        v_pred = self._reverse_model(x, cond, t_embed)
        return v_pred
    

            
    def loss(self, x, cond, _, idx=None, split="train"):
        B = x.shape[0]
        t = self._t_dist.sample((B,)).squeeze(dim=-1)

        mask = self.get_mask(x, cond)
        
        x_1 = self._noise_dist.sample(x.shape).squeeze(dim=-1)
        x_t = (1 - t).view(B,1,1) * x_1 + t.view(B,1,1) * x
        v_t = (x - x_1)

        v_pred = self(x_t, cond, t)

        # FM core loss
        loss = self._loss(v_pred, v_t, mask)
        metrics = {
            "loss": loss.item(),
            "velocity_norm": v_pred.norm().item(),
        }
        return loss, metrics


    def forward_samples(self, x, cond, intermediate_every=0):
        intermediates = []
        mask = None
        if self.mask_key and self.mask_key in cond:
            mask = self.get_mask(x, cond)
        
        step_size = intermediate_every / self.max_diffusion_steps if intermediate_every else 1
        x_0 = self._noise_dist.sample(x.shape).squeeze(dim=-1)  # (B, V, F)
        
        for t in torch.arange(0, 1 + 1e-9, step_size, device=self.device):
            t = t.expand(x.shape[0])
            x_t = (1 - t).view(x.shape[0], 1, 1) * x_0 + t.view(x.shape[0], 1, 1) * x  # (B, V, F)
            x_t = torch.where(mask, x, x_t) if mask is not None else x_t
            intermediates.append(x_t)
        
        return intermediates

    def reverse_samples(self, B, x_in, cond, num_timesteps=-1, intermediate_every=0, mask_override=None, method="euler"):
        # B: batch size, 
        # intermediate_every: how often to save intermediate results,
        # only support B = 1 when use_hard_constraint_in_sampling = True
        method = self.sampling_method.lower()
        if method not in ["euler"]:
            raise ValueError(f"Unknown sampling method: {method}")
        
        batch_shape = (B, cond.x.shape[0], self.input_shape[1])
        mask_shape = (1, x_in.shape[1], 1)

        if num_timesteps <= 0:
            num_timesteps = self.max_diffusion_steps

        x = self._noise_dist.sample(batch_shape).squeeze(dim=-1)  # (B, V, F)
        mask = mask_override.view(*mask_shape) if mask_override is not None else self.get_mask(x_in, cond)
        x = torch.where(mask, x_in, x) if mask is not None else x


        intermediates = [x]
        self._scheduler.set_timesteps(num_timesteps)
        timesteps = self._scheduler.timesteps
        if self.use_hard_constraint_in_sampling:
            assert B == 1, "Hard constraint sampling only supports batch size 1."
            legalizer = Legalizer(cond = cond, grid_res= 128, scores = self.hard_constraint_scores, device=self.device, dtype=x.dtype, padding=0.001)
            legalizer.run = torch.compile(legalizer.run, mode="reduce-overhead")
            print("Initialized Legalizer for hard constraint sampling.")

        
        for i, t in enumerate(timesteps[:-1]):
            t_next = timesteps[i + 1]
            t_vec = torch.tensor(t, device=self.device).expand(B)
            t_next_vec = torch.tensor(t_next, device=self.device).expand(B)

            # 1) Extrapolation: u1 = u_t + (1 - t) * v_theta(u_t)
            v_pred = self(x, cond, t_vec)
            u1 = x + (1 - t_vec.view(B, 1, 1)) * v_pred

            # 2) Correction: project u1 onto constraint manifold
            if self.use_hard_constraint_in_sampling:
                with torch.no_grad():
                    u1_hat = legalizer.run(u1)  
            else:
                u1_hat = u1

            # 3) Interpolation: u_{t'} = (1 - t') * x + t' * u1_hat
            x = (1 - t_next_vec.view(B, 1, 1)) * x + t_next_vec.view(B, 1, 1) * u1_hat

            x = torch.clamp(x, -2, 2)
            x = torch.where(mask, x_in, x)
            if intermediate_every and (i+1) % intermediate_every == 0:
                intermediates.append(x)
        return x, intermediates
    
    def get_mask(self, x, cond):
        if self.mask_key and self.mask_key in cond:
            mask = cond[self.mask_key]
            B, V, F = x.shape
            mask = mask.view(1, V, 1)
            return mask
        else:
            return None

    def _loss(self, prediction, target, mask=None):
        if mask is not None:
            numel = torch.numel(mask)
            squared_error = torch.square(prediction - target)
            squared_error.masked_fill_(mask, 0)
            mse = torch.mean(squared_error) * (numel / (numel - torch.sum(mask)))
            return mse
        else:
            return self._lossfn(prediction, target)


class IMFPlacementModel(nn.Module):
    """Minimal interval-velocity interface over the existing FlowPlace GNN."""

    backbones = FlowMatchingModel.backbones
    time_encoders = FlowMatchingModel.time_encoders

    def __init__(
            self,
            backbone,
            backbone_params,
            input_shape,
            t_encoding_type,
            t_encoding_dim,
            mask_key=None,
            use_mask_as_input=False,
            fixed_xy_key="fixed_xy",
            jvp_attention_implementation="default",
            imf_time_sampler="logit_normal",
            imf_time_mean=-0.4,
            imf_time_std=1.0,
            imf_zero_interval_ratio=0.5,
            imf_sampling_steps=1,
            prior_noise_dist="uniform",
            use_hard_constraint_in_sampling=False,
            device="cpu",
            **kwargs,
    ):
        super().__init__()
        if t_encoding_dim <= 0:
            raise ValueError("IMFPlacementModel requires a positive t_encoding_dim")

        backbone_params = copy.deepcopy(backbone_params)
        configured_attention = backbone_params.get("att_implementation", None)
        if configured_attention == "flash":
            backbone_params["att_implementation"] = jvp_attention_implementation
        with open_dict(backbone_params):
            backbone_params.update({
                "in_node_features": input_shape[1],
                "out_node_features": input_shape[1],
                "t_encoding_dim": t_encoding_dim,
                "device": device,
                "mask_key": mask_key if use_mask_as_input else None,
                "dropout": 0.0,
                "use_autocast": False,
            })

        self.t_encoder = self.time_encoders[t_encoding_type](t_encoding_dim)
        self.h_encoder = self.time_encoders[t_encoding_type](t_encoding_dim)
        self.interval_time_projection = nn.Linear(2 * t_encoding_dim, t_encoding_dim)
        self._reverse_model = self.backbones[backbone](**backbone_params)
        self.input_shape = input_shape
        self.mask_key = mask_key
        self.fixed_xy_key = fixed_xy_key
        self.t_encoding_dim = t_encoding_dim
        self.device = device
        self.configured_attention_implementation = configured_attention
        self.jvp_attention_implementation = backbone_params.get(
            "att_implementation", None
        )
        if imf_time_sampler not in ("debug", "logit_normal"):
            raise ValueError(f"unknown iMF time sampler: {imf_time_sampler}")
        if not 0.0 <= imf_zero_interval_ratio <= 1.0:
            raise ValueError("imf_zero_interval_ratio must be in [0,1]")
        if imf_sampling_steps not in (1, 2):
            raise ValueError("Phase 1 iMF sampling supports only one or two steps")
        if prior_noise_dist != "uniform":
            raise ValueError("Phase 1 iMF requires the uniform FlowPlace prior")
        if use_hard_constraint_in_sampling:
            raise ValueError("Phase 1 iMF sampling must not use hard constraints")
        self.imf_time_sampler = imf_time_sampler
        self.imf_time_mean = float(imf_time_mean)
        self.imf_time_std = float(imf_time_std)
        self.imf_zero_interval_ratio = float(imf_zero_interval_ratio)
        self.imf_sampling_steps = int(imf_sampling_steps)
        self.prior_noise_dist = prior_noise_dist
        self.use_hard_constraint_in_sampling = False
        self.is_raw_sampler = True
        self.last_forward_jvp_error = None
        self.last_sampling_nfe = 0

    @staticmethod
    def _check_time(name, value, batch_size, device):
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.shape != (batch_size,):
            raise ValueError(f"{name} must have shape ({batch_size},), got {tuple(value.shape)}")
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must be FP32, got {value.dtype}")
        if value.device != device:
            raise ValueError(f"{name} must be on {device}, got {value.device}")

    def _fixed_mask(self, z, condition):
        if not self.mask_key or self.mask_key not in condition:
            return None
        mask = condition[self.mask_key].bool()
        if mask.ndim == 1:
            mask = mask.view(1, -1, 1)
        elif mask.ndim == 2 and mask.shape[-1] == 1:
            mask = mask.view(1, mask.shape[0], 1)
        else:
            raise ValueError(
                f"condition[{self.mask_key!r}] must have shape (N,) or (N,1), "
                f"got {tuple(mask.shape)}"
            )
        if mask.shape[1] != z.shape[1]:
            raise ValueError("fixed mask node count does not match z")
        return mask.expand(z.shape[0], -1, z.shape[2])

    def _fixed_xy(self, z, condition, mask):
        if mask is None or not torch.any(mask):
            return None
        if self.fixed_xy_key not in condition:
            raise ValueError(
                f"condition must contain {self.fixed_xy_key!r} when fixed nodes are present"
            )
        fixed_xy = condition[self.fixed_xy_key]
        if fixed_xy.ndim == 2:
            fixed_xy = fixed_xy.unsqueeze(0)
        if fixed_xy.ndim != 3 or fixed_xy.shape[1:] != z.shape[1:]:
            raise ValueError(
                f"condition[{self.fixed_xy_key!r}] must have shape (N,2) or (B,N,2), "
                f"got {tuple(fixed_xy.shape)}"
            )
        if fixed_xy.shape[0] not in (1, z.shape[0]):
            raise ValueError("fixed_xy batch dimension must be one or match z")
        return fixed_xy.to(device=z.device, dtype=z.dtype).expand_as(z)

    def u_theta(self, z, r, t, condition):
        """Return u_theta(z, r, t, condition) with exact fixed-node invariants."""
        if z.ndim != 3 or z.shape[-1] != self.input_shape[1]:
            raise ValueError(
                f"z must have shape (B,N,{self.input_shape[1]}), got {tuple(z.shape)}"
            )
        if z.dtype != torch.float32:
            raise TypeError(f"z must be FP32, got {z.dtype}")
        self._check_time("r", r, z.shape[0], z.device)
        self._check_time("t", t, z.shape[0], z.device)
        h = t - r
        if torch.any(h < 0):
            raise ValueError("iMF requires r <= t for every batch element")

        mask = self._fixed_mask(z, condition)
        fixed_xy = self._fixed_xy(z, condition, mask)
        z_model = torch.where(mask, fixed_xy, z) if fixed_xy is not None else z

        t_embed = self.t_encoder(t)
        h_embed = self.h_encoder(h)
        interval_embed = self.interval_time_projection(
            torch.cat((t_embed, h_embed), dim=-1)
        )
        output = self._reverse_model(z_model, condition, interval_embed)
        return torch.zeros_like(output).where(mask, output) if mask is not None else output

    def forward(self, z, r, t, condition):
        return self.u_theta(z, r, t, condition)

    def _inference_condition(self, x_in, condition, source, mask):
        if mask is None or not torch.any(mask):
            return condition, None

        inference_condition = copy.copy(condition)
        if self.fixed_xy_key in condition:
            fixed_xy = self._fixed_xy(source, condition, mask)
        else:
            if x_in.ndim == 2:
                x_in = x_in.unsqueeze(0)
            if x_in.ndim != 3 or x_in.shape[1:] != source.shape[1:]:
                raise ValueError(
                    "x_in must have shape (N,2) or (B,N,2) for fixed nodes"
                )
            if x_in.shape[0] not in (1, source.shape[0]):
                raise ValueError("x_in batch dimension must be one or match B")
            fixed_xy = x_in.to(
                device=source.device, dtype=source.dtype
            ).expand_as(source)

        fixed_xy = torch.where(mask, fixed_xy, torch.zeros_like(fixed_xy))
        inference_condition[self.fixed_xy_key] = fixed_xy
        return inference_condition, fixed_xy

    @torch.no_grad()
    def reverse_samples(
            self,
            B,
            x_in,
            condition,
            num_timesteps=-1,
            intermediate_every=0,
            mask_override=None,
            method="euler",
            source=None,
            generator=None,
    ):
        """Generate an unclipped raw placement with the frozen backward update."""
        if B <= 0:
            raise ValueError("B must be positive")
        if method.lower() != "euler":
            raise ValueError("Phase 1 iMF supports only the interval Euler update")

        steps = self.imf_sampling_steps if num_timesteps <= 0 else num_timesteps
        if steps not in (1, 2):
            raise ValueError("Phase 1 iMF sampling supports only one or two steps")
        times = (
            (1.0, 0.0)
            if steps == 1
            else (1.0, 0.5, 0.0)
        )

        batch_shape = (B, condition.x.shape[0], self.input_shape[1])
        if source is None:
            z = 2.0 * torch.rand(
                batch_shape,
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            ) - 1.0
        else:
            if source.shape != batch_shape or source.dtype != torch.float32:
                raise ValueError("source must have shape (B,N,2) and use FP32")
            z = source.to(self.device).clone()

        mask = self._fixed_mask(z, condition)
        if mask_override is not None:
            override = mask_override.to(z.device).bool().view(1, z.shape[1], 1)
            override = override.expand_as(z)
            if mask is None or not torch.equal(override, mask):
                raise ValueError(
                    "Phase 1 iMF does not accept movable target coordinates "
                    "through mask_override"
                )

        inference_condition, fixed_xy = self._inference_condition(
            x_in, condition, z, mask
        )
        if fixed_xy is not None:
            z = torch.where(mask, fixed_xy, z)

        intermediates = [z.clone()]
        self.last_sampling_nfe = 0
        for step_index, (t_value, r_value) in enumerate(zip(times[:-1], times[1:])):
            t = torch.full(
                (B,), t_value, dtype=torch.float32, device=z.device
            )
            r = torch.full(
                (B,), r_value, dtype=torch.float32, device=z.device
            )
            interval_velocity = self.u_theta(z, r, t, inference_condition)
            self.last_sampling_nfe += 1
            z = z - (t - r).view(B, 1, 1) * interval_velocity
            if fixed_xy is not None:
                z = torch.where(mask, fixed_xy, z)
            if intermediate_every and (step_index + 1) % intermediate_every == 0:
                intermediates.append(z.clone())

        return z, intermediates

    def sample_time_pairs(self, batch_size, device, generator=None, sampler=None):
        """Sample ``0 <= r <= t <= 1`` for the minimal Phase 1 objective."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        sampler = self.imf_time_sampler if sampler is None else sampler
        if sampler == "debug":
            t = torch.full(
                (batch_size,), 0.75, dtype=torch.float32, device=device
            )
            r = torch.full(
                (batch_size,), 0.25, dtype=torch.float32, device=device
            )
        elif sampler == "logit_normal":
            gaussian = torch.randn(
                (batch_size, 2),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            samples = torch.sigmoid(
                self.imf_time_mean + self.imf_time_std * gaussian
            )
            t = samples.max(dim=1).values
            r = samples.min(dim=1).values
        else:
            raise ValueError(f"unknown iMF time sampler: {sampler}")

        zero_interval_count = int(self.imf_zero_interval_ratio * batch_size)
        if zero_interval_count:
            r = r.clone()
            r[:zero_interval_count] = t[:zero_interval_count]
        return r, t

    def _training_condition(self, x_target, condition, mask):
        if mask is None:
            return condition, None

        training_condition = copy.copy(condition)
        if self.fixed_xy_key in condition:
            fixed_xy = self._fixed_xy(x_target, condition, mask)
        else:
            fixed_xy = torch.where(mask, x_target, torch.zeros_like(x_target))

        if fixed_xy is None:
            fixed_xy = torch.zeros_like(x_target)
        sanitized_fixed_xy = torch.where(
            mask, fixed_xy, torch.zeros_like(fixed_xy)
        )
        training_condition[self.fixed_xy_key] = sanitized_fixed_xy
        return training_condition, fixed_xy

    def _validate_time_pairs(self, r, t, batch_size, device):
        self._check_time("r", r, batch_size, device)
        self._check_time("t", t, batch_size, device)
        if torch.any(r < 0) or torch.any(t > 1) or torch.any(r > t):
            raise ValueError("iMF time pairs must satisfy 0 <= r <= t <= 1")

    def training_objective(
            self,
            x_target,
            condition,
            *,
            time_sampler=None,
            generator=None,
            time_pairs=None,
            source=None,
    ):
        """Compute the frozen minimal boundary-condition iMF objective."""
        if x_target.ndim != 3 or x_target.shape[-1] != self.input_shape[1]:
            raise ValueError(
                "x_target must have shape "
                f"(B,N,{self.input_shape[1]}), got {tuple(x_target.shape)}"
            )
        if x_target.dtype != torch.float32:
            raise TypeError(f"x_target must be FP32, got {x_target.dtype}")

        batch_size = x_target.shape[0]
        mask = self._fixed_mask(x_target, condition)
        training_condition, fixed_xy = self._training_condition(
            x_target, condition, mask
        )
        if mask is None:
            x_clean = x_target
            movable = torch.ones_like(x_target, dtype=torch.bool)
        else:
            x_clean = torch.where(mask, fixed_xy, x_target)
            movable = ~mask

        movable_scalar_count = movable.sum(dim=(1, 2))
        if torch.any(movable_scalar_count == 0):
            raise ValueError("every layout must contain at least one movable coordinate")

        if source is None:
            epsilon = 2.0 * torch.rand(
                x_clean.shape,
                dtype=torch.float32,
                device=x_clean.device,
                generator=generator,
            ) - 1.0
        else:
            if source.shape != x_clean.shape or source.dtype != torch.float32:
                raise ValueError("source must match x_target shape and use FP32")
            epsilon = source.to(device=x_clean.device)
        if mask is not None:
            epsilon = torch.where(mask, fixed_xy, epsilon)

        if time_pairs is None:
            r, t = self.sample_time_pairs(
                batch_size,
                x_clean.device,
                generator=generator,
                sampler=time_sampler,
            )
        else:
            r, t = time_pairs
        self._validate_time_pairs(r, t, batch_size, x_clean.device)

        t_broadcast = t.view(batch_size, 1, 1)
        z_t = (1.0 - t_broadcast) * x_clean + t_broadcast * epsilon
        conditional_velocity = epsilon - x_clean
        if mask is not None:
            z_t = torch.where(mask, fixed_xy, z_t)
            conditional_velocity = torch.zeros_like(
                conditional_velocity
            ).where(mask, conditional_velocity)

        boundary_velocity = self.u_theta(
            z_t, t, t, training_condition
        )
        u_value, total_derivative, jvp_backend = self.jvp(
            z_t, r, t, training_condition
        )
        coefficient = (t - r).view(batch_size, 1, 1)
        compound_prediction = u_value + coefficient * total_derivative.detach()
        if mask is not None:
            compound_prediction = torch.zeros_like(
                compound_prediction
            ).where(mask, compound_prediction)

        squared_error = (compound_prediction - conditional_velocity).square()
        squared_error = squared_error * movable.to(squared_error.dtype)
        per_layout_loss = squared_error.sum(dim=(1, 2)) / movable_scalar_count
        loss = per_layout_loss.mean()

        details = {
            "source": epsilon.detach(),
            "r": r.detach(),
            "t": t.detach(),
            "x_clean": x_clean.detach(),
            "z_t": z_t.detach(),
            "conditional_velocity": conditional_velocity.detach(),
            "boundary_velocity": boundary_velocity.detach(),
            "u": u_value,
            "total_derivative": total_derivative.detach(),
            "coefficient": coefficient.detach(),
            "compound_prediction": compound_prediction,
            "movable_mask": movable.detach(),
            "movable_scalar_count": movable_scalar_count.detach(),
            "per_layout_loss": per_layout_loss,
            "condition_fixed_xy": (
                training_condition[self.fixed_xy_key].detach()
                if mask is not None else None
            ),
            "jvp_backend": jvp_backend,
        }
        return loss, details

    def loss(self, x, condition, _, idx=None, split="train"):
        loss, details = self.training_objective(x, condition)
        metrics = {
            "loss": loss.detach().item(),
            "velocity_norm": details["compound_prediction"].detach().norm().item(),
            "jvp_norm": details["total_derivative"].norm().item(),
            "zero_interval_fraction": (details["r"] == details["t"]).float().mean().item(),
        }
        return loss, metrics

    def jvp(self, z, r, t, condition):
        """Compute the frozen iMF JVP, preferring forward-mode AD.

        The returned primal is a separate ordinary forward so it retains normal
        parameter gradients. The returned JVP is stop-gradient by contract.
        """
        primal = self.u_theta(z, r, t, condition)
        boundary_velocity = self.u_theta(z, t, t, condition).detach()
        tangents = (
            boundary_velocity,
            torch.zeros_like(r),
            torch.ones_like(t),
        )

        def interval_fn(z_value, r_value, t_value):
            return self.u_theta(z_value, r_value, t_value, condition)

        self.last_forward_jvp_error = None
        try:
            ad_primal, jvp_value = torch.func.jvp(
                interval_fn, (z, r, t), tangents
            )
            backend = "torch.func.jvp"
        except (NotImplementedError, RuntimeError) as exc:
            self.last_forward_jvp_error = f"{type(exc).__name__}: {exc}"
            try:
                ad_primal, jvp_value = torch.autograd.functional.jvp(
                    interval_fn,
                    (z, r, t),
                    tangents,
                    create_graph=False,
                    strict=False,
                )
            except (NotImplementedError, RuntimeError) as fallback_exc:
                raise RuntimeError(
                    "Both JVP backends failed. "
                    f"torch.func.jvp: {self.last_forward_jvp_error}; "
                    "torch.autograd.functional.jvp: "
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                ) from fallback_exc
            backend = "torch.autograd.functional.jvp"

        if not torch.allclose(ad_primal, primal.detach(), rtol=1e-4, atol=1e-5):
            transformed_backend = backend
            transformed_max_abs = (
                ad_primal.detach() - primal.detach()
            ).abs().max().item()
            self.last_forward_jvp_error = (
                f"{transformed_backend} primal mismatch "
                f"(max_abs={transformed_max_abs:.6g})"
            )
            try:
                fallback_primal, fallback_jvp = torch.autograd.functional.jvp(
                    interval_fn,
                    (z, r, t),
                    tangents,
                    create_graph=False,
                    strict=False,
                )
            except (NotImplementedError, RuntimeError) as fallback_exc:
                raise RuntimeError(
                    f"{self.last_forward_jvp_error}; "
                    "torch.autograd.functional.jvp fallback failed: "
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                ) from fallback_exc

            if not torch.allclose(
                    fallback_primal, primal.detach(), rtol=1e-4, atol=1e-5
            ):
                fallback_max_abs = (
                    fallback_primal.detach() - primal.detach()
                ).abs().max().item()
                raise RuntimeError(
                    f"{self.last_forward_jvp_error}; "
                    "torch.autograd.functional.jvp primal also mismatches "
                    f"(max_abs={fallback_max_abs:.6g})"
                )
            ad_primal, jvp_value = fallback_primal, fallback_jvp
            backend = "torch.autograd.functional.jvp"

        mask = self._fixed_mask(z, condition)
        if mask is not None:
            jvp_value = torch.zeros_like(jvp_value).where(mask, jvp_value)
        return primal, jvp_value.detach(), backend


MODEL_TYPES = {
    "flow_matching": FlowMatchingModel,
    "imf": IMFPlacementModel,
}


def get_model_class(family):
    if family not in MODEL_TYPES:
        raise ValueError(f"unknown model family: {family}")
    return MODEL_TYPES[family]
