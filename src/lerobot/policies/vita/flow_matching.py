#!/usr/bin/env python

# Portions of this file are derived from VITA
# (https://github.com/ucd-dare/VITA, MIT License, Copyright (c) 2025 the VITA authors)
# and from torchcfm (https://github.com/atong01/conditional-flow-matching, MIT License).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flow matchers for the VITA policy.

These are self-contained ports of `flare/flow/*` from the VITA reference implementation, with the
`torchcfm` dependency inlined so that this module needs nothing beyond torch (plus scipy, lazily, for
the exact-OT coupling).

Every matcher exposes the same two calls::

    loss, metrics = matcher.compute_loss(model, target, start=..., **model_kwargs)
    x1 = matcher.sample(model, shape, device, num_steps=None, start=..., **model_kwargs)

`start` is what makes this VITA rather than ordinary flow matching: instead of Gaussian noise, the
caller passes the *visual latent* as the source of the probability path, so the flow runs
vision -> action and the velocity network needs no conditioning module at all. Passing `start=None`
falls back to standard noise-to-action flow matching, which is useful as an ablation.

Time conventions (they differ between families — do not mix them up):

* CFM family (`conditional`, `exact`): `x0` lives at `t=0`, `x1` at `t=1`,
  `x_t = (1 - t) * x0 + t * x1`, target velocity `u = x1 - x0`. Sampling integrates forward,
  `t: 0 -> 1`. The velocity net is called as `model(x, t, **kwargs)`.
* MeanFlow family (`mean`, `improved_mean`): reversed, as in the MeanFlow paper.
  `z_t = (1 - t) * x1 + t * x0`, so `t=1` is the source and `t=0` the target, and the instantaneous
  velocity is `v = x0 - x1`. Sampling integrates `t: 1 -> 0`. The net is called as
  `model(x=..., timestep=..., h=...)` and must return `(u, v, internal_features)`.
* `consistency` follows the CFM direction but samples with its own stochastic sampler.

The external API hides all of this: `start` is always the source and the returned sample is always
in target space.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def _pad_t_like_x(t: Tensor, x: Tensor) -> Tensor:
    """Reshape a per-sample scalar `(B,)` so it broadcasts against `x` of shape `(B, ...)`."""
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class BaseFlowMatcher:
    """Interface shared by every matcher. Stateless — holds hyper-parameters only, no parameters."""

    num_sampling_steps: int = 1

    def compute_loss(self, model, target: Tensor, start: Tensor | None = None, **kwargs):
        raise NotImplementedError

    def sample(
        self,
        model,
        shape: tuple[int, ...],
        device: torch.device | str,
        num_steps: int | None = None,
        return_traces: bool = False,
        start: Tensor | None = None,
        **kwargs,
    ):
        raise NotImplementedError


class ConditionalFlowMatcher(BaseFlowMatcher):
    """Plain conditional flow matching (Lipman et al.), the `torchcfm` `ConditionalFlowMatcher`.

    With `sigma=0` — the VITA default — the probability path is the straight line
    `x_t = (1 - t) * x0 + t * x1` and the regression target is the constant velocity `x1 - x0`.

    Unlike `ExactOptimalTransportConditionalFlowMatcher` this preserves the pairing between each
    `start[i]` and its own `target[i]`, which is what you want if you are unsure whether the OT
    re-coupling is helping or hurting. See that class's docstring.
    """

    def __init__(self, sigma: float = 0.0, num_sampling_steps: int = 6):
        self.sigma = sigma
        self.num_sampling_steps = num_sampling_steps

    def _sample_t(self, x: Tensor) -> Tensor:
        return torch.rand(x.shape[0], device=x.device, dtype=x.dtype)

    def sample_location_and_conditional_flow(self, x0: Tensor, x1: Tensor):
        """Return `(t, x_t, u_t)` for a randomly drawn time per sample."""
        t = self._sample_t(x0)
        t_pad = _pad_t_like_x(t, x0)
        mu_t = (1.0 - t_pad) * x0 + t_pad * x1
        if self.sigma > 0:
            xt = mu_t + self.sigma * torch.randn_like(mu_t)
        else:
            xt = mu_t
        ut = x1 - x0
        return t, xt, ut

    def couple(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        """Hook for subclasses that re-pair the minibatch. Identity here."""
        return x0, x1

    def compute_loss(self, model, target: Tensor, start: Tensor | None = None, **kwargs):
        x1 = target
        x0 = torch.randn_like(target) if start is None else start
        x0, x1 = self.couple(x0, x1)
        t, xt, ut = self.sample_location_and_conditional_flow(x0, x1)
        vt = model(xt, t, **kwargs)
        loss = F.mse_loss(vt, ut)
        return loss, {"flow_loss": loss.item()}

    def sample(
        self,
        model,
        shape: tuple[int, ...],
        device: torch.device | str,
        num_steps: int | None = None,
        return_traces: bool = False,
        start: Tensor | None = None,
        **kwargs,
    ):
        """Forward Euler integration of the learned velocity field from `t=0` to `t=1`."""
        if num_steps is None:
            num_steps = self.num_sampling_steps
        x = torch.randn(shape, device=device) if start is None else start
        dt = 1.0 / num_steps

        traj_history = [x.detach().clone().cpu()] if return_traces else None
        vel_history = [torch.zeros_like(x).detach().cpu()] if return_traces else None

        for step in range(num_steps):
            t = torch.full((x.shape[0],), step / num_steps, device=x.device, dtype=x.dtype)
            vt = model(x, t, **kwargs)
            x = x + vt * dt
            if return_traces:
                traj_history.append(x.detach().clone().cpu())
                vel_history.append(vt.detach().clone().cpu())

        if return_traces:
            return x, (traj_history, vel_history)
        return x


class ExactOptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    """OT-CFM: re-pair `(x0, x1)` within the minibatch along an exact optimal-transport plan.

    This is the VITA default (`flow_matcher.name: exact`). Understand what it does before trusting a
    training curve from it:

    **It breaks the correspondence between an observation and its own action.** The coupling is
    computed over the whole minibatch, so the flow-matching term generally trains
    `z_img[i] -> z_act[perm[i]]` for some permutation, not `z_img[i] -> z_act[i]`. For ordinary
    noise-to-data flow matching that is harmless and provably straightens paths, because the source
    is exchangeable noise. Here the source carries the observation, so the FM term alone no longer
    pins down which action belongs to which image.

    What restores per-sample correctness in VITA is flow latent decoding (`decode_flow_latents=True`
    with `consistency_weight`/`flow_recon_weight` > 0): those losses sample with
    `start=obs_latents[i]` and compare the decoded result against `gt_actions[i]`, index for index.
    If you turn FLD off, use `flow_matcher_type="conditional"` instead, or you are training an
    unpaired map.

    Ablate `exact` against `conditional` early on your own data.

    The reference uses POT's `ot.emd` and then samples index pairs from the resulting plan. For the
    uniform, equal-size marginals used here the EMD optimum is attained at a permutation matrix, so
    solving the linear assignment problem with scipy gives an equivalent coupling without the extra
    dependency.

    The solve runs on CPU, so on GPU each training step pays one device synchronisation plus an
    O(batch^3) assignment. Negligible at batch 128 against a ResNet forward, but it is not free.
    """

    def __init__(self, sigma: float = 0.0, num_sampling_steps: int = 6):
        super().__init__(sigma=sigma, num_sampling_steps=num_sampling_steps)

    def couple(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise ImportError(
                "The 'exact' (optimal transport) flow matcher needs scipy. Install it with "
                "`pip install 'lerobot[scipy-dep]'`, or set `flow_matcher_type='conditional'` to "
                "disable minibatch OT coupling."
            ) from exc

        if x0.shape[0] < 2:
            return x0, x1

        with torch.no_grad():
            cost = torch.cdist(x0.flatten(1).float(), x1.flatten(1).float()) ** 2
            _, col_ind = linear_sum_assignment(cost.cpu().numpy())
            perm = torch.as_tensor(col_ind, device=x1.device, dtype=torch.long)
        return x0, x1[perm]


def _stopgrad(x: Tensor) -> Tensor:
    return x.detach()


def _adaptive_l2_loss(error: Tensor, gamma: float = 0.5, c: float = 1e-3) -> Tensor:
    """Adaptive L2 loss from the MeanFlow paper: down-weight samples with large residuals."""
    delta_sq = torch.mean(error**2, dim=tuple(range(1, error.ndim)))
    w = 1.0 / (delta_sq + c).pow(1.0 - gamma)
    return (_stopgrad(w) * delta_sq).mean()


def _adaptive_imf_loss(error: Tensor, norm_p: float = 1.0, norm_eps: float = 0.01) -> Tensor:
    """Adaptive loss from Improved MeanFlow."""
    per_sample_loss = torch.sum(error**2, dim=tuple(range(1, error.ndim)))
    adaptive_weight = (per_sample_loss + norm_eps).pow(norm_p)
    return (per_sample_loss / _stopgrad(adaptive_weight)).mean()


def _dispersive_loss(z: Tensor, tau: float = 1.0) -> Tensor:
    """Dispersive loss (MP1): push apart hidden representations to avoid latent collapse."""
    if z.shape[0] <= 1:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    dist_matrix = torch.cdist(z, z, p=2) ** 2
    dist_matrix = dist_matrix / (torch.max(dist_matrix).detach() + 1e-8)
    return torch.log(torch.mean(torch.exp(-dist_matrix / tau)))


class MeanFlowMatcher(BaseFlowMatcher):
    """MeanFlow: regress the *average* velocity over `[r, t]` so a single forward pass generates.

    Reference: Geng et al., "Mean flows for one-step generative modeling" (arXiv 2505.13447), with
    the dispersive loss of MP1 (arXiv 2507.10543). `use_imf=True` switches to Improved MeanFlow
    (arXiv 2512.02012), which additionally regresses the instantaneous velocity as an auxiliary head
    and corrects the bootstrap target with it.

    The training identity needs a Jacobian-vector product, so `model` is differentiated through
    `torch.autograd.functional.jvp`. Two practical consequences:

    * the flow network must return `(u, v, internal_features)`, i.e. `SimpleMeanFlowNet`, not
      `SimpleFlowNet`;
    * `jvp` is numerically fragile in bf16/fp16 and does not compose with fused attention kernels —
      keep the velocity network in fp32.

    Note the reversed time convention documented at the top of this module: `t=1` is the source
    (visual latent), `t=0` the target (action latent).
    """

    def __init__(
        self,
        num_sampling_steps: int = 1,
        flow_ratio: float = 0.5,
        time_dist_mu: float = -0.4,
        time_dist_sigma: float = 1.0,
        adaptive_loss_gamma: float = 0.5,
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
        aux_v_loss_weight: float = 1.0,
        dispersive_loss_tau: float = 1.0,
        dispersive_loss_weight: float = 0.0,
        use_imf: bool = False,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.flow_ratio = flow_ratio
        self.time_dist_mu = time_dist_mu
        self.time_dist_sigma = time_dist_sigma
        self.adaptive_loss_gamma = adaptive_loss_gamma
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.aux_v_loss_weight = aux_v_loss_weight
        self.dispersive_loss_tau = dispersive_loss_tau
        self.dispersive_loss_weight = dispersive_loss_weight
        self.use_imf = use_imf

    @staticmethod
    def _unpack_output(output, require_aux_v: bool = False):
        if not isinstance(output, tuple):
            return output, None, None
        if len(output) == 2:
            prediction, internal_features = output
            if require_aux_v:
                raise ValueError("Improved MeanFlow requires the model to return (u, v, internal_features).")
            return prediction, None, internal_features
        if len(output) == 3:
            return output
        raise ValueError(f"Unexpected MeanFlow model output with {len(output)} values.")

    def sample_t_r(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Draw `(t, r)` with `t >= r` from a logit-normal distribution, tying `r = t` on a subset.

        The tied subset (`flow_ratio` of the batch) reduces to ordinary flow matching, which anchors
        the average-velocity field to the instantaneous one.
        """
        normal_samples = torch.randn(batch_size, 2, device=device) * self.time_dist_sigma + self.time_dist_mu
        samples = torch.sigmoid(normal_samples)
        t = torch.max(samples, dim=1)[0]
        r = torch.min(samples, dim=1)[0]
        num_selected = int(self.flow_ratio * batch_size)
        indices = torch.randperm(batch_size, device=device)[:num_selected]
        r[indices] = t[indices]
        return t, r

    def compute_loss(self, model, target: Tensor, start: Tensor | None = None, **kwargs):
        if start is None:
            raise ValueError(
                "MeanFlowMatcher requires `start` (the visual latent) — it has no noise fallback."
            )

        x1, x0 = target, start
        batch_size, device = x0.shape[0], x0.device

        t, r = self.sample_t_r(batch_size, device)
        t_pad = _pad_t_like_x(t, x0)
        h = t - r
        h_pad = _pad_t_like_x(h, x0)

        # Reversed convention: t=1 -> source, t=0 -> target.
        z_t = (1 - t_pad) * x1 + t_pad * x0
        v = x0 - x1  # ground-truth instantaneous velocity

        def pred_meanflow(z_in, t_in, r_in):
            return model(x=z_in, timestep=t_in, h=t_in - r_in, **kwargs)

        if self.use_imf:
            with torch.no_grad():
                _, v_net, _ = self._unpack_output(pred_meanflow(z_t, t, t), require_aux_v=True)
            dz_tangent = v_net

            def pred_imf(z_in, t_in, r_in):
                u, v_pred, _ = self._unpack_output(pred_meanflow(z_in, t_in, r_in), require_aux_v=True)
                return u, v_pred

            (predicted_mean_vel, predicted_v), (dudt, _) = torch.autograd.functional.jvp(
                pred_imf,
                (z_t, t, r),
                (dz_tangent, torch.ones_like(t), torch.zeros_like(r)),
                create_graph=True,
            )

            compound_velocity = predicted_mean_vel + h_pad * _stopgrad(dudt)
            imf_loss = _adaptive_imf_loss(
                compound_velocity - _stopgrad(v), norm_p=self.norm_p, norm_eps=self.norm_eps
            )
            aux_v_loss = _adaptive_imf_loss(
                predicted_v - _stopgrad(v), norm_p=self.norm_p, norm_eps=self.norm_eps
            )
            loss = imf_loss + self.aux_v_loss_weight * aux_v_loss
            metrics = {
                "imf_loss": imf_loss.item(),
                "aux_v_loss": aux_v_loss.item(),
                "imf_mse": torch.mean((compound_velocity - v) ** 2).item(),
                "aux_v_mse": torch.mean((predicted_v - v) ** 2).item(),
            }
        else:
            def pred_meanflow_u(z_in, t_in, r_in):
                u, _, _ = self._unpack_output(pred_meanflow(z_in, t_in, r_in))
                return u

            predicted_mean_vel, dudt = torch.autograd.functional.jvp(
                pred_meanflow_u,
                (z_t, t, r),
                (v, torch.ones_like(t), torch.zeros_like(r)),
                create_graph=True,
            )

            # MeanFlow identity: u(z_t, r, t) = v - (t - r) * du/dt
            u_tgt = v - h_pad * dudt
            meanflow_loss = _adaptive_l2_loss(
                predicted_mean_vel - _stopgrad(u_tgt), gamma=self.adaptive_loss_gamma
            )
            loss = meanflow_loss
            metrics = {"meanflow_loss": meanflow_loss.item()}

        if self.dispersive_loss_weight > 0:
            # Only worth the extra forward pass when the loss is actually enabled.
            _, _, internal_features = self._unpack_output(
                pred_meanflow(z_t, t, r), require_aux_v=self.use_imf
            )
            if internal_features is not None:
                dis_loss_total = sum(
                    _dispersive_loss(features, tau=self.dispersive_loss_tau)
                    for features in internal_features
                )
                metrics["dispersive_loss"] = dis_loss_total.item()
                loss = loss + self.dispersive_loss_weight * dis_loss_total

        metrics["flow_loss"] = loss.item()
        return loss, metrics

    def sample(
        self,
        model,
        shape: tuple[int, ...],
        device: torch.device | str,
        num_steps: int | None = None,
        return_traces: bool = False,
        start: Tensor | None = None,
        **kwargs,
    ):
        """Integrate `t: 1 -> 0`. With `num_steps=1` this is a single network evaluation (1-NFE)."""
        if start is None:
            raise ValueError(
                "MeanFlowMatcher requires `start` (the visual latent) — it has no noise fallback."
            )
        if num_steps is None:
            num_steps = self.num_sampling_steps

        x = start
        batch_size = x.shape[0]
        traj_history = [x.detach().clone().cpu()] if return_traces else None
        vel_history = [torch.zeros_like(x).detach().cpu()] if return_traces else None

        for step in range(num_steps):
            t_scalar = 1.0 - step / num_steps
            r_scalar = 1.0 - (step + 1) / num_steps
            t = torch.full((batch_size,), t_scalar, device=device, dtype=x.dtype)
            h = torch.full((batch_size,), t_scalar - r_scalar, device=device, dtype=x.dtype)
            mean_velocity, _, _ = self._unpack_output(model(x=x, timestep=t, h=h, **kwargs))
            x = x - _pad_t_like_x(h, x) * mean_velocity
            if return_traces:
                traj_history.append(x.detach().clone().cpu())
                vel_history.append(mean_velocity.detach().clone().cpu())

        if return_traces:
            return x, (traj_history, vel_history)
        return x


class ConsistencyFlowMatcher(BaseFlowMatcher):
    """Consistency flow matching (FlowPolicy-style): piecewise-straight paths for few-step sampling.

    The path is split into `num_segments`; the loss forces the Euler extrapolation from `t` and from
    `t + delta` to land on the same point at the end of the enclosing segment.

    Ported with one deliberate fix: the reference implementation hardcodes rank-3 tensors
    (`t.view(-1, 1, 1).repeat(1, T, D)`), which crashes on VITA's rank-2 `(B, latent_dim)` latents.
    Broadcasting here is shape-generic, so it works for both latent vectors and raw action chunks.
    """

    def __init__(
        self,
        num_sampling_steps: int = 1,
        eps: float = 1e-2,
        num_segments: int = 2,
        boundary: float = 1.0,
        delta: float = 1e-3,
        alpha: float = 1e-5,
        noise_scale: float = 1.0,
        sigma_var: float = 1.0,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.eps = eps
        self.num_segments = num_segments
        self.boundary = boundary
        self.delta = delta
        self.alpha = alpha
        self.noise_scale = noise_scale
        self.sigma_var = sigma_var

    def _sigma_t(self, t: float) -> float:
        return (1.0 - t) * self.sigma_var

    @staticmethod
    def _f_euler(t_pad, segment_ends_pad, xt, vt):
        return xt + (segment_ends_pad - t_pad) * vt

    def _threshold_based_f_euler(self, t_pad, segment_ends_pad, xt, vt, threshold, x_at_segment_ends):
        if threshold == 0:
            return x_at_segment_ends
        less_than_threshold = t_pad < threshold
        return less_than_threshold * self._f_euler(t_pad, segment_ends_pad, xt, vt) + (
            ~less_than_threshold
        ) * x_at_segment_ends

    def _masked_losses_v(self, vt, vr, threshold, segment_ends, t, batch_size):
        if threshold == 0:
            return torch.zeros(batch_size, device=vt.device, dtype=vt.dtype)
        t_pad = _pad_t_like_x(t, vt)
        less_than_threshold = t_pad < threshold
        far_from_segment_ends = _pad_t_like_x((segment_ends - t) > 1.01 * self.delta, vt)
        losses_v = torch.square(vt - vr) * less_than_threshold * far_from_segment_ends
        return torch.mean(losses_v.reshape(batch_size, -1), dim=-1)

    def compute_loss(self, model, target: Tensor, start: Tensor | None = None, **kwargs):
        batch_size, device = target.shape[0], target.device
        a0 = torch.randn_like(target) if start is None else start

        t = torch.rand(batch_size, device=device, dtype=target.dtype) * (1 - self.eps) + self.eps
        r = torch.clamp(t + self.delta, max=1.0)

        t_pad, r_pad = _pad_t_like_x(t, target), _pad_t_like_x(r, target)
        xt = t_pad * target + (1 - t_pad) * a0
        xr = r_pad * target + (1 - r_pad) * a0

        segments = torch.linspace(0, 1, self.num_segments + 1, device=device, dtype=target.dtype)
        seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
        segment_ends = segments[seg_indices]
        segment_ends_pad = _pad_t_like_x(segment_ends, target)
        x_at_segment_ends = segment_ends_pad * target + (1 - segment_ends_pad) * a0

        vt = model(xt, t, **kwargs)
        vr = torch.nan_to_num(model(xr, r, **kwargs))

        ft = self._f_euler(t_pad, segment_ends_pad, xt, vt)
        fr = self._threshold_based_f_euler(
            r_pad, segment_ends_pad, xr, vr, self.boundary, x_at_segment_ends
        )

        losses_f = torch.mean(torch.square(ft - fr).reshape(batch_size, -1), dim=-1)
        losses_v = self._masked_losses_v(vt, vr, self.boundary, segment_ends, t, batch_size)

        loss = torch.mean(losses_f + self.alpha * losses_v)
        return loss, {
            "flow_loss": loss.item(),
            "consistency_f_loss": torch.mean(losses_f).item(),
            "consistency_v_loss": torch.mean(losses_v).item(),
        }

    def sample(
        self,
        model,
        shape: tuple[int, ...],
        device: torch.device | str,
        num_steps: int | None = None,
        return_traces: bool = False,
        start: Tensor | None = None,
        **kwargs,
    ):
        if num_steps is None:
            num_steps = self.num_sampling_steps
        z = (torch.randn(shape, device=device) if start is None else start).detach().clone()
        dt = 1.0 / num_steps

        traj_history = [z.detach().clone().cpu()] if return_traces else None
        vel_history = [torch.zeros_like(z).detach().cpu()] if return_traces else None

        for i in range(num_steps):
            num_t = i / num_steps * (1 - self.eps) + self.eps
            t = torch.full((z.shape[0],), num_t, device=device, dtype=z.dtype)
            vt = model(z, t, **kwargs)
            sigma_t = self._sigma_t(num_t)
            if sigma_t > 0:
                pred_sigma = vt + (sigma_t**2) / (2 * (self.noise_scale**2) * ((1 - num_t) ** 2)) * (
                    0.5 * num_t * (1 - num_t) * vt - 0.5 * (2 - num_t) * z.detach().clone()
                )
                z = (
                    z.detach().clone()
                    + pred_sigma * dt
                    + sigma_t * math.sqrt(dt) * torch.randn_like(pred_sigma)
                )
            else:
                z = z.detach().clone() + vt * dt
            if return_traces:
                traj_history.append(z.detach().clone().cpu())
                vel_history.append(vt.detach().clone().cpu())

        if return_traces:
            return z, (traj_history, vel_history)
        return z


#: Matchers that need a `SimpleMeanFlowNet`-style network returning `(u, v, internal_features)`.
MEAN_FLOW_MATCHERS = ("mean", "improved_mean")

FLOW_MATCHER_CLASSES = {
    "conditional": ConditionalFlowMatcher,
    "exact": ExactOptimalTransportConditionalFlowMatcher,
    "mean": MeanFlowMatcher,
    "improved_mean": MeanFlowMatcher,
    "consistency": ConsistencyFlowMatcher,
}


def make_flow_matcher(name: str = "exact", **kwargs) -> BaseFlowMatcher:
    """Build a flow matcher by name. Unknown keys for the chosen matcher are dropped."""
    try:
        cls = FLOW_MATCHER_CLASSES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(FLOW_MATCHER_CLASSES))
        raise ValueError(f"Invalid flow matcher name: {name}. Expected one of: {valid}") from exc

    if name == "improved_mean":
        kwargs["use_imf"] = True
    elif name == "mean":
        kwargs.setdefault("use_imf", False)

    # Each matcher takes a different subset of hyper-parameters; filter rather than explode.
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})
