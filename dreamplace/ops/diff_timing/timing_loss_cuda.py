"""
timing_loss_cuda.py
"""

import torch
import torch.nn.functional as F
import dreamplace.configure as configure

_cuda_available = False
if configure.compile_configurations.get("CUDA_FOUND") == "TRUE":
    try:
        import dreamplace.ops.diff_timing.timing_loss_cuda_ext as _timing_loss_cuda
        _cuda_available = True
    except ImportError:
        pass


class TimingLossFunction(torch.autograd.Function):
    """
    Custom autograd Function wrapping the fused CUDA forward / backward.
    """
    @staticmethod
    def forward(ctx, AT, endpoints, clock_period, gamma):
        results = _timing_loss_cuda.forward(
            AT.contiguous(), endpoints, float(clock_period), float(gamma))

        wns, tns, neg_slacks, global_max_t, total_exp_sum_t = results

        ctx.save_for_backward(neg_slacks, endpoints, global_max_t, total_exp_sum_t)
        ctx.num_pins = AT.shape[0]
        ctx.gamma    = float(gamma)

        return wns, tns

    @staticmethod
    def backward(ctx, g_wns, g_tns):
        neg_slacks, endpoints, global_max_t, total_exp_sum_t = ctx.saved_tensors

        g_AT = _timing_loss_cuda.backward(
            neg_slacks,
            endpoints,
            g_wns.item(),
            g_tns.item(),
            global_max_t.item(),
            total_exp_sum_t.item(),
            ctx.num_pins
        )
        return g_AT, None, None, None


# ─────────────────────────────────────────────────────────────────────────────

def _compute_slacks_per_ep(AT, endpoints, capture_ck_pins, has_capture_ck, clock_period, gamma):
    """Differentiable per-endpoint worst slack via softmin over rise/fall.
    softmin(a, b) = -γ · LSE(-[a, b] / γ)  →  smooth approximation of min(a, b)
    """
    endpoint_AT = AT[endpoints]                            # [E, 2]
    RAT = torch.full_like(endpoint_AT, clock_period)

    if has_capture_ck.any():
        valid_ck_pins = capture_ck_pins[has_capture_ck]
        clock_AT_rise = AT[valid_ck_pins, 0].unsqueeze(1)  # [V, 1]
        RAT[has_capture_ck] = clock_period + clock_AT_rise

    slacks = RAT - endpoint_AT                             # [E, 2]
    # softmin over rise/fall dim → [E]
    return -gamma * torch.logsumexp(-slacks / gamma, dim=1)


def _timing_loss_python_fallback(AT, endpoints, capture_ck_pins, has_capture_ck, clock_period, gamma):
    """Pure-PyTorch fallback"""
    slacks_per_ep = _compute_slacks_per_ep(
        AT, endpoints, capture_ck_pins, has_capture_ck, clock_period, gamma)  # [E]
    neg_slack_over_gamma = -slacks_per_ep / gamma
    wns_penalty = gamma * torch.logsumexp(neg_slack_over_gamma, dim=0)
    tns_penalty = gamma * F.softplus(neg_slack_over_gamma).sum()
    return wns_penalty, tns_penalty


def compute_timing_loss_cuda(AT, endpoints, capture_ck_pins, has_capture_ck, clock_period, gamma):
    """
    Compute WNS and TNS timing penalties with Clock Skew awareness.
    Rise/Fall 중 더 나쁜 slack을 softmin으로 미분가능하게 선택 후 WNS/TNS 계산.
    """
    if not _cuda_available or not AT.is_cuda:
        return _timing_loss_python_fallback(AT, endpoints, capture_ck_pins, has_capture_ck, clock_period, gamma)

    # Clock Skew 반영: AT_data[ep] -= AT_clock[ep] 를 미리 빼두면
    # PyTorch autograd가 clock 핀까지 gradient를 자동으로 흘려보냄
    if has_capture_ck.any():
        AT_mod = AT.clone()
        valid_ck_pins = capture_ck_pins[has_capture_ck]
        clock_AT_rise = AT[valid_ck_pins, 0].unsqueeze(1)   # [V, 1]
        endpoints_with_ck = endpoints[has_capture_ck]
        AT_mod[endpoints_with_ck] = AT_mod[endpoints_with_ck] - clock_AT_rise
    else:
        AT_mod = AT

    # Differentiable softmin over rise/fall per endpoint
    # clock_period - γ·LSE(AT_mod[ep]/γ, dim=1)  ==  softmin(slack_R, slack_F)  [E]
    endpoint_AT = AT_mod[endpoints]                          # [E, 2]
    slacks_per_ep = clock_period - gamma * torch.logsumexp(endpoint_AT / gamma, dim=1)  # [E]

    # WNS/TNS via LSE / softplus (pure Python — no CUDA kernel needed for this step)
    neg_slack_over_gamma = -slacks_per_ep / gamma
    wns_penalty = gamma * torch.logsumexp(neg_slack_over_gamma, dim=0)
    tns_penalty = gamma * F.softplus(neg_slack_over_gamma).sum()
    return wns_penalty, tns_penalty