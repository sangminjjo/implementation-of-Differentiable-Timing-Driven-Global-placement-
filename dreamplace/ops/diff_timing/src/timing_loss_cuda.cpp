/**
 * @file   timing_loss_cuda.cpp
 * @brief  PyTorch C++ binding for the fused WNS/TNS timing loss CUDA kernels.
 *
 * Forward returns:
 *   [wns_penalty, tns_penalty, neg_slacks, global_max, total_exp_sum]
 *
 * Backward returns:
 *   g_AT : [num_pins, 2]  gradient w.r.t. arrival times
 */

#include <torch/extension.h>
#include <vector>
#include <cmath>

// ── Forward declarations of CUDA launchers (defined in timing_loss_cuda_kernel.cu) ──

void timing_neg_slack_launcher(
    const double* AT, const int32_t* endpoints,
    double clock_period, double inv_gamma, int N,
    double* neg_slacks, double* block_max, int G);

void timing_reduce_launcher(
    const double* neg_slacks, double global_max, int N,
    double* block_exp_sum, double* block_tns_sum, int G);

void timing_bwd_launcher(
    const double* neg_slacks, const int32_t* endpoints,
    double g_wns, double g_tns,
    double global_max, double total_exp_sum,
    int N, double* g_AT, int G);

// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Forward pass.
 *
 * @param AT           [num_pins, 2]  float64 CUDA – arrival times (Rise=0, Fall=1)
 * @param endpoints    [E]            int32  CUDA – endpoint pin indices
 * @param clock_period double         timing constraint
 * @param gamma        double         logsumexp smoothing factor (> 0)
 *
 * @return {wns_penalty, tns_penalty, neg_slacks, global_max, total_exp_sum}
 *   wns_penalty   : scalar CUDA tensor
 *   tns_penalty   : scalar CUDA tensor
 *   neg_slacks    : [E*2] float64  (saved for backward)
 *   global_max    : scalar float64 (saved for backward)
 *   total_exp_sum : scalar float64 (saved for backward)
 */
std::vector<at::Tensor> timing_loss_forward(
    at::Tensor AT,
    at::Tensor endpoints,
    double clock_period,
    double gamma
) {
    TORCH_CHECK(AT.is_cuda(),                           "AT must be on CUDA");
    TORCH_CHECK(AT.scalar_type() == at::kDouble,        "AT must be float64");
    TORCH_CHECK(endpoints.scalar_type() == at::kInt,    "endpoints must be int32");
    TORCH_CHECK(gamma > 0.0,                            "gamma must be positive");

    int E = (int)endpoints.size(0);
    int N = E * 2;              // Rise + Fall per endpoint
    int G = (N + 255) / 256;    // number of CUDA blocks

    auto opts = AT.options();   // float64, CUDA

    // ── Pass 1: compute neg_slacks + partial block maxes ──────────────────
    at::Tensor neg_slacks = at::empty({N},  opts);
    at::Tensor block_max  = at::empty({G},  opts);

    if (N > 0) {
        timing_neg_slack_launcher(
            AT.data_ptr<double>(), endpoints.data_ptr<int32_t>(),
            clock_period, 1.0 / gamma, N,
            neg_slacks.data_ptr<double>(), block_max.data_ptr<double>(), G);
    }

    // ── Find global max (1 device-to-host transfer) ────────────────────────
    double global_max = (N > 0) ? block_max.max().item<double>() : 0.0;

    // ── Pass 2: compute partial exp_sum + softplus_sum ─────────────────────
    at::Tensor block_exp_sum = at::zeros({G}, opts);
    at::Tensor block_tns_sum = at::zeros({G}, opts);

    if (N > 0) {
        timing_reduce_launcher(
            neg_slacks.data_ptr<double>(), global_max, N,
            block_exp_sum.data_ptr<double>(), block_tns_sum.data_ptr<double>(), G);
    }

    double total_exp_sum = (N > 0) ? block_exp_sum.sum().item<double>() : 0.0;
    double total_tns_sum = (N > 0) ? block_tns_sum.sum().item<double>() : 0.0;

    // ── Compute final scalar penalties ─────────────────────────────────────
    double wns_val = gamma * (std::log(total_exp_sum + 1e-12) + global_max);
    double tns_val = gamma * total_tns_sum;

    // Return as CUDA scalar tensors so the Python autograd graph can track them
    at::Tensor wns_t  = at::full({}, wns_val,      opts);
    at::Tensor tns_t  = at::full({}, tns_val,      opts);
    at::Tensor gmax_t = at::full({}, global_max,   opts);
    at::Tensor esum_t = at::full({}, total_exp_sum, opts);

    return {wns_t, tns_t, neg_slacks, gmax_t, esum_t};
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Backward pass.
 *
 * @param neg_slacks    [E*2]  float64  (saved from forward)
 * @param endpoints     [E]    int32
 * @param g_wns         upstream gradient scalar for wns_penalty
 * @param g_tns         upstream gradient scalar for tns_penalty
 * @param global_max    max(neg_slacks) (saved from forward)
 * @param total_exp_sum sum(exp(neg_slacks - global_max)) (saved from forward)
 * @param num_pins      size of first dimension of AT
 *
 * @return g_AT : [num_pins, 2]
 */
at::Tensor timing_loss_backward(
    at::Tensor neg_slacks,
    at::Tensor endpoints,
    double g_wns,
    double g_tns,
    double global_max,
    double total_exp_sum,
    int    num_pins
) {
    int N = (int)neg_slacks.size(0);
    int G = (N + 255) / 256;
    auto opts = neg_slacks.options();

    at::Tensor g_AT = at::zeros({num_pins, 2}, opts);

    if (N > 0) {
        timing_bwd_launcher(
            neg_slacks.data_ptr<double>(), endpoints.data_ptr<int32_t>(),
            g_wns, g_tns, global_max, total_exp_sum,
            N, g_AT.data_ptr<double>(), G);
    }

    return g_AT;
}

// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &timing_loss_forward,  "Timing loss forward  (CUDA)");
    m.def("backward", &timing_loss_backward, "Timing loss backward (CUDA)");
}
