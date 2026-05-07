/**
 * @file   rc_prop_cuda.cpp
 * @brief  PyTorch C++ binding for RC propagation CUDA kernels.
 */

#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA launchers
void scatter_add_fwd(const double* in_buf, const int32_t* src, const int32_t* dst,
                     double* out_buf, int E);
void scatter_set_fwd(const double* in_buf, const double* R, const double* aux,
                     const int32_t* src, const int32_t* dst,
                     double* out_buf, int E);
void scatter_add_bwd(const double* grad_out, const int32_t* src, const int32_t* dst,
                     double* grad_in, int E);
void scatter_set_bwd(const double* grad_out, const double* R, const double* aux,
                     const int32_t* src, const int32_t* dst,
                     double* grad_in, double* grad_R, double* grad_aux, int E);

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Forward pass without snapshots.
 */
std::vector<at::Tensor> rc_prop_cuda_forward(
    at::Tensor node_C,
    at::Tensor edge_R,
    at::Tensor tree_src,
    at::Tensor tree_dst,
    int64_t    num_pins,
    int64_t    max_depth
) {
    TORCH_CHECK(node_C.is_cuda(), "node_C must be a CUDA tensor");
    TORCH_CHECK(node_C.scalar_type() == at::kDouble, "node_C must be float64");

    const int64_t N = node_C.size(0); //number of total nodes
    const int64_t E = edge_R.size(0); //number of edges
    auto src_i32 = tree_src.to(at::kInt);
    auto dst_i32 = tree_dst.to(at::kInt);
    auto opts = node_C.options();

    // Pass 1: Load (Children -> Parent)
    at::Tensor load = node_C.clone();
    at::Tensor load_next = at::empty({N}, opts);
    for (int64_t it = 0; it < max_depth; ++it) {
        load_next.copy_(node_C); //매 iteration마다 node_C로 초기화
        scatter_add_fwd(load.data_ptr<double>(), src_i32.data_ptr<int32_t>(),
                        dst_i32.data_ptr<int32_t>(), load_next.data_ptr<double>(), E);
        std::swap(load, load_next);
    }

    // Pass 2: Delay (Parent -> Children)
    at::Tensor delay = at::zeros({N}, opts);
    at::Tensor delay_next = at::zeros({N}, opts);
    for (int64_t it = 0; it < max_depth; ++it) {
        delay_next.zero_();
        scatter_set_fwd(delay.data_ptr<double>(), edge_R.data_ptr<double>(),
                        load.data_ptr<double>(), src_i32.data_ptr<int32_t>(),
                        dst_i32.data_ptr<int32_t>(), delay_next.data_ptr<double>(), E);
        std::swap(delay, delay_next);
    }

    // Pass 3: ldelay (Children -> Parent)
    at::Tensor base_ldelay = node_C * delay;
    at::Tensor ldelay = base_ldelay.clone();
    at::Tensor ldelay_next = at::empty({N}, opts);
    for (int64_t it = 0; it < max_depth; ++it) {
        ldelay_next.copy_(base_ldelay);
        scatter_add_fwd(ldelay.data_ptr<double>(), src_i32.data_ptr<int32_t>(),
                        dst_i32.data_ptr<int32_t>(), ldelay_next.data_ptr<double>(), E);
        std::swap(ldelay, ldelay_next);
    }

    // Pass 4: Beta (Parent -> Children)
    at::Tensor beta = at::zeros({N}, opts);
    at::Tensor beta_next = at::zeros({N}, opts);
    for (int64_t it = 0; it < max_depth; ++it) {
        beta_next.zero_();
        scatter_set_fwd(beta.data_ptr<double>(), edge_R.data_ptr<double>(),
                        ldelay.data_ptr<double>(), src_i32.data_ptr<int32_t>(),
                        dst_i32.data_ptr<int32_t>(), beta_next.data_ptr<double>(), E);
        std::swap(beta, beta_next);
    }

    at::Tensor impulse_sq = 2.0 * beta - delay.pow(2);

    return {
        delay.narrow(0, 0, num_pins),
        impulse_sq.narrow(0, 0, num_pins),
        load.narrow(0, 0, num_pins)
    };
}

/**
 * RC propagation backward pass.
 */
std::vector<at::Tensor> rc_prop_cuda_backward(
    at::Tensor grad_delay_out,    // [N]
    at::Tensor grad_impulse_sq,   // [N]
    at::Tensor grad_load_out,      // [N]
    at::Tensor load_snapshots,     // [max_depth+1, N]
    at::Tensor delay_snapshots,    // [max_depth+1, N]
    at::Tensor ldelay_snapshots,   // [max_depth+1, N]
    at::Tensor beta_snapshots,     // [max_depth+1, N]
    at::Tensor node_C,
    at::Tensor edge_R,
    at::Tensor tree_src,
    at::Tensor tree_dst,
    int64_t    max_depth
) {
    const int64_t N = node_C.size(0);
    const int64_t E = edge_R.size(0);
    auto src_i32 = tree_src.to(at::kInt);
    auto dst_i32 = tree_dst.to(at::kInt);
    const int32_t* sp = src_i32.data_ptr<int32_t>();
    const int32_t* dp = dst_i32.data_ptr<int32_t>();

    auto opts = node_C.options();
    at::Tensor grad_node_C = at::zeros({N}, opts);
    at::Tensor grad_edge_R = at::zeros({E}, opts);

    // Initial gradients from impulse_sq = 2.0 * beta - delay^2
    at::Tensor last_delay = delay_snapshots[max_depth];
    at::Tensor g_beta  = 2.0 * grad_impulse_sq;
    at::Tensor g_delay = grad_delay_out - 2.0 * last_delay * grad_impulse_sq;

    // Pass 4 Backward: Beta (scatter_set)
    at::Tensor g_ldelay = at::zeros({N}, opts);
    at::Tensor g_beta_it = g_beta.clone();
    at::Tensor final_ldelay = ldelay_snapshots[max_depth]; // Pass 4 forward used final ldelay as aux
    for (int64_t it = max_depth - 1; it >= 0; --it) {
        at::Tensor g_beta_prev = at::zeros({N}, opts);
        scatter_set_bwd(g_beta_it.data_ptr<double>(), edge_R.data_ptr<double>(),
                        final_ldelay.data_ptr<double>(), sp, dp,
                        g_beta_prev.data_ptr<double>(), grad_edge_R.data_ptr<double>(),
                        g_ldelay.data_ptr<double>(), static_cast<int>(E));
        g_beta_it = g_beta_prev;
    }

    // Pass 3 Backward: ldelay (scatter_add)
    at::Tensor g_base_ldelay = at::zeros({N}, opts);
    at::Tensor g_ldelay_it = g_ldelay.clone();
    for (int64_t it = max_depth - 1; it >= 0; --it) {
        g_base_ldelay.add_(g_ldelay_it);
        at::Tensor g_ldelay_prev = at::zeros({N}, opts);
        scatter_add_bwd(g_ldelay_it.data_ptr<double>(), sp, dp,
                        g_ldelay_prev.data_ptr<double>(), static_cast<int>(E));
        g_ldelay_it = g_ldelay_prev;
    }
    // base_ldelay = node_C * delay
    grad_node_C.add_(g_base_ldelay * last_delay);
    g_delay.add_(g_base_ldelay * node_C);

    // Pass 2 Backward: Delay (scatter_set)
    at::Tensor g_load = grad_load_out.clone();
    at::Tensor g_delay_it = g_delay.clone();
    at::Tensor final_load = load_snapshots[max_depth]; // Pass 2 forward used final load as aux
    for (int64_t it = max_depth - 1; it >= 0; --it) {
        at::Tensor g_delay_prev = at::zeros({N}, opts);
        scatter_set_bwd(g_delay_it.data_ptr<double>(), edge_R.data_ptr<double>(),
                        final_load.data_ptr<double>(), sp, dp,
                        g_delay_prev.data_ptr<double>(), grad_edge_R.data_ptr<double>(),
                        g_load.data_ptr<double>(), static_cast<int>(E));
        g_delay_it = g_delay_prev;
    }

    // Pass 1 Backward: Load (scatter_add)
    at::Tensor g_load_it = g_load.clone();
    for (int64_t it = max_depth - 1; it >= 0; --it) {
        grad_node_C.add_(g_load_it);
        at::Tensor g_load_prev = at::zeros({N}, opts);
        scatter_add_bwd(g_load_it.data_ptr<double>(), sp, dp,
                        g_load_prev.data_ptr<double>(), static_cast<int>(E));
        g_load_it = g_load_prev;
    }

    return {grad_node_C, grad_edge_R};
}

/**
 * Forward pass with snapshots.
 */
std::vector<at::Tensor> rc_prop_cuda_forward_with_snapshots(
    at::Tensor node_C,
    at::Tensor edge_R,
    at::Tensor tree_src,
    at::Tensor tree_dst,
    int64_t    num_pins,
    int64_t    max_depth
) {
    const int64_t N = node_C.size(0);
    const int64_t E = edge_R.size(0);
    auto src_i32 = tree_src.to(at::kInt);
    auto dst_i32 = tree_dst.to(at::kInt);
    auto opts = node_C.options();

    auto mk_snap = [&]() { return at::zeros({max_depth + 1, N}, opts); };
    at::Tensor load_snap   = mk_snap();
    at::Tensor delay_snap  = mk_snap();
    at::Tensor ldelay_snap = mk_snap();
    at::Tensor beta_snap   = mk_snap();

    // Pass 1: Load
    at::Tensor load = node_C.clone();
    load_snap[0].copy_(load);
    for (int64_t it = 0; it < max_depth; ++it) {
        at::Tensor load_next = node_C.clone();
        scatter_add_fwd(load.data_ptr<double>(), src_i32.data_ptr<int32_t>(),
                        dst_i32.data_ptr<int32_t>(), load_next.data_ptr<double>(), E);
        load = load_next;
        load_snap[it + 1].copy_(load);
    }

    // Pass 2: Delay
    at::Tensor delay = at::zeros({N}, opts);
    delay_snap[0].copy_(delay);
    for (int64_t it = 0; it < max_depth; ++it) {
        at::Tensor delay_next = at::zeros({N}, opts);
        scatter_set_fwd(delay.data_ptr<double>(), edge_R.data_ptr<double>(),
                        load.data_ptr<double>(), src_i32.data_ptr<int32_t>(),
                        dst_i32.data_ptr<int32_t>(), delay_next.data_ptr<double>(), E);
        delay = delay_next;
        delay_snap[it + 1].copy_(delay);
    }

    // Pass 3: ldelay
    at::Tensor base_ldelay = node_C * delay;
    at::Tensor ldelay = base_ldelay.clone();
    ldelay_snap[0].copy_(ldelay);
    for (int64_t it = 0; it < max_depth; ++it) {
        at::Tensor ldelay_next = base_ldelay.clone();
        scatter_add_fwd(ldelay.data_ptr<double>(), src_i32.data_ptr<int32_t>(),
                        dst_i32.data_ptr<int32_t>(), ldelay_next.data_ptr<double>(), E);
        ldelay = ldelay_next;
        ldelay_snap[it + 1].copy_(ldelay);
    }

    // Pass 4: Beta
    at::Tensor beta = at::zeros({N}, opts);
    beta_snap[0].copy_(beta);
    for (int64_t it = 0; it < max_depth; ++it) {
        at::Tensor beta_next = at::zeros({N}, opts);
        scatter_set_fwd(beta.data_ptr<double>(), edge_R.data_ptr<double>(),
                        ldelay.data_ptr<double>(), src_i32.data_ptr<int32_t>(),
                        dst_i32.data_ptr<int32_t>(), beta_next.data_ptr<double>(), E);
        beta = beta_next;
        beta_snap[it + 1].copy_(beta);
    }

    at::Tensor impulse_sq = 2.0 * beta - delay.pow(2);

    return {
        delay.narrow(0, 0, num_pins),
        impulse_sq.narrow(0, 0, num_pins),
        load.narrow(0, 0, num_pins),
        load_snap, delay_snap, ldelay_snap, beta_snap
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",               &rc_prop_cuda_forward);
    m.def("forward_with_snapshots",&rc_prop_cuda_forward_with_snapshots);
    m.def("backward",              &rc_prop_cuda_backward);
}
