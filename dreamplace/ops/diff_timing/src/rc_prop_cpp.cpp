#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> rc_prop_forward(
    at::Tensor node_C,
    at::Tensor edge_R,
    at::Tensor tree_edge_src,
    at::Tensor tree_edge_dst,
    int64_t    num_pins,
    int64_t    max_depth)
{
    const int64_t N = node_C.size(0);

    auto src = tree_edge_src.to(at::kLong);
    auto dst = tree_edge_dst.to(at::kLong);

    // ── Loop 1: Load 
    auto load = node_C.clone();
    for (int64_t i = 0; i < max_depth; ++i) {
        auto msg_sum = torch::zeros_like(load)
                           .scatter_add_(0, src, load.index_select(0, dst));
        load = node_C + msg_sum;
    }

    // ── Loop 2: Delay (zeros -> clone으로 수정)
    auto delay = torch::zeros({N}, node_C.options());
    for (int64_t i = 0; i < max_depth; ++i) {
        auto msg = delay.index_select(0, src) + edge_R * load.index_select(0, dst);
        auto new_delay = delay.clone(); // <--- 여기 수정
        new_delay.scatter_(0, dst, msg);
        delay = new_delay;
    }

    // ── Loop 3: ldelay
    auto base_ldelay = node_C * delay;
    auto ldelay      = base_ldelay.clone();
    for (int64_t i = 0; i < max_depth; ++i) {
        auto msg_sum = torch::zeros_like(ldelay)
                           .scatter_add_(0, src, ldelay.index_select(0, dst));
        ldelay = base_ldelay + msg_sum;
    }

    // ── Loop 4: Beta (zeros -> clone으로 수정)
    auto beta = torch::zeros({N}, node_C.options());
    for (int64_t i = 0; i < max_depth; ++i) {
        auto msg = beta.index_select(0, src) + edge_R * ldelay.index_select(0, dst);
        auto new_beta = beta.clone(); // <--- 여기 수정
        new_beta.scatter_(0, dst, msg);
        beta = new_beta;
    }

    // ── Impulse
    auto impulse_sq = 2.0 * beta - delay.pow(2);
    auto impulse    = torch::sqrt(torch::clamp(impulse_sq, 1e-12));

    return {
        delay.narrow(0, 0, num_pins),
        impulse.narrow(0, 0, num_pins),
        load.narrow(0, 0, num_pins)
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rc_prop_forward, "Elmore delay RC tree propagation");
}