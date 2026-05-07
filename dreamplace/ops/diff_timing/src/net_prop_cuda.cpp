/**
 * @file   net_prop_cuda.cpp
 * @brief  PyTorch C++ binding for net (wire AT/slew) propagation CUDA kernels.
 *
 * Exposes two entry points consumed by level_prop_cuda.py:
 *
 *   forward(level_pins, driver_pins, AT, slew, pin_delays, pin_impulses_sq)
 *     -> [new_AT_l, new_slew_l, sv_slew_u, sv_slew_v, sv_impulse_sq]
 *
 *   backward(level_pins, driver_pins, sv_slew_u, sv_slew_v, sv_impulse_sq,
 *            g_AT, g_slew, num_pins)
 *     -> [g_delay, g_impulse_sq]
 *     Note: g_AT and g_slew are also modified in-place.
 */

#include <torch/extension.h>
#include <vector>

// Forward declarations from net_prop_cuda_kernel.cu
void net_prop_fwd_launcher(
    const int32_t* level_pins, const int32_t* driver_pins,
    const double* AT_in, const double* slew_in,
    const double* pin_delays, const double* pin_impulses_sq,
    double* AT_out, double* slew_out,
    double* sv_slew_u, double* sv_slew_v, double* sv_impulse_sq,
    int N
);

void net_prop_bwd_launcher(
    const int32_t* level_pins, const int32_t* driver_pins,
    const double* sv_slew_u, const double* sv_slew_v, const double* sv_impulse_sq,
    int N,
    double* g_AT, double* g_slew,
    double* g_delay, double* g_impulse_sq
);

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Net propagation forward.
 *
 * level_pins      [N]           int32   sink pin indices at this level
 * driver_pins     [N]           int32   corresponding driver pin indices
 * AT              [num_pins, 2] float64 arrival times (Rise/Fall), full tensor
 * slew            [num_pins, 2] float64 slews (Rise/Fall, signed), full tensor
 * pin_delays      [num_pins]    float64 Elmore delay per pin (ps)
 * pin_impulses_sq [num_pins]    float64 impulse² per pin (ps²)
 *
 * Returns [new_AT_l, new_slew_l, sv_slew_u, sv_slew_v, sv_impulse_sq],
 * all shaped [N, 2] or [N].
 */
std::vector<at::Tensor> net_prop_forward(
    at::Tensor level_pins,
    at::Tensor driver_pins,
    at::Tensor AT,
    at::Tensor slew,
    at::Tensor pin_delays,
    at::Tensor pin_impulses_sq
) {
    TORCH_CHECK(AT.is_cuda(),                       "AT must be a CUDA tensor");
    TORCH_CHECK(AT.scalar_type() == at::kDouble,    "AT must be float64");
    TORCH_CHECK(AT.is_contiguous(),                 "AT must be contiguous");
    TORCH_CHECK(slew.is_contiguous(),               "slew must be contiguous");
    TORCH_CHECK(pin_delays.is_contiguous(),         "pin_delays must be contiguous");
    TORCH_CHECK(pin_impulses_sq.is_contiguous(),    "pin_impulses_sq must be contiguous");

    const int N = static_cast<int>(level_pins.size(0));
    auto opts = AT.options();

    at::Tensor new_AT_l      = at::empty({N, 2}, opts);
    at::Tensor new_slew_l    = at::empty({N, 2}, opts);
    at::Tensor sv_slew_u     = at::empty({N, 2}, opts);
    at::Tensor sv_slew_v     = at::empty({N, 2}, opts);
    at::Tensor sv_impulse_sq = at::empty({N},    opts);

    net_prop_fwd_launcher(
        level_pins.data_ptr<int32_t>(),
        driver_pins.data_ptr<int32_t>(),
        AT.data_ptr<double>(),
        slew.data_ptr<double>(),
        pin_delays.data_ptr<double>(),
        pin_impulses_sq.data_ptr<double>(),
        new_AT_l.data_ptr<double>(),
        new_slew_l.data_ptr<double>(),
        sv_slew_u.data_ptr<double>(),
        sv_slew_v.data_ptr<double>(),
        sv_impulse_sq.data_ptr<double>(),
        N
    );

    return {new_AT_l, new_slew_l, sv_slew_u, sv_slew_v, sv_impulse_sq};
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Net propagation backward.
 *
 * level_pins      [N]           int32
 * driver_pins     [N]           int32
 * sv_slew_u       [N, 2]        float64  saved driver slew from forward
 * sv_slew_v       [N, 2]        float64  saved sink slew from forward
 * sv_impulse_sq   [N]           float64  saved impulse² from forward
 * g_AT            [num_pins, 2] float64  grad w.r.t. AT (in-place modified)
 * g_slew          [num_pins, 2] float64  grad w.r.t. slew (in-place modified)
 * num_pins        int64
 *
 * Returns [g_delay, g_impulse_sq], both shaped [num_pins].
 * g_AT and g_slew are also updated in-place by the kernel.
 */
std::vector<at::Tensor> net_prop_backward(
    at::Tensor level_pins,
    at::Tensor driver_pins,
    at::Tensor sv_slew_u,
    at::Tensor sv_slew_v,
    at::Tensor sv_impulse_sq,
    at::Tensor g_AT,
    at::Tensor g_slew,
    int64_t    num_pins
) {
    const int N = static_cast<int>(level_pins.size(0));
    auto opts = g_AT.options();

    at::Tensor g_delay      = at::zeros({num_pins}, opts);
    at::Tensor g_impulse_sq = at::zeros({num_pins}, opts);

    net_prop_bwd_launcher(
        level_pins.data_ptr<int32_t>(),
        driver_pins.data_ptr<int32_t>(),
        sv_slew_u.data_ptr<double>(),
        sv_slew_v.data_ptr<double>(),
        sv_impulse_sq.data_ptr<double>(),
        N,
        g_AT.data_ptr<double>(),
        g_slew.data_ptr<double>(),
        g_delay.data_ptr<double>(),
        g_impulse_sq.data_ptr<double>()
    );

    return {g_delay, g_impulse_sq};
}

// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &net_prop_forward,  "Net AT/slew propagation forward");
    m.def("backward", &net_prop_backward, "Net AT/slew propagation backward");
}
