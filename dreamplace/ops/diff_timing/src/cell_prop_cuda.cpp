/**
 * @file   cell_prop_cuda.cpp
 * @brief  PyTorch C++ binding for the cell propagation CUDA kernels.
 */

#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA launchers defined in cell_prop_cuda_kernel.cu
void cell_prop_fwd_launcher(
    const int32_t* level_pins, const int32_t* src_matrix,
    const int32_t* arc_id_matrix, const bool* fanin_mask,
    const double* AT, const double* slew, const double* pin_load_caps,
    const double* lut_slew_axis, const double* lut_cap_axis,
    const double* lut_cell_rise, const double* lut_cell_fall,
    const double* lut_rise_tran, const double* lut_fall_tran,
    const int32_t* lut_sense,
    double* out_AT, double* out_slew,
    double* sv_wx_R, double* sv_wx_F, double* sv_wy,
    int32_t* sv_ix_R, int32_t* sv_ix_F, int32_t* sv_iy,
    bool* sv_relu, double* sv_lse_w,
    int N, int max_fanin, int S, int C, double gamma);

void cell_prop_bwd_launcher(
    const int32_t* level_pins, const int32_t* src_matrix,
    const int32_t* arc_id_matrix, const bool* fanin_mask,
    const double* g_out_AT, const double* g_out_slew,
    const double* lut_slew_axis, const double* lut_cap_axis,
    const double* lut_cell_rise, const double* lut_cell_fall,
    const double* lut_rise_tran, const double* lut_fall_tran,
    const int32_t* lut_sense,
    const double* sv_wx_R, const double* sv_wx_F, const double* sv_wy,
    const int32_t* sv_ix_R, const int32_t* sv_ix_F, const int32_t* sv_iy,
    const bool* sv_relu, const double* sv_lse_w,
    double* g_AT, double* g_slew, double* g_caps,
    int N, int max_fanin, int S, int C, double gamma);

// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Forward pass.
 *
 * Returns [out_AT, out_slew,
 *          sv_wx_R, sv_wx_F, sv_wy, sv_ix_R, sv_ix_F, sv_iy,
 *          sv_relu, sv_lse_w]
 *
 * out_AT   : [N, 2]  new arrival times for level_pins
 * out_slew : [N, 2]  new slews     for level_pins
 * sv_*     : saved intermediates for the backward pass
 */
std::vector<at::Tensor> cell_prop_forward(
    at::Tensor level_pins,       // [N]          int32
    at::Tensor src_matrix,       // [num_pins, max_fanin]  int32
    at::Tensor arc_id_matrix,    // [num_pins, max_fanin]  int32
    at::Tensor fanin_mask,       // [num_pins, max_fanin]  bool
    at::Tensor AT,               // [num_pins, 2]  float64
    at::Tensor slew,             // [num_pins, 2]  float64
    at::Tensor pin_load_caps,    // [num_pins]     float64
    at::Tensor lut_slew_axis,    // [num_luts, S]  float64
    at::Tensor lut_cap_axis,     // [num_luts, C]  float64
    at::Tensor lut_cell_rise,    // [num_luts, S, C]
    at::Tensor lut_cell_fall,
    at::Tensor lut_rise_tran,
    at::Tensor lut_fall_tran,
    at::Tensor lut_sense,        // [num_luts]  int32
    double gamma
) {
    TORCH_CHECK(AT.is_cuda(),          "AT must be a CUDA tensor");
    TORCH_CHECK(AT.scalar_type()    == at::kDouble, "float64 only");
    TORCH_CHECK(gamma > 0,             "gamma must be positive");

    int N         = level_pins.size(0);
    int max_fanin = src_matrix.size(1);
    int S         = lut_slew_axis.size(1);
    int C         = lut_cap_axis.size(1);
    int tc        = max_fanin * 2;

    TORCH_CHECK(max_fanin <= 32,
        "max_fanin exceeds compile-time MAX_FANIN=32; increase it in the .cu file");

    auto opts_f = AT.options();
    auto opts_i = src_matrix.options();                        // int32, CUDA
    auto opts_b = fanin_mask.options();                        // bool,  CUDA

    at::Tensor out_AT   = at::zeros({N, 2}, opts_f);
    at::Tensor out_slew = at::zeros({N, 2}, opts_f);

    // saved intermediates
    at::Tensor sv_wx_R = at::empty({N, max_fanin}, opts_f);
    at::Tensor sv_wx_F = at::empty({N, max_fanin}, opts_f);
    at::Tensor sv_wy   = at::empty({N, max_fanin}, opts_f);
    at::Tensor sv_ix_R = at::empty({N, max_fanin}, opts_i);
    at::Tensor sv_ix_F = at::empty({N, max_fanin}, opts_i);
    at::Tensor sv_iy   = at::empty({N, max_fanin}, opts_i);
    at::Tensor sv_relu = at::empty({N, max_fanin, 8}, opts_b);
    at::Tensor sv_lse_w= at::empty({N, 4, tc},    opts_f);

    cell_prop_fwd_launcher(
        level_pins.data_ptr<int32_t>(),
        src_matrix.data_ptr<int32_t>(),
        arc_id_matrix.data_ptr<int32_t>(),
        fanin_mask.data_ptr<bool>(),
        AT.data_ptr<double>(),
        slew.data_ptr<double>(),
        pin_load_caps.data_ptr<double>(),
        lut_slew_axis.data_ptr<double>(),
        lut_cap_axis.data_ptr<double>(),
        lut_cell_rise.data_ptr<double>(),
        lut_cell_fall.data_ptr<double>(),
        lut_rise_tran.data_ptr<double>(),
        lut_fall_tran.data_ptr<double>(),
        lut_sense.data_ptr<int32_t>(),
        out_AT.data_ptr<double>(),
        out_slew.data_ptr<double>(),
        sv_wx_R.data_ptr<double>(),
        sv_wx_F.data_ptr<double>(),
        sv_wy.data_ptr<double>(),
        sv_ix_R.data_ptr<int32_t>(),
        sv_ix_F.data_ptr<int32_t>(),
        sv_iy.data_ptr<int32_t>(),
        sv_relu.data_ptr<bool>(),
        sv_lse_w.data_ptr<double>(),
        N, max_fanin, S, C, gamma
    );

    return {out_AT, out_slew,
            sv_wx_R, sv_wx_F, sv_wy,
            sv_ix_R, sv_ix_F, sv_iy,
            sv_relu, sv_lse_w};
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Backward pass.
 *
 * Returns [grad_AT, grad_slew, grad_pin_load_caps]
 *   grad_AT            : [num_pins, 2]  gradient w.r.t. AT
 *   grad_slew          : [num_pins, 2]  gradient w.r.t. slew
 *   grad_pin_load_caps : [num_pins]     gradient w.r.t. pin_load_caps
 */
std::vector<at::Tensor> cell_prop_backward(
    at::Tensor level_pins,
    at::Tensor src_matrix,
    at::Tensor arc_id_matrix,
    at::Tensor fanin_mask,
    at::Tensor g_out_AT,          // [N, 2]   grad of loss w.r.t. out_AT
    at::Tensor g_out_slew,        // [N, 2]   grad of loss w.r.t. out_slew
    at::Tensor lut_slew_axis,
    at::Tensor lut_cap_axis,
    at::Tensor lut_cell_rise,
    at::Tensor lut_cell_fall,
    at::Tensor lut_rise_tran,
    at::Tensor lut_fall_tran,
    at::Tensor lut_sense,
    at::Tensor sv_wx_R, at::Tensor sv_wx_F, at::Tensor sv_wy,
    at::Tensor sv_ix_R, at::Tensor sv_ix_F, at::Tensor sv_iy,
    at::Tensor sv_relu, at::Tensor sv_lse_w,
    int num_pins,
    double gamma
) {
    int N         = level_pins.size(0);
    int max_fanin = src_matrix.size(1);
    int S         = lut_slew_axis.size(1);
    int C         = lut_cap_axis.size(1);

    auto opts_f = g_out_AT.options();

    // Initialize grad buffers to zero (backward uses atomicAdd for AT/slew)
    at::Tensor g_AT   = at::zeros({num_pins, 2}, opts_f);
    at::Tensor g_slew = at::zeros({num_pins, 2}, opts_f);
    at::Tensor g_caps = at::zeros({num_pins},    opts_f);

    cell_prop_bwd_launcher(
        level_pins.data_ptr<int32_t>(),
        src_matrix.data_ptr<int32_t>(),
        arc_id_matrix.data_ptr<int32_t>(),
        fanin_mask.data_ptr<bool>(),
        g_out_AT.data_ptr<double>(),
        g_out_slew.data_ptr<double>(),
        lut_slew_axis.data_ptr<double>(),
        lut_cap_axis.data_ptr<double>(),
        lut_cell_rise.data_ptr<double>(),
        lut_cell_fall.data_ptr<double>(),
        lut_rise_tran.data_ptr<double>(),
        lut_fall_tran.data_ptr<double>(),
        lut_sense.data_ptr<int32_t>(),
        sv_wx_R.data_ptr<double>(),
        sv_wx_F.data_ptr<double>(),
        sv_wy.data_ptr<double>(),
        sv_ix_R.data_ptr<int32_t>(),
        sv_ix_F.data_ptr<int32_t>(),
        sv_iy.data_ptr<int32_t>(),
        sv_relu.data_ptr<bool>(),
        sv_lse_w.data_ptr<double>(),
        g_AT.data_ptr<double>(),
        g_slew.data_ptr<double>(),
        g_caps.data_ptr<double>(),
        N, max_fanin, S, C, gamma
    );

    return {g_AT, g_slew, g_caps};
}

// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &cell_prop_forward,  "Cell propagation forward  (CUDA)");
    m.def("backward", &cell_prop_backward, "Cell propagation backward (CUDA)");
}
