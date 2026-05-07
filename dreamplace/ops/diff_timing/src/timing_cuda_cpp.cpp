#include <torch/extension.h>
#include <vector>

// ── CUDA launcher declarations ────────────────────────────────────────────────

void cell_prop_fwd_launcher(
    const int32_t *level_pins, const int32_t *src_matrix,
    const int32_t *arc_id_matrix, const bool *fanin_mask,
    const double *AT, const double *slew, const double *pin_load_caps,
    const double *lut_slew_axis, const double *lut_cap_axis,
    const double *lut_cell_rise, const double *lut_cell_fall,
    const double *lut_rise_tran, const double *lut_fall_tran,
    const int32_t *lut_sense,
    double *out_AT, double *out_slew,
    double *sv_wx_R, double *sv_wx_F, double *sv_wy,
    int32_t *sv_ix_R, int32_t *sv_ix_F, int32_t *sv_iy,
    bool *sv_relu, double *sv_lse_w,
    int N, int max_fanin, int S, int C, double gamma);

void net_prop_launcher(
    const int32_t *pins, const int32_t *driver_map,
    double *AT, double *slew,
    const double *delays, const double *impulses_sq, int N);

// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Forward pass over all timing levels (net + cell propagation).
 *
 * Level convention (matches Python):
 *   level 0        : PI/FF-Q  (no propagation needed, skip)
 *   odd  levels    : net propagation
 *   even levels≥2  : cell propagation
 *
 * For cell propagation, cell_prop_fwd_launcher writes a dense [N, 2] result
 * indexed by thread position, NOT by pin id.  We therefore:
 *   1. Allocate temporary output buffers  (out_AT_tmp, out_slew_tmp)
 *   2. Allocate the sv_* buffers required by the kernel (discarded afterwards
 *      because this C++ loop is not wrapped in an autograd Function – gradients
 *      flow through the PyTorch ops in Python instead)
 *   3. Scatter the results back via index_put_
 *
 * @param flat_level_pins  [total_pins]       int32  all level pins concatenated
 * @param level_ptr        [num_levels+1]     int32  start offset per level
 * @param cell_src_matrix  [num_pins, F]      int32
 * @param cell_arc_id_matrix [num_pins, F]    int32
 * @param cell_fanin_mask  [num_pins, F]      bool
 * @param net_driver_map   [num_pins]         int32
 * @param AT               [num_pins, 2]      float64  (read + written in-place)
 * @param slew             [num_pins, 2]      float64  (read + written in-place)
 * @param pin_delays       [num_pins]         float64
 * @param pin_impulses_sq     [num_pins]         float64
 * @param pin_capacitance  [num_pins]         float64
 * @param lut_*            LUT tables
 * @param gamma            double
 *
 * @return {AT, slew}
 */
std::vector<at::Tensor> forward_all_levels(
    at::Tensor flat_level_pins,
    at::Tensor level_ptr,
    at::Tensor cell_src_matrix,
    at::Tensor cell_arc_id_matrix,
    at::Tensor cell_fanin_mask,
    at::Tensor net_driver_map,
    at::Tensor AT,
    at::Tensor slew,
    at::Tensor pin_delays,
    at::Tensor pin_impulses_sq,
    at::Tensor pin_capacitance,
    at::Tensor lut_slew_axis,
    at::Tensor lut_cap_axis,
    at::Tensor lut_cell_rise,
    at::Tensor lut_cell_fall,
    at::Tensor lut_rise_tran,
    at::Tensor lut_fall_tran,
    at::Tensor lut_sense,
    double gamma)
{
    int num_levels = (int)level_ptr.size(0) - 1;
    int max_fanin  = (int)cell_src_matrix.size(1);
    int S          = (int)lut_slew_axis.size(1);
    int C          = (int)lut_cap_axis.size(1);
    int tc         = max_fanin * 2;   // candidates per output in cell prop

    auto opts_f = AT.options();                          // float64, CUDA
    auto opts_i = cell_src_matrix.options();             // int32,   CUDA
    auto opts_b = cell_fanin_mask.options();             // bool,    CUDA

    // level 0 is skipped (PI/FF-Q, no fanin to propagate)
    for (int l = 1; l < num_levels; ++l)
    {
        int start = level_ptr[l].item<int>();
        int end   = level_ptr[l + 1].item<int>();
        int N     = end - start;
        if (N <= 0) continue;

        const int32_t *lp_ptr = flat_level_pins.data_ptr<int32_t>() + start;

        if (l % 2 == 1)
        {
            // ── Net propagation ──────────────────────────────────────────
            // AT[v] = AT[driver[v]] + delay[v]
            // slew[v] = sqrt(slew[driver[v]]^2 + impulse[v]^2)
            net_prop_launcher(
                lp_ptr,
                net_driver_map.data_ptr<int32_t>(),
                AT.data_ptr<double>(),
                slew.data_ptr<double>(),
                pin_delays.data_ptr<double>(),
                pin_impulses_sq.data_ptr<double>(),
                N);
        }
        else
        {
            // ── Cell propagation ─────────────────────────────────────────
            // The kernel writes dense [N, 2] output indexed by thread pos,
            // not by pin id → use separate temp buffers and scatter back.

            at::Tensor out_AT_tmp   = at::zeros({N, 2}, opts_f);
            at::Tensor out_slew_tmp = at::zeros({N, 2}, opts_f);

            // Saved-variable buffers required by the kernel signature.
            // We don't need them for backward here (no autograd Function),
            // so allocate and immediately discard.
            at::Tensor sv_wx_R = at::empty({N, max_fanin}, opts_f);
            at::Tensor sv_wx_F = at::empty({N, max_fanin}, opts_f);
            at::Tensor sv_wy   = at::empty({N, max_fanin}, opts_f);
            at::Tensor sv_ix_R = at::empty({N, max_fanin}, opts_i);
            at::Tensor sv_ix_F = at::empty({N, max_fanin}, opts_i);
            at::Tensor sv_iy   = at::empty({N, max_fanin}, opts_i);
            at::Tensor sv_relu = at::empty({N, max_fanin, 8}, opts_b);
            at::Tensor sv_lse_w= at::empty({N, 4, tc},    opts_f);

            cell_prop_fwd_launcher(
                lp_ptr,
                cell_src_matrix.data_ptr<int32_t>(),
                cell_arc_id_matrix.data_ptr<int32_t>(),
                cell_fanin_mask.data_ptr<bool>(),
                AT.data_ptr<double>(),           // input: full [num_pins, 2]
                slew.data_ptr<double>(),         // input: full [num_pins, 2]
                pin_capacitance.data_ptr<double>(),
                lut_slew_axis.data_ptr<double>(),
                lut_cap_axis.data_ptr<double>(),
                lut_cell_rise.data_ptr<double>(),
                lut_cell_fall.data_ptr<double>(),
                lut_rise_tran.data_ptr<double>(),
                lut_fall_tran.data_ptr<double>(),
                lut_sense.data_ptr<int32_t>(),
                out_AT_tmp.data_ptr<double>(),   // output: [N, 2]
                out_slew_tmp.data_ptr<double>(), // output: [N, 2]
                sv_wx_R.data_ptr<double>(),
                sv_wx_F.data_ptr<double>(),
                sv_wy.data_ptr<double>(),
                sv_ix_R.data_ptr<int32_t>(),
                sv_ix_F.data_ptr<int32_t>(),
                sv_iy.data_ptr<int32_t>(),
                sv_relu.data_ptr<bool>(),
                sv_lse_w.data_ptr<double>(),
                N, max_fanin, S, C, gamma);

            // Scatter results back into the full AT / slew tensors.
            // level_pins are distinct → no conflict, no atomics needed.
            auto lp_long = flat_level_pins.slice(0, start, end).to(at::kLong);
            AT.index_put_({lp_long}, out_AT_tmp);
            slew.index_put_({lp_long}, out_slew_tmp);
        }
    }

    return {AT, slew};
}

// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward_all_levels", &forward_all_levels, "Timing forward all levels (CUDA)");
}
