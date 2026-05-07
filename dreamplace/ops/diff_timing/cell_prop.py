"""
cell_prop.py

Drop-in CUDA replacement for DifferentiableTimingLoss._cell_propagation().

Usage (inside DifferentiableTimingLoss):
    from dreamplace.ops.diff_timing.cell_prop import cell_propagation_cuda
    ...
    AT, slew = cell_propagation_cuda(self, level_pins, AT, slew, pin_load_caps, self.gamma)
"""

import torch
import torch.nn.functional as F
import dreamplace.configure as configure

_cuda_available = False
if configure.compile_configurations.get("CUDA_FOUND") == "TRUE":
    try:
        import dreamplace.ops.diff_timing.cell_prop_cuda as _cell_prop_cuda
        _cuda_available = True
    except ImportError:
        pass


class CellPropFunction(torch.autograd.Function):
    """
    Custom autograd Function wrapping the fused CUDA forward/backward kernels.

    forward inputs
    ──────────────
    AT               [num_pins, 2]         arrival times  (R/F)
    slew             [num_pins, 2]         slews
    pin_load_caps    [num_pins]            output pin load cap (from Elmore)
    level_pins       [N]  int32            output pins at this level
    cell_src_matrix  [num_pins, max_fanin] source pins per fanin
    cell_arc_id_mat  [num_pins, max_fanin] LUT arc id per fanin
    cell_fanin_mask  [num_pins, max_fanin] valid fanin mask
    lut_*            LUT tables
    gamma            float                 logsumexp smoothing parameter

    forward outputs
    ───────────────
    new_AT    [N, 2]   updated AT for level_pins
    new_slew  [N, 2]   updated slew for level_pins
    """

    @staticmethod
    def forward(ctx,
                AT, slew, pin_load_caps,
                level_pins,
                cell_src_matrix, cell_arc_id_matrix, cell_fanin_mask,
                lut_slew_axis, lut_cap_axis,
                lut_cell_rise, lut_cell_fall,
                lut_rise_tran, lut_fall_tran,
                lut_sense,
                gamma):

        results = _cell_prop_cuda.forward(
            level_pins,
            cell_src_matrix, cell_arc_id_matrix, cell_fanin_mask,
            AT, slew, pin_load_caps,
            lut_slew_axis, lut_cap_axis,
            lut_cell_rise, lut_cell_fall, lut_rise_tran, lut_fall_tran,
            lut_sense,
            float(gamma)
        )
        new_AT, new_slew = results[0], results[1]
        saved = results[2:]   # sv_wx_R, sv_wx_F, sv_wy, sv_ix_R, sv_ix_F, sv_iy, sv_relu, sv_lse_w

        ctx.save_for_backward(
            level_pins,
            cell_src_matrix, cell_arc_id_matrix, cell_fanin_mask,
            lut_slew_axis, lut_cap_axis,
            lut_cell_rise, lut_cell_fall, lut_rise_tran, lut_fall_tran,
            lut_sense,
            *saved
        )
        ctx.num_pins = AT.shape[0]
        ctx.gamma    = float(gamma)

        return new_AT, new_slew

    @staticmethod
    def backward(ctx, g_new_AT, g_new_slew):
        (level_pins,
         cell_src_matrix, cell_arc_id_matrix, cell_fanin_mask,
         lut_slew_axis, lut_cap_axis,
         lut_cell_rise, lut_cell_fall, lut_rise_tran, lut_fall_tran,
         lut_sense,
         sv_wx_R, sv_wx_F, sv_wy,
         sv_ix_R, sv_ix_F, sv_iy,
         sv_relu, sv_lse_w) = ctx.saved_tensors

        g_AT, g_slew, g_caps = _cell_prop_cuda.backward(
            level_pins,
            cell_src_matrix, cell_arc_id_matrix, cell_fanin_mask,
            g_new_AT.contiguous(), g_new_slew.contiguous(),
            lut_slew_axis, lut_cap_axis,
            lut_cell_rise, lut_cell_fall, lut_rise_tran, lut_fall_tran,
            lut_sense,
            sv_wx_R, sv_wx_F, sv_wy,
            sv_ix_R, sv_ix_F, sv_iy,
            sv_relu, sv_lse_w,
            ctx.num_pins,
            ctx.gamma
        )

        # Return grads matching forward()'s positional args:
        #  AT, slew, pin_load_caps, level_pins(no grad),
        #  src_matrix, arc_id_matrix, fanin_mask (no grad),
        #  lut_* (no grad),  gamma (no grad)
        return (g_AT, g_slew, g_caps,
                None,           # level_pins
                None, None, None,  # src_matrix, arc_id_matrix, fanin_mask
                None, None,     # lut_slew_axis, lut_cap_axis
                None, None, None, None,  # lut tables
                None,           # lut_sense
                None)           # gamma


def _cell_propagation_python_fallback(self_obj, level_pins, AT, slew, pin_load_caps, gamma):
    """Original Python implementation — used when CUDA is not available."""
    return self_obj._cell_propagation(level_pins, AT, slew, pin_load_caps, gamma)


def cell_propagation_cuda(self_obj, level_pins, AT, slew, pin_load_caps, gamma):
    """
    CUDA-accelerated cell propagation.  Falls back to Python if CUDA op is
    not compiled or if tensors are on CPU.

    Parameters
    ──────────
    self_obj      : DifferentiableTimingLoss instance  (holds LUT tensors)
    level_pins    : [N] int64 tensor of output pins at this level
    AT            : [num_pins, 2] arrival time tensor
    slew          : [num_pins, 2] slew tensor
    pin_load_caps : [num_pins] capacitance per pin
    gamma         : float logsumexp smoothing

    Returns
    ───────
    AT, slew  (updated in-place at level_pins positions)
    """
    if not _cuda_available or not AT.is_cuda:
        return self_obj._cell_propagation(level_pins, AT, slew, pin_load_caps, gamma)

    mask = self_obj.cell_fanin_mask[level_pins.long()]
    if not mask.any():
        return AT, slew

    # CellPropFunction expects int32 level_pins
    lp_i32 = level_pins.to(torch.int32)

    new_AT, new_slew = CellPropFunction.apply(
        AT, slew, pin_load_caps,
        lp_i32,
        self_obj.cell_src_matrix,
        self_obj.cell_arc_id_matrix,
        self_obj.cell_fanin_mask,
        self_obj.lut_slew_axis,
        self_obj.lut_cap_axis,
        self_obj.lut_cell_rise,
        self_obj.lut_cell_fall,
        self_obj.lut_rise_tran,
        self_obj.lut_fall_tran,
        self_obj.lut_sense,
        gamma
    )

    # Scatter results back into the full AT / slew tensors.
    # level_pins positions are written; all other positions are unchanged.
    AT   = AT.index_put((level_pins,), new_AT)
    slew = slew.index_put((level_pins,), new_slew)

    return AT, slew
