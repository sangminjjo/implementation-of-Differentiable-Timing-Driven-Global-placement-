"""
level_prop_cuda.py

Differentiable level-by-level timing propagation for DifferentiableTimingLoss.
"""

import torch
import torch.nn.functional as F
import dreamplace.configure as configure

_net_prop_cuda  = None
_cell_prop_cuda = None

if configure.compile_configurations.get("CUDA_FOUND") == "TRUE":
    try:
        import dreamplace.ops.diff_timing.net_prop_cuda_ext as _net_prop_cuda
    except ImportError:
        pass
    try:
        import dreamplace.ops.diff_timing.diff_timing_cell_prop_cuda as _cell_prop_cuda
    except ImportError:
        pass

_cuda_available = (_net_prop_cuda is not None) and (_cell_prop_cuda is not None)

class AllLevelsPropFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                AT, slew, pin_delays, pin_impulses_sq, pin_capacitance,
                pin_levels, net_driver_map,
                cell_src_matrix, cell_arc_id_matrix, cell_fanin_mask,
                lut_slew_axis, lut_cap_axis,
                lut_cell_rise, lut_cell_fall,
                lut_rise_tran, lut_fall_tran,
                lut_sense,
                gamma):

        num_pins = AT.shape[0]
        AT_cur   = AT.clone()
        slew_cur = slew.clone()

        saved_per_level = []

        for level_idx, level_pins_i64 in enumerate(pin_levels):
            if level_idx == 0:
                saved_per_level.append(None)
                continue

            level_pins_i32 = level_pins_i64.to(torch.int32)
            N = level_pins_i32.shape[0]
            if N == 0:
                saved_per_level.append(None)
                continue

            if level_idx % 2 == 1:
                # Net propagation
                driver_pins = net_driver_map[level_pins_i64].to(torch.int32)

                results = _net_prop_cuda.forward(
                    level_pins_i32, driver_pins,
                    AT_cur, slew_cur,
                    pin_delays, pin_impulses_sq)

                new_AT_l, new_slew_l = results[0], results[1]
                sv_slew_u, sv_slew_v, sv_impulse_sq = results[2], results[3], results[4]

                AT_cur   = AT_cur.index_put((level_pins_i64,), new_AT_l)
                slew_cur = slew_cur.index_put((level_pins_i64,), new_slew_l)

                saved_per_level.append(
                    ('net', level_pins_i32, driver_pins,
                     sv_slew_u, sv_slew_v, sv_impulse_sq))
            else:
                # Cell propagation
                results = _cell_prop_cuda.forward(
                    level_pins_i32,
                    cell_src_matrix, cell_arc_id_matrix, cell_fanin_mask,
                    AT_cur, slew_cur, pin_capacitance,
                    lut_slew_axis, lut_cap_axis,
                    lut_cell_rise, lut_cell_fall,
                    lut_rise_tran, lut_fall_tran,
                    lut_sense,
                    float(gamma))

                new_AT_l, new_slew_l = results[0], results[1]
                sv = results[2:]

                AT_cur   = AT_cur.index_put((level_pins_i64,), new_AT_l)
                slew_cur = slew_cur.index_put((level_pins_i64,), new_slew_l)

                saved_per_level.append(
                    ('cell', level_pins_i32, level_pins_i64, *sv))

        ctx.saved_per_level = saved_per_level
        ctx.num_pins        = num_pins
        ctx.gamma           = float(gamma)
        ctx.save_for_backward(
            cell_src_matrix, cell_arc_id_matrix, cell_fanin_mask,
            lut_slew_axis, lut_cap_axis,
            lut_cell_rise, lut_cell_fall,
            lut_rise_tran, lut_fall_tran,
            lut_sense)

        return AT_cur, slew_cur

    @staticmethod
    def backward(ctx, g_AT_in, g_slew_in):
        lut_tensors = ctx.saved_tensors
        cell_src_matrix, cell_arc_id_matrix, cell_fanin_mask = lut_tensors[:3]
        lut_slew_axis, lut_cap_axis = lut_tensors[3:5]
        lut_cell_rise, lut_cell_fall, lut_rise_tran, lut_fall_tran = lut_tensors[5:9]
        lut_sense = lut_tensors[9]

        num_pins = ctx.num_pins
        gamma    = ctx.gamma

        g_AT   = g_AT_in.clone()
        g_slew = g_slew_in.clone()

        g_delays      = torch.zeros(num_pins, device=g_AT.device, dtype=g_AT.dtype)
        g_impulses_sq = torch.zeros(num_pins, device=g_AT.device, dtype=g_AT.dtype)
        g_caps        = torch.zeros(num_pins, device=g_AT.device, dtype=g_AT.dtype)

        for entry in reversed(ctx.saved_per_level):
            if entry is None:
                continue

            kind = entry[0]

            if kind == 'net':
                _, level_pins_i32, driver_pins, sv_slew_u, sv_slew_v, sv_impulse_sq = entry
                level_pins_i64 = level_pins_i32.to(torch.long)

                # grads: [g_delay, g_impulse_sq]
                grads = _net_prop_cuda.backward(
                    level_pins_i32, driver_pins,
                    sv_slew_u, sv_slew_v, sv_impulse_sq,
                    g_AT, g_slew, num_pins)

                g_d, g_imp = grads[0], grads[1]
                g_delays[level_pins_i64]      += g_d[level_pins_i64]
                g_impulses_sq[level_pins_i64] += g_imp[level_pins_i64]

            else:  # 'cell'
                _, level_pins_i32, level_pins_i64 = entry[:3]
                sv = entry[3:]
                (sv_wx_R, sv_wx_F, sv_wy, sv_ix_R, sv_ix_F, sv_iy, sv_relu, sv_lse_w) = sv

                g_out_AT   = g_AT[level_pins_i64].contiguous()
                g_out_slew = g_slew[level_pins_i64].contiguous()

                cell_grads = _cell_prop_cuda.backward(
                    level_pins_i32,
                    cell_src_matrix, cell_arc_id_matrix, cell_fanin_mask,
                    g_out_AT, g_out_slew,
                    lut_slew_axis, lut_cap_axis,
                    lut_cell_rise, lut_cell_fall,
                    lut_rise_tran, lut_fall_tran,
                    lut_sense,
                    sv_wx_R, sv_wx_F, sv_wy,
                    sv_ix_R, sv_ix_F, sv_iy,
                    sv_relu, sv_lse_w,
                    num_pins, gamma)

                g_AT_cell, g_slew_cell, g_caps_cell = cell_grads[:3]

                # Consumed level entries zeroing
                g_AT.index_put_((level_pins_i64,), torch.zeros_like(g_out_AT))
                g_slew.index_put_((level_pins_i64,), torch.zeros_like(g_out_slew))
                
                g_AT.add_(g_AT_cell)
                g_slew.add_(g_slew_cell)
                g_caps.add_(g_caps_cell)

        return (g_AT, g_slew, g_delays, g_impulses_sq, g_caps,
                None, None, None, None, None, None, None, None, None, None, None, None, None)

def level_propagation(AT, slew, pin_delays, pin_impulses_sq, pin_capacitance, timing_obj):
    if not _cuda_available or not AT.is_cuda:
        for level_idx, level_pins in enumerate(timing_obj.pin_levels):
            if level_idx == 0:
                continue
            if level_idx % 2 == 1:
                AT, slew = timing_obj._net_propagation(level_pins, AT, slew, pin_delays, pin_impulses_sq)
            else:
                AT, slew = timing_obj._cell_propagation(level_pins, AT, slew, pin_capacitance, timing_obj.gamma)
        return AT, slew

    return AllLevelsPropFunction.apply(
        AT, slew, pin_delays, pin_impulses_sq, pin_capacitance,
        timing_obj.pin_levels,
        timing_obj.net_driver_map,
        timing_obj.cell_src_matrix,
        timing_obj.cell_arc_id_matrix,
        timing_obj.cell_fanin_mask,
        timing_obj.lut_slew_axis,
        timing_obj.lut_cap_axis,
        timing_obj.lut_cell_rise,
        timing_obj.lut_cell_fall,
        timing_obj.lut_rise_tran,
        timing_obj.lut_fall_tran,
        timing_obj.lut_sense,
        timing_obj.gamma,
    )
