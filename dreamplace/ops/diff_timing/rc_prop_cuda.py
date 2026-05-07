import torch
from torch.autograd import Function
import dreamplace.ops.diff_timing.rc_prop_cuda_ext as rc_prop_cuda_ext

class RCPropFunction(Function):
    @staticmethod
    def forward(ctx, node_C, edge_R, src, dst, num_pins, max_depth):
        # forward_with_snapshots returns: [delay, impulse_sq, load, load_snap, delay_snap, ldelay_snap, beta_snap]
        results = rc_prop_cuda_ext.forward_with_snapshots(
            node_C, edge_R, src, dst, num_pins, max_depth
        )
        
        delay, impulse_sq, load = results[0], results[1], results[2]
        snaps = results[3:] # load, delay, ldelay, beta snapshots
        
        ctx.save_for_backward(node_C, edge_R, src, dst)
        ctx.snaps = snaps
        ctx.num_pins = num_pins
        ctx.max_depth = max_depth
        
        return delay, impulse_sq, load

    @staticmethod
    def backward(ctx, grad_delay, grad_impulse_sq, grad_load):
        node_C, edge_R, src, dst = ctx.saved_tensors
        load_snap, delay_snap, ldelay_snap, beta_snap = ctx.snaps
        
        # C++ extension backward
        grads = rc_prop_cuda_ext.backward(
            grad_delay, grad_impulse_sq, grad_load,
            load_snap, delay_snap, ldelay_snap, beta_snap,
            node_C, edge_R, src, dst, ctx.max_depth
        )
        
        # return grads for [node_C, edge_R, src, dst, num_pins, max_depth]
        return grads[0], grads[1], None, None, None, None

def rc_prop_cuda(node_C, edge_R, src, dst, num_pins, max_depth):
    return RCPropFunction.apply(node_C, edge_R, src, dst, num_pins, max_depth)
