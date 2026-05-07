/**
 * @file   rc_prop_cuda_kernel.cu
 * @brief  CUDA kernels for Elmore delay RC tree propagation.
 *         All computation in double precision to match OpenTimer.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK 256

// ─────────────────────────────────────────────────────────────────────────────
// Forward kernels
// ─────────────────────────────────────────────────────────────────────────────

__global__ void scatter_add_fwd_kernel(
    const double*  __restrict__ in_buf,
    const int32_t* __restrict__ src,
    const int32_t* __restrict__ dst,
    double*        __restrict__ out_buf,
    int E
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < E) {
        atomicAdd(&out_buf[src[e]], in_buf[dst[e]]);
    }
}

__global__ void scatter_set_fwd_kernel(
    const double*  __restrict__ in_buf,
    const double*  __restrict__ R,
    const double*  __restrict__ aux,
    const int32_t* __restrict__ src,
    const int32_t* __restrict__ dst,
    double*        __restrict__ out_buf,
    int E
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < E) {
        out_buf[dst[e]] = in_buf[src[e]] + R[e] * aux[dst[e]];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward kernels
// ─────────────────────────────────────────────────────────────────────────────

__global__ void scatter_add_bwd_kernel(
    const double*  __restrict__ grad_out,
    const int32_t* __restrict__ src,
    const int32_t* __restrict__ dst,
    double*        __restrict__ grad_in,
    int E
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < E) {
        atomicAdd(&grad_in[dst[e]], grad_out[src[e]]);
    }
}

__global__ void scatter_set_bwd_kernel(
    const double*  __restrict__ grad_out,
    const double*  __restrict__ R,
    const double*  __restrict__ aux,
    const int32_t* __restrict__ src,
    const int32_t* __restrict__ dst,
    double*        __restrict__ grad_in,
    double*        __restrict__ grad_R,
    double*        __restrict__ grad_aux,
    int E
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < E) {
        double g = grad_out[dst[e]];
        atomicAdd(&grad_in[src[e]], g);
        atomicAdd(&grad_R[e], g * aux[dst[e]]);
        atomicAdd(&grad_aux[dst[e]], g * R[e]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Launcher helpers
// ─────────────────────────────────────────────────────────────────────────────

void scatter_add_fwd(const double* in_buf, const int32_t* src, const int32_t* dst,
                     double* out_buf, int E) {
    if (E <= 0) return;
    scatter_add_fwd_kernel<<<(E+BLOCK-1)/BLOCK, BLOCK>>>(in_buf, src, dst, out_buf, E);
}

void scatter_set_fwd(const double* in_buf, const double* R, const double* aux,
                     const int32_t* src, const int32_t* dst,
                     double* out_buf, int E) {
    if (E <= 0) return;
    scatter_set_fwd_kernel<<<(E+BLOCK-1)/BLOCK, BLOCK>>>(in_buf, R, aux, src, dst, out_buf, E);
}

void scatter_add_bwd(const double* grad_out, const int32_t* src, const int32_t* dst,
                     double* grad_in, int E) {
    if (E <= 0) return;
    scatter_add_bwd_kernel<<<(E+BLOCK-1)/BLOCK, BLOCK>>>(grad_out, src, dst, grad_in, E);
}

void scatter_set_bwd(const double* grad_out, const double* R, const double* aux,
                     const int32_t* src, const int32_t* dst,
                     double* grad_in, double* grad_R, double* grad_aux, int E) {
    if (E <= 0) return;
    scatter_set_bwd_kernel<<<(E+BLOCK-1)/BLOCK, BLOCK>>>(
        grad_out, R, aux, src, dst, grad_in, grad_R, grad_aux, E);
}
