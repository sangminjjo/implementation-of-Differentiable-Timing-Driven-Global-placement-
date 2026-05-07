/**
 * @file   timing_loss_cuda_kernel.cu
 * @brief  Fused CUDA kernels for differentiable WNS/TNS timing penalties.
 *         All computation in double precision to match OpenTimer.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>

#define BLOCK 256
#define EPS   1e-12

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1: compute neg_slacks + block-level partial max
// ─────────────────────────────────────────────────────────────────────────────
__global__ void timing_neg_slack_kernel(
    const double*  __restrict__ AT,
    const int32_t* __restrict__ endpoints,
    double clock_period,
    double inv_gamma,
    int   N,
    double* __restrict__ neg_slacks,
    double* __restrict__ block_max
) {
    __shared__ double smem[BLOCK];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double x = -DBL_MAX;
    if (gid < N) {
        int ep  = gid >> 1;
        int rf  = gid &  1;
        double at_val = AT[endpoints[ep] * 2 + rf];
        x = (at_val - clock_period) * inv_gamma;
        neg_slacks[gid] = x;
    }
    smem[tid] = x;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmax(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    if (tid == 0) block_max[blockIdx.x] = smem[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2: compute block-level partial exp_sum + softplus_sum
// ─────────────────────────────────────────────────────────────────────────────
__global__ void timing_reduce_kernel(
    const double* __restrict__ neg_slacks,
    double global_max,
    int   N,
    double* __restrict__ block_exp_sum,
    double* __restrict__ block_tns_sum
) {
    __shared__ double smem_exp[BLOCK];
    __shared__ double smem_tns[BLOCK];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double e = 0.0, t = 0.0;
    if (gid < N) {
        double x = neg_slacks[gid];
        e = exp(x - global_max);
        t = (x >= 0.0) ? x + log(1.0 + exp(-x))
                       : log(1.0 + exp(x));
    }
    smem_exp[tid] = e;
    smem_tns[tid] = t;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem_exp[tid] += smem_exp[tid + s];
            smem_tns[tid] += smem_tns[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_exp_sum[blockIdx.x] = smem_exp[0];
        block_tns_sum[blockIdx.x] = smem_tns[0];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward: scatter d_loss/d_AT back to the full AT tensor
// ─────────────────────────────────────────────────────────────────────────────
__global__ void timing_bwd_kernel(
    const double*  __restrict__ neg_slacks,
    const int32_t* __restrict__ endpoints,
    double g_wns,
    double g_tns,
    double global_max,
    double total_exp_sum,
    int   N,
    double* __restrict__ g_AT
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    int ep = gid >> 1;
    int rf = gid &  1;
    double x = neg_slacks[gid];

    double softmax_i = exp(x - global_max) / (total_exp_sum + EPS);
    double sigmoid_i = 1.0 / (1.0 + exp(-x));

    double grad = g_wns * softmax_i + g_tns * sigmoid_i;
    atomicAdd(&g_AT[endpoints[ep] * 2 + rf], grad);
}

// ─────────────────────────────────────────────────────────────────────────────
// C++ launcher wrappers
// ─────────────────────────────────────────────────────────────────────────────

void timing_neg_slack_launcher(
    const double* AT, const int32_t* endpoints,
    double clock_period, double inv_gamma, int N,
    double* neg_slacks, double* block_max, int G
) {
    timing_neg_slack_kernel<<<G, BLOCK>>>(
        AT, endpoints, clock_period, inv_gamma, N, neg_slacks, block_max);
}

void timing_reduce_launcher(
    const double* neg_slacks, double global_max, int N,
    double* block_exp_sum, double* block_tns_sum, int G
) {
    timing_reduce_kernel<<<G, BLOCK>>>(
        neg_slacks, global_max, N, block_exp_sum, block_tns_sum);
}

void timing_bwd_launcher(
    const double* neg_slacks, const int32_t* endpoints,
    double g_wns, double g_tns,
    double global_max, double total_exp_sum,
    int N, double* g_AT, int G
) {
    timing_bwd_kernel<<<G, BLOCK>>>(
        neg_slacks, endpoints, g_wns, g_tns,
        global_max, total_exp_sum, N, g_AT);
}
