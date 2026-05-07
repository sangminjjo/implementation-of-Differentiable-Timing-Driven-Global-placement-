/**
 * @file   net_prop_cuda_kernel.cu
 * @brief  CUDA kernels for differentiable net (wire) timing propagation.
 *         All computation in double precision to match OpenTimer.
 *
 * Forward:
 *   AT[v, rf]   = AT[driver[v], rf] + delay[v]
 *   slew[v, rf] = copysign(sqrt(slew[driver[v], rf]^2 + impulse[v]^2 + eps), slew[driver[v], rf])
 *
 * Backward:
 *   g_AT[driver[v]]   += g_AT[v]
 *   g_delay[v]         = g_AT[v,R] + g_AT[v,F]
 *   g_slew[driver[v]] += g_slew[v,rf] * slew_u[rf] / slew_v[rf]
 *   g_impulse_sq[v]    = sum_rf g_slew[v,rf] / (2 * slew_v[rf])
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK 256
#define EPS   1e-12

// ─────────────────────────────────────────────────────────────────────────────
// Forward kernel
// ─────────────────────────────────────────────────────────────────────────────
__global__ void net_prop_fwd_kernel(
    const int32_t* __restrict__ level_pins,
    const int32_t* __restrict__ driver_pins,
    const double*  __restrict__ AT_in,
    const double*  __restrict__ slew_in,
    const double*  __restrict__ pin_delays,
    const double*  __restrict__ pin_impulses_sq,
    double*        __restrict__ AT_out,
    double*        __restrict__ slew_out,
    double*        __restrict__ sv_slew_u,
    double*        __restrict__ sv_slew_v,
    double*        __restrict__ sv_impulse_sq,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int v = level_pins[i];
    int u = driver_pins[i];

    double d       = pin_delays[v];
    double imp_sq  = pin_impulses_sq[v];

    double at_u_R  = AT_in[u * 2 + 0];
    double at_u_F  = AT_in[u * 2 + 1];
    double sl_u_R  = slew_in[u * 2 + 0];
    double sl_u_F  = slew_in[u * 2 + 1];

    double sl_v_R  = copysign(sqrt(sl_u_R * sl_u_R + imp_sq + EPS), sl_u_R);
    double sl_v_F  = copysign(sqrt(sl_u_F * sl_u_F + imp_sq + EPS), sl_u_F);

    AT_out[i * 2 + 0] = at_u_R + d;
    AT_out[i * 2 + 1] = at_u_F + d;
    slew_out[i * 2 + 0] = sl_v_R;
    slew_out[i * 2 + 1] = sl_v_F;

    sv_slew_u[i * 2 + 0] = sl_u_R;
    sv_slew_u[i * 2 + 1] = sl_u_F;
    sv_slew_v[i * 2 + 0] = sl_v_R;
    sv_slew_v[i * 2 + 1] = sl_v_F;
    sv_impulse_sq[i]     = imp_sq;
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward kernel
// ─────────────────────────────────────────────────────────────────────────────
__global__ void net_prop_bwd_kernel(
    const int32_t* __restrict__ level_pins,
    const int32_t* __restrict__ driver_pins,
    const double*  __restrict__ sv_slew_u,
    const double*  __restrict__ sv_slew_v,
    const double*  __restrict__ sv_impulse_sq,
    int N,
    double* __restrict__ g_AT,
    double* __restrict__ g_slew,
    double* __restrict__ g_delay,
    double* __restrict__ g_impulse_sq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int v = level_pins[i];
    int u = driver_pins[i];

    // AT backward
    double gAR = g_AT[v * 2 + 0];
    double gAF = g_AT[v * 2 + 1];

    g_AT[v * 2 + 0] = 0.0;
    g_AT[v * 2 + 1] = 0.0;

    atomicAdd(&g_AT[u * 2 + 0], gAR);
    atomicAdd(&g_AT[u * 2 + 1], gAF);

    g_delay[v] = gAR + gAF;

    // slew backward
    double gSR = g_slew[v * 2 + 0];
    double gSF = g_slew[v * 2 + 1];

    double su_R = sv_slew_u[i * 2 + 0];
    double su_F = sv_slew_u[i * 2 + 1];
    double sv_R = sv_slew_v[i * 2 + 0];
    double sv_F = sv_slew_v[i * 2 + 1];

    g_slew[v * 2 + 0] = 0.0;
    g_slew[v * 2 + 1] = 0.0;

    atomicAdd(&g_slew[u * 2 + 0], gSR * (su_R / sv_R));
    atomicAdd(&g_slew[u * 2 + 1], gSF * (su_F / sv_F));

    g_impulse_sq[v] = (gSR * 0.5 / sv_R) + (gSF * 0.5 / sv_F);
}

// ─────────────────────────────────────────────────────────────────────────────
// C++ launcher wrappers
// ─────────────────────────────────────────────────────────────────────────────

void net_prop_fwd_launcher(
    const int32_t* level_pins, const int32_t* driver_pins,
    const double* AT_in, const double* slew_in,
    const double* pin_delays, const double* pin_impulses_sq,
    double* AT_out, double* slew_out,
    double* sv_slew_u, double* sv_slew_v, double* sv_impulse_sq,
    int N
) {
    if (N <= 0) return;
    int blocks = (N + BLOCK - 1) / BLOCK;
    net_prop_fwd_kernel<<<blocks, BLOCK>>>(
        level_pins, driver_pins,
        AT_in, slew_in, pin_delays, pin_impulses_sq,
        AT_out, slew_out,
        sv_slew_u, sv_slew_v, sv_impulse_sq, N);
}

void net_prop_bwd_launcher(
    const int32_t* level_pins, const int32_t* driver_pins,
    const double* sv_slew_u, const double* sv_slew_v, const double* sv_impulse_sq,
    int N,
    double* g_AT, double* g_slew,
    double* g_delay, double* g_impulse_sq
) {
    if (N <= 0) return;
    int blocks = (N + BLOCK - 1) / BLOCK;
    net_prop_bwd_kernel<<<blocks, BLOCK>>>(
        level_pins, driver_pins,
        sv_slew_u, sv_slew_v, sv_impulse_sq, N,
        g_AT, g_slew, g_delay, g_impulse_sq);
}

// ── In-place launcher ──

__global__ void net_prop_inplace_kernel(
    const int32_t* __restrict__ level_pins,
    const int32_t* __restrict__ driver_map,
    double*        __restrict__ AT,
    double*        __restrict__ slew,
    const double*  __restrict__ delays,
    const double*  __restrict__ impulses_sq,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int v = level_pins[i];
    int u = driver_map[v];

    double d      = delays[v];
    double imp_sq = impulses_sq[v];

    double sl_u_R = slew[u * 2 + 0];
    double sl_u_F = slew[u * 2 + 1];

    double sl_v_R = copysign(sqrt(sl_u_R * sl_u_R + imp_sq + EPS), sl_u_R);
    double sl_v_F = copysign(sqrt(sl_u_F * sl_u_F + imp_sq + EPS), sl_u_F);

    AT[v * 2 + 0] = AT[u * 2 + 0] + d;
    AT[v * 2 + 1] = AT[u * 2 + 1] + d;
    slew[v * 2 + 0] = sl_v_R;
    slew[v * 2 + 1] = sl_v_F;
}

void net_prop_launcher(
    const int32_t* pins, const int32_t* driver_map,
    double* AT, double* slew,
    const double* delays, const double* impulses_sq, int N
) {
    if (N <= 0) return;
    int blocks = (N + BLOCK - 1) / BLOCK;
    net_prop_inplace_kernel<<<blocks, BLOCK>>>(
        pins, driver_map, AT, slew, delays, impulses_sq, N);
}
