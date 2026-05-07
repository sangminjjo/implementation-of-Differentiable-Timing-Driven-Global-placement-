/**
 * @file   cell_prop_cuda_kernel.cu
 * @brief  CUDA kernel for differentiable cell timing propagation.
 *         All computation in double precision to match OpenTimer.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <torch/extension.h>

#define MAX_FANIN 64
#define MAX_CANDS 128   // MAX_FANIN * 2

#define EPS     1e-9
#define MIN_VAL -1e9
#define BLOCK   256

// -----------------------------------------------------------------------------
// Device helpers
// -----------------------------------------------------------------------------

__device__ __forceinline__ int left_bracket(const double* axis, int n, double val) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (axis[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return max(0, min(lo - 1, n - 2));
}

__device__ __forceinline__ double bilinear_val(
        const double* grid, int C,
        int ix, int iy,
        double wx, double wy) {
    double v00 = grid[ ix      * C + iy    ];
    double v01 = grid[ ix      * C + iy + 1];
    double v10 = grid[(ix + 1) * C + iy    ];
    double v11 = grid[(ix + 1) * C + iy + 1];
    return (1.0 - wx) * (1.0 - wy) * v00
         + (1.0 - wx) *        wy  * v01
         +        wx  * (1.0 - wy) * v10
         +        wx  *        wy  * v11;
}

__device__ __forceinline__ double bilinear_grad_wx(
        const double* grid, int C,
        int ix, int iy, double wy) {
    double v00 = grid[ ix      * C + iy    ];
    double v01 = grid[ ix      * C + iy + 1];
    double v10 = grid[(ix + 1) * C + iy    ];
    double v11 = grid[(ix + 1) * C + iy + 1];
    return -(1.0 - wy) * v00 - wy * v01
           +(1.0 - wy) * v10 + wy * v11;
}

__device__ __forceinline__ double bilinear_grad_wy(
        const double* grid, int C,
        int ix, int iy, double wx) {
    double v00 = grid[ ix      * C + iy    ];
    double v01 = grid[ ix      * C + iy + 1];
    double v10 = grid[(ix + 1) * C + iy    ];
    double v11 = grid[(ix + 1) * C + iy + 1];
    return -(1.0 - wx) * v00 + (1.0 - wx) * v01
           -       wx  * v10 +        wx  * v11;
}

__device__ __forceinline__ double stable_lse(
        const double* cands, int nc,
        double gamma, double inv_gamma,
        double* lse_w) {
    double maxv = cands[0];
    for (int i = 1; i < nc; i++) maxv = fmax(maxv, cands[i]);

    double sum_e = 0.0;
    for (int i = 0; i < nc; i++) {
        double e = (cands[i] <= MIN_VAL) ? 0.0 : exp((cands[i] - maxv) * inv_gamma);
        lse_w[i] = e;
        sum_e += e;
    }
    double inv_sum = (sum_e > 0.0) ? (1.0 / sum_e) : 0.0;
    for (int i = 0; i < nc; i++) lse_w[i] *= inv_sum;

    return (sum_e > 0.0) ? (gamma * (log(sum_e) + maxv * inv_gamma)) : MIN_VAL;
}

// -----------------------------------------------------------------------------
// Forward Kernel
// -----------------------------------------------------------------------------

__global__ void cell_prop_fwd_kernel(
    const int32_t* __restrict__ level_pins,
    const int32_t* __restrict__ src_matrix,
    const int32_t* __restrict__ arc_id_matrix,
    const bool*    __restrict__ fanin_mask,
    const double*  __restrict__ AT,
    const double*  __restrict__ slew,
    const double*  __restrict__ pin_load_caps,
    const double*  __restrict__ lut_slew_axis,
    const double*  __restrict__ lut_cap_axis,
    const double*  __restrict__ lut_cell_rise,
    const double*  __restrict__ lut_cell_fall,
    const double*  __restrict__ lut_rise_tran,
    const double*  __restrict__ lut_fall_tran,
    const int32_t* __restrict__ lut_sense,
    double* __restrict__ out_AT,
    double* __restrict__ out_slew,
    double*  __restrict__ sv_wx_R,
    double*  __restrict__ sv_wx_F,
    double*  __restrict__ sv_wy,
    int32_t* __restrict__ sv_ix_R,
    int32_t* __restrict__ sv_ix_F,
    int32_t* __restrict__ sv_iy,
    bool*    __restrict__ sv_relu,
    double*  __restrict__ sv_lse_w,
    int N, int max_fanin, int S, int C,
    double gamma, double inv_gamma
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    int pin    = level_pins[t];
    double ocap = pin_load_caps[pin];
    int   tc   = max_fanin * 2;

    double cAR[MAX_CANDS], cAF[MAX_CANDS], cSR[MAX_CANDS], cSF[MAX_CANDS];
    for (int i = 0; i < tc; i++) {
        cAR[i] = MIN_VAL;
        cAF[i] = MIN_VAL;
        cSR[i] = MIN_VAL;
        cSF[i] = MIN_VAL;
    }

    for (int f = 0; f < max_fanin; f++) {
        int mat = pin * max_fanin + f;
        if (!fanin_mask[mat]) continue;

        int src    = src_matrix[mat];
        int arc_id = arc_id_matrix[mat];
        int sense  = lut_sense[arc_id];

        double in_AT_R  = AT[src * 2 + 0];
        double in_AT_F  = AT[src * 2 + 1];
        double in_sl_R  = slew[src * 2 + 0];
        double in_sl_F  = slew[src * 2 + 1];

        const double* s_ax = lut_slew_axis + arc_id * S;
        const double* c_ax = lut_cap_axis  + arc_id * C;
        const double* gDR  = lut_cell_rise + arc_id * S * C;
        const double* gDF  = lut_cell_fall + arc_id * S * C;
        const double* gSR  = lut_rise_tran + arc_id * S * C;
        const double* gSF  = lut_fall_tran + arc_id * S * C;

        int ix_R = left_bracket(s_ax, S, in_sl_R);
        int ix_F = left_bracket(s_ax, S, in_sl_F);
        int iy   = left_bracket(c_ax, C, ocap);

        double x0R = s_ax[ix_R], x1R = s_ax[ix_R + 1];
        double x0F = s_ax[ix_F], x1F = s_ax[ix_F + 1];
        double y0  = c_ax[iy],   y1  = c_ax[iy  + 1];

        double wx_R = (in_sl_R - x0R) / (x1R - x0R + EPS);
        double wx_F = (in_sl_F - x0F) / (x1F - x0F + EPS);
        double wy   = (ocap    - y0 ) / (y1  - y0  + EPS);

        int si = t * max_fanin + f;
        sv_wx_R[si] = wx_R;  sv_wx_F[si] = wx_F;  sv_wy[si] = wy;
        sv_ix_R[si] = ix_R;  sv_ix_F[si] = ix_F;  sv_iy[si] = iy;

        double d_RR = bilinear_val(gDR, C, ix_R, iy, wx_R, wy);
        double s_RR = bilinear_val(gSR, C, ix_R, iy, wx_R, wy);
        double d_FR = bilinear_val(gDR, C, ix_F, iy, wx_F, wy);
        double s_FR = bilinear_val(gSR, C, ix_F, iy, wx_F, wy);
        double d_RF = bilinear_val(gDF, C, ix_R, iy, wx_R, wy);
        double s_RF = bilinear_val(gSF, C, ix_R, iy, wx_R, wy);
        double d_FF = bilinear_val(gDF, C, ix_F, iy, wx_F, wy);
        double s_FF = bilinear_val(gSF, C, ix_F, iy, wx_F, wy);

        bool vRRFF = (sense == 0 || sense == 2);
        bool vRFFR = (sense == 1 || sense == 2);

        cAR[f*2+0] = vRRFF ? (in_AT_R + d_RR) : MIN_VAL;
        cAR[f*2+1] = vRFFR ? (in_AT_F + d_FR) : MIN_VAL;
        cAF[f*2+0] = vRRFF ? (in_AT_F + d_FF) : MIN_VAL;
        cAF[f*2+1] = vRFFR ? (in_AT_R + d_RF) : MIN_VAL;

        cSR[f*2+0] = vRRFF ? s_RR : MIN_VAL;
        cSR[f*2+1] = vRFFR ? s_FR : MIN_VAL;
        cSF[f*2+0] = vRRFF ? s_FF : MIN_VAL;
        cSF[f*2+1] = vRFFR ? s_RF : MIN_VAL;
    }

    double* lw = sv_lse_w + t * 4 * tc;
    out_AT[t*2+0] = stable_lse(cAR, tc, gamma, inv_gamma, lw);
    out_AT[t*2+1] = stable_lse(cAF, tc, gamma, inv_gamma, lw + tc);

    double* lw_SR = lw + 2 * tc;
    double* lw_SF = lw + 3 * tc;

    out_slew[t*2+0] = stable_lse(cSR, tc, gamma, inv_gamma, lw_SR);
    out_slew[t*2+1] = stable_lse(cSF, tc, gamma, inv_gamma, lw_SF);
}

// -----------------------------------------------------------------------------
// Backward Kernel
// -----------------------------------------------------------------------------

__global__ void cell_prop_bwd_kernel(
    const int32_t* __restrict__ level_pins,
    const int32_t* __restrict__ src_matrix,
    const int32_t* __restrict__ arc_id_matrix,
    const bool*    __restrict__ fanin_mask,
    const double*  __restrict__ g_out_AT,
    const double*  __restrict__ g_out_slew,
    const double*  __restrict__ lut_slew_axis,
    const double*  __restrict__ lut_cap_axis,
    const double*  __restrict__ lut_cell_rise,
    const double*  __restrict__ lut_cell_fall,
    const double*  __restrict__ lut_rise_tran,
    const double*  __restrict__ lut_fall_tran,
    const int32_t* __restrict__ lut_sense,
    const double*  __restrict__ sv_wx_R,
    const double*  __restrict__ sv_wx_F,
    const double*  __restrict__ sv_wy,
    const int32_t* __restrict__ sv_ix_R,
    const int32_t* __restrict__ sv_ix_F,
    const int32_t* __restrict__ sv_iy,
    const bool*    __restrict__ sv_relu,
    const double*  __restrict__ sv_lse_w,
    double*        __restrict__ g_AT,
    double*        __restrict__ g_slew,
    double*        __restrict__ g_caps,
    int N, int max_fanin, int S, int C,
    double inv_gamma
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    int  pin = level_pins[t];
    int  tc  = max_fanin * 2;

    double gAR = g_out_AT[t*2+0];
    double gAF = g_out_AT[t*2+1];
    double gSR = g_out_slew[t*2+0];
    double gSF = g_out_slew[t*2+1];

    const double* lw = sv_lse_w + t * 4 * tc;
    const double* lw_AR = lw;
    const double* lw_AF = lw + tc;

    double g_cap_acc = 0.0;

    for (int f = 0; f < max_fanin; f++) {
        int mat = pin * max_fanin + f;
        if (!fanin_mask[mat]) continue;

        int src    = src_matrix[mat];
        int arc_id = arc_id_matrix[mat];
        int sense  = lut_sense[arc_id];

        int si = t * max_fanin + f;
        double wx_R = sv_wx_R[si], wx_F = sv_wx_F[si], wy = sv_wy[si];
        int   ix_R = sv_ix_R[si], ix_F = sv_ix_F[si], iy = sv_iy[si];

        bool vRRFF = (sense == 0 || sense == 2);
        bool vRFFR = (sense == 1 || sense == 2);

        const double* lw_SR = lw + 2 * tc;
        const double* lw_SF = lw + 3 * tc;
        double g_dRR = vRRFF ? lw_AR[f*2+0] * gAR : 0.0;
        double g_sRR = vRRFF ? lw_SR[f*2+0] * gSR : 0.0;
        double g_dFR = vRFFR ? lw_AR[f*2+1] * gAR : 0.0;
        double g_sFR = vRFFR ? lw_SR[f*2+1] * gSR : 0.0;
        double g_dRF = vRFFR ? lw_AF[f*2+1] * gAF : 0.0;
        double g_sRF = vRFFR ? lw_SF[f*2+1] * gSF : 0.0;
        double g_dFF = vRRFF ? lw_AF[f*2+0] * gAF : 0.0;
        double g_sFF = vRRFF ? lw_SF[f*2+0] * gSF : 0.0;

        // AT 역전파
        atomicAdd(&g_AT[src*2+0], g_dRR + g_dRF);
        atomicAdd(&g_AT[src*2+1], g_dFR + g_dFF);

        const double* s_ax = lut_slew_axis + arc_id * S;
        const double* c_ax = lut_cap_axis  + arc_id * C;
        double dx_R  = s_ax[ix_R+1] - s_ax[ix_R] + EPS;
        double dx_F  = s_ax[ix_F+1] - s_ax[ix_F] + EPS;
        double dy    = c_ax[iy  +1] - c_ax[iy  ] + EPS;
        double inv_dx_R = 1.0 / dx_R;
        double inv_dx_F = 1.0 / dx_F;
        double inv_dy   = 1.0 / dy;

        const double* gDR     = lut_cell_rise + arc_id * S * C;
        const double* gDF     = lut_cell_fall + arc_id * S * C;
        const double* gSR_lut = lut_rise_tran + arc_id * S * C;
        const double* gSF_lut = lut_fall_tran + arc_id * S * C;

        // Slew 역전파
        double g_sl_R = (g_dRR * bilinear_grad_wx(gDR, C, ix_R, iy, wy) +
                         g_sRR * bilinear_grad_wx(gSR_lut, C, ix_R, iy, wy) +
                         g_dRF * bilinear_grad_wx(gDF, C, ix_R, iy, wy) +
                         g_sRF * bilinear_grad_wx(gSF_lut, C, ix_R, iy, wy)) * inv_dx_R;
        atomicAdd(&g_slew[src*2+0], g_sl_R);

        double g_sl_F = (g_dFR * bilinear_grad_wx(gDR, C, ix_F, iy, wy) +
                         g_sFR * bilinear_grad_wx(gSR_lut, C, ix_F, iy, wy) +
                         g_dFF * bilinear_grad_wx(gDF, C, ix_F, iy, wy) +
                         g_sFF * bilinear_grad_wx(gSF_lut, C, ix_F, iy, wy)) * inv_dx_F;
        atomicAdd(&g_slew[src*2+1], g_sl_F);

        // Load Cap 역전파
        g_cap_acc += (g_dRR * bilinear_grad_wy(gDR, C, ix_R, iy, wx_R) +
                      g_sRR * bilinear_grad_wy(gSR_lut, C, ix_R, iy, wx_R) +
                      g_dFR * bilinear_grad_wy(gDR, C, ix_F, iy, wx_F) +
                      g_sFR * bilinear_grad_wy(gSR_lut, C, ix_F, iy, wx_F) +
                      g_dRF * bilinear_grad_wy(gDF, C, ix_R, iy, wx_R) +
                      g_sRF * bilinear_grad_wy(gSF_lut, C, ix_R, iy, wx_R) +
                      g_dFF * bilinear_grad_wy(gDF, C, ix_F, iy, wx_F) +
                      g_sFF * bilinear_grad_wy(gSF_lut, C, ix_F, iy, wx_F)) * inv_dy;
    }

    g_caps[pin] = g_cap_acc;
}

// -----------------------------------------------------------------------------
// Launcher functions
// -----------------------------------------------------------------------------

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
    int N, int max_fanin, int S, int C, double gamma
) {
    if (N <= 0) return;
    int blocks = (N + BLOCK - 1) / BLOCK;
    cell_prop_fwd_kernel<<<blocks, BLOCK>>>(
        level_pins, src_matrix, arc_id_matrix, fanin_mask,
        AT, slew, pin_load_caps,
        lut_slew_axis, lut_cap_axis,
        lut_cell_rise, lut_cell_fall, lut_rise_tran, lut_fall_tran,
        lut_sense,
        out_AT, out_slew,
        sv_wx_R, sv_wx_F, sv_wy,
        sv_ix_R, sv_ix_F, sv_iy,
        sv_relu, sv_lse_w,
        N, max_fanin, S, C, gamma, 1.0 / gamma);
}

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
    int N, int max_fanin, int S, int C, double gamma
) {
    if (N <= 0) return;
    int blocks = (N + BLOCK - 1) / BLOCK;
    cell_prop_bwd_kernel<<<blocks, BLOCK>>>(
        level_pins, src_matrix, arc_id_matrix, fanin_mask,
        g_out_AT, g_out_slew,
        lut_slew_axis, lut_cap_axis,
        lut_cell_rise, lut_cell_fall, lut_rise_tran, lut_fall_tran,
        lut_sense,
        sv_wx_R, sv_wx_F, sv_wy,
        sv_ix_R, sv_ix_F, sv_iy,
        sv_relu, sv_lse_w,
        g_AT, g_slew, g_caps,
        N, max_fanin, S, C, 1.0 / gamma);
}
