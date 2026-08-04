import Foundation
import MLX
import MLXFast

// Certified two-pass lm_head elision for the final-token projection (notes/68).

private let lagunaLmHeadPruneVocab = 100_352
private let lagunaLmHeadPruneHidden = 2048

/// Master switch for the certified two-pass final-row lm_head (notes/68).
let lagunaLmHeadPruneEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LM_HEAD_PRUNE"] != "0"

/// Same-binary A/B switch for applying the certified pruner to prefill's
let lagunaLmHeadPrunePrefillEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_LM_HEAD_PRUNE_PREFILL"] != "0"

/// Inline candidate testing for the certified exact pass. The exact kernel
private let lagunaLmHeadInlineMaskEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_INLINE_MASK"] != "0"


/// v5 coarse copy (exp-hybridcoarse section 7): planar-packed symmetric int5
private let lagunaLmHeadCoarseV5Enabled =
    // DEFAULT ON (2026-08-01): int5 planar 1344 B/row + exact-winner
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_COARSE_V5"] != "0"

private let lagunaLmHeadV5StatsEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_V5_STATS"] == "1"

/// Tight v5 assembly threshold: use the BF16 predecessor of the exact coarse-
private let lagunaLmHeadBF16PredecessorThresholdEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_LMHEAD_BF16_PREDECESSOR_THRESHOLD"] != "0"

/// Refine the predecessor threshold to the exact FP32 midpoint between the
private let lagunaLmHeadBF16MidpointThresholdEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_LMHEAD_BF16_MIDPOINT_THRESHOLD"] != "0"

private let lagunaTraceFusionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_TRACE_FUSION"] == "1"

/// Replace the per-element `m = sum |x|*|what|` accumulation with its exact
private let lagunaLmHeadRatioBoundEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_RATIO_BOUND"] != "0"

/// Narrow the certified bound's round-trip buffer from FP32 to BF16, rounding
private let lagunaLmHeadDeltaBF16Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_DELTA_BF16"] != "0"

/// Precompute the 64 activation-group L1 sums once, then reuse them across
private let lagunaLmHeadPrecomputeAbsGroupsEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_LMHEAD_PRECOMPUTE_ABS_GROUPS"] != "0"

private let lagunaLmHeadPruneHeader = """
    // e4m3fn decode, identical to fp8.h:32-38 (half bit pattern (b&127)<<7,
    // times 256, sign from bit 7). Exact in half/float for all 256 codes.
    static inline float laguna_e4m3_decode(uint8_t b) {
        half converted = as_type<half>(ushort(uint(b & 127u) << 7));
        converted = converted * (half)256.0f;
        return (b & 128u) ? -float(converted) : float(converted);
    }

    // e8m0 decode, identical to fp8.h:70-77 (bits<<7 as bf16; bits==0 ->
    // 0x40 as bf16 = 2^-127). Exponent-bit construction, exact.
    static inline float laguna_e8m0_decode(uint8_t b) {
        if (b == 0u) {
            return as_type<float>(0x00400000u);  // 2^-127
        }
        return as_type<float>(uint(b) << 23);
    }

    // Certified |ratio - code| bound for an e4m3 element: half the enclosing
    // RNE cell (denormal half-ulp 2^-10; normal half-ulp 2^(e-11)), except the
    // saturated top code 0x7E whose cell is open: the e8m0 scale may round
    // down by up to a factor 2^0.5, so ratio <= 448*2^0.5 and the bound is
    // 448*(2^0.5-1) = 185.6, rounded up to 186.
    //
    // Max-form: the denormal branch 2^-10 equals 2^(1-11), so both non-top
    // cases collapse to 2^(max(e,1)-11) -- identical float for all 256 codes
    // to the original three-branch form (e==0 -> 2^-10; e>0 -> 2^(e-11)).
    static inline float laguna_hs8(uint8_t b) {
        uint mag = uint(b) & 127u;
        uint e = mag >> 3;
        float h = as_type<float>((metal::max(e, 1u) + 116u) << 23);  // 2^(max(e,1)-11)
        return (mag == 126u) ? 186.0f : h;
    }

    // Bit-parallel e4m3 decode of one packed word (4 codes) into 4 floats.
    // Per byte b the half bit pattern is sign<<15 | (b&127)<<7, i.e. exactly
    // fp8.h's (b&127)<<7 construction with the sign applied as the half sign
    // bit instead of a post-float negate. IEEE multiply is sign-magnitude
    // symmetric, so (sign-packed half)*256h == sign*((b&127)-half * 256h)
    // bit-for-bit for every code, including -0 (code 0x80). The four decoded
    // floats are byte-order b0,b1,b2,b3 in out.x,out.y,out.z,out.w.
    static inline float4 laguna_e4m3_decode4(uint w) {
        uint lo = ((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8);
        uint hs = w >> 8;
        uint hi = ((hs & 0x007F007Fu) << 7) | ((hs & 0x00800080u) << 8);
        half2 h02 = as_type<half2>(lo) * half2((half)256.0f);
        half2 h13 = as_type<half2>(hi) * half2((half)256.0f);
        return float4(float(h02.x), float(h13.x), float(h02.y), float(h13.y));
    }
    """

/// Fused MXFP8 coarse GEMV + certified bound + BF16 pre-fill.
private let lagunaLmHeadCoarseKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_coarse_pack16_v3",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta", "coarse_bf"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* crow = codes + size_t(row) * 2048;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        float m_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            const device uint4* cptr = (const device uint4*)(crow + g * 32);
            uint4 packed0 = cptr[0];
            uint4 packed1 = cptr[1];
            float cg = 0.0f;
            float dg = 0.0f;
            float mg = 0.0f;
            const device ushort4* xrow = (const device ushort4*)(x + g * 32);
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 8; ++w) {
                uint word = (w < 4u) ? packed0[w & 3u] : packed1[w & 3u];
                float4 cv4 = laguna_e4m3_decode4(word);
                // bf16 -> f32 is exactly bits<<16 for every value class.
                float4 xv4 = as_type<float4>(uint4(xrow[w]) << 16);
                float4 ax4 = metal::abs(xv4);
                uint4 b4 = (uint4(word) >> uint4(0u, 8u, 16u, 24u)) & 255u;
                uint4 mag4 = b4 & 127u;
                uint4 e4 = mag4 >> 3;
                float4 hsf = as_type<float4>((metal::max(e4, uint4(1u)) + 116u) << 23);
                float4 hs4 = metal::select(hsf, float4(186.0f), mag4 == 126u);
                float4 acv4 = metal::abs(cv4);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    float cv = cv4[k];
                    float xv = xv4[k];
                    float ax = ax4[k];
                    cg += xv * cv;
                    dg += ax * hs4[k];
                    mg += ax * acv4[k];
                }
            }
            c_acc += sd * cg;
            d_acc += sd * dg;
            m_acc += sd * mg;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        m_acc = simd_sum(m_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            delta[row] = d_acc * (1.0f + GAMMA) + (2.0f * GAMMA) * m_acc;
            coarse_bf[row] = bfloat(c_acc);
        }
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

/// Same MXFP8 coarse logits and quantization-error sum as the pack16 kernel
private let lagunaLmHeadCoarseRatioBoundKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_coarse_ratio_bound_pack16_v1",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta", "coarse_bf"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* crow = codes + size_t(row) * 2048;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            const device uint4* cptr = (const device uint4*)(crow + g * 32);
            uint4 packed0 = cptr[0];
            uint4 packed1 = cptr[1];
            float cg = 0.0f;
            float dg = 0.0f;
            const device ushort4* xrow = (const device ushort4*)(x + g * 32);
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 8; ++w) {
                uint word = (w < 4u) ? packed0[w & 3u] : packed1[w & 3u];
                float4 cv4 = laguna_e4m3_decode4(word);
                float4 xv4 = as_type<float4>(uint4(xrow[w]) << 16);
                float4 ax4 = metal::abs(xv4);
                uint4 b4 = (uint4(word) >> uint4(0u, 8u, 16u, 24u)) & 255u;
                uint4 mag4 = b4 & 127u;
                uint4 e4 = mag4 >> 3;
                float4 hsf =
                    as_type<float4>((metal::max(e4, uint4(1u)) + 116u) << 23);
                float4 hs4 =
                    metal::select(hsf, float4(186.0f), mag4 == 126u);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    cg += xv4[k] * cv4[k];
                    dg += ax4[k] * hs4[k];
                }
            }
            c_acc += sd * cg;
            d_acc += sd * dg;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            delta[row] = d_acc * (1.0f + 61.0f * GAMMA);
            coarse_bf[row] = bfloat(c_acc);
        }
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

/// v1 coarse kernel, kept verbatim for same-binary A/B (the paired
private let lagunaLmHeadCoarseKernelV1 = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_coarse_v1",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta", "coarse_bf"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 8 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* crow = codes + size_t(row) * 2048;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        float m_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            const device uint4* cptr = (const device uint4*)(crow + g * 32);
            uint4 packed0 = cptr[0];
            uint4 packed1 = cptr[1];
            float cg = 0.0f;
            float dg = 0.0f;
            float mg = 0.0f;
            for (uint j = 0; j < 32; ++j) {
                uint word = (j < 16) ? packed0[j / 4] : packed1[(j - 16) / 4];
                uint8_t b = uint8_t(word >> (8 * (j % 4)));
                float cv = laguna_e4m3_decode(b);
                float xv = float(x[g * 32 + j]);
                float ax = metal::abs(xv);
                cg += xv * cv;
                dg += ax * laguna_hs8(b);
                mg += ax * metal::abs(cv);
            }
            c_acc += sd * cg;
            d_acc += sd * dg;
            m_acc += sd * mg;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        m_acc = simd_sum(m_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            delta[row] = d_acc * (1.0f + GAMMA) + (2.0f * GAMMA) * m_acc;
            coarse_bf[row] = bfloat(c_acc);
        }
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

/// Two-output coarse kernels for the inline-mask path. Their accumulation and
private let lagunaLmHeadInlineCoarseKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_inline_coarse_pack16_v3",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* crow = codes + size_t(row) * 2048;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        float m_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            const device uint4* cptr = (const device uint4*)(crow + g * 32);
            uint4 packed0 = cptr[0];
            uint4 packed1 = cptr[1];
            float cg = 0.0f;
            float dg = 0.0f;
            float mg = 0.0f;
            const device ushort4* xrow = (const device ushort4*)(x + g * 32);
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 8; ++w) {
                uint word = (w < 4u) ? packed0[w & 3u] : packed1[w & 3u];
                float4 cv4 = laguna_e4m3_decode4(word);
                // bf16 -> f32 is exactly bits<<16 for every value class.
                float4 xv4 = as_type<float4>(uint4(xrow[w]) << 16);
                float4 ax4 = metal::abs(xv4);
                uint4 b4 = (uint4(word) >> uint4(0u, 8u, 16u, 24u)) & 255u;
                uint4 mag4 = b4 & 127u;
                uint4 e4 = mag4 >> 3;
                float4 hsf = as_type<float4>((metal::max(e4, uint4(1u)) + 116u) << 23);
                float4 hs4 = metal::select(hsf, float4(186.0f), mag4 == 126u);
                float4 acv4 = metal::abs(cv4);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    float cv = cv4[k];
                    float xv = xv4[k];
                    float ax = ax4[k];
                    cg += xv * cv;
                    dg += ax * hs4[k];
                    mg += ax * acv4[k];
                }
            }
            c_acc += sd * cg;
            d_acc += sd * dg;
            m_acc += sd * mg;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        m_acc = simd_sum(m_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            delta[row] = d_acc * (1.0f + GAMMA) + (2.0f * GAMMA) * m_acc;
        }
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

/// Same MXFP8 coarse logits and quantization-error sum as the pack16 inline
private let lagunaLmHeadInlineCoarseRatioBoundKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_inline_coarse_ratio_bound_pack16_v1",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* crow = codes + size_t(row) * 2048;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            const device uint4* cptr = (const device uint4*)(crow + g * 32);
            uint4 packed0 = cptr[0];
            uint4 packed1 = cptr[1];
            float cg = 0.0f;
            float dg = 0.0f;
            const device ushort4* xrow = (const device ushort4*)(x + g * 32);
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 8; ++w) {
                uint word = (w < 4u) ? packed0[w & 3u] : packed1[w & 3u];
                float4 cv4 = laguna_e4m3_decode4(word);
                float4 xv4 = as_type<float4>(uint4(xrow[w]) << 16);
                float4 ax4 = metal::abs(xv4);
                uint4 b4 = (uint4(word) >> uint4(0u, 8u, 16u, 24u)) & 255u;
                uint4 mag4 = b4 & 127u;
                uint4 e4 = mag4 >> 3;
                float4 hsf =
                    as_type<float4>((metal::max(e4, uint4(1u)) + 116u) << 23);
                float4 hs4 =
                    metal::select(hsf, float4(186.0f), mag4 == 126u);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    cg += xv4[k] * cv4[k];
                    dg += ax4[k] * hs4[k];
                }
            }
            c_acc += sd * cg;
            d_acc += sd * dg;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            delta[row] = d_acc * (1.0f + 61.0f * GAMMA);
        }
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

/// The shipped default: the ratio-bound inline coarse kernel with `delta`
private let lagunaLmHeadInlineCoarseRatioBoundDeltaBF16Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_inline_coarse_ratio_bound_delta_bf16_pack16_v1",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* crow = codes + size_t(row) * 2048;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            const device uint4* cptr = (const device uint4*)(crow + g * 32);
            uint4 packed0 = cptr[0];
            uint4 packed1 = cptr[1];
            float cg = 0.0f;
            float dg = 0.0f;
            const device ushort4* xrow = (const device ushort4*)(x + g * 32);
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 8; ++w) {
                uint word = (w < 4u) ? packed0[w & 3u] : packed1[w & 3u];
                float4 cv4 = laguna_e4m3_decode4(word);
                float4 xv4 = as_type<float4>(uint4(xrow[w]) << 16);
                float4 ax4 = metal::abs(xv4);
                uint4 b4 = (uint4(word) >> uint4(0u, 8u, 16u, 24u)) & 255u;
                uint4 mag4 = b4 & 127u;
                uint4 e4 = mag4 >> 3;
                float4 hsf =
                    as_type<float4>((metal::max(e4, uint4(1u)) + 116u) << 23);
                float4 hs4 =
                    metal::select(hsf, float4(186.0f), mag4 == 126u);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    cg += xv4[k] * cv4[k];
                    dg += ax4[k] * hs4[k];
                }
            }
            c_acc += sd * cg;
            d_acc += sd * dg;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            // Same FP32 bound as the kernel above, then rounded UP to BF16.
            float d_up = d_acc * (1.0f + 61.0f * GAMMA);
            uint dbits = as_type<uint>(d_up);
            uint dtrunc = dbits & 0xFFFF0000u;
            if (dtrunc != dbits) {
                dtrunc += 0x00010000u;
            }
            delta[row] = as_type<bfloat>(ushort(dtrunc >> 16));
        }
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

private let lagunaLmHeadInlineCoarseKernelV1 = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_inline_coarse_v1",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 8 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* crow = codes + size_t(row) * 2048;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        float m_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            const device uint4* cptr = (const device uint4*)(crow + g * 32);
            uint4 packed0 = cptr[0];
            uint4 packed1 = cptr[1];
            float cg = 0.0f;
            float dg = 0.0f;
            float mg = 0.0f;
            for (uint j = 0; j < 32; ++j) {
                uint word = (j < 16) ? packed0[j / 4] : packed1[(j - 16) / 4];
                uint8_t b = uint8_t(word >> (8 * (j % 4)));
                float cv = laguna_e4m3_decode(b);
                float xv = float(x[g * 32 + j]);
                float ax = metal::abs(xv);
                cg += xv * cv;
                dg += ax * laguna_hs8(b);
                mg += ax * metal::abs(cv);
            }
            c_acc += sd * cg;
            d_acc += sd * dg;
            m_acc += sd * mg;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        m_acc = simd_sum(m_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            delta[row] = d_acc * (1.0f + GAMMA) + (2.0f * GAMMA) * m_acc;
        }
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

/// One thread per 32-element activation group. Each thread reproduces the
private let lagunaLmHeadAbsGroupSumsKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_abs_group_sums_v1",
    inputNames: ["x"],
    outputNames: ["abs_group_sums"],
    source: """
        uint g = thread_position_in_grid.x;
        const device ushort4* xrow =
            (const device ushort4*)(x + g * 32);
        float ag = 0.0f;
        #pragma clang loop unroll(full)
        for (uint w = 0; w < 4; ++w) {
            // bf16 -> f32 is exactly bits<<16 for every value class.
            float4 xa = as_type<float4>(uint4(xrow[2 * w]) << 16);
            float4 xb = as_type<float4>(uint4(xrow[2 * w + 1]) << 16);
            float4 xe = float4(xa.x, xa.z, xb.x, xb.z);
            float4 xo = float4(xa.y, xa.w, xb.y, xb.w);
            float4 axe = metal::abs(xe);
            float4 axo = metal::abs(xo);
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 4; ++k) {
                ag += axe[k];
                ag += axo[k];
            }
        }
        abs_group_sums[g] = ag;
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

/// Same launch geometry as the v4 int6 kernels (16 rows/threadgroup, one
private let lagunaLmHeadInt5CoarseRatioBoundDeltaBF16Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_int5_inline_coarse_ratio_bound_delta_bf16_v5",
    inputNames: ["x", "codes_lo", "codes_hi", "scales"],
    outputNames: ["coarse", "delta"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* lorow = codes_lo + size_t(row) * 1024;
        const device uint8_t* hirow = codes_hi + size_t(row) * 256;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            uint4 lo4 = ((const device uint4*)(lorow + g * 16))[0];
            uint hb = ((const device uint*)(hirow + g * 4))[0];
            const device ushort4* xrow = (const device ushort4*)(x + g * 32);
            float cg = 0.0f;
            float ag = 0.0f;
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 4; ++w) {
                // Word w: elements 8w..8w+7 of the group. Nibble plane byte
                // b holds elements 2b (low) / 2b+1 (high); 1-bit plane bit j
                // of the group's word holds element j's bit 4.
                uint lw = lo4[w];
                uint hw = hb >> (8u * w);
                uint4 ne = (uint4(lw) >> uint4(0u, 8u, 16u, 24u)) & 15u;
                uint4 no = (uint4(lw) >> uint4(4u, 12u, 20u, 28u)) & 15u;
                uint4 he = (uint4(hw) >> uint4(0u, 2u, 4u, 6u)) & 1u;
                uint4 ho = (uint4(hw) >> uint4(1u, 3u, 5u, 7u)) & 1u;
                // Offset-binary decode: value = u - 16 in [-15, 15], exact.
                float4 ve = float4(ne | (he << 4u)) - 16.0f;
                float4 vo = float4(no | (ho << 4u)) - 16.0f;
                // bf16 -> f32 is exactly bits<<16 for every value class.
                float4 xa = as_type<float4>(uint4(xrow[2 * w]) << 16);
                float4 xb = as_type<float4>(uint4(xrow[2 * w + 1]) << 16);
                float4 xe = float4(xa.x, xa.z, xb.x, xb.z);
                float4 xo = float4(xa.y, xa.w, xb.y, xb.w);
                float4 axe = metal::abs(xe);
                float4 axo = metal::abs(xo);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    cg += xe[k] * ve[k];
                    cg += xo[k] * vo[k];
                    ag += axe[k];
                    ag += axo[k];
                }
            }
            c_acc += sd * cg;
            d_acc += (0.5f * sd) * ag;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            // FP32 bound, then rounded UP to BF16 (mask-and-bump, sign clear).
            float d_up = d_acc * (1.0f + 61.0f * GAMMA);
            uint dbits = as_type<uint>(d_up);
            uint dtrunc = dbits & 0xFFFF0000u;
            if (dtrunc != dbits) {
                dtrunc += 0x00010000u;
            }
            delta[row] = as_type<bfloat>(ushort(dtrunc >> 16));
        }
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

/// Same-binary A/B selector for the coarse kernel (v2 default).
private let lagunaLmHeadCoarseUseV1 =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_COARSE"] == "v1"

/// `lower.max()` uses MLX's two-pass `all_reduce_max` for this 100352-element
private let lagunaLmHeadLowerMaxHeader = """
    static inline float laguna_lmhead_max_pair(float a, float b) {
        if (metal::isnan(a) || metal::isnan(b)) {
            return NAN;
        }
        return a > b ? a : b;
    }

    static inline float laguna_lmhead_simd_max(float value) {
        if (simd_any(value != value)) {
            return NAN;
        }
        return simd_max(value);
    }
    """

/// Pass one of the fused lower-bound reduction. Its launch shape and read
private let lagunaLmHeadLowerMaxStage1Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_lower_max_stage1_v1",
    inputNames: ["coarse", "delta"],
    outputNames: ["partial_max"],
    source: """
        constexpr uint ROW_SIZE = 784;
        constexpr uint READS = 4;
        constexpr uint ACTIVE_THREADS = ROW_SIZE / READS;
        constexpr uint SIMD_GROUPS = 7;

        uint row = threadgroup_position_in_grid.y;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        threadgroup float shared_vals[32];

        float total = -metal::numeric_limits<float>::infinity();
        if (lid < ACTIVE_THREADS) {
            uint base = row * ROW_SIZE + lid * READS;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < READS; ++i) {
                float lower = coarse[base + i] - delta[base + i];
                total = laguna_lmhead_max_pair(lower, total);
            }
        }

        total = laguna_lmhead_simd_max(total);
        if (simd_lane == 0) {
            shared_vals[simd_group] = total;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        total = lid < SIMD_GROUPS
            ? shared_vals[lid]
            : -metal::numeric_limits<float>::infinity();
        total = laguna_lmhead_simd_max(total);
        if (lid == 0) {
            partial_max[row] = total;
        }
        """,
    header: lagunaLmHeadLowerMaxHeader,
    ensureRowContiguous: true
)

/// Stage one over a BF16 `delta`. Textually the kernel above with the single
private let lagunaLmHeadLowerMaxStage1DeltaBF16Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_lower_max_stage1_delta_bf16_v1",
    inputNames: ["coarse", "delta"],
    outputNames: ["partial_max"],
    source: """
        constexpr uint ROW_SIZE = 784;
        constexpr uint READS = 4;
        constexpr uint ACTIVE_THREADS = ROW_SIZE / READS;
        constexpr uint SIMD_GROUPS = 7;

        uint row = threadgroup_position_in_grid.y;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        threadgroup float shared_vals[32];

        float total = -metal::numeric_limits<float>::infinity();
        if (lid < ACTIVE_THREADS) {
            uint base = row * ROW_SIZE + lid * READS;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < READS; ++i) {
                float lower = coarse[base + i] - float(delta[base + i]);
                total = laguna_lmhead_max_pair(lower, total);
            }
        }

        total = laguna_lmhead_simd_max(total);
        if (simd_lane == 0) {
            shared_vals[simd_group] = total;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        total = lid < SIMD_GROUPS
            ? shared_vals[lid]
            : -metal::numeric_limits<float>::infinity();
        total = laguna_lmhead_simd_max(total);
        if (lid == 0) {
            partial_max[row] = total;
        }
        """,
    header: lagunaLmHeadLowerMaxHeader,
    ensureRowContiguous: true
)

/// Pass two reduces the 128 partials with the same four-values-per-lane
private let lagunaLmHeadLowerMaxThresholdKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_lower_max_threshold_v1",
    inputNames: ["partial_max"],
    outputNames: ["threshold"],
    source: """
        constexpr uint READS = 4;
        uint lid = thread_position_in_threadgroup.x;
        threadgroup float rounded_beta[1];

        float total = -metal::numeric_limits<float>::infinity();
        uint base = lid * READS;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < READS; ++i) {
            total = laguna_lmhead_max_pair(partial_max[base + i], total);
        }
        total = laguna_lmhead_simd_max(total);

        if (lid == 0) {
            rounded_beta[0] = metal::abs(total) * 0x1p-6f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            threshold[0] = total - rounded_beta[0];
        }
        """,
    header: lagunaLmHeadLowerMaxHeader,
    ensureRowContiguous: true
)

/// v5 stage one: partial ARGMAX of `coarse` (value + index), replacing the
private let lagunaLmHeadCoarseArgmaxStage1Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_coarse_argmax_stage1_v5",
    inputNames: ["coarse"],
    outputNames: ["partial_max", "partial_idx"],
    source: """
        constexpr uint ROW_SIZE = 784;
        constexpr uint READS = 4;
        constexpr uint ACTIVE_THREADS = ROW_SIZE / READS;
        constexpr uint SIMD_GROUPS = 7;

        uint row = threadgroup_position_in_grid.y;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        threadgroup float shared_vals[32];
        threadgroup uint shared_idxs[32];

        float best = -metal::numeric_limits<float>::infinity();
        uint best_idx = 0xFFFFFFFFu;
        if (lid < ACTIVE_THREADS) {
            uint base = row * ROW_SIZE + lid * READS;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < READS; ++i) {
                float v = coarse[base + i];
                if (v > best || (v == best && base + i < best_idx)) {
                    best = v;
                    best_idx = base + i;
                }
            }
        }

        #pragma clang loop unroll(full)
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            float ov = simd_shuffle_down(best, sn);
            uint oi = simd_shuffle_down(best_idx, sn);
            if (ov > best || (ov == best && oi < best_idx)) {
                best = ov;
                best_idx = oi;
            }
        }
        if (simd_lane == 0) {
            shared_vals[simd_group] = best;
            shared_idxs[simd_group] = best_idx;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        best = lid < SIMD_GROUPS
            ? shared_vals[lid]
            : -metal::numeric_limits<float>::infinity();
        best_idx = lid < SIMD_GROUPS ? shared_idxs[lid] : 0xFFFFFFFFu;
        #pragma clang loop unroll(full)
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            float ov = simd_shuffle_down(best, sn);
            uint oi = simd_shuffle_down(best_idx, sn);
            if (ov > best || (ov == best && oi < best_idx)) {
                best = ov;
                best_idx = oi;
            }
        }
        if (lid == 0) {
            partial_max[row] = best;
            partial_idx[row] = best_idx;
        }
        """,
    ensureRowContiguous: true
)

/// v5 pass two: EXACT-WINNER threshold. Reduces the 128 stage-one partials
private let lagunaLmHeadExactWinnerThresholdKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_winner_threshold_v5",
    inputNames: ["partial_max", "partial_idx", "lm_head", "x"],
    outputNames: ["threshold"],
    source: """
        constexpr uint VOCAB = 100352;
        constexpr uint K = 2048;
        constexpr uint READS = 4;
        uint lid = thread_position_in_threadgroup.x;
        threadgroup float rounded_beta[1];
        threadgroup uint winner_row[1];

        // Final argmax over the 128 partials, four per lane in the stock
        // second-pass read order; value primary, lowest index on ties.
        float best = -metal::numeric_limits<float>::infinity();
        uint best_idx = 0xFFFFFFFFu;
        uint base = lid * READS;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < READS; ++i) {
            float v = partial_max[base + i];
            uint idx = partial_idx[base + i];
            if (v > best || (v == best && idx < best_idx)) {
                best = v;
                best_idx = idx;
            }
        }
        #pragma clang loop unroll(full)
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            float ov = simd_shuffle_down(best, sn);
            uint oi = simd_shuffle_down(best_idx, sn);
            if (ov > best || (ov == best && oi < best_idx)) {
                best = ov;
                best_idx = oi;
            }
        }
        if (lid == 0) {
            winner_row[0] = metal::min(best_idx, uint(VOCAB - 1));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r = winner_row[0];

        // --- stock gemv_al replica begin (single row r; gemv.h:151-289) ---
        float result = 0.0f;
        thread bfloat inter[4];
        thread float v_coeff[4];
        uint bn = lid * 4;
        const device bfloat* mrow = lm_head + size_t(r) * K;
        for (uint i = 0; i < 16; ++i) {
            vec<bfloat, 4> xv =
                *((const device vec<bfloat, 4>*)(x + bn));
            v_coeff[0] = float(xv.x);
            v_coeff[1] = float(xv.y);
            v_coeff[2] = float(xv.z);
            v_coeff[3] = float(xv.w);
            vec<bfloat, 4> mv =
                *((const device vec<bfloat, 4>*)(mrow + bn));
            inter[0] = mv.x;
            inter[1] = mv.y;
            inter[2] = mv.z;
            inter[3] = mv.w;
            result += inter[0] * v_coeff[0];
            result += inter[1] * v_coeff[1];
            result += inter[2] * v_coeff[2];
            result += inter[3] * v_coeff[3];
            bn += 128;
        }
        #pragma unroll
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            result += simd_shuffle_down(result, sn);
        }
        // --- stock gemv_al replica end ---
        if (lid == 0) {
            rounded_beta[0] = metal::abs(result) * 0x1p-6f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            threshold[0] = result - rounded_beta[0];
        }
        """,
    ensureRowContiguous: true
)

/// Tight exact-winner threshold. The argmax reduction and single-row stock
private let lagunaLmHeadExactWinnerBF16PredecessorThresholdKernel = MLXFast.metalKernel(
    name: lagunaLmHeadBF16MidpointThresholdEnabled
        ? "laguna_lmhead_exact_winner_bf16_midpoint_threshold_v1"
        : "laguna_lmhead_exact_winner_bf16_predecessor_threshold_v1",
    inputNames: ["partial_max", "partial_idx", "lm_head", "x"],
    outputNames: ["threshold"],
    source: """
        constexpr uint VOCAB = 100352;
        constexpr uint K = 2048;
        constexpr uint READS = 4;
        uint lid = thread_position_in_threadgroup.x;
        threadgroup uint winner_row[1];

        // Verbatim final argmax over the retained 128 partials.
        float best = -metal::numeric_limits<float>::infinity();
        uint best_idx = 0xFFFFFFFFu;
        uint base = lid * READS;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < READS; ++i) {
            float v = partial_max[base + i];
            uint idx = partial_idx[base + i];
            if (v > best || (v == best && idx < best_idx)) {
                best = v;
                best_idx = idx;
            }
        }
        #pragma clang loop unroll(full)
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            float ov = simd_shuffle_down(best, sn);
            uint oi = simd_shuffle_down(best_idx, sn);
            if (ov > best || (ov == best && oi < best_idx)) {
                best = ov;
                best_idx = oi;
            }
        }
        if (lid == 0) {
            winner_row[0] = metal::min(best_idx, uint(VOCAB - 1));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r = winner_row[0];

        // --- stock gemv_al replica begin (single row r; gemv.h:151-289) ---
        float result = 0.0f;
        thread bfloat inter[4];
        thread float v_coeff[4];
        uint bn = lid * 4;
        const device bfloat* mrow = lm_head + size_t(r) * K;
        for (uint i = 0; i < 16; ++i) {
            vec<bfloat, 4> xv =
                *((const device vec<bfloat, 4>*)(x + bn));
            v_coeff[0] = float(xv.x);
            v_coeff[1] = float(xv.y);
            v_coeff[2] = float(xv.z);
            v_coeff[3] = float(xv.w);
            vec<bfloat, 4> mv =
                *((const device vec<bfloat, 4>*)(mrow + bn));
            inter[0] = mv.x;
            inter[1] = mv.y;
            inter[2] = mv.z;
            inter[3] = mv.w;
            result += inter[0] * v_coeff[0];
            result += inter[1] * v_coeff[1];
            result += inter[2] * v_coeff[2];
            result += inter[3] * v_coeff[3];
            bn += 128;
        }
        #pragma unroll
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            result += simd_shuffle_down(result, sn);
        }
        // --- stock gemv_al replica end ---
        if (lid == 0) {
            bfloat rounded = bfloat(result);
            // Expand through the numeric BF16->FP32 conversion, whose bits are
            // exactly `bf16_bits << 16`; do not reinterpret the Metal wrapper.
            ushort bits = ushort(as_type<uint>(float(rounded)) >> 16);
            ushort magnitude = bits & 0x7FFFu;
            ushort predecessor_bits;
            if (magnitude == 0u) {
                predecessor_bits = 0x8001u;  // predecessor of either zero
            } else if ((bits & 0x8000u) == 0u) {
                predecessor_bits = bits - 1u;
            } else {
                predecessor_bits = bits + 1u;
            }
            float predecessor =
                as_type<float>(uint(predecessor_bits) << 16);
            if (\(lagunaLmHeadBF16MidpointThresholdEnabled ? "true" : "false")) {
                float rounded_value = as_type<float>(uint(bits) << 16);
                threshold[0] = predecessor +
                    (rounded_value - predecessor) * 0.5f;
            } else {
                threshold[0] = predecessor;
            }
        }
        """,
    ensureRowContiguous: true
)

/// GPU candidate marking: one byte per vocabulary row, set when the row's
private let lagunaLmHeadSelectKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_select_v2",
    inputNames: ["coarse", "delta", "thr"],
    outputNames: ["is_cand"],
    source: """
        uint i = thread_position_in_grid.x;
        is_cand[i] = (coarse[i] + delta[i] >= thr[0]) ? uint8_t(1) : uint8_t(0);
        """,
    ensureRowContiguous: true
)

/// Exact pass. Each simdgroup owns a FIXED block of four output rows -- the
private let lagunaLmHeadExactKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_block_v2",
    inputNames: ["coarse_bf", "lm_head", "x", "is_cand"],
    outputNames: ["assembled"],
    source: """
        constexpr uint VOCAB = 100352;
        constexpr uint K = 2048;

        uint tgid = threadgroup_position_in_grid.x;
        uint sgid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        // This simdgroup's fixed four output rows. VOCAB is 3136 * 32, so the
        // grid tiles it exactly; the bounds test is belt-and-braces.
        uint base = tgid * 32 + sgid * 4;

        // Simdgroup-uniform: every lane reads the same four mask bytes.
        bool any_candidate = false;
        #pragma unroll
        for (uint tm = 0; tm < 4; ++tm) {
            uint r = base + tm;
            any_candidate = any_candidate || (r < VOCAB && is_cand[r] != 0);
        }

        if (!any_candidate) {
            if (lane < 4 && base + lane < VOCAB) {
                assembled[base + lane] = coarse_bf[base + lane];
            }
            return;
        }

        // --- stock gemv_al replica begin (gemv.h:151-289) ---
        thread float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        thread bfloat inter[4];
        thread float v_coeff[4];
        uint bn = lane * 4;
        for (uint i = 0; i < 16; ++i) {
            vec<bfloat, 4> xv =
                *((const device vec<bfloat, 4>*)(x + bn));
            v_coeff[0] = float(xv.x);
            v_coeff[1] = float(xv.y);
            v_coeff[2] = float(xv.z);
            v_coeff[3] = float(xv.w);
            #pragma unroll
            for (uint tm = 0; tm < 4; ++tm) {
                const device bfloat* mrow = lm_head + size_t(base + tm) * K;
                vec<bfloat, 4> mv =
                    *((const device vec<bfloat, 4>*)(mrow + bn));
                inter[0] = mv.x;
                inter[1] = mv.y;
                inter[2] = mv.z;
                inter[3] = mv.w;
                result[tm] += inter[0] * v_coeff[0];
                result[tm] += inter[1] * v_coeff[1];
                result[tm] += inter[2] * v_coeff[2];
                result[tm] += inter[3] * v_coeff[3];
            }
            bn += 128;
        }
        #pragma unroll
        for (uint tm = 0; tm < 4; ++tm) {
            #pragma unroll
            for (ushort sn = 16; sn >= 1; sn >>= 1) {
                result[tm] += simd_shuffle_down(result[tm], sn);
            }
        }
        // --- stock gemv_al replica end ---
        if (lane == 0) {
            #pragma unroll
            for (uint tm = 0; tm < 4; ++tm) {
                uint r = base + tm;
                if (r < VOCAB) {
                    assembled[r] = (is_cand[r] != 0)
                        ? bfloat(result[tm])
                        : coarse_bf[r];
                }
            }
        }
        """,
    ensureRowContiguous: true
)

/// Default exact pass with candidate testing inlined. The membership
private let lagunaLmHeadInlineExactKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_inline_mask_block_v1",
    inputNames: ["coarse", "delta", "thr", "lm_head", "x"],
    outputNames: ["assembled"],
    source: """
        constexpr uint VOCAB = 100352;
        constexpr uint K = 2048;

        uint tgid = threadgroup_position_in_grid.x;
        uint sgid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        // This simdgroup's fixed four output rows. VOCAB is 3136 * 32, so the
        // grid tiles it exactly; the bounds test is belt-and-braces.
        uint base = tgid * 32 + sgid * 4;

        // Simdgroup-uniform. This is textually the selector's predicate; the
        // fixed row mapping still gives one owner per output slot.
        bool any_candidate = false;
        #pragma unroll
        for (uint tm = 0; tm < 4; ++tm) {
            uint r = base + tm;
            any_candidate = any_candidate ||
                (r < VOCAB && coarse[r] + delta[r] >= thr[0]);
        }

        if (!any_candidate) {
            if (lane < 4 && base + lane < VOCAB) {
                assembled[base + lane] = bfloat(coarse[base + lane]);
            }
            return;
        }

        // --- stock gemv_al replica begin (gemv.h:151-289) ---
        thread float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        thread bfloat inter[4];
        thread float v_coeff[4];
        uint bn = lane * 4;
        for (uint i = 0; i < 16; ++i) {
            vec<bfloat, 4> xv =
                *((const device vec<bfloat, 4>*)(x + bn));
            v_coeff[0] = float(xv.x);
            v_coeff[1] = float(xv.y);
            v_coeff[2] = float(xv.z);
            v_coeff[3] = float(xv.w);
            #pragma unroll
            for (uint tm = 0; tm < 4; ++tm) {
                const device bfloat* mrow = lm_head + size_t(base + tm) * K;
                vec<bfloat, 4> mv =
                    *((const device vec<bfloat, 4>*)(mrow + bn));
                inter[0] = mv.x;
                inter[1] = mv.y;
                inter[2] = mv.z;
                inter[3] = mv.w;
                result[tm] += inter[0] * v_coeff[0];
                result[tm] += inter[1] * v_coeff[1];
                result[tm] += inter[2] * v_coeff[2];
                result[tm] += inter[3] * v_coeff[3];
            }
            bn += 128;
        }
        #pragma unroll
        for (uint tm = 0; tm < 4; ++tm) {
            #pragma unroll
            for (ushort sn = 16; sn >= 1; sn >>= 1) {
                result[tm] += simd_shuffle_down(result[tm], sn);
            }
        }
        // --- stock gemv_al replica end ---
        if (lane == 0) {
            #pragma unroll
            for (uint tm = 0; tm < 4; ++tm) {
                uint r = base + tm;
                if (r < VOCAB) {
                    assembled[r] = (coarse[r] + delta[r] >= thr[0])
                        ? bfloat(result[tm])
                        : bfloat(coarse[r]);
                }
            }
        }
        """,
    ensureRowContiguous: true
)

/// The shipped default: the inline-mask exact pass over a BF16 `delta`.
private let lagunaLmHeadInlineExactDeltaBF16Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_inline_mask_block_delta_bf16_lane0_mask_v1",
    inputNames: ["coarse", "delta", "thr", "lm_head", "x"],
    outputNames: ["assembled"],
    source: """
        constexpr uint VOCAB = 100352;
        constexpr uint K = 2048;

        uint tgid = threadgroup_position_in_grid.x;
        uint sgid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        // This simdgroup's fixed four output rows. VOCAB is 3136 * 32, so the
        // grid tiles it exactly; the bounds test is belt-and-braces.
        uint base = tgid * 32 + sgid * 4;

        // The predicate is simdgroup-uniform, so lane 0 forms it once and
        // broadcasts the four row decisions. Reusing the mask below removes
        // the same coarse/delta/threshold reads from the final write path.
        uint candidate_mask = 0;
        if (lane == 0) {
            #pragma unroll
            for (uint tm = 0; tm < 4; ++tm) {
                uint r = base + tm;
                if (r < VOCAB && coarse[r] + float(delta[r]) >= thr[0]) {
                    candidate_mask |= 1u << tm;
                }
            }
        }
        candidate_mask = simd_broadcast(candidate_mask, 0);

        if (candidate_mask == 0) {
            if (lane < 4 && base + lane < VOCAB) {
                assembled[base + lane] = bfloat(coarse[base + lane]);
            }
            return;
        }

        // --- stock gemv_al replica begin (gemv.h:151-289) ---
        thread float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        thread bfloat inter[4];
        thread float v_coeff[4];
        uint bn = lane * 4;
        for (uint i = 0; i < 16; ++i) {
            vec<bfloat, 4> xv =
                *((const device vec<bfloat, 4>*)(x + bn));
            v_coeff[0] = float(xv.x);
            v_coeff[1] = float(xv.y);
            v_coeff[2] = float(xv.z);
            v_coeff[3] = float(xv.w);
            #pragma unroll
            for (uint tm = 0; tm < 4; ++tm) {
                const device bfloat* mrow = lm_head + size_t(base + tm) * K;
                vec<bfloat, 4> mv =
                    *((const device vec<bfloat, 4>*)(mrow + bn));
                inter[0] = mv.x;
                inter[1] = mv.y;
                inter[2] = mv.z;
                inter[3] = mv.w;
                result[tm] += inter[0] * v_coeff[0];
                result[tm] += inter[1] * v_coeff[1];
                result[tm] += inter[2] * v_coeff[2];
                result[tm] += inter[3] * v_coeff[3];
            }
            bn += 128;
        }
        #pragma unroll
        for (uint tm = 0; tm < 4; ++tm) {
            #pragma unroll
            for (ushort sn = 16; sn >= 1; sn >>= 1) {
                result[tm] += simd_shuffle_down(result[tm], sn);
            }
        }
        // --- stock gemv_al replica end ---
        if (lane == 0) {
            #pragma unroll
            for (uint tm = 0; tm < 4; ++tm) {
                uint r = base + tm;
                if (r < VOCAB) {
                    assembled[r] = (candidate_mask & (1u << tm)) != 0
                        ? bfloat(result[tm])
                        : bfloat(coarse[r]);
                }
            }
        }
        """,
    ensureRowContiguous: true
)

/// Retained init-time MXFP8 coarse copy of lm_head plus the pruned final-row
final class LagunaLmHeadPruner {
    let codes: MLXArray?   // [100352, 2048] uint8 e4m3 elements
    let scales: MLXArray?  // [100352, 64] uint8 e8m0 group scales
    /// v5 planar int5 coarse copy (DARKBLOOM_LMHEAD_COARSE_V5=1): nibble
    let int5CodesLo: MLXArray?
    let int5CodesHi: MLXArray?
    let int5Scales: MLXArray?

    var residentArrays: [MLXArray] {
        if let lo = int5CodesLo, let hi = int5CodesHi, let s5 = int5Scales {
            return [lo, hi, s5]
        }
        return [codes, scales].compactMap { $0 }
    }

    init?(lmHeadWeight: MLXArray) {
        guard lmHeadWeight.shape == [lagunaLmHeadPruneVocab, lagunaLmHeadPruneHidden],
            lmHeadWeight.dtype == .bfloat16
        else {
            FileHandle.standardError.write(
                Data("mlxfast: lm_head prune: unrecognized lm_head shape/dtype; disabled\n".utf8))
            return nil
        }
        if lagunaLmHeadCoarseV5Enabled {
            if lagunaLmHeadInlineMaskEnabled, lagunaLmHeadRatioBoundEnabled,
                lagunaLmHeadDeltaBF16Enabled,
                let planes = LagunaLmHeadPruner.buildInt5Planes(lmHeadWeight)
            {
                // v5: the int5 copy replaces every other coarse copy.
                self.int5CodesLo = planes.lo
                self.int5CodesHi = planes.hi
                self.int5Scales = planes.scales
                self.codes = nil
                self.scales = nil
                if lagunaTraceFusionEnabled {
                    FileHandle.standardError.write(
                        Data("fusion active: lmhead-int5-winner-coarse-v5\n".utf8))
                }
                return
            }
            // Either a nested selector is off (v5's single kernel folds the
            FileHandle.standardError.write(
                Data(
                    "mlxfast: lm_head coarse v5 declined; using v4/MXFP8 arm\n"
                        .utf8))
        }
        self.int5CodesLo = nil
        self.int5CodesHi = nil
        self.int5Scales = nil
        // The repo's own quantizer (ops.cpp fp_quantize gs32/bits8 ->
        let (wq, scales, _) = quantized(
            lmHeadWeight, groupSize: 32, bits: 8, mode: .mxfp8)
        self.codes = wq.view(dtype: .uint8)
        self.scales = scales
    }

    /// Builds the v5 planar int5 copy (untimed init).
    private static func buildInt5Planes(
        _ lmHeadWeight: MLXArray
    ) -> (lo: MLXArray, hi: MLXArray, scales: MLXArray)? {
        let vocab = lagunaLmHeadPruneVocab
        let hidden = lagunaLmHeadPruneHidden
        let w = lmHeadWeight.asType(.float32).reshaped([vocab, hidden / 32, 32])
        let gmax = MLX.abs(w).max(axis: 2)  // [V, 64] float32, contiguous
        let gbits = gmax.view(dtype: .uint32)
        let biasedE = (gbits >> 23).asType(.int32)
        let mant = gbits & MLXArray(UInt32(0x007F_FFFF))
        // bump when mantissa >= 0.9375 * 2^23 (i.e. m >= 15.5/8).
        let bump = (mant .>= MLXArray(UInt32(0x78_0000))).asType(.int32)
        let sdByte = clip(biasedE - 3 + bump, min: 0, max: 255)
        let sd = which(
            sdByte .== 0,
            MLXArray(Float(bitPattern: 0x0040_0000)),  // 2^-127, e8m0 semantics
            (sdByte.asType(.uint32) << 23).view(dtype: .float32))
        let q = (w / sd.expandedDimensions(axis: 2)).round()
        let maxCode = MLX.abs(q).max().item(Float.self)
        guard maxCode <= 15.0 else {
            FileHandle.standardError.write(
                Data(
                    "mlxfast: lm_head coarse v5: int5 code overflow (\(maxCode)); using v4/MXFP8\n"
                        .utf8))
            return nil
        }
        // Offset-binary u = q + 16 in [1, 31]; planar-pack 4+1 bits.
        let u = (q + 16).asType(.uint8).reshaped([vocab, hidden])
        let u16 = u.view(dtype: .uint16)  // [V, 1024]: elem 2b low byte
        let lo =
            ((u16 & MLXArray(UInt16(0x000F)))
            | ((u16 >> 4) & MLXArray(UInt16(0x00F0)))).asType(.uint8)
        // 1-bit plane: bit 4 of each code; element j of each 32-element
        let u32 = u.view(dtype: .uint32)  // [V, 512]: elem 4t..4t+3
        let nib =
            (((u32 >> 4) & MLXArray(UInt32(0x01)))
            | ((u32 >> 11) & MLXArray(UInt32(0x02)))
            | ((u32 >> 18) & MLXArray(UInt32(0x04)))
            | ((u32 >> 25) & MLXArray(UInt32(0x08)))).asType(.uint8)
        // Step 2: merge nibble pairs into bytes (byte s = elements 8s..8s+7).
        let nib16 = nib.view(dtype: .uint16)  // [V, 256]
        let hi =
            ((nib16 & MLXArray(UInt16(0x000F)))
            | ((nib16 >> 4) & MLXArray(UInt16(0x00F0)))).asType(.uint8)
        return (lo, hi, sdByte.asType(.uint8))
    }

    func logits(hidden: MLXArray, lmHeadWeight: MLXArray) -> MLXArray {
        precondition(hidden.dtype == .bfloat16 && hidden.size == lagunaLmHeadPruneHidden)
        let vocab = lagunaLmHeadPruneVocab
        let x = hidden.reshaped([lagunaLmHeadPruneHidden])
        // v5 arm: int5 coarse pass + exact-winner threshold. Early return so
        if let lo5 = int5CodesLo, let hi5 = int5CodesHi, let s5 = int5Scales {
            let coarseOut5 = lagunaLmHeadInt5CoarseRatioBoundDeltaBF16Kernel(
                [x, lo5, hi5, s5],
                grid: (vocab / 16 * 512, 1, 1),
                threadGroup: (512, 1, 1),
                outputShapes: [[vocab], [vocab]],
                outputDTypes: [.float32, .bfloat16]
            )
            let coarse5 = coarseOut5[0]
            let delta5 = coarseOut5[1]
            let argmaxPartials = lagunaLmHeadCoarseArgmaxStage1Kernel(
                [coarse5],
                grid: (224, 128, 1),
                threadGroup: (224, 1, 1),
                outputShapes: [[128], [128]],
                outputDTypes: [.float32, .uint32]
            )
            let thresholdKernel =
                lagunaLmHeadBF16PredecessorThresholdEnabled
                ? lagunaLmHeadExactWinnerBF16PredecessorThresholdKernel
                : lagunaLmHeadExactWinnerThresholdKernel
            let thr5 = thresholdKernel(
                [argmaxPartials[0], argmaxPartials[1], lmHeadWeight, x],
                grid: (32, 1, 1),
                threadGroup: (32, 1, 1),
                outputShapes: [[1]],
                outputDTypes: [.float32]
            )[0]
            if lagunaLmHeadV5StatsEnabled {
                // Debug-only: forces a per-step GPU sync; never on timing runs.
                let count = (coarse5 + delta5.asType(.float32) .>= thr5)
                    .asType(.int32).sum().item(Int32.self)
                FileHandle.standardError.write(
                    Data("lmhead-v5 candidates: \(count)\n".utf8))
            }
            let assembled5 = lagunaLmHeadInlineExactDeltaBF16Kernel(
                [coarse5, delta5, thr5, lmHeadWeight, x],
                grid: (vocab / 32 * 256, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[vocab]],
                outputDTypes: [.bfloat16]
            )[0]
            return assembled5.reshaped([1, 1, vocab])
        }
        let useCoarseV1 = lagunaLmHeadCoarseUseV1
        let coarseRowsPerThreadgroup = useCoarseV1 ? 8 : 16
        let coarseThreadsPerThreadgroup = coarseRowsPerThreadgroup * 32
        // The BF16 `delta` round trip is nested inside the ratio bound and, on
        let useDeltaBF16 =
            (lagunaLmHeadInlineMaskEnabled && !useCoarseV1)
            && lagunaLmHeadRatioBoundEnabled && lagunaLmHeadDeltaBF16Enabled

        let coarseOut: [MLXArray]
        if lagunaLmHeadInlineMaskEnabled {
            let coarseKernel: MLXFast.MLXFastKernel =
                if useCoarseV1 {
                    lagunaLmHeadInlineCoarseKernelV1
                } else if useDeltaBF16 {
                    lagunaLmHeadInlineCoarseRatioBoundDeltaBF16Kernel
                } else if lagunaLmHeadRatioBoundEnabled {
                    lagunaLmHeadInlineCoarseRatioBoundKernel
                } else {
                    lagunaLmHeadInlineCoarseKernel
                }
            coarseOut = coarseKernel(
                [x, codes!, scales!],
                grid: (
                    vocab / coarseRowsPerThreadgroup * coarseThreadsPerThreadgroup,
                    1,
                    1
                ),
                threadGroup: (coarseThreadsPerThreadgroup, 1, 1),
                outputShapes: [[vocab], [vocab]],
                outputDTypes: [.float32, useDeltaBF16 ? .bfloat16 : .float32]
            )
        } else {
            let coarseKernel: MLXFast.MLXFastKernel =
                if useCoarseV1 {
                    lagunaLmHeadCoarseKernelV1
                } else if lagunaLmHeadRatioBoundEnabled {
                    lagunaLmHeadCoarseRatioBoundKernel
                } else {
                    lagunaLmHeadCoarseKernel
                }
            coarseOut = coarseKernel(
                [x, codes!, scales!],
                grid: (
                    vocab / coarseRowsPerThreadgroup * coarseThreadsPerThreadgroup,
                    1,
                    1
                ),
                threadGroup: (coarseThreadsPerThreadgroup, 1, 1),
                outputShapes: [[vocab], [vocab], [vocab]],
                outputDTypes: [.float32, .float32, .bfloat16]
            )
        }
        let coarse = coarseOut[0]
        let delta = coarseOut[1]

        let lowerMaxStage1Kernel =
            useDeltaBF16
            ? lagunaLmHeadLowerMaxStage1DeltaBF16Kernel
            : lagunaLmHeadLowerMaxStage1Kernel
        let lowerMaxPartials = lowerMaxStage1Kernel(
            [coarse, delta],
            grid: (224, 128, 1),
            threadGroup: (224, 1, 1),
            outputShapes: [[128]],
            outputDTypes: [.float32]
        )[0]
        let thr = lagunaLmHeadLowerMaxThresholdKernel(
            [lowerMaxPartials],
            grid: (32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[1]],
            outputDTypes: [.float32]
        )[0]

        let assembled: MLXArray
        if lagunaLmHeadInlineMaskEnabled {
            let exactKernel =
                useDeltaBF16
                ? lagunaLmHeadInlineExactDeltaBF16Kernel
                : lagunaLmHeadInlineExactKernel
            assembled = exactKernel(
                [coarse, delta, thr, lmHeadWeight, x],
                grid: (vocab / 32 * 256, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[vocab]],
                outputDTypes: [.bfloat16]
            )[0]
        } else {
            let coarseBF = coarseOut[2]
            let isCandidate = lagunaLmHeadSelectKernel(
                [coarse, delta, thr],
                grid: (vocab, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[vocab]],
                outputDTypes: [.uint8]
            )[0]
            assembled = lagunaLmHeadExactKernel(
                [coarseBF, lmHeadWeight, x, isCandidate],
                grid: (vocab / 32 * 256, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[vocab]],
                outputDTypes: [.bfloat16]
            )[0]
        }
        return assembled.reshaped([1, 1, vocab])
    }
}
