import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
@testable import MLXFastModel
import Testing

/// Micro-benchmark of the decode MoE SwiGLU kernels using synthetic tensors
/// with the exact Laguna XS 2.1 shapes. Runs on real Metal hardware; guard
/// with MLXFAST_RUN_MLX_RUNTIME_TESTS=1 like the other kernel tests.
///
/// Measures wall-clock GPU time (via mx.eval forcing materialization) for:
///   1. Separated path: lagunaRoutedSwiGLUQMV + lagunaSharedSwiGLUQMV
///   2. Merged path:    lagunaRoutedSharedSwiGLUQMV (one dispatch)
/// Prints per-iteration ms for each so profiling the difference is trivial.
@Test
func profileDecodeSwiGLUMergedVsSeparatedWhenRuntimeTestsAreEnabled() {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }
    let hidden = LagunaConstants.hiddenSize          // 2048
    let moe = LagunaConstants.moeIntermediateSize    // 512
    let shared = LagunaConstants.sharedExpertIntermediateSize // 512
    let experts = LagunaConstants.numExperts         // 256
    let topK = LagunaConstants.numExpertsPerTok      // 8

    // Input: one token, hidden 2048, BF16.
    let input = MLXArray(
        Array(repeating: Float(0.01), count: hidden),
        [1, 1, hidden]).asType(.bfloat16)

    // Routed fused [gate;up] weights: [256, 1024, 256] BF16 -> quantize NVFP4
    // group-16 -> (uint32 packed, uint8 scales).
    let routedBF16 = MLXArray(
        Array(repeating: Float(0.02), count: experts * 2 * moe * (hidden / 8)),
        [experts, 2 * moe, hidden / 8]).asType(.bfloat16)
    let (routedCodes, routedScales, _) = quantized(
        routedBF16, groupSize: 16, bits: 4, mode: .nvfp4)

    // Shared fused [gate;up]: [1024, 256] BF16.
    let sharedBF16 = MLXArray(
        Array(repeating: Float(0.02), count: 2 * shared * (hidden / 8)),
        [2 * shared, hidden / 8]).asType(.bfloat16)
    let (sharedCodes, sharedScales, _) = quantized(
        sharedBF16, groupSize: 16, bits: 4, mode: .nvfp4)

    // Indices: 8 experts, deterministic.
    let indices = MLXArray(
        [UInt32(0), UInt32(1), UInt32(2), UInt32(3),
         UInt32(4), UInt32(5), UInt32(6), UInt32(7)],
        [1, 1, topK])

    // Warmup both paths once (kernels JIT-compile on first launch).
    let warmRouted = lagunaRoutedSwiGLUQMV(
        input, fusedWeight: routedCodes, fusedScales: routedScales, indices: indices)
    let warmShared = lagunaSharedSwiGLUQMV(
        input, fusedWeight: sharedCodes, fusedScales: sharedScales)
    let warmMerged = lagunaRoutedSharedSwiGLUQMV(
        input, fusedWeight: routedCodes, fusedScales: routedScales, indices: indices,
        sharedWeight: sharedCodes, sharedScales: sharedScales)
    eval(warmRouted, warmShared, warmMerged.0, warmMerged.1)

    // Bit-exactness check: merged routed half must equal separated routed.
    let routedDiff = maxAbsDiff(warmRouted, warmMerged.0)
    let sharedDiff = maxAbsDiff(warmShared, warmMerged.1)
    #expect(routedDiff <= 1e-6)
    #expect(sharedDiff <= 1e-6)

    let iterations = 50
    func time(_ body: () -> Void) -> Double {
        let start = DispatchTime.now()
        body()
        let end = DispatchTime.now()
        return Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1e6
    }

    // Separated path timing.
    var separatedMs = 0.0
    for _ in 0..<iterations {
        separatedMs += time {
            let r = lagunaRoutedSwiGLUQMV(
                input, fusedWeight: routedCodes, fusedScales: routedScales, indices: indices)
            let s = lagunaSharedSwiGLUQMV(
                input, fusedWeight: sharedCodes, fusedScales: sharedScales)
            eval(r, s)
        }
    }
    separatedMs /= Double(iterations)

    // Merged path timing.
    var mergedMs = 0.0
    for _ in 0..<iterations {
        mergedMs += time {
            let m = lagunaRoutedSharedSwiGLUQMV(
                input, fusedWeight: routedCodes, fusedScales: routedScales, indices: indices,
                sharedWeight: sharedCodes, sharedScales: sharedScales)
            eval(m.0, m.1)
        }
    }
    mergedMs /= Double(iterations)

    let speedup = separatedMs / mergedMs
    print("PROFILE_SWIGLU separated=\(String(format: "%.3f", separatedMs))ms "
        + "merged=\(String(format: "%.3f", mergedMs))ms "
        + "speedup=\(String(format: "%.3f", speedup))x "
        + "routedDiff=\(routedDiff) sharedDiff=\(sharedDiff)")
}

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    let diff = MLX.abs(a - b)
    eval(diff)
    return diff.asArray(Float.self).max() ?? .infinity
}
