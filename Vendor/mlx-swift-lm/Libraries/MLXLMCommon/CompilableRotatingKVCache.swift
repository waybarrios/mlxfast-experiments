// CompilableRotatingKVCache: compile-traceable rotating KV cache.

import Foundation
import MLX
import MLXNN

/// Compile-traceable specialisation of ``RotatingKVCache``.
public final class CompilableRotatingKVCache: RotatingKVCache, @unchecked Sendable {

    public var idxArray: MLXArray

    /// Total valid tokens seen, as `MLXArray[1] int32`. In the linear
    public var offsetArray: MLXArray

    private lazy var maskRinds: MLXArray = MLXArray(Int32(0) ..< Int32(maxCacheSize))

    /// Promotion-time proof that the physical ring is already full. Once true,
    private var canElideFullWindowDecodeMask = false

    // MARK: - Init

    /// Direct constructor matching the parent. Primarily for testing.
    public override init(maxSize: Int, keep: Int = 0, step: Int = 256) {
        self.idxArray = MLXArray([Int32(0)])
        self.offsetArray = MLXArray([Int32(0)])
        super.init(maxSize: maxSize, keep: keep, step: step)
    }

    /// Promote an existing populated ``RotatingKVCache`` to a compile-
    public convenience init(from rotating: RotatingKVCache) {
        self.init(
            maxSize: rotating.maxCacheSize,
            keep: rotating.keep,
            step: rotating.step
        )

        self.idx = rotating.idx
        self.offset = rotating.offset
        self.canElideFullWindowDecodeMask = rotating.offset >= maxCacheSize

        if let srcK = rotating.keys, let srcV = rotating.values {
            let B = srcK.dim(0)
            let H = srcK.dim(1)
            let kD = srcK.dim(3)
            let vD = srcV.dim(3)
            let curLen = srcK.dim(2)

            if curLen < maxCacheSize {
                let padLen = maxCacheSize - curLen
                let padK = MLXArray.zeros([B, H, padLen, kD], dtype: srcK.dtype)
                let padV = MLXArray.zeros([B, H, padLen, vD], dtype: srcV.dtype)
                self.keys = concatenated([srcK, padK], axis: 2)
                self.values = concatenated([srcV, padV], axis: 2)
            } else {
                self.keys = srcK
                self.values = srcV
            }
        }

        self.idxArray = MLXArray([Int32(self.idx)])
        self.offsetArray = MLXArray([Int32(self.offset)])
    }

    /// Static promote helper for symmetry with `CompilableKVCache.promote`.
    public static func promote(from cache: RotatingKVCache, maxLength: Int) -> CompilableRotatingKVCache {
        return CompilableRotatingKVCache(from: cache)
    }

    // MARK: - Overridden update

    /// Compile-traceable append. Writes new tokens at `idxArray` position
    public override func update(
        keys newKeys: MLXArray, values newValues: MLXArray
    ) -> (MLXArray, MLXArray) {
        let nTokens = newKeys.dim(2)

        // Lazy-allocate the unified buffer if empty (first-call init).
        if keys == nil {
            let B = newKeys.dim(0)
            let H = newKeys.dim(1)
            let kD = newKeys.dim(3)
            let vD = newValues.dim(3)
            keys = MLXArray.zeros([B, H, maxCacheSize, kD], dtype: newKeys.dtype)
            values = MLXArray.zeros([B, H, maxCacheSize, vD], dtype: newValues.dtype)
        }

        // Write new tokens at idxArray position.
        keys!._updateInternal(
            dynamicSliceUpdate(keys!, update: newKeys, start: idxArray, axes: [2]))
        values!._updateInternal(
            dynamicSliceUpdate(values!, update: newValues, start: idxArray, axes: [2]))

        let advance = MLXArray([Int32(nTokens)])
        let advancedIdx = idxArray + advance
        let maxSz = MLXArray([Int32(maxCacheSize)])
        let keepArr = MLXArray([Int32(keep)])
        let cycleLen = maxSz - keepArr  // number of rotating slots

        let rotatedIdx: MLXArray
        if keep > 0 {
            rotatedIdx = keepArr + ((advancedIdx - keepArr) % cycleLen)
        } else {
            rotatedIdx = advancedIdx % maxSz
        }
        // where_(cond, true_branch, false_branch)
        let newIdx = MLX.`where`(advancedIdx .< maxSz, advancedIdx, rotatedIdx)

        idxArray._updateInternal(newIdx)
        offsetArray._updateInternal(offsetArray + advance)

        // DELIBERATELY no Swift-Int mirror updates here:

        return (keys!, values!)
    }

    // MARK: - makeMask

    /// Build an attention mask over the full `[B, H, maxCacheSize, D]`
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1, windowSize == maxCacheSize, canElideFullWindowDecodeMask {
            return .none
        }

        let linds: MLXArray
        if n == 1 {
            linds = offsetArray.reshaped(1, 1)
        } else {
            linds = (MLXArray(Int32(0) ..< Int32(n)) + offsetArray).reshaped(n, 1)
        }

        let rinds = maskRinds.reshaped(1, maxCacheSize)
        // Causal: attend to positions j <= query_position.
        let causal = linds .>= rinds

        let maxSzArr = MLXArray([Int32(maxCacheSize)]).reshaped(1, 1)
        let allTrueMask = MLX.broadcast(
            MLXArray([true]).reshaped(1, 1),
            to: [linds.dim(0), rinds.dim(1)]
        )
        var mask = MLX.`where`(linds .>= maxSzArr, allTrueMask, causal)

        if let windowSize {
            // After ring wrap, the recent window may be split across buffer
            let tokenInds = (rinds - idxArray + MLXArray(Int32(maxCacheSize))) % Int32(maxCacheSize)
            let windowFilter = tokenInds .>= Int32(maxCacheSize - windowSize)
            mask = mask & windowFilter
        }

        return .array(mask)
    }

    // MARK: - innerState

    public override func innerState() -> [MLXArray] {
        var state = [MLXArray]()
        if let k = keys { state.append(k) }
        if let v = values { state.append(v) }
        state.append(idxArray)
        state.append(offsetArray)
        return state
    }
}
