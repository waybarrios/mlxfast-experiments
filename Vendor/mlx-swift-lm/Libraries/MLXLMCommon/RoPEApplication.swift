// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - BatchPositionedKVCache

/// Protocol for KV caches that expose per-sequence RoPE offsets.
public protocol BatchPositionedKVCache: KVCache {
    /// Per-sequence RoPE offsets with shape `[B]`.
    var batchOffset: MLXArray { get }
}

// MARK: - graphOffsetArray Helper

/// Returns a graph-visible cache offset when the cache exposes one.
public func graphOffsetArray(for cache: KVCache?) -> MLXArray? {
    if let compilableRot = cache as? CompilableRotatingKVCache {
        return compilableRot.offsetArray + 0
    }
    if let compilable = cache as? CompilableKVCache {
        return compilable.offsetArray + 0
    }
    if let batchCache = cache as? BatchPositionedKVCache {
        return batchCache.batchOffset + 0
    }
    return nil
}

// MARK: - applyRotaryPosition Helper

/// Apply rotary position embeddings, using the cache offset when available.
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, cache: KVCache?)
    -> MLXArray
{
    if let offsetArray = graphOffsetArray(for: cache) {
        return rope(x, offset: offsetArray)
    }
    return rope(x, offset: cache?.offset ?? 0)
}
