import Foundation
import MLX
import MLXFastCore
import MLXLMCommon
import MLXNN

public enum LagunaWeightNames {
    private static let prefix = "model"

    public static let embedTokens = "\(prefix).embed_tokens.weight"
    public static let finalNorm = "\(prefix).norm.weight"
    public static let lmHead = "lm_head.weight"

    public static func layer(_ layerIndex: Int, _ suffix: String) -> String {
        "\(prefix).layers.\(layerIndex).\(suffix)"
    }

    public static func attention(_ layerIndex: Int, _ suffix: String) -> String {
        layer(layerIndex, "self_attn.\(suffix)")
    }

    public static func mlp(_ layerIndex: Int, _ suffix: String) -> String {
        layer(layerIndex, "mlp.\(suffix)")
    }
}

/// Metadata-level access and validation for the transformed Laguna weights
public struct LagunaWeightLoader {
    public let denseStore: DenseTensorStore

    public init(weightsPath: String) throws {
        self.denseStore = try DenseTensorStore(weightsPath: weightsPath)
    }

    public init(denseStore: DenseTensorStore) {
        self.denseStore = denseStore
    }

    public func validateRequiredMetadata(config: LagunaConfig) throws {
        let tensorNames = denseStore.tensorNames
        let forbiddenSuffixes = [
            ".weight_packed",
            ".input_global_scale",
            ".weight_global_scale",
            ".k_scale",
            ".v_scale",
            ".biases",
        ]
        if let forbiddenName = tensorNames.first(where: { name in
            forbiddenSuffixes.contains(where: name.hasSuffix)
        }) {
            throw MLXFastError.invalidInput(
                "Poolside Laguna MLX checkpoint must not contain compressed-tensors/global-scale, "
                    + "FP8 KV-scale, or affine-bias tensor \(forbiddenName)"
            )
        }
        guard tensorNames.count == LagunaConstants.tensorCount else {
            throw MLXFastError.invalidInput(
                "Poolside Laguna tensor inventory contains \(tensorNames.count) tensors; "
                    + "expected exactly \(LagunaConstants.tensorCount)"
            )
        }
        var dtypeCounts: [String: Int] = [:]
        for name in tensorNames {
            guard let record = denseStore.record(named: name) else {
                throw MLXFastError.invalidInput("dense tensor not found: \(name)")
            }
            dtypeCounts[record.dtype, default: 0] += 1
        }
        let expectedDTypeCounts = [
            "BF16": LagunaConstants.bfloat16TensorCount,
            "F32": LagunaConstants.float32TensorCount,
            "U32": LagunaConstants.packedUInt32TensorCount,
            "U8": LagunaConstants.e4m3ScaleUInt8TensorCount,
        ]
        guard dtypeCounts == expectedDTypeCounts else {
            throw MLXFastError.invalidInput(
                "Poolside Laguna tensor dtype inventory \(dtypeCounts) does not match "
                    + "expected \(expectedDTypeCounts)"
            )
        }

        try validateBFloat16ProjectionMetadata(
            named: LagunaWeightNames.embedTokens,
            expectedShape: [config.vocabSize, config.hiddenSize]
        )
        try validateDenseTensorMetadata(
            named: LagunaWeightNames.finalNorm,
            expectedShape: [config.hiddenSize],
            expectedDType: "BF16"
        )
        if !config.tieWordEmbeddings {
            try validateBFloat16ProjectionMetadata(
                named: LagunaWeightNames.lmHead,
                expectedShape: [config.vocabSize, config.hiddenSize]
            )
        }

        for layerIndex in 0..<config.numHiddenLayers {
            let layerHeads = config.heads(forLayer: layerIndex)

            for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                try validateDenseTensorMetadata(
                    named: LagunaWeightNames.layer(layerIndex, suffix),
                    expectedShape: [config.hiddenSize],
                    expectedDType: "BF16"
                )
            }

            try validateBFloat16ProjectionMetadata(
                named: LagunaWeightNames.attention(layerIndex, "q_proj.weight"),
                expectedShape: [layerHeads * config.headDim, config.hiddenSize]
            )
            for suffix in ["k_proj.weight", "v_proj.weight"] {
                try validateBFloat16ProjectionMetadata(
                    named: LagunaWeightNames.attention(layerIndex, suffix),
                    expectedShape: [
                        config.numKeyValueHeads * config.headDim,
                        config.hiddenSize,
                    ]
                )
            }
            try validateBFloat16ProjectionMetadata(
                named: LagunaWeightNames.attention(layerIndex, "o_proj.weight"),
                expectedShape: [config.hiddenSize, layerHeads * config.headDim]
            )
            if let gateDim = config.gateProjectionOutputDim(forLayer: layerIndex) {
                try validateBFloat16ProjectionMetadata(
                    named: LagunaWeightNames.attention(layerIndex, "g_proj.weight"),
                    expectedShape: [gateDim, config.hiddenSize]
                )
            }
            for suffix in ["q_norm.weight", "k_norm.weight"] {
                try validateDenseTensorMetadata(
                    named: LagunaWeightNames.attention(layerIndex, suffix),
                    expectedShape: [config.headDim],
                    expectedDType: "BF16"
                )
            }

            if config.isSparse(layer: layerIndex) {
                try validateBFloat16ProjectionMetadata(
                    named: LagunaWeightNames.mlp(layerIndex, "gate.weight"),
                    expectedShape: [config.numExperts, config.hiddenSize]
                )
                try validateDenseTensorMetadata(
                    named: LagunaWeightNames.mlp(layerIndex, "gate.e_score_correction_bias"),
                    expectedShape: [config.numExperts],
                    expectedDType: "F32"
                )
                for suffix in ["switch_mlp.gate_proj.weight", "switch_mlp.up_proj.weight"] {
                    try validateNVFP4TensorMetadata(
                        named: LagunaWeightNames.mlp(layerIndex, suffix),
                        expectedLeadingShape: [config.numExperts, config.moeIntermediateSize],
                        expectedInputFeatures: config.hiddenSize,
                        quantization: config.quantization
                    )
                }
                try validateNVFP4TensorMetadata(
                    named: LagunaWeightNames.mlp(layerIndex, "switch_mlp.down_proj.weight"),
                    expectedLeadingShape: [config.numExperts, config.hiddenSize],
                    expectedInputFeatures: config.moeIntermediateSize,
                    quantization: config.quantization
                )
                for suffix in ["shared_expert.gate_proj.weight", "shared_expert.up_proj.weight"] {
                    try validateNVFP4TensorMetadata(
                        named: LagunaWeightNames.mlp(layerIndex, suffix),
                        expectedLeadingShape: [config.sharedExpertIntermediateSize],
                        expectedInputFeatures: config.hiddenSize,
                        quantization: config.quantization
                    )
                }
                try validateNVFP4TensorMetadata(
                    named: LagunaWeightNames.mlp(layerIndex, "shared_expert.down_proj.weight"),
                    expectedLeadingShape: [config.hiddenSize],
                    expectedInputFeatures: config.sharedExpertIntermediateSize,
                    quantization: config.quantization
                )
            } else {
                for suffix in ["gate_proj.weight", "up_proj.weight"] {
                    try validateBFloat16ProjectionMetadata(
                        named: LagunaWeightNames.mlp(layerIndex, suffix),
                        expectedShape: [config.intermediateSize, config.hiddenSize]
                    )
                }
                try validateBFloat16ProjectionMetadata(
                    named: LagunaWeightNames.mlp(layerIndex, "down_proj.weight"),
                    expectedShape: [config.hiddenSize, config.intermediateSize]
                )
            }
        }

        var expectedTensorCount = config.tieWordEmbeddings ? 2 : 3
        for layerIndex in 0..<config.numHiddenLayers {
            expectedTensorCount += 8
            if config.gateProjectionOutputDim(forLayer: layerIndex) != nil {
                expectedTensorCount += 1
            }
            expectedTensorCount += config.isSparse(layer: layerIndex) ? 14 : 3
        }
        guard expectedTensorCount == LagunaConstants.tensorCount else {
            throw MLXFastError.invalidInput(
                "internal Poolside Laguna tensor contract computed \(expectedTensorCount) tensors; "
                    + "expected \(LagunaConstants.tensorCount)"
            )
        }
    }

    /// Validates a plain tensor's exact dtype and shape without materializing it.
    private func validateDenseTensorMetadata(
        named name: String,
        expectedShape: [Int],
        expectedDType: String
    ) throws {
        guard let record = denseStore.record(named: name) else {
            throw MLXFastError.invalidInput("dense tensor not found: \(name)")
        }
        guard record.dtype == expectedDType, record.shape == expectedShape else {
            throw MLXFastError.invalidInput(
                "tensor \(name) dtype/shape \(record.dtype) \(record.shape) does not match expected \(expectedDType) \(expectedShape)"
            )
        }
    }

    private func validateBFloat16ProjectionMetadata(
        named name: String,
        expectedShape: [Int]
    ) throws {
        try validateDenseTensorMetadata(
            named: name,
            expectedShape: expectedShape,
            expectedDType: "BF16"
        )
        for suffix in ["scales", "biases"] {
            let companionName = Self.companionName(for: name, suffix: suffix)
            guard denseStore.record(named: companionName) == nil else {
                throw MLXFastError.invalidInput(
                    "BF16 Poolside projection \(name) must not contain \(companionName)"
                )
            }
        }
    }

    private func validateNVFP4TensorMetadata(
        named name: String,
        expectedLeadingShape: [Int],
        expectedInputFeatures: Int,
        quantization: LagunaQuantizationSpec
    ) throws {
        let (groupSize, bits) = quantization.expected(forTensorStem: Self.tensorStem(name))
        guard expectedInputFeatures > 0,
              quantization.mode == LagunaConstants.quantizationMode,
              groupSize == LagunaConstants.quantizationGroupSize,
              bits == LagunaConstants.quantizationBits,
              (expectedInputFeatures * bits).isMultiple(of: 32),
              expectedInputFeatures.isMultiple(of: groupSize)
        else {
            throw MLXFastError.invalidInput(
                "NVFP4 tensor \(name) logical input \(expectedInputFeatures) is incompatible with 4-bit group size 16"
            )
        }
        guard let record = denseStore.record(named: name) else {
            throw MLXFastError.invalidInput("dense tensor not found: \(name)")
        }
        guard record.dtype == "U32" else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) must use U32 packed codes, found \(record.dtype)"
            )
        }
        let expectedWeightShape = expectedLeadingShape + [expectedInputFeatures * bits / 32]
        guard record.shape == expectedWeightShape else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) shape \(record.shape) does not match expected shape \(expectedWeightShape)"
            )
        }

        let scalesName = Self.companionName(for: name, suffix: "scales")
        guard let scales = denseStore.record(named: scalesName) else {
            throw MLXFastError.invalidInput(
                "NVFP4 tensor \(name) is missing U8 scales \(scalesName)"
            )
        }
        let expectedCompanionShape = expectedLeadingShape + [expectedInputFeatures / groupSize]
        guard scales.dtype == "U8", scales.shape == expectedCompanionShape else {
            throw MLXFastError.invalidInput(
                "NVFP4 tensor \(name) scales dtype/shape \(scales.dtype) \(scales.shape) does not match expected U8 \(expectedCompanionShape)"
            )
        }
        let biasesName = Self.companionName(for: name, suffix: "biases")
        guard denseStore.record(named: biasesName) == nil else {
            throw MLXFastError.invalidInput(
                "NVFP4 tensor \(name) must not contain affine biases \(biasesName)"
            )
        }
    }

    static func tensorStem(_ name: String) -> String {
        guard name.hasSuffix(".weight") else {
            return name
        }
        return String(name.dropLast(".weight".count))
    }

    private static func companionName(for baseName: String, suffix: String) -> String {
        "\(tensorStem(baseName)).\(suffix)"
    }
}

/// Eagerly-prepared, RAM-resident weight cache for the Laguna text tower. The
public final class LagunaRuntimeWeightCache {
    public let loader: LagunaWeightLoader
    public let config: LagunaConfig

    /// The Laguna runtime model this benchmark's reference runs. Loaded once
    public let libraryModel: LagunaRuntimeModel?
    public let loadError: Error?

    public init(loader: LagunaWeightLoader, config: LagunaConfig) {
        self.loader = loader
        self.config = config
        // Select the startup memory profile BEFORE the model load. Laguna
        let startupMemoryPolicy: RuntimeStartupMemoryPolicy?
        if config.numHiddenLayers >= 16 {
            let policy = RuntimeStartupMemoryPolicy.resolve(
                physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                requestedProfile: ProcessInfo.processInfo.environment[
                    RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName
                ]
            )
            if policy.isLowMemory {
                policy.apply()
                startupMemoryPolicy = policy
            } else {
                // The full 128 GiB ranked profile wires the complete live
                let env = ProcessInfo.processInfo.environment
                setenv("MLX_BFS_MAX_WIDTH", "50", 0)
                if env["DARKBLOOM_POST_WIRE_COMMAND_BUFFER"] != "0" {
                    setenv("MLX_MAX_MB_PER_BUFFER", "200", 0)
                    setenv("MLX_MAX_OPS_PER_BUFFER", "200", 0)
                }
                startupMemoryPolicy = nil
            }
        } else {
            startupMemoryPolicy = nil
        }
        do {
            libraryModel = try LagunaRuntimeWeightCache.loadLibraryModel(
                loader: loader,
                config: config
            )
            loadError = nil
        } catch {
            libraryModel = nil
            loadError = error
        }
        // Constructor-time warmup: the runtime worker builds this cache
        if let model = libraryModel, config.numHiddenLayers >= 16 {
            Self.warmLibraryModel(model)
            if startupMemoryPolicy?.clearAllocatorCacheAfterWarmup == true {
                Memory.clearCache()
            }
        }
        // Zero-headroom wired residency (notes/47 §4e follow-up, session
        if libraryModel != nil, config.numHiddenLayers >= 16 {
            Self.wireResidentWeightsIfEnabled()
        }
    }

    /// One prefill-shaped forward (512 tokens) and one single-token decode
    private static func warmLibraryModel(_ model: LagunaRuntimeModel) {
        let bosToken = Int32(LagunaConstants.bosTokenID)
        let warmupCache = model.newCache(parameters: nil)
        let prefillTokens = MLXArray(
            Array(repeating: bosToken, count: 512),
            [1, 512]
        )
        eval(model(prefillTokens, cache: warmupCache))
        let decodeToken = MLXArray([bosToken], [1, 1])
        var warmDecodeLogits = model(decodeToken, cache: warmupCache)
        eval(warmDecodeLogits)
        // The historical full-attention bundle coupled this second whole-model
        if lagunaFusedFullAttentionEnabled,
            lagunaFusedFullAttentionWholeModelWarmupEnabled
        {
            warmDecodeLogits = model(decodeToken, cache: warmupCache)
            eval(warmDecodeLogits)
        }
        if lagunaFusedFullAttentionEnabled,
            lagunaFusedFullAttentionKernelWarmupEnabled
        {
            lagunaWarmFullFusedAttentionKernel()
        }
        // Warm the greedy-token pipeline too. Every scored worker request ends
        if ProcessInfo.processInfo.environment["DARKBLOOM_WARM_GREEDY_ARGMAX"] != "0",
            let vocabSize = warmDecodeLogits.shape.last, vocabSize > 0
        {
            let rows = warmDecodeLogits.reshaped([-1, vocabSize])
            eval(rows[-1].argMax())
        }
    }
    /// See the construction-time comment: one `set_wired_limit` call sized
    private static let wiredZHDefaultFraction = 1.0
    private static let wiredZHDefaultSlackMB = 64

    private static func wireResidentWeightsIfEnabled() {
        let env = ProcessInfo.processInfo.environment
        guard env["DARKBLOOM_WIRED_ZH"] != "0" else { return }
        guard ProcessInfo.processInfo.physicalMemory >= (96 << 30) else { return }

        // Evict cached warmup transients FIRST so only live buffers
        Memory.clearCache()

        let active = Memory.activeMemory
        guard active > 0 else { return }

        let fraction =
            env["DARKBLOOM_WIRED_ZH_FRACTION"].flatMap(Double.init)
            ?? wiredZHDefaultFraction
        let slackMB =
            env["DARKBLOOM_WIRED_ZH_SLACK_MB"].flatMap(Int.init)
            ?? wiredZHDefaultSlackMB
        var target = Int(Double(active) * min(max(fraction, 0.0), 1.0))
        target += max(0, slackMB) << 20

        if let maxRec = GPU.maxRecommendedWorkingSetBytes() {
            target = min(target, maxRec - (256 << 20))
        }
        guard target > 0 else { return }

        let ticket = WiredMemoryTicket(
            size: target,
            policy: MLXLMCommon.WiredSumPolicy(cap: target),
            manager: .shared,
            kind: .active
        )
        let appliedBox = LagunaWiredLimitBox()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            let applied = await ticket.start()
            appliedBox.value = applied
            semaphore.signal()
        }
        // Bounded wait: a manager stall must not hang worker construction.
        let outcome = semaphore.wait(timeout: .now() + .seconds(30))
        Self.wiredTicketRetainer = ticket
        let applied = outcome == .success ? appliedBox.value : -1
        let maxRec = GPU.maxRecommendedWorkingSetBytes() ?? -1
        var line = "mlxfast: wired-zh request=\(target) applied=\(applied)"
        line += " active=\(active)"
        line += " slack_mb=\(max(0, slackMB))"
        line += " fraction=\(fraction)"
        line += " maxrec=\(maxRec)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    nonisolated(unsafe) private static var wiredTicketRetainer: WiredMemoryTicket?
    private final class LagunaWiredLimitBox: @unchecked Sendable {
        var value: Int = 0
    }

    /// Construct and weight-load the Laguna runtime model from the
    private static func loadLibraryModel(
        loader: LagunaWeightLoader,
        config: LagunaConfig
    ) throws -> LagunaRuntimeModel {
        try loader.validateRequiredMetadata(config: config)
        let model = LagunaRuntimeModel(config)

        let loadedWeights = try loadRuntimeWeightArrays(denseStore: loader.denseStore)
        let sanitized = model.sanitize(weights: loadedWeights)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        eval(model)
        // Build the retained fused weight layouts (fused QKV, fused
        model.prepareFusedRuntimeWeights()
        return model
    }

    public func requireLibraryModel() throws -> LagunaRuntimeModel {
        guard let libraryModel else {
            throw loadError
                ?? MLXFastError.invalidInput("Laguna runtime model was not loaded")
        }
        return libraryModel
    }
}
