import Foundation
import MLXFastCore
#if canImport(Darwin)
import Darwin
#endif

public struct TransformOptions: Equatable {
    public let referencePath: String
    public let outputPath: String

    public init(referencePath: String, outputPath: String) {
        self.referencePath = referencePath
        self.outputPath = outputPath
    }
}

public struct TransformReport: Equatable {
    public let referencePath: String
    public let outputPath: String
    public let denseTensorCount: Int
    public let denseShardCount: Int
    public let configPath: String
    public let indexPath: String

    public init(
        referencePath: String,
        outputPath: String,
        denseTensorCount: Int,
        denseShardCount: Int,
        configPath: String,
        indexPath: String
    ) {
        self.referencePath = referencePath
        self.outputPath = outputPath
        self.denseTensorCount = denseTensorCount
        self.denseShardCount = denseShardCount
        self.configPath = configPath
        self.indexPath = indexPath
    }
}

/// Model family of the source reference checkpoint, detected from its
enum TransformModelFamily: Equatable {
    case gemma4
    case laguna
}

/// Offline transform for the pinned reference checkpoint: selects ONLY the
public enum SwiftTransform {
    /// Tensor name prefix that marks a checkpoint tensor as part of the text
    static let textTowerPrefix = "language_model."

    public static func run(_ options: TransformOptions) throws -> TransformReport {
        try run(
            options,
            beforeSidecarGeneration: nil,
            beforeSourceRevalidation: nil
        )
    }

    static func run(
        _ options: TransformOptions,
        beforeSidecarGeneration: (() throws -> Void)? = nil,
        beforeSourceRevalidation: (() throws -> Void)?
    ) throws -> TransformReport {
        let referenceDirectory = canonicalURL(
            try findReferenceDirectory(URL(fileURLWithPath: options.referencePath))
        )
        let outputDirectory = canonicalURL(URL(fileURLWithPath: options.outputPath))
        try validateDistinctDirectories(
            referenceDirectory: referenceDirectory,
            outputDirectory: outputDirectory
        )
        var outputIsDirectory = ObjCBool(false)
        if FileManager.default.fileExists(
            atPath: outputDirectory.path,
            isDirectory: &outputIsDirectory
        ), !outputIsDirectory.boolValue {
            throw MLXFastError.invalidInput(
                "transform output exists and is not a directory: \(outputDirectory.path)"
            )
        }

        let referenceConfigPath = referenceDirectory.appendingPathComponent("config.json")
        try requireFile(
            referenceConfigPath.path,
            description: "reference checkpoint config"
        )
        let sourceConfigRoot = try loadReferenceConfigRoot(referenceConfigPath)
        let modelFamily = try detectModelFamily(sourceConfigRoot: sourceConfigRoot)
        let runtimeConfigData = try makeRuntimeConfigData(
            sourceConfigRoot: sourceConfigRoot,
            family: modelFamily
        )
        let metadataSnapshot = try captureMetadataFiles(from: referenceDirectory)

        let index = try loadIndex(referenceDirectory)
        let indexSnapshot = try index.canonicalData()
        let validatedHeaders = try validateCheckpointIndex(
            index,
            referenceDirectory: referenceDirectory
        )
        let textKeys = Set(
            index.weightMap.keys.filter { isSelectedTextTowerKey($0, family: modelFamily) }
        )
        guard !textKeys.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint index contains no text-tower tensors")
        }

        let textKeysByShard = Dictionary(grouping: textKeys) { key in
            index.weightMap[key] ?? ""
        }
        var totalTensorByteCount = 0
        for key in textKeys.sorted() {
            guard let shardName = index.weightMap[key],
                  let info = validatedHeaders[shardName]?.tensors[key]
            else {
                throw MLXFastError.invalidInput(
                    "missing validated tensor metadata for \(key)"
                )
            }
            let (nextTotal, overflow) = totalTensorByteCount.addingReportingOverflow(
                info.byteCount
            )
            guard !overflow else {
                throw MLXFastError.invalidInput(
                    "transformed tensor byte count overflows Int"
                )
            }
            totalTensorByteCount = nextTotal
        }

        if modelFamily == .laguna {
            // Fail before the multi-GB copy if the selected tensor set is
            try LagunaCheckpointValidation.validateSelectedTensors(
                selectedKeys: textKeys,
                index: index,
                headers: validatedHeaders,
                quantization: LagunaCheckpointValidation.quantizationSpec(
                    fromConfigRoot: sourceConfigRoot
                )
            )
        }

        let fileManager = FileManager.default
        let stagingDirectory = outputDirectory.deletingLastPathComponent().appendingPathComponent(
            ".\(outputDirectory.lastPathComponent).mlxfast-transform-\(UUID().uuidString)",
            isDirectory: true
        )
        try fileManager.createDirectory(
            at: stagingDirectory.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try fileManager.createDirectory(at: stagingDirectory, withIntermediateDirectories: false)
        var installed = false
        defer {
            if !installed {
                try? fileManager.removeItem(at: stagingDirectory)
            }
        }

        var copiedTensors = 0
        var stagedHeaders: [String: SafetensorsHeader] = [:]
        for shardName in textKeysByShard.keys.sorted() {
            let source = referenceDirectory.appendingPathComponent(shardName)
            let destination = stagingDirectory.appendingPathComponent(shardName)
            guard let header = validatedHeaders[shardName] else {
                throw MLXFastError.invalidInput("missing validated header for checkpoint shard \(shardName)")
            }
            let selectedNames = textKeysByShard[shardName, default: []].sorted()
            if Set(selectedNames) == Set(header.tensors.keys) {
                // Poolside's reference is already text-only. Preserve the
                try cloneOrCopyShard(from: source, to: destination)
                copiedTensors += selectedNames.count
            } else {
                copiedTensors += try Safetensors.copySubset(
                    from: source,
                    to: destination,
                    tensorNames: selectedNames,
                    validatedHeader: header
                )
            }
            stagedHeaders[shardName] = try Safetensors.readHeader(destination)
        }

        try beforeSidecarGeneration?()
        let generatedProjectionMetadata: GeneratedAffineMetadataReport
        let generatedTiedHeadMetadata: GeneratedAffineMetadataReport
        switch modelFamily {
        case .gemma4:
            generatedProjectionMetadata = try AffineMetadataCoding.writeProjectionSidecar(
                sourceDirectory: stagingDirectory,
                index: index,
                sourceHeaders: stagedHeaders,
                selectedKeys: textKeys,
                destinationDirectory: stagingDirectory
            )
            generatedTiedHeadMetadata = try TiedHeadMetadataCoding.writeSidecar(
                sourceDirectory: stagingDirectory,
                index: index,
                sourceHeaders: stagedHeaders,
                selectedKeys: textKeys,
                destinationDirectory: stagingDirectory
            )
        case .laguna:
            // docs/laguna-weight-contract.md forbids derived layouts and
            generatedProjectionMetadata = GeneratedAffineMetadataReport(
                weightMap: [:],
                tensorByteCount: 0
            )
            generatedTiedHeadMetadata = GeneratedAffineMetadataReport(
                weightMap: [:],
                tensorByteCount: 0
            )
        }
        let (projectionOutputByteCount, projectionSizeOverflow) =
            totalTensorByteCount.addingReportingOverflow(
                generatedProjectionMetadata.tensorByteCount
            )
        let (outputTensorByteCount, tiedHeadSizeOverflow) =
            projectionOutputByteCount.addingReportingOverflow(
                generatedTiedHeadMetadata.tensorByteCount
            )
        guard !projectionSizeOverflow, !tiedHeadSizeOverflow else {
            throw MLXFastError.invalidInput("transformed tensor byte count overflows Int")
        }
        let generatedWeightMap = generatedProjectionMetadata.weightMap.merging(
            generatedTiedHeadMetadata.weightMap
        ) { _, _ in
            preconditionFailure("generated metadata tensor names collide")
        }

        try writeMetadataFiles(metadataSnapshot, to: stagingDirectory)
        try index.writeStripped(
            to: stagingDirectory.appendingPathComponent("model.safetensors.index.json"),
            keeping: textKeys,
            totalTensorByteCount: outputTensorByteCount,
            additionalWeightMap: generatedWeightMap
        )

        try runtimeConfigData.write(
            to: stagingDirectory.appendingPathComponent("config.json")
        )
        try beforeSourceRevalidation?()
        try validateConfigAndIndexSnapshot(
            referenceDirectory: referenceDirectory,
            referenceConfigPath: referenceConfigPath,
            runtimeConfigData: runtimeConfigData,
            indexSnapshot: indexSnapshot,
            metadataSnapshot: metadataSnapshot
        )
        for shardName in validatedHeaders.keys.sorted() {
            guard let header = validatedHeaders[shardName] else {
                throw MLXFastError.invalidInput(
                    "missing validated header for checkpoint shard \(shardName)"
                )
            }
            try Safetensors.validateSourceIdentity(
                referenceDirectory.appendingPathComponent(shardName),
                against: header
            )
        }
        try validateConfigAndIndexSnapshot(
            referenceDirectory: referenceDirectory,
            referenceConfigPath: referenceConfigPath,
            runtimeConfigData: runtimeConfigData,
            indexSnapshot: indexSnapshot,
            metadataSnapshot: metadataSnapshot
        )
        try installTransformedDirectory(
            stagingDirectory,
            at: outputDirectory,
            fileManager: fileManager
        )
        installed = true

        let indexPath = outputDirectory.appendingPathComponent("model.safetensors.index.json")
        let configPath = outputDirectory.appendingPathComponent("config.json")

        return TransformReport(
            referencePath: referenceDirectory.path,
            outputPath: outputDirectory.path,
            denseTensorCount: copiedTensors
                + generatedProjectionMetadata.tensorCount
                + generatedTiedHeadMetadata.tensorCount,
            denseShardCount: textKeysByShard.count
                + generatedProjectionMetadata.shardCount
                + generatedTiedHeadMetadata.shardCount,
            configPath: configPath.path,
            indexPath: indexPath.path
        )
    }

    private static func loadIndex(_ referenceDirectory: URL) throws -> CheckpointIndex {
        let indexPath = referenceDirectory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexPath.path) {
            return try CheckpointIndex.load(from: indexPath)
        }
        return try CheckpointIndex.buildFromSafetensors(in: referenceDirectory)
    }

    private static func validateCheckpointIndex(
        _ index: CheckpointIndex,
        referenceDirectory: URL
    ) throws -> [String: SafetensorsHeader] {
        guard !index.weightMap.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint index contains no tensors")
        }

        let keysByShard = Dictionary(grouping: index.weightMap.keys.sorted()) { key in
            index.weightMap[key] ?? ""
        }
        var headersByShard: [String: SafetensorsHeader] = [:]
        for shardName in keysByShard.keys.sorted() {
            try validateSafetensorsShardName(shardName, context: "checkpoint index")

            let shardURL = referenceDirectory.appendingPathComponent(shardName)
            try requireFile(shardURL.path, description: "checkpoint shard \(shardName)")
            let header = try Safetensors.readHeader(shardURL)
            headersByShard[shardName] = header

            for key in keysByShard[shardName, default: []].sorted() {
                guard let info = header.tensors[key] else {
                    throw MLXFastError.invalidInput(
                        "checkpoint index lists tensor \(key) in \(shardName), but the shard header does not contain it"
                    )
                }
                let dtype = try TensorDType.parse(info.dtype)
                let expectedByteLength = try expectedTensorByteCount(
                    name: key,
                    dtype: dtype,
                    shape: info.shape
                )
                guard info.byteCount == expectedByteLength else {
                    throw MLXFastError.invalidInput(
                        "checkpoint tensor \(key) byte length \(info.byteCount) does not match dtype \(info.dtype) and shape \(info.shape) expected \(expectedByteLength)"
                    )
                }
                guard info.dataStart >= 0, info.byteCount > 0 else {
                    throw MLXFastError.invalidInput(
                        "checkpoint tensor \(key) has an empty or invalid byte range"
                    )
                }
            }
        }
        return headersByShard
    }

    private static func canonicalURL(_ url: URL) -> URL {
        url.standardizedFileURL.resolvingSymlinksInPath()
    }

    static func validateDistinctDirectories(
        referenceDirectory: URL,
        outputDirectory: URL,
        workingDirectory: URL = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true
        )
    ) throws {
        guard referenceDirectory.path != outputDirectory.path else {
            throw MLXFastError.invalidInput(
                "transform reference and output directories must be different: \(referenceDirectory.path)"
            )
        }
        let outputPrefix = outputDirectory.path == "/" ? "/" : outputDirectory.path + "/"
        guard !referenceDirectory.path.hasPrefix(outputPrefix) else {
            throw MLXFastError.invalidInput(
                "transform output directory cannot contain the reference directory: \(outputDirectory.path)"
            )
        }
        let referencePrefix = referenceDirectory.path == "/"
            ? "/"
            : referenceDirectory.path + "/"
        guard !outputDirectory.path.hasPrefix(referencePrefix) else {
            throw MLXFastError.invalidInput(
                "transform output directory cannot be inside the reference directory: \(outputDirectory.path)"
            )
        }

        let canonicalWorkingDirectory = canonicalURL(workingDirectory)
        guard canonicalWorkingDirectory.path != outputDirectory.path,
              !canonicalWorkingDirectory.path.hasPrefix(outputPrefix)
        else {
            throw MLXFastError.invalidInput(
                "transform output directory cannot contain the current working directory: \(outputDirectory.path)"
            )
        }
    }

    private static func installTransformedDirectory(
        _ stagedDirectory: URL,
        at outputDirectory: URL,
        fileManager: FileManager
    ) throws {
        if fileManager.fileExists(atPath: outputDirectory.path) {
            _ = try fileManager.replaceItemAt(outputDirectory, withItemAt: stagedDirectory)
        } else {
            try fileManager.moveItem(at: stagedDirectory, to: outputDirectory)
        }
    }

    private static func findReferenceDirectory(_ base: URL) throws -> URL {
        if FileManager.default.fileExists(
            atPath: base.appendingPathComponent("config.json").path
        ) {
            return base
        }

        guard let enumerator = FileManager.default.enumerator(
            at: base,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw MLXFastError.missingFile("reference path not found at \(base.path)")
        }

        for case let url as URL in enumerator {
            if url.lastPathComponent == "config.json" {
                return url.deletingLastPathComponent()
            }
        }

        throw MLXFastError.missingFile(
            "no config.json found under \(base.path); place the pinned reference checkpoint there"
        )
    }

    static func isTextTowerKey(_ key: String) -> Bool {
        key.hasPrefix(textTowerPrefix)
    }

    static func isSelectedTextTowerKey(_ key: String, family: TransformModelFamily) -> Bool {
        switch family {
        case .gemma4:
            return isTextTowerKey(key)
        case .laguna:
            guard key.hasPrefix("model.") || key.hasPrefix("lm_head.") else {
                return false
            }
            return !key.contains("rotary_emb.inv_freq")
        }
    }

    private static func captureMetadataFiles(from source: URL) throws -> [String: Data] {
        let files = try FileManager.default.contentsOfDirectory(
            at: source,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey]
        )
        var snapshot: [String: Data] = [:]
        for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            if file.lastPathComponent == "model.safetensors.index.json" || file.lastPathComponent == "config.json" {
                continue
            }
            if shouldCopyMetadataFile(file) {
                let values = try file.resourceValues(
                    forKeys: [.isRegularFileKey, .isSymbolicLinkKey]
                )
                guard values.isRegularFile == true, values.isSymbolicLink != true else {
                    throw MLXFastError.invalidInput(
                        "reference metadata is not a regular file: \(file.path)"
                    )
                }
                snapshot[file.lastPathComponent] = try Data(contentsOf: file)
            }
        }
        return snapshot
    }

    private static func writeMetadataFiles(
        _ snapshot: [String: Data],
        to destination: URL
    ) throws {
        for name in snapshot.keys.sorted() {
            try snapshot[name]?.write(to: destination.appendingPathComponent(name))
        }
    }

    private static func shouldCopyMetadataFile(_ url: URL) -> Bool {
        let name = url.lastPathComponent
        if name.hasSuffix(".safetensors") {
            return false
        }
        switch url.pathExtension {
        case "json", "model", "tiktoken", "txt":
            return true
        default:
            return name == "tokenizer" || name == "vocab"
        }
    }

    private static func cloneOrCopyShard(from source: URL, to destination: URL) throws {
        #if canImport(Darwin)
        let cloned = source.path.withCString { sourcePath in
            destination.path.withCString { destinationPath in
                clonefile(sourcePath, destinationPath, 0) == 0
            }
        }
        if !cloned {
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.copyItem(at: source, to: destination)
        }
        #else
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.copyItem(at: source, to: destination)
        #endif
        let values = try destination.resourceValues(
            forKeys: [.isRegularFileKey, .isSymbolicLinkKey]
        )
        guard values.isRegularFile == true, values.isSymbolicLink != true else {
            throw MLXFastError.invalidInput(
                "transformed shard copy is not an independent regular file: \(destination.path)"
            )
        }
    }

    private static func loadReferenceConfigRoot(_ sourceConfigPath: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: sourceConfigPath)
        let object = try JSONSerialization.jsonObject(with: data)
        guard let root = object as? [String: Any] else {
            throw MLXFastError.invalidInput("reference config.json must be a JSON object")
        }
        return root
    }

    static func detectModelFamily(
        sourceConfigRoot root: [String: Any]
    ) throws -> TransformModelFamily {
        if root["text_config"] is [String: Any] {
            return .gemma4
        }
        if let modelType = root["model_type"] as? String, modelType == "laguna" {
            return .laguna
        }
        throw MLXFastError.invalidInput(
            "reference config.json is missing text_config and does not declare model_type laguna"
        )
    }

    static func makeRuntimeConfigData(sourceConfigPath: URL) throws -> Data {
        let root = try loadReferenceConfigRoot(sourceConfigPath)
        return try makeRuntimeConfigData(
            sourceConfigRoot: root,
            family: detectModelFamily(sourceConfigRoot: root)
        )
    }

    /// Writes the runtime's `config.json`.
    static func makeRuntimeConfigData(
        sourceConfigRoot root: [String: Any],
        family: TransformModelFamily
    ) throws -> Data {
        var runtimeConfig: [String: Any]
        switch family {
        case .gemma4:
            guard let textConfig = root["text_config"] as? [String: Any] else {
                throw MLXFastError.invalidInput("reference config.json is missing text_config")
            }
            runtimeConfig = textConfig
            if let quantization = root["quantization"] {
                runtimeConfig["quantization"] = quantization
            } else if let quantizationConfig = root["quantization_config"] {
                runtimeConfig["quantization"] = quantizationConfig
            }
        case .laguna:
            _ = try LagunaCheckpointValidation.quantizationSpec(fromConfigRoot: root)
            runtimeConfig = root
            runtimeConfig["vision_config"] = nil
        }

        return try JSONSerialization.data(
            withJSONObject: runtimeConfig,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
    }

    private static func validateConfigAndIndexSnapshot(
        referenceDirectory: URL,
        referenceConfigPath: URL,
        runtimeConfigData: Data,
        indexSnapshot: Data,
        metadataSnapshot: [String: Data]
    ) throws {
        guard try makeRuntimeConfigData(sourceConfigPath: referenceConfigPath)
            == runtimeConfigData
        else {
            throw MLXFastError.invalidInput(
                "reference config changed while transform was running"
            )
        }
        guard try loadIndex(referenceDirectory).canonicalData() == indexSnapshot else {
            throw MLXFastError.invalidInput(
                "checkpoint index changed while transform was running"
            )
        }
        guard try captureMetadataFiles(from: referenceDirectory) == metadataSnapshot else {
            throw MLXFastError.invalidInput(
                "reference tokenizer metadata changed while transform was running"
            )
        }
    }
}
