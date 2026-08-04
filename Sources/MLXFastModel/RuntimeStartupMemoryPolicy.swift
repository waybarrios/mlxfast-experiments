import Darwin
import MLX

/// Selects a model-startup profile from the machine's physical-memory budget.
public struct RuntimeStartupMemoryPolicy: Equatable, Sendable {
    public static let fullProfileMinimumPhysicalMemoryBytes = UInt64(64) << 30

    /// Environment name for the explicit profile override. It must keep the
    public static let profileOverrideEnvironmentName =
        "DARKBLOOM_STARTUP_MEMORY_PROFILE"

    public let isLowMemory: Bool
    /// Why this profile was selected; quoted in the stderr notice.
    public let selectionReason: String
    public let cacheLimitBytes: Int
    public let maxMegabytesPerCommandBuffer: Int
    public let maxOperationsPerCommandBuffer: Int
    public let clearAllocatorCacheAfterWarmup: Bool
    public let environmentOverrides: [String: String]

    public static func resolve(
        physicalMemoryBytes: UInt64,
        requestedProfile: String? = nil
    ) -> RuntimeStartupMemoryPolicy {
        let lowMemory: Bool
        let selectionReason: String
        switch requestedProfile?.lowercased() ?? "" {
        case "", "auto":
            lowMemory = physicalMemoryBytes < fullProfileMinimumPhysicalMemoryBytes
            selectionReason = "physical memory \(physicalMemoryBytes >> 30) GiB "
                + (lowMemory ? "is below" : "meets")
                + " the \(fullProfileMinimumPhysicalMemoryBytes >> 30) GiB full-profile minimum"
        case "full":
            lowMemory = false
            selectionReason = "\(profileOverrideEnvironmentName)=full"
        case "low":
            lowMemory = true
            selectionReason = "\(profileOverrideEnvironmentName)=low"
        default:
            preconditionFailure(
                "\(profileOverrideEnvironmentName) must be auto, full, or low"
            )
        }

        if lowMemory {
            return RuntimeStartupMemoryPolicy(
                isLowMemory: true,
                selectionReason: selectionReason,
                cacheLimitBytes: 6 << 30,
                // Half the full profile's referenced-byte and op budgets:
                maxMegabytesPerCommandBuffer: 128,
                maxOperationsPerCommandBuffer: 64,
                clearAllocatorCacheAfterWarmup: true,
                // No feature-disable defaults. Compiled decode
                environmentOverrides: [:]
            )
        }

        return RuntimeStartupMemoryPolicy(
            isLowMemory: false,
            selectionReason: selectionReason,
            // Ranked/full profile -- byte-identical to the constants this
            cacheLimitBytes: 32 << 30,
            // The MLX M5 Max default commits a command buffer after
            maxMegabytesPerCommandBuffer: 320,
            maxOperationsPerCommandBuffer: 128,
            clearAllocatorCacheAfterWarmup: false,
            environmentOverrides: [:]
        )
    }

    /// The environment work `apply()` will perform, split into defaults to
    func environmentPlan(
        existingValue: (String) -> String?
    ) -> RuntimeStartupMemoryEnvironmentPlan {
        var defaultsToApply: [String: String] = [:]
        var preservedUserValues: [String: String] = [:]
        for (name, value) in environmentOverrides {
            if let existing = existingValue(name) {
                preservedUserValues[name] = existing
            } else {
                defaultsToApply[name] = value
            }
        }
        var noticeLines: [String] = []
        if isLowMemory {
            noticeLines.append(
                "mlxfast: low-memory startup profile active (\(selectionReason)): "
                    + "capping the MLX allocator cache at \(cacheLimitBytes >> 30) GiB and "
                    + "clearing free warmup buffers; compiled decode and every other "
                    + "ranked code path stay enabled; set "
                    + "\(Self.profileOverrideEnvironmentName)=full to opt out"
            )
            noticeLines.append(
                "mlxfast: a machine too small for the model plus the decode working "
                    + "set fails with an out-of-memory error instead of silently "
                    + "skipping ranked code paths; verify on a "
                    + "\(Self.fullProfileMinimumPhysicalMemoryBytes >> 30) GiB+ machine "
                    + "or rely on the ranked run"
            )
            if !preservedUserValues.isEmpty {
                let preserved = preservedUserValues.keys.sorted()
                    .map { name in "\(name)=\(preservedUserValues[name] ?? "")" }
                    .joined(separator: " ")
                noticeLines.append(
                    "mlxfast: low-memory startup profile preserved user-set flags: "
                        + preserved
                )
            }
        }
        return RuntimeStartupMemoryEnvironmentPlan(
            defaultsToApply: defaultsToApply,
            preservedUserValues: preservedUserValues,
            noticeLines: noticeLines
        )
    }

    func apply() {
        setenv(
            "MLX_MAX_MB_PER_BUFFER",
            String(maxMegabytesPerCommandBuffer),
            1
        )
        setenv(
            "MLX_MAX_OPS_PER_BUFFER",
            String(maxOperationsPerCommandBuffer),
            1
        )
        let plan = environmentPlan { name in
            getenv(name).map { String(cString: $0) }
        }
        for (name, value) in plan.defaultsToApply {
            setenv(name, value, 0)
        }
        for line in plan.noticeLines {
            fputs(line + "\n", stderr)
        }
        Memory.cacheLimit = cacheLimitBytes
    }
}

/// See `RuntimeStartupMemoryPolicy.environmentPlan(existingValue:)`.
struct RuntimeStartupMemoryEnvironmentPlan: Equatable, Sendable {
    let defaultsToApply: [String: String]
    let preservedUserValues: [String: String]
    let noticeLines: [String]
}
