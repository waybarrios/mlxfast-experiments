// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// A `LogitSampler` is responsible for sampling `logits` produced by
public protocol LogitSampler {

    /// Given `logits` produce a new `MLXArray` with the token.
    func sample(logits: MLXArray) -> MLXArray
}

/// A `LogitProcessor` is an optional visitor of `logits`.
public protocol LogitProcessor {

    /// Called before token generation starts with the text tokens of the prompt
    mutating func prompt(_ prompt: MLXArray)

    /// Called to visit and possibly modify the logits
    func process(logits: MLXArray) -> MLXArray

    /// Called to provide the sampled token
    mutating func didSample(token: MLXArray)
}

/// Parameters for text generation, see ``TokenIterator``.
public struct GenerateParameters: Sendable {

    /// Step size for processing the prompt
    public var prefillStepSize: Int

    /// Maximum tokens to generate
    public var maxTokens: Int?

    public var maxKVSize: Int?

    /// Number of bits to use for KV cache quantization. nil implies no cache quantization.
    public var kvBits: Int?

    /// Group size for KV cache quantization (default: 64)
    public var kvGroupSize: Int

    /// Step to begin using a quantized KV cache when kvBits is non-nil (default: 0)
    public var quantizedKVStart: Int

    /// Sampling temperature
    public var temperature: Float

    /// Top-p sampling
    public var topP: Float

    /// Top-k sampling (0 disables)
    public var topK: Int

    /// Min-p sampling threshold relative to the highest probability token (0 disables)
    public var minP: Float

    /// Penalty factor for repeating tokens
    public var repetitionPenalty: Float?

    /// Number of tokens to consider for repetition penalty
    public var repetitionContextSize: Int

    /// additive penalty for tokens that appear in recent context
    public var presencePenalty: Float?

    /// number of tokens to consider for presence penalty
    public var presenceContextSize: Int

    /// additive penalty that scales with token frequency in recent context
    public var frequencyPenalty: Float?

    /// number of tokens to consider for frequency penalty
    public var frequencyContextSize: Int

    public init(
        maxTokens: Int? = nil,
        maxKVSize: Int? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20,
        presencePenalty: Float? = nil,
        presenceContextSize: Int = 20,
        frequencyPenalty: Float? = nil,
        frequencyContextSize: Int = 20,
        prefillStepSize: Int = 512
    ) {
        self.maxTokens = maxTokens
        self.maxKVSize = maxKVSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.presencePenalty = presencePenalty
        self.presenceContextSize = presenceContextSize
        self.frequencyPenalty = frequencyPenalty
        self.frequencyContextSize = frequencyContextSize
        self.prefillStepSize = prefillStepSize
    }

    public func sampler() -> LogitSampler {
        let usesTopP = topP > 0 && topP < 1
        let usesTopK = topK > 0
        let usesMinP = minP > 0

        if temperature == 0 {
            return ArgMaxSampler()
        } else if usesTopP || usesTopK || usesMinP {
            return TopPSampler(temperature: temperature, topP: topP, topK: topK, minP: minP)
        } else {
            return CategoricalSampler(temperature: temperature)
        }
    }

    public func processor() -> LogitProcessor? {
        let repetitionContext: RepetitionContext?
        if let repetitionPenalty, repetitionPenalty != 0, repetitionPenalty != 1,
            repetitionContextSize > 0
        {
            repetitionContext = RepetitionContext(
                repetitionPenalty: repetitionPenalty,
                repetitionContextSize: repetitionContextSize
            )
        } else {
            repetitionContext = nil
        }

        let presenceContext: PresencePenaltyContext?
        if let presencePenalty, presencePenalty != 0, presenceContextSize > 0 {
            presenceContext = PresencePenaltyContext(
                presencePenalty: presencePenalty,
                presenceContextSize: presenceContextSize
            )
        } else {
            presenceContext = nil
        }

        let frequencyContext: FrequencyPenaltyContext?
        if let frequencyPenalty, frequencyPenalty != 0, frequencyContextSize > 0 {
            frequencyContext = FrequencyPenaltyContext(
                frequencyPenalty: frequencyPenalty,
                frequencyContextSize: frequencyContextSize
            )
        } else {
            frequencyContext = nil
        }

        if repetitionContext == nil && presenceContext == nil && frequencyContext == nil {
            return nil
        }

        return PenaltyProcessor(
            repetitionContext: repetitionContext,
            presenceContext: presenceContext,
            frequencyContext: frequencyContext
        )
    }
}

/// Sampler that uses `argMax` (most likely) to sample the logits.
public struct ArgMaxSampler: LogitSampler {
    public init() {}

    public func sample(logits: MLXArray) -> MLXArray {
        argMax(logits, axis: -1)
    }
}

/// Sampler that uses probability filters (`topP`, `topK`, `minP`) and `temperature`
public struct TopPSampler: LogitSampler {
    let temp: MLXArray
    let topP: MLXArray?
    let topK: Int?
    let minP: MLXArray?
    let negInf: MLXArray
    let randomState: MLXRandom.RandomState

    public init(temperature: Float, topP: Float = 1.0, topK: Int = 0, minP: Float = 0.0) {
        self.temp = MLXArray(temperature)
        if topP > 0 && topP < 1 {
            self.topP = MLXArray(topP)
        } else {
            self.topP = nil
        }
        self.topK = topK > 0 ? topK : nil
        self.minP = minP > 0 ? MLXArray(minP) : nil
        self.negInf = MLXArray(-Float.infinity)
        self.randomState = MLXRandom.RandomState()
    }

    public func sample(logits: MLXArray) -> MLXArray {
        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        return withRandomState(randomState) {
            var logprobs = logSoftmax(logits)

            // Apply filters in Python mlx-lm order: top_p → min_p → top_k.
            if let topP {
                logprobs = applyTopP(logprobs, topP: topP)
            }
            if let minP {
                logprobs = applyMinP(logprobs, minP: minP)
            }
            if let topK {
                logprobs = applyTopK(logprobs, topK: topK)
            }

            return categorical(logprobs * (1 / temp))
        }
    }

    private func applyTopP(_ logprobs: MLXArray, topP: MLXArray) -> MLXArray {
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let sortedProbs = exp(sortedLogprobs)
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)

        // Mask low-probability tail in sorted order, scatter back to original vocab order.
        let filtered = MLX.where(cumulativeProbs .> (1 - topP), sortedLogprobs, negInf)
        return putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }

    private func applyMinP(_ logprobs: MLXArray, minP: MLXArray) -> MLXArray {
        // threshold in log-space: log(maxProb * minP) = maxLogprob + log(minP)
        let maxLogprob = logprobs.max(axis: -1, keepDims: true)
        let threshold = maxLogprob + log(minP)
        return MLX.where(logprobs .>= threshold, logprobs, negInf)
    }

    private func applyTopK(_ logprobs: MLXArray, topK: Int) -> MLXArray {
        let vocabularySize = logprobs.dim(-1)
        guard topK < vocabularySize else { return logprobs }
        let maskIndices = argPartition(-logprobs, kth: topK - 1, axis: -1)[0..., topK...]
        return putAlong(logprobs, maskIndices, values: negInf, axis: -1)
    }
}

/// Sampler that uses `temperature` to sample the logits.
public struct CategoricalSampler: LogitSampler {
    let temp: MLXArray
    let randomState: MLXRandom.RandomState

    public init(temperature: Float) {
        self.temp = MLXArray(temperature)
        self.randomState = MLXRandom.RandomState()
    }

    public func sample(logits: MLXArray) -> MLXArray {
        return withRandomState(randomState) {
            categorical(logits * (1 / temp))
        }
    }
}

/// GPU-resident ring buffer of recent token IDs.
struct TokenRing {
    private(set) var buffer: MLXArray
    private(set) var count = 0
    private var writeIndex = 0
    let capacity: Int
    private let positions: MLXArray

    init(capacity: Int) {
        precondition(capacity > 0)
        self.capacity = capacity
        self.buffer = MLXArray.zeros([capacity], type: Int32.self)
        self.positions = MLXArray.arange(capacity)
    }

    /// The valid portion of the ring (all of it once full), or `nil` if empty.
    var validTokens: MLXArray? {
        guard count > 0 else { return nil }
        return count < capacity ? buffer[..<count] : buffer
    }

    /// Bulk-load from a prompt, keeping the last `capacity` tokens.
    mutating func loadPrompt(_ prompt: MLXArray) {
        let promptTokens = prompt.reshaped(-1).asType(.int32)
        let n = promptTokens.dim(0)
        if n <= capacity {
            if n < capacity {
                let padding = MLXArray.zeros([capacity - n], type: Int32.self)
                buffer = concatenated([promptTokens, padding])
            } else {
                buffer = promptTokens
            }
            count = n
            writeIndex = n % capacity
        } else {
            buffer = promptTokens[(n - capacity)...]
            count = capacity
            writeIndex = 0
        }
    }

    /// Append a single token using GPU-only mask write (no CPU←GPU sync).
    mutating func append(_ token: MLXArray) {
        let mask = positions .== Int32(writeIndex)
        buffer = MLX.where(mask, token.asType(.int32), buffer)
        writeIndex = (writeIndex + 1) % capacity
        count = min(count + 1, capacity)
    }
}

/// Processor that implements a `repetitionPenalty`.
public struct RepetitionContext: LogitProcessor {
    private var ring: TokenRing
    let repetitionPenalty: Float

    public init(repetitionPenalty: Float, repetitionContextSize: Int) {
        self.repetitionPenalty = repetitionPenalty
        self.ring = TokenRing(capacity: repetitionContextSize)
    }

    mutating public func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let indices = ring.validTokens?.asType(.uint32) else { return logits }
        var selectedLogits = logits[0..., indices]

        selectedLogits = MLX.where(
            selectedLogits .< 0, selectedLogits * repetitionPenalty,
            selectedLogits / repetitionPenalty)

        logits[0..., indices] = selectedLogits
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that applies an additive presence penalty to tokens in a recent context window.
public struct PresencePenaltyContext: LogitProcessor {
    private var ring: TokenRing
    let presencePenalty: Float

    public init(presencePenalty: Float, presenceContextSize: Int) {
        self.presencePenalty = presencePenalty
        self.ring = TokenRing(capacity: presenceContextSize)
    }

    mutating public func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let indices = ring.validTokens?.asType(.uint32) else { return logits }
        logits[0..., indices] = logits[0..., indices] - presencePenalty
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that applies an additive frequency penalty to tokens in a recent context window.
public struct FrequencyPenaltyContext: LogitProcessor {
    private var ring: TokenRing
    let frequencyPenalty: Float

    public init(frequencyPenalty: Float, frequencyContextSize: Int) {
        self.frequencyPenalty = frequencyPenalty
        self.ring = TokenRing(capacity: frequencyContextSize)
    }

    mutating public func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let validTokens = ring.validTokens else { return logits }

        let vocabSize = logits.dim(-1)
        let ones = MLXArray.ones([validTokens.dim(0)], type: Float32.self)
        let histogram = MLXArray.zeros([vocabSize], type: Float32.self)
            .at[validTokens.asType(.int32)].add(ones)

        return logits - (histogram * frequencyPenalty).reshaped(1, -1)
    }

    mutating public func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that composes penalty processors in Python mlx-lm order.
public struct PenaltyProcessor: LogitProcessor {
    var repetitionContext: RepetitionContext?
    var presenceContext: PresencePenaltyContext?
    var frequencyContext: FrequencyPenaltyContext?

    public init(
        repetitionContext: RepetitionContext?,
        presenceContext: PresencePenaltyContext?,
        frequencyContext: FrequencyPenaltyContext?
    ) {
        self.repetitionContext = repetitionContext
        self.presenceContext = presenceContext
        self.frequencyContext = frequencyContext
    }

    mutating public func prompt(_ prompt: MLXArray) {
        repetitionContext?.prompt(prompt)
        presenceContext?.prompt(prompt)
        frequencyContext?.prompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        var logits = logits
        logits = repetitionContext?.process(logits: logits) ?? logits
        logits = presenceContext?.process(logits: logits) ?? logits
        logits = frequencyContext?.process(logits: logits) ?? logits
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        repetitionContext?.didSample(token: token)
        presenceContext?.didSample(token: token)
        frequencyContext?.didSample(token: token)
    }
}

/// Common properties shared by token-generating iterators.
public protocol TokenIteratorProtocol: Sequence, IteratorProtocol where Element == Int {
    var maxTokens: Int? { get }
    var tokenCount: Int { get }
    var promptPrefillTime: TimeInterval { get }
}

/// Generator of tokens.
public struct TokenIterator: TokenIteratorProtocol {
    let model: any LanguageModel
    var state: LMOutput.State?

    var y: LMInput.Text
    var cache: [KVCache]
    var processor: LogitProcessor?
    let sampler: LogitSampler

    public var tokenCount = 0
    public let maxTokens: Int?

    // Cache quantization parameters
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int

    // Internal metrics
    public var promptPrefillTime: TimeInterval = 0.0

    /// Initialize a `TokenIterator` with the given tokens. Note: this has been
    @available(*, deprecated, message: "please use init(input:model:cache:parameters:)")
    public init(
        prompt: MLXArray, model: any LanguageModel, cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        self.model = model
        self.y = .init(tokens: prompt)
        self.cache = cache ?? model.newCache(parameters: parameters)

        self.processor = parameters.processor()
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens

        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart

        self.promptPrefillTime = try measure {
            try prepare(input: .init(text: y), windowSize: parameters.prefillStepSize)
        }
    }

    /// Initialize a `TokenIterator` with the given input.
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        self.model = model
        self.y = input.text
        self.cache = cache ?? model.newCache(parameters: parameters)

        self.processor = parameters.processor()
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens

        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: parameters.prefillStepSize)
        }
    }

    /// Initialize a `TokenIterator` with the given input and logit handling.
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        processor: LogitProcessor?, sampler: LogitSampler, prefillStepSize: Int = 512,
        maxTokens: Int? = nil
    ) throws {
        self.model = model
        self.y = input.text
        self.cache = cache ?? model.newCache(parameters: nil)

        self.processor = processor
        self.sampler = sampler
        self.maxTokens = maxTokens

        // No cache quantization for this direct initialization
        self.kvBits = nil
        self.kvGroupSize = 64
        self.quantizedKVStart = 0

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: prefillStepSize)
        }
    }

    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        switch try model.prepare(input, cache: cache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens

            // evaluate the remainder of the prompt -- this primes the pump
            let token = step(previous: y)
            y = .init(tokens: token)
            asyncEval(y.tokens)

        case .logits(let result):
            y = .init(tokens: convertToToken(logits: result.logits))
            asyncEval(y.tokens)

            break
        }
    }

    mutating func convertToToken(logits: MLXArray) -> MLXArray {
        // process the logits (one hot array of possible tokens)
        var logits = logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits

        // transform logits back to a token
        let y = sampler.sample(logits: logits)

        processor?.didSample(token: y)

        return y
    }

    /// Evaluate the next token and return the new token (y), updating cache state
    mutating func step(previous: LMInput.Text) -> MLXArray {
        let result = model(
            previous[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: state)
        self.state = result.state

        // Apply dynamic cache quantization after each step
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart
        )

        return convertToToken(logits: result.logits)
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // save current value -- this will be returned
        let previousY = y

        // compute the next state and async eval the next token
        let token = step(previous: previousY)
        y = .init(tokens: token)
        asyncEval(token)

        tokenCount += 1

        return previousY.tokens.item(Int.self)
    }
}

/// Generator of tokens using speculative decoding.
public struct SpeculativeTokenIterator: TokenIteratorProtocol {

    var y: LMInput.Text
    var draftY: LMInput.Text

    let mainModel: any LanguageModel
    let draftModel: any LanguageModel

    var mainState: LMOutput.State?
    var mainCache: [KVCache]
    var draftCache: [KVCache]
    let quantizeKVCache: (inout [KVCache]) -> Void

    var processor: LogitProcessor?
    let sampler: LogitSampler

    public var tokenCount = 0
    public let maxTokens: Int?
    let numDraftTokens: Int

    // Buffer of accepted tokens from the current speculation round
    private var pendingTokens = [Int]()
    private var pendingIndex = 0

    // Internal metrics
    public var promptPrefillTime: TimeInterval = 0.0

    /// Initialize a `SpeculativeTokenIterator` with the given input.
    public init(
        input: LMInput,
        mainModel: any LanguageModel,
        draftModel: any LanguageModel,
        mainCache: [KVCache]? = nil,
        draftCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int
    ) throws {
        self.y = input.text
        self.draftY = input.text
        self.mainModel = mainModel
        self.draftModel = draftModel

        self.mainCache = mainCache ?? mainModel.newCache(parameters: parameters)
        self.draftCache = draftCache ?? draftModel.newCache(parameters: parameters)
        guard canTrimPromptCache(self.mainCache), canTrimPromptCache(self.draftCache) else {
            throw KVCacheError(message: "Speculative decoding requires trimmable KV caches.")
        }

        self.sampler = parameters.sampler()
        self.processor = parameters.processor()

        self.maxTokens = parameters.maxTokens
        self.numDraftTokens = numDraftTokens

        self.quantizeKVCache = { cache in
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart
            )
        }

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: parameters.prefillStepSize)
        }
    }

    /// Prefill both main and draft models with the prompt, priming caches for generation
    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        // Prefill main model
        switch try mainModel.prepare(input, cache: mainCache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens
        case .logits(let result):
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            mainState = result.state
        }

        // Prefill draft model, don't call didSample here --
        switch try draftModel.prepare(input, cache: draftCache, windowSize: windowSize) {
        case .tokens(let tokens):
            draftY = tokens
        case .logits(let result):
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            draftY = .init(tokens: token)
            asyncEval(draftY.tokens)
        }
    }

    /// Run one round of speculative decoding: draft, verify, accept/reject
    mutating func speculateRound() {
        let remaining = maxTokens.map { $0 - tokenCount } ?? numDraftTokens
        let numDraft = Swift.min(remaining, numDraftTokens)
        guard numDraft > 0 else {
            return
        }

        // Draft generation: autoregressive loop with draft model
        var draftProcessor = processor  // Copy to discard later
        var draftTokens = [MLXArray]()
        for _ in 0 ..< numDraft {
            let draftResult = draftModel(draftY[text: .newAxis], cache: draftCache, state: nil)
            var draftLogits = draftResult.logits[0..., -1, 0...]
            draftLogits = draftProcessor?.process(logits: draftLogits) ?? draftLogits
            let draftToken = sampler.sample(logits: draftLogits)
            draftProcessor?.didSample(token: draftToken)
            asyncEval(draftToken)
            draftTokens.append(draftToken)
            draftY = .init(tokens: draftToken)
        }

        // Verification: main model processes proposals in one pass
        let verifyTokens = [y.tokens] + draftTokens
        let verifyInput = LMInput.Text(tokens: concatenated(verifyTokens))
        let verifyStart = verifyInput.tokens.dim(0) - (numDraft + 1)
        let mainResult = mainModel(verifyInput[text: .newAxis], cache: mainCache, state: mainState)
        let mainLogits = mainResult.logits
        mainState = mainResult.state

        let mainTokens: MLXArray
        if var verifyProcessor = processor {
            // Process each position sequentially so that th
            var sampled = [MLXArray]()
            for i in 0 ..< (numDraft + 1) {
                var logits = mainLogits[0..., verifyStart + i, 0...]
                logits = verifyProcessor.process(logits: logits)
                let token = sampler.sample(logits: logits)
                verifyProcessor.didSample(token: token)
                sampled.append(token)
            }
            mainTokens = concatenated(sampled)
        } else {
            // Batch-sample all verify tokens from main model in one operation
            let verifyLogits = mainLogits[0..., verifyStart..., 0...].squeezed(axis: 0)
            mainTokens = sampler.sample(logits: verifyLogits)
        }

        // Compare and accept proposed tokens
        eval(mainTokens, draftTokens)
        let mainTokensList = mainTokens.asArray(Int.self)
        let draftTokensList = concatenated(draftTokens).asArray(Int.self)
        var accepted = 0
        for i in 0 ..< numDraft {
            guard mainTokensList[i] == draftTokensList[i] else {
                break
            }

            processor?.didSample(token: draftTokens[i])
            pendingTokens.append(mainTokensList[i])
            accepted += 1
        }

        let finalToken = mainTokens[accepted ... accepted]
        processor?.didSample(token: finalToken)
        pendingTokens.append(mainTokensList[accepted])

        // Rewind caches for rejected tokens
        trimPromptCache(mainCache, numTokens: numDraft - accepted)
        trimPromptCache(draftCache, numTokens: Swift.max(numDraft - accepted - 1, 0))

        // Apply dynamic cache quantization after rewind
        quantizeKVCache(&mainCache)
        quantizeKVCache(&draftCache)

        // Set y/draftY for the next round
        y = .init(tokens: finalToken)
        draftY = .init(tokens: finalToken)

        if accepted == numDraft {
            draftY = .init(
                tokens: concatenated([
                    draftTokens[numDraft - 1].reshaped([1]),
                    finalToken,
                ])
            )
        }
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // Drain the pending buffer first
        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return token
        }

        // Run a new speculation round
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        speculateRound()

        if pendingTokens.isEmpty {
            return nil
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }
}

/// Result of a call to a deprecated callback-based generate function.
public struct GenerateResult {

    /// Initializes a new `GenerateResult` instance.
    public init(
        inputText: LMInput.Text, tokenIds: [Int], output: String, promptTime: TimeInterval,
        generateTime: TimeInterval
    ) {
        self.inputText = inputText
        self.tokenIds = tokenIds
        self.output = output
        self.promptTime = promptTime
        self.generateTime = generateTime
    }

    @available(*, deprecated, renamed: "init(inputText:tokenIds:output:promptTime:generateTime:)")
    public init(
        inputText: LMInput.Text, tokens: [Int], output: String, promptTime: TimeInterval,
        generateTime: TimeInterval
    ) {
        self.init(
            inputText: inputText, tokenIds: tokens, output: output, promptTime: promptTime,
            generateTime: generateTime)
    }

    /// input (prompt, images, etc.)
    public let inputText: LMInput.Text

    /// The token IDs of the input prompt.
    public var promptTokenIds: [Int] {
        inputText.tokens.asArray(Int.self)
    }

    @available(*, deprecated, renamed: "promptTokenIds")
    public var promptTokens: [Int] { promptTokenIds }

    /// Generated token IDs
    public let tokenIds: [Int]

    @available(*, deprecated, renamed: "tokenIds")
    public var tokens: [Int] { tokenIds }

    /// Output text
    public let output: String

    /// The number of tokens included in the input prompt.
    public var promptTokenCount: Int { inputText.tokens.size }

    /// The number of tokens generated by the language model.
    public var generationTokenCount: Int { tokenIds.count }

    /// Time to process the prompt (generate the first token)
    public let promptTime: TimeInterval

    /// Time to generate the remaining tokens
    public let generateTime: TimeInterval

    /// The number of tokens processed per second during the prompt phase.
    public var promptTokensPerSecond: Double {
        Double(inputText.tokens.size) / promptTime
    }

    /// The number of tokens generated per second during the generation phase.
    public var tokensPerSecond: Double {
        Double(tokenIds.count) / generateTime
    }

    public func summary() -> String {
        """
        Prompt:     \(promptTokenCount) tokens, \(promptTokensPerSecond.formatted()) tokens/s, \(promptTime.formatted())s
        Generation: \(generationTokenCount) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
    }
}

/// Action from token visitor callback in deprecated callback-based generate functions.
public enum GenerateDisposition: Sendable {
    /// Keep producing tokens until an EOS token is produced
    case more

    /// Stop producing tokens, e.g. a token limit has been hit
    case stop
}

private struct SynchronousGenerationLoopResult {
    let generatedTokenIds: [Int]
    let promptTime: TimeInterval
    let generateTime: TimeInterval
    let promptPrefillTime: TimeInterval
    let stopReason: GenerateStopReason
}

private func buildStopTokenIds(
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer
) -> Set<Int> {
    // Build complete EOS token set from all sources.
    var stopTokenIds = modelConfiguration.eosTokenIds
    if let tokenizerEOS = tokenizer.eosTokenId {
        stopTokenIds.insert(tokenizerEOS)
    }
    for token in modelConfiguration.extraEOSTokens {
        if let id = tokenizer.convertTokenToId(token) {
            stopTokenIds.insert(id)
        }
    }
    return stopTokenIds
}

private func runSynchronousGenerationLoop(
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: TokenIterator,
    didGenerate: (_ token: Int, _ generatedTokenIds: [Int]) -> GenerateDisposition
) -> SynchronousGenerationLoopResult {
    var start = Date.timeIntervalSinceReferenceDate
    var promptTime: TimeInterval = 0

    let stopTokenIds = buildStopTokenIds(
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer
    )

    var generatedTokenIds = [Int]()
    var iterator = iterator
    var stopReason: GenerateStopReason?

    while let token = iterator.next() {
        // Compute the timing for the prompt.
        if promptTime == 0 {
            let now = Date.timeIntervalSinceReferenceDate
            promptTime = now - start
            start = now
        }

        // Check for end-of-sequence tokens.
        if token == tokenizer.unknownTokenId || stopTokenIds.contains(token) {
            stopReason = .stop
            break
        }

        generatedTokenIds.append(token)

        if didGenerate(token, generatedTokenIds) == .stop {
            stopReason = .cancelled
            break
        }
    }

    // If the iterator ends naturally, the max-token limit was reached.
    if stopReason == nil {
        if let maxTokens = iterator.maxTokens, iterator.tokenCount >= maxTokens {
            stopReason = .length
        } else {
            stopReason = .cancelled
        }
    }

    let now = Date.timeIntervalSinceReferenceDate
    let generateTime = now - start

    // TokenIterator uses `asyncEval()` to keep the pipeline full. If the caller
    Stream().synchronize()

    return SynchronousGenerationLoopResult(
        generatedTokenIds: generatedTokenIds,
        promptTime: promptTime,
        generateTime: generateTime,
        promptPrefillTime: iterator.promptPrefillTime,
        stopReason: stopReason ?? .cancelled
    )
}

/// Given prompt tokens generate text using the given model and parameters.
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    promptTokens: [Int], parameters: GenerateParameters, model: any LanguageModel,
    tokenizer: Tokenizer,
    extraEOSTokens: Set<String>? = nil,
    didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
    let tokens = MLXArray(promptTokens)
    let iterator = try TokenIterator(
        prompt: tokens, model: model, parameters: parameters)

    let input = LMInput(tokens: tokens)
    let configuration = ModelConfiguration(id: "stand-in", extraEOSTokens: extraEOSTokens ?? [])
    let context = ModelContext(
        configuration: configuration, model: model, processor: StandInUserInputProcessor(),
        tokenizer: tokenizer)

    return generate(
        input: input, context: context, iterator: iterator,
        didGenerate: didGenerate)
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
    let iterator = try TokenIterator(
        input: input, model: context.model, parameters: parameters)
    return generate(
        input: input, context: context, iterator: iterator,
        didGenerate: didGenerate)
}

/// Low-level token generation using a ``TokenIterator``.
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: ([Int]) -> GenerateDisposition
) -> GenerateResult {
    let result = runSynchronousGenerationLoop(
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator
    ) { _, generatedTokens in
        didGenerate(generatedTokens)
    }

    return GenerateResult(
        inputText: input.text, tokenIds: result.generatedTokenIds,
        output: context.tokenizer.decode(tokenIds: result.generatedTokenIds),
        promptTime: result.promptTime + result.promptPrefillTime,
        generateTime: result.generateTime
    )
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: (Int) -> GenerateDisposition
) throws -> GenerateCompletionInfo {
    let iterator = try TokenIterator(
        input: input, model: context.model, parameters: parameters)
    return generate(
        input: input, context: context, iterator: iterator,
        didGenerate: didGenerate)
}

/// Low-level token generation using a ``TokenIterator``.
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: (Int) -> GenerateDisposition
) -> GenerateCompletionInfo {
    let result = runSynchronousGenerationLoop(
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator
    ) { token, _ in
        didGenerate(token)
    }

    return GenerateCompletionInfo(
        promptTokenCount: input.text.tokens.size,
        generationTokenCount: result.generatedTokenIds.count,
        promptTime: result.promptTime + result.promptPrefillTime,
        generationTime: result.generateTime,
        stopReason: result.stopReason
    )
}

/// Generates tokens asynchronously using the provided langu
public func generate(
    input: LMInput, cache: [KVCache]? = nil, parameters: GenerateParameters, context: ModelContext,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<Generation> {
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters)
    let (stream, _) = generateTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket)
    return stream
}

/// Generates text and tool calls asynchronously using speculative decoding with a draft model.
public func generate(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    draftModel: any LanguageModel,
    draftCache: [KVCache]? = nil,
    numDraftTokens: Int = 2,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<Generation> {
    let iterator = try SpeculativeTokenIterator(
        input: input,
        mainModel: context.model,
        draftModel: draftModel,
        mainCache: cache,
        draftCache: draftCache,
        parameters: parameters,
        numDraftTokens: numDraftTokens
    )
    let (stream, _) = generateLoopTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: TextToolTokenLoopHandler(
            tokenizer: context.tokenizer,
            format: context.configuration.toolCallFormat ?? .json
        )
    )
    return stream
}

@available(
    *, deprecated,
    message: "use a higher level generate() call or use generateTask() for fine grained control"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) -> AsyncStream<Generation> {
    let (stream, _) = generateTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket)
    return stream
}

/// Low-level token generation using a ``TokenIterator``, returning an
public func generateTask(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming TokenIterator,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) -> (AsyncStream<Generation>, Task<Void, Never>) {
    generateLoopTask(
        promptTokenCount: promptTokenCount,
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: TextToolTokenLoopHandler(
            tokenizer: tokenizer,
            format: modelConfiguration.toolCallFormat ?? .json
        )
    )
}

/// Low-level token generation accepting any ``TokenIteratorProtocol``
public func generateTask(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming any TokenIteratorProtocol,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) -> (AsyncStream<Generation>, Task<Void, Never>) {
    generateLoopTask(
        promptTokenCount: promptTokenCount,
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: TextToolTokenLoopHandler(
            tokenizer: tokenizer,
            format: modelConfiguration.toolCallFormat ?? .json
        )
    )
}

/// Generates raw token IDs asynchronously using the provide
public func generateTokens(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    includeStopToken: Bool = false,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<TokenGeneration> {
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters)
    let (stream, _) = generateTokenTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        includeStopToken: includeStopToken,
        wiredMemoryTicket: wiredMemoryTicket
    )
    return stream
}

/// Generates raw token IDs asynchronously using speculative decoding with a draft model.
public func generateTokens(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    draftModel: any LanguageModel,
    draftCache: [KVCache]? = nil,
    numDraftTokens: Int = 2,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<TokenGeneration> {
    let iterator = try SpeculativeTokenIterator(
        input: input,
        mainModel: context.model,
        draftModel: draftModel,
        mainCache: cache,
        draftCache: draftCache,
        parameters: parameters,
        numDraftTokens: numDraftTokens
    )
    let (stream, _) = generateLoopTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: RawTokenLoopHandler()
    )
    return stream
}

/// Generates raw token IDs asynchronously and returns the stream plus a `Task`.
public func generateTokensTask(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    includeStopToken: Bool = false,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters)
    return generateTokenTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        includeStopToken: includeStopToken,
        wiredMemoryTicket: wiredMemoryTicket
    )
}

/// Low-level raw token generation using a `TokenIterator`, returning an
public func generateTokenTask(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming TokenIterator,
    includeStopToken: Bool = false,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {
    generateLoopTask(
        promptTokenCount: promptTokenCount,
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        includeStopToken: includeStopToken,
        handler: RawTokenLoopHandler()
    )
}

private func generateLoopTask<Handler: TokenLoopHandler>(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming any TokenIteratorProtocol,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    includeStopToken: Bool = false,
    handler: consuming Handler
) -> (AsyncStream<Handler.Output>, Task<Void, Never>) {

    let (stream, continuation) = AsyncStream<Handler.Output>.makeStream()

    let iterator = SendableBox(iterator)
    let handler = SendableBox(handler)

    // Launch a Task to perform iteration asynchronously.
    let task = Task {
        let performIteration = {
            let iterator = iterator.consume()
            var handler = handler.consume()

            var start = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0
            var tokenCount = 0
            var stopReason: GenerateStopReason?

            let stopTokenIds = buildStopTokenIds(
                modelConfiguration: modelConfiguration,
                tokenizer: tokenizer
            )

            for token in iterator {
                // Check for cancellation on every loop iteration.
                if Task.isCancelled {
                    stopReason = .cancelled
                    break
                }

                if promptTime == 0 {
                    let now = Date.timeIntervalSinceReferenceDate
                    promptTime = now - start
                    start = now
                }

                // Check for end-of-sequence tokens
                if token == tokenizer.unknownTokenId || stopTokenIds.contains(token) {
                    if includeStopToken {
                        tokenCount += 1
                        if !handler.onStopToken(token, emit: continuation.yield) {
                            stopReason = .cancelled
                            break
                        }
                    }
                    stopReason = .stop
                    break
                }

                tokenCount += 1
                if !handler.onToken(token, emit: continuation.yield) {
                    stopReason = .cancelled
                    break
                }
            }

            if stopReason == nil {
                if Task.isCancelled {
                    stopReason = .cancelled
                } else if let maxTokens = iterator.maxTokens, iterator.tokenCount >= maxTokens {
                    stopReason = .length
                } else {
                    stopReason = .cancelled
                }
            }

            handler.onGenerationEnd(emit: continuation.yield)

            let now = Date.timeIntervalSinceReferenceDate
            let generateTime = now - start

            let info = GenerateCompletionInfo(
                promptTokenCount: promptTokenCount,
                generationTokenCount: tokenCount,
                promptTime: promptTime + iterator.promptPrefillTime,
                generationTime: generateTime,
                stopReason: stopReason ?? .cancelled
            )
            _ = continuation.yield(handler.infoEvent(info))

            // Synchronize with the stream to ensure tasks are completed
            Stream().synchronize()

            // Finalize the stream
            continuation.finish()
        }

        if let ticket = wiredMemoryTicket {
            await WiredMemoryTicket.withWiredLimit(ticket) {
                performIteration()
            }
        } else {
            performIteration()
        }
    }

    // When the consumer cancels (or ends) the stream, cancel our underlying task.
    continuation.onTermination = { termination in
        if case .cancelled = termination {
            task.cancel()
        }
    }

    return (stream, task)
}

/// Measures the execution time of a closure.
private func measure(_ closure: () throws -> Void) rethrows -> TimeInterval {
    let start = Date.timeIntervalSinceReferenceDate
    try closure()
    return Date.timeIntervalSinceReferenceDate - start
}

// MARK: - Generation structs

/// Reason why token generation stopped.
public enum GenerateStopReason: Sendable {
    /// Generation stopped because an EOS/unknown stop token was encountered.
    case stop

    /// Generation stopped because the configured max token limit was reached.
    case length

    /// Generation stopped due to explicit task cancellation or early stream termination.
    case cancelled
}

public struct GenerateCompletionInfo: Sendable {
    /// The number of tokens included in the input prompt.
    public let promptTokenCount: Int

    /// The number of tokens generated by the language model.
    public let generationTokenCount: Int

    /// The time interval (in seconds) taken to process the input prompt.
    public let promptTime: TimeInterval

    /// The time interval (in seconds) taken to generate the output tokens.
    public let generateTime: TimeInterval

    /// Reason generation stopped.
    public let stopReason: GenerateStopReason

    /// The number of tokens processed per second during the prompt phase.
    public var promptTokensPerSecond: Double {
        Double(promptTokenCount) / promptTime
    }

    /// The number of tokens generated per second during the generation phase.
    public var tokensPerSecond: Double {
        Double(generationTokenCount) / generateTime
    }

    public init(
        promptTokenCount: Int,
        generationTokenCount: Int,
        promptTime: TimeInterval,
        generationTime: TimeInterval,
        stopReason: GenerateStopReason = .stop
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptTime = promptTime
        self.generateTime = generationTime
        self.stopReason = stopReason
    }

    public func summary() -> String {
        """
        Prompt:     \(promptTokenCount) tokens, \(promptTokensPerSecond.formatted()) tokens/s, \(promptTime.formatted())s
        Generation: \(generationTokenCount) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
    }
}

/// Represents the different stages or outputs of the token generation process.
public enum Generation: Sendable {
    /// A generated text chunk as a String.
    case chunk(String)

    /// Completion information summarizing token counts and performance metrics.
    case info(GenerateCompletionInfo)

    /// A tool call from the language model.
    case toolCall(ToolCall)

    /// Generated text or nil
    public var chunk: String? {
        switch self {
        case .chunk(let string): string
        case .info: nil
        case .toolCall: nil
        }
    }

    /// Completion info or nil
    public var info: GenerateCompletionInfo? {
        switch self {
        case .chunk: nil
        case .info(let info): info
        case .toolCall: nil
        }
    }

    /// Tool call or nil
    public var toolCall: ToolCall? {
        switch self {
        case .chunk: nil
        case .info: nil
        case .toolCall(let toolCall): toolCall
        }
    }

    /// Reducer that can be used with `throttle()` to gather elements into a batch
    @Sendable
    public static func collect(_ batch: [Generation]?, _ element: Generation) -> [Generation] {
        (batch ?? []) + [element]
    }
}

public enum TokenGeneration: Sendable {
    /// A generated token ID.
    case token(Int)

    /// Completion information summarizing token counts and performance metrics.
    case info(GenerateCompletionInfo)

    /// Token ID or nil
    public var token: Int? {
        switch self {
        case .token(let token): token
        case .info: nil
        }
    }

    /// Completion info or nil
    public var info: GenerateCompletionInfo? {
        switch self {
        case .token: nil
        case .info(let info): info
        }
    }

    /// Reducer that can be used with `throttle()` to gather elements into a batch
    @Sendable
    public static func collect(_ batch: [TokenGeneration]?, _ element: TokenGeneration)
        -> [TokenGeneration]
    {
        (batch ?? []) + [element]
    }
}

// MARK: - TokenLoopHandlers

private protocol TokenLoopHandler: Sendable {
    associatedtype Output

    /// Return false to stop the loop early.
    mutating func onToken(
        _ token: Int,
        emit: (sending Output) -> AsyncStream<Output>.Continuation.YieldResult
    ) -> Bool

    /// Called only when includeStopToken == true and a stop token was hit.
    mutating func onStopToken(
        _ token: Int,
        emit: (sending Output) -> AsyncStream<Output>.Continuation.YieldResult
    ) -> Bool

    /// Called after the token loop finishes, before the info event.
    mutating func onGenerationEnd(
        emit: (sending Output) -> AsyncStream<Output>.Continuation.YieldResult
    )

    func infoEvent(_ info: GenerateCompletionInfo) -> Output
}

private struct TextToolTokenLoopHandler: TokenLoopHandler, @unchecked Sendable {
    typealias Output = Generation

    var detokenizer: NaiveStreamingDetokenizer
    let toolCallProcessor: ToolCallProcessor

    init(tokenizer: Tokenizer, format: ToolCallFormat) {
        detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        toolCallProcessor = ToolCallProcessor(format: format)
    }

    mutating func onToken(
        _ token: Int,
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        detokenizer.append(token: token)
        if let chunk = detokenizer.next() {
            // Process chunk through the tool call processor.
            if let textToYield = toolCallProcessor.processChunk(chunk) {
                if case .terminated = emit(.chunk(textToYield)) {
                    return false
                }
            }

            // Emit all complete tool calls in parse order.
            for toolCall in toolCallProcessor.drainToolCalls() {
                if case .terminated = emit(.toolCall(toolCall)) {
                    return false
                }
            }
        }

        return true
    }

    mutating func onStopToken(
        _ token: Int,
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        true
    }

    mutating func onGenerationEnd(
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) {
        if let bufferedText = toolCallProcessor.processEOS(returnBufferedText: true),
            !bufferedText.isEmpty
        {
            if case .terminated = emit(.chunk(bufferedText)) {
                return
            }
        }

        for toolCall in toolCallProcessor.drainToolCalls() {
            if case .terminated = emit(.toolCall(toolCall)) {
                break
            }
        }
    }

    func infoEvent(_ info: GenerateCompletionInfo) -> Generation {
        .info(info)
    }
}

private struct RawTokenLoopHandler: TokenLoopHandler {
    typealias Output = TokenGeneration

    mutating func onToken(
        _ token: Int,
        emit: (sending TokenGeneration) -> AsyncStream<TokenGeneration>.Continuation.YieldResult
    ) -> Bool {
        if case .terminated = emit(.token(token)) {
            return false
        }
        return true
    }

    mutating func onStopToken(
        _ token: Int,
        emit: (sending TokenGeneration) -> AsyncStream<TokenGeneration>.Continuation.YieldResult
    ) -> Bool {
        if case .terminated = emit(.token(token)) {
            return false
        }
        return true
    }

    mutating func onGenerationEnd(
        emit: (sending TokenGeneration) -> AsyncStream<TokenGeneration>.Continuation.YieldResult
    ) {}

    func infoEvent(_ info: GenerateCompletionInfo) -> TokenGeneration {
        .info(info)
    }
}
