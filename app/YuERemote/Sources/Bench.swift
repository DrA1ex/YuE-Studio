import CoreML
import Foundation
import os
import UIKit

struct Bucket: Identifiable, Hashable {
    let s: Int, p: Int
    var q = 1024, k = 2048          // attention tile: query block, key chunk
    var suffix = ""
    var layers = 2                  // layers per program
    var id: String { resource }
    var resource: String { "layers\(layers)_\(s)_\(p)\(suffix)" }
    var label: String { "S \(s) × P \(p) \(suffix.contains("_spi") ? "spread-in" : "plain") \(q)/\(k) L\(layers)" }
}

@MainActor
final class Bench: ObservableObject {
    static let buckets = [Bucket(s: 512, p: 1024), Bucket(s: 2048, p: 2048), Bucket(s: 2048, p: 4096, suffix: "_q1024_k2048"),
                          Bucket(s: 4096, p: 4096, q: 512, k: 1024, suffix: "_q512_k1024"),
                          Bucket(s: 6656, p: 4096, q: 512, k: 1024, suffix: "_spi_q512_k1024", layers: 1),
                          Bucket(s: 6656, p: 4096, q: 512, k: 1024, suffix: "_spi_q512_k1024")]
    static let D = 2048, H = 16, KV = 8, HD = 128, F = 6144
    nonisolated static var weightShapes: [(String, [Int])] { Engine.weightShapes }

    @Published var log: [String] = []
    @Published var busy = false
    @Published var memoryLine = ""
    @Published var deviceLine = ""
    private var held: [[String: MLMultiArray]] = []

    func start() {
        deviceLine = "\(Self.machine())  iOS \(ProcessInfo.processInfo.operatingSystemVersionString)  RAM \(ProcessInfo.processInfo.physicalMemory / 1_048_576) MB"
        let died = UserDefaults.standard.integer(forKey: "holding")
        if died > 0 { say("previous run ended while holding \(died) layers of weights (killed by the system?)") }
        UserDefaults.standard.set(0, forKey: "holding")
        refreshMemory()
    }

    static let logURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("bench.log")

    func say(_ line: String) {
        log.append(line); print(line)
        let stamped = "\(ISO8601DateFormatter().string(from: Date())) \(line)\n"
        if let h = try? FileHandle(forWritingTo: Self.logURL) { h.seekToEndOfFile(); h.write(stamped.data(using: .utf8)!); try? h.close() }
        else { try? stamped.write(to: Self.logURL, atomically: true, encoding: .utf8) }
    }

    func refreshMemory() { memoryLine = "footprint \(Self.footprintMB()) MB, available \(Self.availableMB()) MB" }

    func run(bucket: Bucket, hold: Int) async {
        busy = true; UIApplication.shared.isIdleTimerDisabled = true
        defer { busy = false; UIApplication.shared.isIdleTimerDisabled = false }
        do { try await bench(bucket: bucket, hold: hold) } catch { say("error: \(error)") }
        refreshMemory()
    }

    func runAll(from first: Bucket, hold: Int) async {
        busy = true; UIApplication.shared.isIdleTimerDisabled = true
        defer { busy = false; UIApplication.shared.isIdleTimerDisabled = false }
        guard let start = Self.buckets.firstIndex(of: first) else { return }
        for bucket in Self.buckets[start...] {
            do { try await bench(bucket: bucket, hold: hold) } catch { say("error: \(error)") }
            refreshMemory()
        }
        say("— run all finished")
    }

    private func bench(bucket: Bucket, hold: Int) async throws {
        let S = bucket.s, P = bucket.p
        say("— bucket S=\(S) P=\(P), \(Self.availableMB()) MB available")
        guard let url = Bundle.main.url(forResource: bucket.resource, withExtension: "mlmodelc") else {
            say("compiled model \(bucket.resource).mlmodelc missing from bundle"); return
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        var t0 = Date()
        var allANE = false
        do {
            let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
            if case .program(let program) = plan.modelStructure, let main = program.functions["main"] {
                var counts: [String: Int] = [:]
                var offloaded: [String] = []
                for op in main.block.operations {
                    let name: String
                    switch plan.deviceUsage(for: op)?.preferred {
                    case .neuralEngine: name = "ANE"
                    case .gpu: name = "GPU"
                    case .cpu: name = "CPU"
                    case .none: name = "none"
                    @unknown default: name = "?"
                    }
                    counts[name, default: 0] += 1
                    if name == "CPU" || name == "GPU", offloaded.count < 40 {
                        let outs = op.outputs.map { "\($0.name) \(String(describing: $0.type))" }.joined(separator: "; ")
                        offloaded.append("  \(name) \(op.operatorName): \(outs)")
                    }
                }
                for line in offloaded { say(line) }
                allANE = counts.keys.allSatisfy { $0 == "ANE" || $0 == "none" }
                say(String(format: "compute plan %.1f s: ", Date().timeIntervalSince(t0)) + counts.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", "))
            }
        } catch { say("compute plan failed: \(error.localizedDescription)") }
        guard allANE else { say("not entirely on the Neural Engine — skipping this bucket"); return }
        t0 = Date()
        let model = try await Task.detached { try MLModel(contentsOf: url, configuration: config) }.value
        say(String(format: "load %.1f s", Date().timeIntervalSince(t0)))
        refreshMemory()

        // Hold N layers of weights (fp16, ~100 MB each) to see whether the whole stack fits.
        if held.count > hold { held.removeSubrange(hold...) }
        while held.count < hold {
            UserDefaults.standard.set(held.count + 1, forKey: "holding")
            t0 = Date()
            held.append(try Self.randomWeights(seed: UInt64(held.count + 1)))
            say(String(format: "holding %d layers: footprint %d MB, available %d MB (%.2f s)", held.count, Self.footprintMB(), Self.availableMB(), Date().timeIntervalSince(t0)))
            refreshMemory()
            await Task.yield()
        }
        UserDefaults.standard.set(0, forKey: "holding")
        let scratch = held.isEmpty ? [try Self.randomWeights(seed: 1), try Self.randomWeights(seed: 2)] : []

        // Activations and prefix K/V.
        let x = try Self.array([1, 1, S, Self.D], seed: 11, scale: 1)
        let cos = try Self.array([1, 1, S, Self.HD / 2], seed: 12, scale: 1)
        let sin = try Self.array([1, 1, S, Self.HD / 2], seed: 13, scale: 1)
        let bias = try Self.array([1, 1, 1, S + P], seed: 0, scale: 0)
        let kv = try (0..<(2 * bucket.layers)).map { try Self.array([1, Self.KV, P, Self.HD], seed: UInt64(20 + $0), scale: 0.5) }

        var fixed: [String: MLMultiArray] = ["cos": cos, "sin": sin, "bias": bias]
        for j in 0..<bucket.layers { fixed["pk\(j)"] = kv[2 * j]; fixed["pv\(j)"] = kv[2 * j + 1] }
        if bucket.suffix.contains("_spi") { for (name, arr) in try Engine.spreadMatrices() { fixed[name] = arr } }
        let sets = held.isEmpty ? scratch : held
        func weights(_ layer: Int) -> [[String: MLMultiArray]] { (0..<bucket.layers).map { sets[(layer + $0) % sets.count] } }

        t0 = Date()
        let w0 = weights(0)
        var h = try await Task.detached { try Self.predict(model, x: x, fixed: fixed, weights: w0) }.value
        say(String(format: "first call %.0f ms", 1000 * Date().timeIntervalSince(t0)))
        let calls = 28 / bucket.layers
        t0 = Date()
        for i in 0..<calls {
            let input = h, w = weights(bucket.layers * i)
            h = try await Task.detached { try Self.predict(model, x: input, fixed: fixed, weights: w) }.value
        }
        let pass = Date().timeIntervalSince(t0)
        let absmax = Self.absmax(h)
        say(String(format: "%d calls (one 28-layer pass): %.2f s, %.0f ms per call, |h|max %.2f, finite %@", calls, pass, 1000 * pass / Double(calls), absmax, absmax.isFinite ? "yes" : "NO"))
        // 8 draft steps = 16 passes; 32 full steps = 64 passes.
        say(String(format: "estimate for a song filling this bucket: draft %.0f s, full %.0f s", 16 * pass, 64 * pass))
    }

    nonisolated static func predict(_ model: MLModel, x: MLMultiArray, fixed: [String: MLMultiArray], weights: [[String: MLMultiArray]]) throws -> MLMultiArray {
        var dict: [String: MLFeatureValue] = ["x": .init(multiArray: x)]
        for (name, arr) in fixed { dict[name] = .init(multiArray: arr) }
        for (j, w) in weights.enumerated() {
            for (name, _) in weightShapes { dict["w\(j)_\(name)"] = .init(multiArray: w[name]!) }
        }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: dict))
        guard let h = out.featureValue(for: "h")?.multiArrayValue else {
            throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "no output h"])
        }
        return h
    }

    // MARK: - arrays

    static func randomWeights(seed: UInt64) throws -> [String: MLMultiArray] {
        var out: [String: MLMultiArray] = [:]
        for (name, shape) in weightShapes {
            out[name] = try array(shape, seed: seed &* 31 &+ UInt64(name.hashValue & 0xffff), scale: name.hasSuffix("norm") ? 0 : 0.02, offset: name.hasSuffix("norm") ? 1 : 0)
        }
        return out
    }

    /// fp16 array filled with uniform noise in ±scale (plus offset); a 4096-element random block repeated.
    static func array(_ shape: [Int], seed: UInt64, scale: Float, offset: Float = 0) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        let n = arr.count
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float16.self)
        var state = seed | 1
        let block = min(n, 4096)
        for i in 0..<block {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            let u = Float(state >> 40) / Float(1 << 24) * 2 - 1
            ptr[i] = Float16(offset + scale * u)
        }
        var filled = block
        while filled < n {
            let chunk = min(filled, n - filled)
            memcpy(ptr + filled, ptr, chunk * 2)
            filled += chunk
        }
        return arr
    }

    static func absmax(_ a: MLMultiArray) -> Float {
        let ptr = a.dataPointer.assumingMemoryBound(to: Float16.self)
        var m: Float = 0
        for i in stride(from: 0, to: a.count, by: 7) { let v = abs(Float(ptr[i])); if !(v <= m) { m = v } }
        return m
    }

    // MARK: - system

    static func machine() -> String {
        var u = utsname(); uname(&u)
        return withUnsafePointer(to: &u.machine) { $0.withMemoryRebound(to: CChar.self, capacity: 256) { String(cString: $0) } }
    }

    static func availableMB() -> Int { Int(os_proc_available_memory() / 1_048_576) }

    static func footprintMB() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let r = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return r == KERN_SUCCESS ? Int(info.phys_footprint / 1_048_576) : -1
    }
}
