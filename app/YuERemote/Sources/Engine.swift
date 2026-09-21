import CoreML
import Foundation

/// The phone's share of YuE2 synthesis: the 28 decoder layers of one solver pass, run two per
/// Core ML call on the Neural Engine. Weights arrive once from the Mac and live in files
/// (memory-mapped at use); one session at a time holds a song's prefix K/V and tables.
final class Engine {
    static let D = 2048, H = 16, KV = 8, HD = 128, F = 6144, layers = 28
    static let weightShapes: [(String, [Int])] = [
        ("in_norm", [1, 1, 1, D]), ("q", [1, 1, H * HD, D]), ("k", [1, 1, KV * HD, D]), ("v", [1, 1, KV * HD, D]),
        ("o", [1, 1, D, H * HD]), ("q_norm", [1, 1, 1, HD]), ("k_norm", [1, 1, 1, HD]), ("mlp_norm", [1, 1, 1, D]),
        ("gate", [1, 1, F, D]), ("up", [1, 1, F, D]), ("down", [1, 1, D, F])]
    static var layerBytes: Int { weightShapes.reduce(0) { $0 + $1.1.reduce(1, *) * 2 } }

    let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    var log: (String) -> Void = { print($0) }
    var status: (String) -> Void = { _ in }

    // Weights on disk and mapped.
    private(set) var weightsIdentity: String?
    private var mapped: [[String: MLMultiArray]] = []
    private var mappings: [UnsafeMutableRawPointer] = []

    // Session.
    struct Session {
        let S: Int, P: Int, sReal: Int, pReal: Int, program: String
        let perCall: Int                 // layers per program call
        let model: MLModel
        var kv: [(MLMultiArray, MLMultiArray)?]
        var cos: MLMultiArray?, sin: MLMultiArray?, bias: MLMultiArray?
        var spread: [String: MLMultiArray] = [:]   // mq/tq/mk/tk when the program's head norm is the "spread" form
        var passes = 0
    }
    private(set) var session: Session?

    init() {
        try? FileManager.default.createDirectory(at: root.appendingPathComponent("weights"), withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: root.appendingPathComponent("programs"), withIntermediateDirectories: true)
        weightsIdentity = storedWeightsIdentity()
    }

    // MARK: weights

    private func weightsDir(_ identity: String) -> URL { root.appendingPathComponent("weights").appendingPathComponent(identity) }

    func storedWeightsIdentity() -> String? {
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("weights").path) else { return nil }
        for name in names where FileManager.default.fileExists(atPath: weightsDir(name).appendingPathComponent("complete").path) { return name }
        return nil
    }

    func beginWeights(identity: String) throws {
        let dir = weightsDir(identity)
        try? FileManager.default.removeItem(at: dir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        unmapWeights(); weightsIdentity = nil
    }

    func storeLayer(identity: String, index: Int, data: Data) throws {
        guard data.count == Self.layerBytes else { throw EngineError.message("layer \(index): expected \(Self.layerBytes) bytes, got \(data.count)") }
        try data.write(to: weightsDir(identity).appendingPathComponent("layer\(index).bin"))
    }

    func endWeights(identity: String) throws {
        for i in 0..<Self.layers where !FileManager.default.fileExists(atPath: weightsDir(identity).appendingPathComponent("layer\(i).bin").path) {
            throw EngineError.message("layer \(i) missing")
        }
        try Data().write(to: weightsDir(identity).appendingPathComponent("complete"))
        for other in (try? FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("weights").path)) ?? [] where other != identity {
            try? FileManager.default.removeItem(at: weightsDir(other))
        }
        weightsIdentity = identity
    }

    /// Memory-map every layer's file and wrap the 11 arrays without copying.
    func mapWeights() throws {
        guard mapped.isEmpty, let identity = weightsIdentity else { return }
        for i in 0..<Self.layers {
            let path = weightsDir(identity).appendingPathComponent("layer\(i).bin").path
            let fd = Darwin.open(path, O_RDONLY)
            guard fd >= 0 else { throw EngineError.message("cannot open \(path)") }
            defer { close(fd) }
            guard let base = mmap(nil, Self.layerBytes, PROT_READ, MAP_PRIVATE, fd, 0), base != MAP_FAILED else { throw EngineError.message("mmap failed for layer \(i)") }
            mappings.append(base)
            var offset = 0
            var arrays: [String: MLMultiArray] = [:]
            for (name, shape) in Self.weightShapes {
                let count = shape.reduce(1, *)
                var strides: [Int] = []; var s = 1
                for d in shape.reversed() { strides.insert(s, at: 0); s *= d }
                arrays[name] = try MLMultiArray(dataPointer: base + offset, shape: shape.map { NSNumber(value: $0) }, dataType: .float16,
                                                strides: strides.map { NSNumber(value: $0) }, deallocator: nil)
                offset += count * 2
            }
            mapped.append(arrays)
        }
    }

    func unmapWeights() {
        mapped.removeAll()
        for base in mappings { munmap(base, Self.layerBytes) }
        mappings.removeAll()
    }

    // MARK: programs

    private func programDir(_ name: String) -> URL { root.appendingPathComponent("programs").appendingPathComponent(name + ".mlmodelc") }

    func hasProgram(_ name: String) -> Bool { FileManager.default.fileExists(atPath: programDir(name).path) }

    func programNames() -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("programs").path)) ?? [])
            .filter { $0.hasSuffix(".mlmodelc") }.map { String($0.dropLast(9)) }
    }

    /// Store the mlpackage files sent by the Mac and compile them (minutes for long songs).
    func installProgram(name: String, files: [(String, Int)], payload: Data, progress: @escaping (String) -> Void) throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(name + ".mlpackage")
        try? FileManager.default.removeItem(at: tmp)
        var offset = 0
        for (path, size) in files {
            let url = tmp.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try payload.subdata(in: offset..<offset + size).write(to: url)
            offset += size
        }
        let t0 = Date()
        let ticker = DispatchSource.makeTimerSource(queue: .global())
        ticker.schedule(deadline: .now() + 5, repeating: 5)
        ticker.setEventHandler { progress(String(format: "compiling %@ on the iPhone (%.0f s)", name, Date().timeIntervalSince(t0))) }
        ticker.resume()
        defer { ticker.cancel() }
        let compiled = try MLModel.compileModel(at: tmp)
        try? FileManager.default.removeItem(at: programDir(name))
        try FileManager.default.moveItem(at: compiled, to: programDir(name))
        try? FileManager.default.removeItem(at: tmp)
        log(String(format: "compiled %@ in %.0f s", name, Date().timeIntervalSince(t0)))
    }

    // MARK: session

    func open(S: Int, P: Int, sReal: Int, pReal: Int, program: String, perCall: Int = 2, progress: @escaping (String) -> Void) async throws {
        closeSession()
        try mapWeights()
        let url = programDir(program)
        let config = MLModelConfiguration(); config.computeUnits = .cpuAndNeuralEngine
        let t0 = Date()
        let ticker = DispatchSource.makeTimerSource(queue: .global())
        ticker.schedule(deadline: .now() + 5, repeating: 5)
        ticker.setEventHandler { progress(String(format: "loading %@ onto the iPhone's Neural Engine (%.0f s)", program, Date().timeIntervalSince(t0))) }
        ticker.resume()
        defer { ticker.cancel() }
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
        var off = 0, total = 0
        var rejected: [String] = []
        if case .program(let prog) = plan.modelStructure, let main = prog.functions["main"] {
            for op in main.block.operations {
                if let device = plan.deviceUsage(for: op)?.preferred {
                    total += 1
                    if case .neuralEngine = device {} else {
                        off += 1
                        if rejected.count < 5 { rejected.append(op.operatorName) }
                    }
                }
            }
        }
        guard off == 0, total > 0 else {
            log("Neural Engine placement rejected; examples: " + rejected.joined(separator: ", "))
            throw EngineError.message("program \(program): \(off) of \(total) ops would not run on the Neural Engine; update YuE Studio on the Mac to regenerate compatible programs")
        }
        let model = try MLModel(contentsOf: url, configuration: config)
        log(String(format: "program %@ ready in %.0f s (%d ops on the Neural Engine)", program, Date().timeIntervalSince(t0), total))
        var s = Session(S: S, P: P, sReal: sReal, pReal: pReal, program: program, perCall: perCall, model: model, kv: Array(repeating: nil, count: Self.layers))
        if model.modelDescription.inputDescriptionsByName["mq"] != nil { s.spread = try Self.spreadMatrices() }
        session = s
    }

    func setKV(layer: Int, data: Data) throws {
        guard var s = session else { throw EngineError.message("no session") }
        let rows = s.pReal, per = Self.KV * rows * Self.HD * 2
        guard data.count == 2 * per else { throw EngineError.message("kv \(layer): expected \(2 * per) bytes, got \(data.count)") }
        func padded(_ src: Data) throws -> MLMultiArray {
            let arr = try MLMultiArray(shape: [1, Self.KV, s.P, Self.HD].map { NSNumber(value: $0) }, dataType: .float16)
            let dst = arr.dataPointer
            memset(dst, 0, Self.KV * s.P * Self.HD * 2)
            src.withUnsafeBytes { raw in
                for h in 0..<Self.KV {
                    memcpy(dst + h * s.P * Self.HD * 2, raw.baseAddress! + h * rows * Self.HD * 2, rows * Self.HD * 2)
                }
            }
            return arr
        }
        s.kv[layer] = (try padded(data.subdata(in: 0..<per)), try padded(data.subdata(in: per..<2 * per)))
        session = s
    }

    func setTables(data: Data) throws {
        guard var s = session else { throw EngineError.message("no session") }
        let cs = s.S * (Self.HD / 2) * 2, bs = (s.S + s.P) * 2
        guard data.count == 2 * cs + bs else { throw EngineError.message("tables: expected \(2 * cs + bs) bytes, got \(data.count)") }
        func arr(_ shape: [Int], _ range: Range<Int>) throws -> MLMultiArray {
            let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
            data.subdata(in: range).withUnsafeBytes { memcpy(a.dataPointer, $0.baseAddress!, range.count) }
            return a
        }
        s.cos = try arr([1, 1, s.S, Self.HD / 2], 0..<cs)
        s.sin = try arr([1, 1, s.S, Self.HD / 2], cs..<2 * cs)
        s.bias = try arr([1, 1, 1, s.S + s.P], 2 * cs..<2 * cs + bs)
        session = s
    }

    /// One pass: x [S, D] fp16 -> h [S, D] fp16.
    func velocity(_ x: Data) throws -> (Data, Double) {
        guard var s = session, let cos = s.cos, let sin = s.sin, let bias = s.bias else { throw EngineError.message("session not ready") }
        guard x.count == s.S * Self.D * 2 else { throw EngineError.message("x: expected \(s.S * Self.D * 2) bytes, got \(x.count)") }
        for i in 0..<Self.layers where s.kv[i] == nil { throw EngineError.message("prefix K/V for layer \(i) missing") }
        let t0 = Date()
        var h = try MLMultiArray(shape: [1, 1, s.S, Self.D].map { NSNumber(value: $0) }, dataType: .float16)
        x.withUnsafeBytes { memcpy(h.dataPointer, $0.baseAddress!, x.count) }
        for first in stride(from: 0, to: Self.layers, by: s.perCall) {
            var dict: [String: MLFeatureValue] = ["x": .init(multiArray: h), "cos": .init(multiArray: cos), "sin": .init(multiArray: sin), "bias": .init(multiArray: bias)]
            for (name, arr) in s.spread { dict[name] = .init(multiArray: arr) }
            for j in 0..<s.perCall {
                let (pk, pv) = s.kv[first + j]!
                dict["pk\(j)"] = .init(multiArray: pk); dict["pv\(j)"] = .init(multiArray: pv)
                for (name, _) in Self.weightShapes { dict["w\(j)_\(name)"] = .init(multiArray: mapped[first + j][name]!) }
            }
            let out = try s.model.prediction(from: MLDictionaryFeatureProvider(dictionary: dict))
            guard let next = out.featureValue(for: "h")?.multiArrayValue else { throw EngineError.message("no output h") }
            h = next
        }
        s.passes += 1; session = s
        let seconds = Date().timeIntervalSince(t0)
        return (Data(bytes: h.dataPointer, count: s.S * Self.D * 2), seconds)
    }

    func closeSession() { session = nil }

    /// The "spread" head-norm matrices as inputs: M [n*HD, n*HD] block-diagonal ones and Tt [n*HD, HD]
    /// with Tt[j, j % HD] = 1, for the query heads (mq, tq) and the key heads (mk, tk).
    static func spreadMatrices() throws -> [String: MLMultiArray] {
        var out: [String: MLMultiArray] = [:]
        for (prefix, n) in [("q", H), ("k", KV)] {
            let m = try MLMultiArray(shape: [1, 1, n * HD, n * HD].map { NSNumber(value: $0) }, dataType: .float16)
            let mp = m.dataPointer.assumingMemoryBound(to: Float16.self)
            memset(m.dataPointer, 0, n * HD * n * HD * 2)
            for h in 0..<n { for i in 0..<HD { for j in 0..<HD { mp[(h * HD + i) * n * HD + h * HD + j] = 1 } } }
            let tt = try MLMultiArray(shape: [1, 1, n * HD, HD].map { NSNumber(value: $0) }, dataType: .float16)
            let tp = tt.dataPointer.assumingMemoryBound(to: Float16.self)
            memset(tt.dataPointer, 0, n * HD * HD * 2)
            for j in 0..<(n * HD) { tp[j * HD + j % HD] = 1 }
            out["m" + prefix] = m; out["t" + prefix] = tt
        }
        return out
    }

    /// Matmul work of one 28-layer pass over the real lengths (2 FLOP per multiply-add):
    /// projections and MLP per row, plus QK^T and PV over the attended keys.
    static func teraflopsPerPass(rows: Int, prefix: Int) -> Double {
        let S = Double(rows), K = Double(prefix + rows)
        let linear = 2 * S * Double(D * H * HD + 2 * D * KV * HD + H * HD * D + 3 * D * F)
        let attention = 4 * Double(H) * S * K * Double(HD)
        return Double(layers) * (linear + attention) / 1e12
    }
}

enum EngineError: Error, LocalizedError {
    case message(String)
    var errorDescription: String? { if case .message(let m) = self { return m }; return nil }
}
