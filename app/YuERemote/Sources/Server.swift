import Foundation
import Network
import UIKit

/// Advertises the phone on the local network as a YuE Studio synthesis engine and serves one
/// Mac at a time. Requests and replies are Frames (see Protocol.swift).
@MainActor
final class Server: ObservableObject {
    static let serviceType = "_yuestudio._tcp"
    static let version = 1

    @Published var lines: [String] = []
    @Published var state = "starting"
    @Published var peer = "no Mac connected"
    @Published var weights = "none"
    @Published var session = "idle"
    @Published var passLine = ""
    @Published var tflops = 0.0          // last pass
    @Published var odometer = 0.0        // TFLOP done since launch
    @Published var running = false

    let engine = Engine()
    private var listener: NWListener?
    private var current: NWConnection?
    private let queue = DispatchQueue(label: "yue.remote")

    func start() {
        // A companion app: keep the screen awake whenever it is in front, since a locked phone
        // suspends the app and closes the listening socket.
        UIApplication.shared.isIdleTimerDisabled = true
        engine.log = { [weak self] line in Task { @MainActor in self?.say(line) } }
        weights = engine.weightsIdentity.map { "cached (\($0))" } ?? "none yet"
        do {
            // Keep the multi-gigabyte weight upload on infrastructure Wi-Fi instead of
            // opting into Apple's peer-to-peer path when both devices already share a LAN.
            let params = NWParameters.tcp
            params.allowLocalEndpointReuse = true
            let l = try NWListener(using: params)
            let txt = NWTXTRecord(["version": String(Self.version), "model": Bench.machine()])
            l.service = NWListener.Service(name: UIDevice.current.name, type: Self.serviceType, txtRecord: txt)
            l.stateUpdateHandler = { [weak self] st in
                Task { @MainActor in
                    switch st {
                    case .ready: self?.state = "advertising as “\(UIDevice.current.name)” on port \(l.port?.rawValue ?? 0)"
                    case .failed(let e): self?.state = "listener failed: \(e)"
                    default: break
                    }
                }
            }
            l.newConnectionHandler = { [weak self] conn in Task { @MainActor in self?.accept(conn) } }
            l.start(queue: queue)
            listener = l
        } catch { say("cannot start listener: \(error)") }
    }

    func say(_ line: String) {
        lines.append(line)
        if lines.count > 300 { lines.removeFirst(lines.count - 300) }
        print(line)
    }

    private func accept(_ conn: NWConnection) {
        current?.cancel()
        current = conn
        peer = "Mac connected (\(conn.endpoint))"
        conn.stateUpdateHandler = { [weak self] st in
            if case .failed(let e) = st { Task { @MainActor in self?.say("connection failed: \(e)") } }
        }
        conn.start(queue: queue)
        Task { await serve(conn) }
    }

    private func serve(_ conn: NWConnection) async {
        say("Mac connected")
        while true {
            let frame: Frame
            do { frame = try await conn.receiveFrame() } catch { break }
            let reply = await handle(frame, conn)
            do { try await conn.sendFrame(reply) } catch { break }
        }
        conn.cancel()
        engine.closeSession()
        if current === conn {
            current = nil; peer = "no Mac connected"; session = "idle"; passLine = ""; running = false; tflops = 0
        }
        say("Mac disconnected")
    }

    private func progress(_ conn: NWConnection, _ text: String) {
        conn.send(content: Frame(header: ["op": "progress", "text": text], payload: Data()).encoded(), completion: .idempotent)
        Task { @MainActor in self.session = text }
    }

    private func handle(_ f: Frame, _ conn: NWConnection) async -> Frame {
        let op = f.op
        func ok(_ extra: [String: Any] = [:], payload: Data = Data()) -> Frame {
            var h: [String: Any] = ["op": op, "ok": true]; extra.forEach { h[$0] = $1 }
            return Frame(header: h, payload: payload)
        }
        func fail(_ message: String) -> Frame {
            if op == "open" || op == "program" {
                engine.closeSession()
                session = "unavailable: " + message
                passLine = ""; running = false; tflops = 0
            }
            say("\(op): \(message)")
            return Frame(header: ["op": op, "ok": false, "error": message], payload: Data())
        }
        do {
            switch op {
            case "hello":
                return ok(["version": Self.version, "device": UIDevice.current.name, "model": Bench.machine(),
                           "memory_available_mb": Bench.availableMB(), "weights": engine.weightsIdentity ?? NSNull(),
                           "programs": engine.programNames()])
            case "ping":
                return ok()
            case "weights_begin":
                guard let identity = f.header["identity"] as? String else { return fail("identity missing") }
                try engine.beginWeights(identity: identity)
                weights = "receiving…"
                say("receiving weights \(identity)")
                return ok()
            case "weights_layer":
                guard let identity = f.header["identity"] as? String, let index = f.header["layer"] as? Int else { return fail("bad header") }
                try engine.storeLayer(identity: identity, index: index, data: f.payload)
                weights = "receiving layer \(index + 1)/\(Engine.layers)"
                return ok()
            case "weights_end":
                guard let identity = f.header["identity"] as? String else { return fail("identity missing") }
                try engine.endWeights(identity: identity)
                weights = "cached (\(identity))"
                say("weights \(identity) stored")
                return ok()
            case "program":
                guard let name = f.header["name"] as? String, let files = f.header["files"] as? [[String: Any]] else { return fail("bad header") }
                let list = files.compactMap { d -> (String, Int)? in guard let p = d["path"] as? String, let n = d["size"] as? Int else { return nil }; return (p, n) }
                if !engine.hasProgram(name) {
                    session = "compiling \(name)"
                    let t0 = Date()
                    try await Task.detached { [engine] in try engine.installProgram(name: name, files: list, payload: f.payload) { text in
                        Task { @MainActor in self.progress(conn, text) } } }.value
                    say(String(format: "program %@ compiled in %.0f s", name, Date().timeIntervalSince(t0)))
                }
                return ok()
            case "open":
                guard let S = f.header["S"] as? Int, let P = f.header["P"] as? Int, let sReal = f.header["S_real"] as? Int,
                      let pReal = f.header["P_real"] as? Int, let program = f.header["program"] as? String else { return fail("bad header") }
                guard engine.weightsIdentity != nil else { return fail("no weights on the phone") }
                guard engine.hasProgram(program) else { return fail("program \(program) not installed") }
                session = "loading \(program)"
                let perCall = f.header["layers"] as? Int ?? 2
                try await engine.open(S: S, P: P, sReal: sReal, pReal: pReal, program: program, perCall: perCall) { text in Task { @MainActor in self.progress(conn, text) } }
                session = "S \(S) × P \(P) (\(sReal) frames, \(pReal) prefix)"
                return ok()
            case "kv":
                guard let layer = f.header["layer"] as? Int else { return fail("bad header") }
                try engine.setKV(layer: layer, data: f.payload)
                return ok()
            case "tables":
                try engine.setTables(data: f.payload)
                return ok()
            case "velocity":
                running = true
                let (h, seconds) = try await Task.detached { [engine] in try engine.velocity(f.payload) }.value
                let passes = engine.session?.passes ?? 0
                let calls = Double(Engine.layers / (engine.session?.perCall ?? 2))
                passLine = String(format: "pass %d: %.1f s (%.0f ms per call), %d MB free", passes, seconds, 1000 * seconds / calls, Bench.availableMB())
                if let s = engine.session {
                    let work = Engine.teraflopsPerPass(rows: s.sReal, prefix: s.pReal)
                    tflops = work / max(seconds, 1e-3); odometer += work
                }
                return ok(["seconds": seconds], payload: h)
            case "close":
                engine.closeSession()
                session = "idle"; passLine = ""; running = false; tflops = 0
                return ok()
            default:
                return fail("unknown op \(op)")
            }
        } catch {
            return fail(error.localizedDescription)
        }
    }
}
