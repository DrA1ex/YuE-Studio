import Foundation
import Network
import UIKit

/// A small Bonjour service dedicated to voice memo exchange. It is separate from the
/// synthesis listener so downloading an idea cannot interrupt an active Neural Engine run.
@MainActor
final class MemoServer: ObservableObject {
    static let serviceType = "_yuevoice._tcp"
    @Published var state = "starting"
    let store: VoiceMemoStore
    private var listener: NWListener?
    private var connection: NWConnection?
    private let queue = DispatchQueue(label: "yue.voice-memos")

    init(store: VoiceMemoStore? = nil) { self.store = store ?? VoiceMemoStore() }

    func start() {
        guard listener == nil else { return }
        do {
            let params = NWParameters.tcp; params.allowLocalEndpointReuse = true; params.includePeerToPeer = true
            let l = try NWListener(using: params)
            l.service = NWListener.Service(name: UIDevice.current.name, type: Self.serviceType,
                                           txtRecord: NWTXTRecord(["version": "1"]))
            l.stateUpdateHandler = { [weak self] update in
                Task { @MainActor in
                    switch update {
                    case .ready: self?.state = "ready for YuE Studio"
                    case .failed(let error): self?.state = "voice memo service failed: \(error.localizedDescription)"
                    default: break
                    }
                }
            }
            l.newConnectionHandler = { [weak self] connection in Task { @MainActor in self?.accept(connection) } }
            l.start(queue: queue); listener = l
        } catch { state = "voice memo service failed: \(error.localizedDescription)" }
    }

    private func accept(_ connection: NWConnection) {
        self.connection?.cancel(); self.connection = connection
        connection.start(queue: queue)
        Task { await serve(connection) }
    }

    private func serve(_ connection: NWConnection) async {
        while true {
            do {
                let frame = try await connection.receiveFrame()
                let reply = handle(frame)
                try await connection.sendFrame(reply)
            } catch { break }
        }
        connection.cancel()
        if self.connection === connection { self.connection = nil }
    }

    private func handle(_ frame: Frame) -> Frame {
        let op = frame.op
        func ok(_ extra: [String: Any] = [:], payload: Data = Data()) -> Frame {
            var header: [String: Any] = ["op": op, "ok": true]; extra.forEach { header[$0] = $1 }
            return Frame(header: header, payload: payload)
        }
        func fail(_ message: String) -> Frame { Frame(header: ["op": op, "ok": false, "error": message], payload: Data()) }
        switch op {
        case "hello": return ok(["version": 1, "device": UIDevice.current.name])
        case "list":
            let value: [[String: Any]] = store.memos.map { memo in
                ["id": memo.id, "title": memo.title, "created_at": memo.createdAt.timeIntervalSince1970,
                 "seconds": memo.seconds, "file_name": memo.fileName, "source": memo.source]
            }
            return ok(["memos": value])
        case "get":
            guard let id = frame.header["id"] as? String, let memo = store.memos.first(where: { $0.id == id }) else { return fail("memo not found") }
            guard let data = try? Data(contentsOf: store.url(for: memo)) else { return fail("memo audio is unavailable") }
            return ok(["id": memo.id, "file_name": memo.fileName], payload: data)
        default: return fail("unknown op \(op)")
        }
    }
}
