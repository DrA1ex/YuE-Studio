import Foundation
import Network

private struct MemoFrame {
    var header: [String: Any]
    var payload: Data

    func encoded() -> Data {
        let json = try! JSONSerialization.data(withJSONObject: header)
        var headerLength = UInt32(json.count).bigEndian
        var payloadLength = UInt64(payload.count).bigEndian
        var data = Data(bytes: &headerLength, count: 4)
        data.append(json); data.append(Data(bytes: &payloadLength, count: 8)); data.append(payload)
        return data
    }
}

private enum MemoWireError: Error { case closed, badHeader }

private extension NWConnection {
    func receiveMemoExactly(_ count: Int) async throws -> Data {
        var output = Data(capacity: count)
        while output.count < count {
            let want = min(4 << 20, count - output.count)
            let chunk: Data = try await withCheckedThrowingContinuation { continuation in
                receive(minimumIncompleteLength: want, maximumLength: want) { data, _, _, error in
                    if let error { continuation.resume(throwing: error) }
                    else if let data, data.count == want { continuation.resume(returning: data) }
                    else { continuation.resume(throwing: MemoWireError.closed) }
                }
            }
            output.append(chunk)
        }
        return output
    }

    func receiveMemoFrame() async throws -> MemoFrame {
        let headerLength = try await receiveMemoExactly(4).withUnsafeBytes { UInt32(bigEndian: $0.loadUnaligned(as: UInt32.self)) }
        let headerData = try await receiveMemoExactly(Int(headerLength))
        guard let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else { throw MemoWireError.badHeader }
        let payloadLength = try await receiveMemoExactly(8).withUnsafeBytes { UInt64(bigEndian: $0.loadUnaligned(as: UInt64.self)) }
        let payload = payloadLength == 0 ? Data() : try await receiveMemoExactly(Int(payloadLength))
        return MemoFrame(header: header, payload: payload)
    }

    func sendMemoFrame(_ frame: MemoFrame) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            send(content: frame.encoded(), completion: .contentProcessed { error in
                if let error { continuation.resume(throwing: error) } else { continuation.resume() }
            })
        }
    }
}

/// Discovers the iPhone's voice memo service independently from the Neural Engine service.
@MainActor
final class VoiceMemoRemote: ObservableObject {
    @Published private(set) var phoneName: String?
    @Published private(set) var memos: [VoiceMemo] = []
    @Published var detail = "looking for saved ideas on YuE Remote…"
    private var browser: NWBrowser?
    private var endpoint: NWEndpoint?
    private var connection: NWConnection?

    func start() {
        guard browser == nil else { return }
        let params = NWParameters.tcp; params.includePeerToPeer = true
        let b = NWBrowser(for: .bonjour(type: "_yuevoice._tcp", domain: nil), using: params)
        b.stateUpdateHandler = { [weak self] state in
            Task { @MainActor in if case .failed(let error) = state { self?.detail = "voice memo discovery failed: \(error.localizedDescription)" } }
        }
        b.browseResultsChangedHandler = { [weak self] results, _ in
            Task { @MainActor in self?.update(results) }
        }
        b.start(queue: .main); browser = b
    }

    func stop() {
        browser?.cancel(); browser = nil; connection?.cancel(); connection = nil
        endpoint = nil; phoneName = nil; memos = []; detail = "voice memo service stopped"
    }

    func refresh() {
        guard let endpoint else { detail = "no iPhone voice memo service found"; return }
        connection?.cancel()
        let c = NWConnection(to: endpoint, using: .tcp); connection = c
        c.stateUpdateHandler = { [weak self] state in
            Task { @MainActor in if case .failed(let error) = state { self?.detail = "cannot connect to voice memos: \(error.localizedDescription)" } }
        }
        c.start(queue: .main)
        Task { [weak self] in await self?.load(c) }
    }

    func importMemo(_ memo: VoiceMemo, into store: VoiceMemoStore, completion: ((URL?) -> Void)? = nil) {
        guard let endpoint else { detail = "no iPhone voice memo service found"; return }
        connection?.cancel()
        let c = NWConnection(to: endpoint, using: .tcp); connection = c; c.start(queue: .main)
        Task { [weak self] in
            do {
                let reply = try await self?.request(c, MemoFrame(header: ["op": "get", "id": memo.id], payload: Data()))
                guard let reply, reply.header["ok"] as? Bool == true else { throw CocoaError(.fileReadUnknown) }
                let imported = store.addTransferred(memo, data: reply.payload)
                await MainActor.run {
                    self?.detail = imported == nil ? "could not save \(memo.title)" : "imported \(memo.title)"
                    completion?(imported.map { store.url(for: $0) })
                }
                c.cancel()
            } catch { await MainActor.run { self?.detail = "voice memo download failed: \(error.localizedDescription)" }; c.cancel() }
        }
    }

    private func update(_ results: Set<NWBrowser.Result>) {
        guard let result = results.first else {
            endpoint = nil; phoneName = nil; memos = []; detail = "no iPhone voice memo service found"; return
        }
        endpoint = result.endpoint
        if case .service(let name, _, _, _) = result.endpoint { phoneName = name }
        detail = "iPhone found · refreshing saved ideas…"
        refresh()
    }

    private func load(_ connection: NWConnection) async {
        do {
            _ = try await request(connection, MemoFrame(header: ["op": "hello"], payload: Data()))
            let reply = try await request(connection, MemoFrame(header: ["op": "list"], payload: Data()))
            let parsed = (reply.header["memos"] as? [[String: Any]] ?? []).compactMap(Self.memo)
            await MainActor.run { [weak self] in self?.memos = parsed; self?.detail = parsed.isEmpty ? "iPhone connected · no saved ideas" : "\(parsed.count) saved idea(s) on iPhone" }
            connection.cancel()
        } catch { await MainActor.run { [weak self] in self?.detail = "voice memo list failed: \(error.localizedDescription)" } }
    }

    private func request(_ connection: NWConnection, _ frame: MemoFrame) async throws -> MemoFrame {
        try await connection.sendMemoFrame(frame); return try await connection.receiveMemoFrame()
    }

    private static func memo(_ value: [String: Any]) -> VoiceMemo? {
        guard let id = value["id"] as? String, let title = value["title"] as? String,
              let created = value["created_at"] as? Double, let seconds = value["seconds"] as? Double,
              let fileName = value["file_name"] as? String else { return nil }
        return VoiceMemo(id: id, title: title, createdAt: Date(timeIntervalSince1970: created), seconds: seconds,
                         fileName: fileName, source: value["source"] as? String ?? "iPhone")
    }
}
