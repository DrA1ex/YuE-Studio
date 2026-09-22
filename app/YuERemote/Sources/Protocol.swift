import Foundation
import Network

/// Wire format shared with tools/yue2 (src/yue2/remote/protocol.py):
///   u32 header length | header JSON | u64 payload length | payload
/// Every request gets one terminal reply with the same "op"; long operations may send
/// {"op":"progress","text":...} frames before it.
struct Frame {
    var header: [String: Any]
    var payload: Data

    var op: String { header["op"] as? String ?? "" }

    func encoded() -> Data {
        let json = try! JSONSerialization.data(withJSONObject: header)
        var out = Data(capacity: 12 + json.count + payload.count)
        var hl = UInt32(json.count).bigEndian
        var pl = UInt64(payload.count).bigEndian
        out.append(Data(bytes: &hl, count: 4)); out.append(json)
        out.append(Data(bytes: &pl, count: 8)); out.append(payload)
        return out
    }
}

enum WireError: Error { case closed, badHeader }

extension NWConnection {
    /// Receive exactly n bytes (chunked; payloads reach 100 MB).
    func receiveExactly(_ n: Int) async throws -> Data {
        var out = Data(capacity: n)
        while out.count < n {
            let maximum = min(4 << 20, n - out.count)
            let minimum = min(64 << 10, maximum)
            let chunk: Data = try await withCheckedThrowingContinuation { cont in
                receive(minimumIncompleteLength: minimum, maximumLength: maximum) { data, _, complete, error in
                    if let error { cont.resume(throwing: error); return }
                    if let data, !data.isEmpty { cont.resume(returning: data); return }
                    if complete { cont.resume(throwing: WireError.closed); return }
                    cont.resume(throwing: WireError.closed)
                }
            }
            out.append(chunk)
        }
        return out
    }

    func receiveFrame() async throws -> Frame {
        let hl = try await receiveExactly(4).withUnsafeBytes { UInt32(bigEndian: $0.loadUnaligned(as: UInt32.self)) }
        let json = try await receiveExactly(Int(hl))
        guard let header = try JSONSerialization.jsonObject(with: json) as? [String: Any] else { throw WireError.badHeader }
        let pl = try await receiveExactly(8).withUnsafeBytes { UInt64(bigEndian: $0.loadUnaligned(as: UInt64.self)) }
        let payload = pl > 0 ? try await receiveExactly(Int(pl)) : Data()
        return Frame(header: header, payload: payload)
    }

    func sendFrame(_ frame: Frame) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            send(content: frame.encoded(), completion: .contentProcessed { error in
                if let error { cont.resume(throwing: error) } else { cont.resume() }
            })
        }
    }
}
