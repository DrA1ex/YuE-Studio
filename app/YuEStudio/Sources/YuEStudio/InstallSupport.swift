import Foundation

/// Runs a process streaming merged stdout+stderr: JSON lines go to onLine, everything else
/// (noise-filtered, capped) to append. Throws on nonzero exit. Shared by both installers.
@MainActor
func runStreaming(_ exe: String, _ args: [String], _ env: [String: String],
                  register: (Process?) -> Void, append: @escaping @MainActor (String) -> Void,
                  onLine: (@Sendable (String) -> Void)? = nil) async throws {
    let p = Process(); p.executableURL = URL(fileURLWithPath: exe); p.arguments = args; p.environment = env
    register(p)
    let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
    final class LineBuffer: @unchecked Sendable { var pending = "" }
    let buffer = LineBuffer()                          // buffers partial lines between chunks
    pipe.fileHandleForReading.readabilityHandler = { h in
        buffer.pending += String(decoding: h.availableData, as: UTF8.self)
        var lines: [String] = []
        while let r = buffer.pending.firstIndex(where: { $0 == "\n" || $0 == "\r" }) {
            let line = String(buffer.pending[..<r]).trimmingCharacters(in: .whitespaces)
            buffer.pending = String(buffer.pending[buffer.pending.index(after: r)...])
            if line.hasPrefix("{"), let onLine { onLine(line); continue }        // structured progress, not log
            if !line.isEmpty && !line.contains("Fetching ") && !line.contains("it/s]") && !line.contains("not on your PATH") { lines.append(String(line.prefix(300))) }
        }
        if !lines.isEmpty { Task { @MainActor in for l in lines { append(l) } } }
    }
    try p.run()
    await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in p.terminationHandler = { _ in c.resume() } }
    pipe.fileHandleForReading.readabilityHandler = nil
    register(nil)
    if p.terminationStatus != 0 { throw NSError(domain: "YuEStudio", code: Int(p.terminationStatus), userInfo: [NSLocalizedDescriptionKey: "\(URL(fileURLWithPath: exe).lastPathComponent) exited with status \(p.terminationStatus)"]) }
}

/// Parses a download script's {"bytes","total","rate_mbps"} line into a fraction and human detail.
func downloadProgress(_ line: String) -> (fraction: Double, detail: String, bytes: Double, rate: Double)? {
    guard line.hasPrefix("{"), let d = line.data(using: .utf8),
          let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
          let bytes = o["bytes"] as? Double, let total = o["total"] as? Double, total > 0 else { return nil }
    let rate = o["rate_mbps"] as? Double ?? 0
    var remaining = ""
    if rate > 1 {
        let seconds = max(0, (total - bytes) / (rate * 1e6))
        remaining = seconds < 60 ? " · under a minute left" : String(format: " · about %.0f min left", seconds / 60)
    }
    return (min(0.99, bytes / total), String(format: "%.2f of %.1f GB · %.0f MB/s%@", bytes / 1e9, total / 1e9, rate, remaining), bytes, rate)
}
