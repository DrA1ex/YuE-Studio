import Foundation
import Combine

@MainActor
final class Installer: ObservableObject {
    enum State: Equatable { case checking, needed, running, ready, failed(String) }
    struct Step: Identifiable { let id: Int; let title: String; var done = false }
    @Published var state: State = .checking
    @Published var steps: [Step] = [Step(id: 0, title: "Copy YuE source"), Step(id: 1, title: "Install Python 3.12"), Step(id: 2, title: "Create environment"),
                                     Step(id: 3, title: "Install packages (about 1 GB)"), Step(id: 4, title: "Download the music model (about 7 GB)"), Step(id: 5, title: "Finish")]
    @Published var current = 0
    @Published var progress = 0.0
    @Published var detail = ""
    @Published var log: [LogLine] = []
    private var task: Task<Void, Never>?
    private var running: Process?
    private var lastRateLog = Date.distantPast

    func cancel() { running?.terminate(); task?.cancel() }

    func check() {
        guard Paths.packaged else { state = .ready; return }
        let fm = FileManager.default
        let installed = (try? JSONSerialization.jsonObject(with: Data(contentsOf: Paths.installedMarker)) as? [String: String])?["version"]
        let modelsPresent = fm.fileExists(atPath: Paths.models.appendingPathComponent("hub/models--m-a-p--YuE2-3B").path)
        state = (installed == Paths.bundledVersion && fm.fileExists(atPath: Paths.python.path) && modelsPresent) ? .ready : .needed
    }

    func append(_ message: String) {
        let f = DateFormatter(); f.dateFormat = "HH:mm:ss"
        log.append(LogLine(time: f.string(from: Date()), message: message))
    }

    func repair() {
        try? FileManager.default.removeItem(at: Paths.installedMarker)
        try? FileManager.default.removeItem(at: Paths.support.appendingPathComponent("env"))
        for i in steps.indices { steps[i].done = false }
        state = .needed; log.removeAll()
        install()
    }

    func install() {
        guard let payload = Paths.payload else { state = .ready; return }
        state = .running; progress = 0; current = 0
        let uv = payload.appendingPathComponent("uv").path
        let support = Paths.support
        let env: [String: String] = ["UV_PYTHON_INSTALL_DIR": support.appendingPathComponent("python").path,
                                     "UV_CACHE_DIR": support.appendingPathComponent("uv-cache").path,
                                     "HF_HOME": Paths.models.path, "HF_HUB_DISABLE_TELEMETRY": "1",
                                     "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": NSHomeDirectory()]
        task = Task { [weak self] in
            guard let self else { return }
            do {
                try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
                try FileManager.default.createDirectory(at: Paths.output, withIntermediateDirectories: true)
                try await step(0) { try await self.run("/usr/bin/rsync", ["-a", "--delete", payload.appendingPathComponent("yue2-src").path + "/", Paths.src.path + "/"], env) }
                try await step(1) { try await self.run(uv, ["python", "install", "3.12"], env) }
                try await step(2) { try await self.run(uv, ["venv", support.appendingPathComponent("env").path, "--python", "3.12", "--clear"], env) }
                try await step(3) { try await self.run(uv, ["pip", "install", "--python", Paths.python.path, Paths.src.path + "[apple]"], env) }
                try await step(4) {
                    // The download script reports byte progress from the Hub client's own callbacks.
                    try await self.run(Paths.python.path, [Paths.src.appendingPathComponent("tools/download_models.py").path], env) { [weak self] line in
                        guard let p = downloadProgress(line) else { return }
                        Task { @MainActor in
                            guard let self else { return }
                            self.progress = p.fraction; self.detail = p.detail
                            if Date().timeIntervalSince(self.lastRateLog) > 15 && p.bytes > 0 {
                                self.lastRateLog = Date(); self.append(String(format: "Downloaded %.2f GB at %.0f MB/s", p.bytes / 1e9, p.rate))
                            }
                        }
                    }
                }
                try await step(5) {
                    try? FileManager.default.removeItem(at: support.appendingPathComponent("uv-cache"))   // ~750 MB, not needed after install
                    let data = try JSONSerialization.data(withJSONObject: ["version": Paths.bundledVersion])
                    try data.write(to: Paths.installedMarker)
                }
                state = .ready
            } catch {
                append("Setup failed: \(error.localizedDescription)")
                state = .failed(error.localizedDescription)
            }
        }
    }

    private func step(_ i: Int, _ body: () async throws -> Void) async throws {
        current = i; detail = ""; append("Step \(i + 1): \(steps[i].title)")
        try await body()
        steps[i].done = true; progress = Double(i + 1) / Double(steps.count)
    }

    private func run(_ exe: String, _ args: [String], _ env: [String: String], onLine: (@Sendable (String) -> Void)? = nil) async throws {
        try await runStreaming(exe, args, env, register: { running = $0 }, append: { [weak self] in self?.append($0) }, onLine: onLine)
    }

    nonisolated static func directorySize(_ url: URL) -> Int64 {
        guard let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey], options: []) else { return 0 }
        var total: Int64 = 0
        for case let f as URL in e {
            if let v = try? f.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]), v.isRegularFile == true { total += Int64(v.fileSize ?? 0) }
        }
        return total
    }
}
