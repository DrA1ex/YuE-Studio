import SwiftUI
import AVFoundation
import AppKit
import UniformTypeIdentifiers

@MainActor
final class Installer: ObservableObject {
    static let runtimeSchema = 1
    enum State: Equatable { case checking, needed, running, ready, failed(String) }
    struct Step: Identifiable { let id: Int; let title: String; var done = false }
    @Published var state: State = .checking
    @Published var steps: [Step] = [Step(id: 0, title: "Copy YuE source"), Step(id: 1, title: "Install Python 3.12"), Step(id: 2, title: "Create environment"),
                                     Step(id: 3, title: "Install or update packages"), Step(id: 4, title: "Install FFmpeg"),
                                     Step(id: 5, title: "Download the music model (about 7 GB)"), Step(id: 6, title: "Finish")]
    @Published var current = 0
    @Published var progress = 0.0
    @Published var detail = ""
    @Published var log: [LogLine] = []
    private var task: Task<Void, Never>?
    private var running: Process?
    private var lastRateLog = Date.distantPast

    private var mainModelURL: URL { Paths.models.appendingPathComponent("hub/models--m-a-p--YuE2-3B") }
    private var vaeModelURL: URL { Paths.models.appendingPathComponent("hub/models--m-a-p--YuE2-Vae") }

    static func installationIsUsable(pythonPresent: Bool, modelsPresent: Bool, ffmpegPresent: Bool,
                                     installedSchema: Int?, versionCurrent: Bool) -> Bool {
        pythonPresent && modelsPresent && ffmpegPresent && versionCurrent
            && (installedSchema == nil || installedSchema == runtimeSchema)
    }

    private func installedInfo() -> [String: Any] {
        guard let data = try? Data(contentsOf: Paths.installedMarker),
              let info = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
        return info
    }

    private func writeInstalledMarker() throws {
        let data = try JSONSerialization.data(withJSONObject: ["version": Paths.bundledVersion, "runtime_schema": Self.runtimeSchema])
        try data.write(to: Paths.installedMarker, options: .atomic)
    }

    /// Ordinary app updates should refresh bundled source without redownloading the runtime or models.
    private func refreshBundledSource() throws {
        guard let payload = Paths.payload else { return }
        try FileManager.default.createDirectory(at: Paths.support, withIntermediateDirectories: true)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/rsync")
        p.arguments = ["-a", "--delete", payload.appendingPathComponent("yue2-src").path + "/", Paths.src.path + "/"]
        try p.run(); p.waitUntilExit()
        guard p.terminationStatus == 0 else {
            throw NSError(domain: "YuEStudio.Install", code: Int(p.terminationStatus), userInfo: [NSLocalizedDescriptionKey: "Could not refresh bundled YuE source."])
        }
    }

    func cancel() { running?.terminate(); task?.cancel() }

    func check() {
        guard Paths.packaged else { state = .ready; return }
        let fm = FileManager.default
        let info = installedInfo()
        let pythonPresent = fm.isExecutableFile(atPath: Paths.python.path)
        let modelsPresent = fm.fileExists(atPath: mainModelURL.path) && fm.fileExists(atPath: vaeModelURL.path)
        let schema = info["runtime_schema"] as? Int
        let versionCurrent = info["version"] as? String == Paths.bundledVersion
        guard Self.installationIsUsable(pythonPresent: pythonPresent, modelsPresent: modelsPresent,
                                        ffmpegPresent: Paths.ffmpeg != nil, installedSchema: schema,
                                        versionCurrent: versionCurrent) else {
            state = .needed
            return
        }
        state = .ready
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
                                     "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                                     "HOME": NSHomeDirectory()]
        task = Task { [weak self] in
            guard let self else { return }
            do {
                try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
                try FileManager.default.createDirectory(at: Paths.output, withIntermediateDirectories: true)
                try await step(0) {
                    try await self.run("/usr/bin/rsync", ["-a", "--delete",
                        payload.appendingPathComponent("yue2-src").path + "/", Paths.src.path + "/"], env)
                }
                try await step(1) { try await self.run(uv, ["python", "install", "3.12"], env) }
                try await step(2) {
                    if FileManager.default.isExecutableFile(atPath: Paths.python.path) {
                        self.detail = "Existing environment found · reusing it"
                        self.append("Python environment already exists; keeping installed packages")
                    } else {
                        try await self.run(uv, ["venv", support.appendingPathComponent("env").path,
                                                "--python", "3.12", "--clear"], env)
                    }
                }
                try await step(3) {
                    try await self.run(uv, ["pip", "install", "--python", Paths.python.path,
                                            Paths.src.path + "[apple]"], env)
                }
                try await step(4) { try await self.ensureFFmpeg(env) }
                try await step(5) {
                    // The downloader first verifies both pinned snapshots locally. A healthy existing
                    // installation therefore needs no network; missing or corrupt files are repaired.
                    try await self.run(Paths.python.path, [Paths.src.appendingPathComponent("tools/download_models.py").path], env) { [weak self] line in
                        guard line.hasPrefix("{"), let d = line.data(using: .utf8),
                              let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return }
                        if o["cached"] as? Bool == true {
                            Task { @MainActor in
                                self?.detail = "Existing pinned models verified · no download needed"
                                self?.append("YuE2 and VAE model snapshots verified")
                            }
                            return
                        }
                        guard let bytes = o["bytes"] as? Double, let total = o["total"] as? Double, total > 0 else { return }
                        let rate = o["rate_mbps"] as? Double ?? 0
                        var remaining = ""
                        if rate > 1 {
                            let seconds = max(0, (total - bytes) / (rate * 1e6))
                            remaining = seconds < 60 ? " · under a minute left" : String(format: " · about %.0f min left", seconds / 60)
                        }
                        Task { @MainActor in
                            guard let self else { return }
                            self.progress = min(0.99, bytes / total)
                            self.detail = String(format: "%.2f of %.1f GB · %.0f MB/s%@", bytes / 1e9, total / 1e9, rate, remaining)
                            if Date().timeIntervalSince(self.lastRateLog) > 15 && bytes > 0 {
                                self.lastRateLog = Date(); self.append(String(format: "Downloaded %.2f GB at %.0f MB/s", bytes / 1e9, rate))
                            }
                        }
                    }
                }
                try await step(6) {
                    try? FileManager.default.removeItem(at: support.appendingPathComponent("uv-cache"))
                    try self.writeInstalledMarker()
                }
                state = .ready
            } catch {
                append("Setup failed: \(error.localizedDescription)")
                state = .failed(error.localizedDescription)
            }
        }
    }

    private func ensureFFmpeg(_ env: [String: String]) async throws {
        if let ffmpeg = Paths.ffmpeg {
            detail = "Using \(ffmpeg.path)"
            append("FFmpeg already installed; reusing \(ffmpeg.path)")
            return
        }

        let candidates = ["/opt/homebrew/bin/brew", "/usr/local/bin/brew"]
        guard let brew = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            throw NSError(
                domain: "YuEStudio.Install",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey:
                    "FFmpeg is required for audio import and MP3 export. Install Homebrew, then retry setup."]
            )
        }

        detail = "Installing with Homebrew"
        append("FFmpeg is missing; installing it with Homebrew")
        try await run(brew, ["install", "ffmpeg"], env)
        guard Paths.ffmpeg != nil else {
            throw NSError(
                domain: "YuEStudio.Install",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Homebrew completed, but FFmpeg could not be found."]
            )
        }
    }

    private func step(_ i: Int, _ body: () async throws -> Void) async throws {
        current = i; detail = ""; append("Step \(i + 1): \(steps[i].title)")
        try await body()
        steps[i].done = true; progress = Double(i + 1) / Double(steps.count)
    }

    private func run(_ exe: String, _ args: [String], _ env: [String: String], onLine: (@Sendable (String) -> Void)? = nil) async throws {
        let p = Process(); p.executableURL = URL(fileURLWithPath: exe); p.arguments = args; p.environment = env
        running = p
        let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
        final class LineBuffer: @unchecked Sendable { var pending = "" }
        let buffer = LineBuffer()                          // buffers partial lines between chunks
        pipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            buffer.pending += String(decoding: h.availableData, as: UTF8.self)
            var lines: [String] = []
            while let r = buffer.pending.firstIndex(where: { $0 == "\n" || $0 == "\r" }) {
                let line = String(buffer.pending[..<r]).trimmingCharacters(in: .whitespaces)
                buffer.pending = String(buffer.pending[buffer.pending.index(after: r)...])
                if line.hasPrefix("{"), let onLine { onLine(line); continue }        // structured progress, not log
                if !line.isEmpty && !line.contains("Fetching ") && !line.contains("it/s]") && !line.contains("not on your PATH") { lines.append(String(line.prefix(300))) }
            }
            if !lines.isEmpty { Task { @MainActor in for l in lines { self?.append(l) } } }
        }
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
            p.terminationHandler = { _ in c.resume() }
            do { try p.run() } catch { c.resume(throwing: error) }
        }
        pipe.fileHandleForReading.readabilityHandler = nil
        running = nil
        if p.terminationStatus != 0 { throw NSError(domain: "YuEStudio", code: Int(p.terminationStatus), userInfo: [NSLocalizedDescriptionKey: "\(URL(fileURLWithPath: exe).lastPathComponent) exited with status \(p.terminationStatus)"]) }
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

// MARK: - Model
