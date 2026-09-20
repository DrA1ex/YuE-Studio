import SwiftUI
import AVFoundation
import AppKit
import UniformTypeIdentifiers

// MARK: - Paths

/// Where things live. Packaged app: payload in the bundle, runtime under Application Support.
/// Development (swift run from the repo): the repo's .venv and tools/ are used directly.
struct Paths {
    static let support: URL = {
        if let o = ProcessInfo.processInfo.environment["YUE_STUDIO_SUPPORT"] { return URL(fileURLWithPath: o) }
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("YuE Studio")
    }()
    static let payload: URL? = {
        guard let r = Bundle.main.resourceURL?.appendingPathComponent("payload"),
              FileManager.default.fileExists(atPath: r.appendingPathComponent("uv").path) else { return nil }
        return r
    }()
    static var packaged: Bool { payload != nil }
    static let repoRoot: URL = { var u = URL(fileURLWithPath: #filePath); for _ in 0..<5 { u.deleteLastPathComponent() }; return u }()
    static var python: URL {
        if let override = ProcessInfo.processInfo.environment["YUE_STUDIO_PYTHON"] { return URL(fileURLWithPath: override) }
        let local = repoRoot.appendingPathComponent(".venv/bin/python")
        return !packaged && FileManager.default.isExecutableFile(atPath: local.path) ? local : support.appendingPathComponent("env/bin/python")
    }
    static var worker: URL {
        if let o = ProcessInfo.processInfo.environment["YUE_STUDIO_WORKER"] { return URL(fileURLWithPath: o) }   // tests
        return packaged ? support.appendingPathComponent("src/tools/yue2_worker.py") : repoRoot.appendingPathComponent("tools/yue2_worker.py")
    }
    static var transcriber: URL {
        if let o = ProcessInfo.processInfo.environment["YUE_STUDIO_TRANSCRIBER"] { return URL(fileURLWithPath: o) }
        return packaged ? support.appendingPathComponent("src/tools/transcribe_cover.py") : repoRoot.appendingPathComponent("tools/transcribe_cover.py")
    }
    static var src: URL { support.appendingPathComponent("src") }
    static var models: URL { ProcessInfo.processInfo.environment["YUE_STUDIO_HF_HOME"].map { URL(fileURLWithPath: $0) } ?? support.appendingPathComponent("models") }
    static var aneCache: URL { support.appendingPathComponent("ane-cache") }
    static var output: URL {
        if let override = ProcessInfo.processInfo.environment["YUE_STUDIO_OUTPUT"] { return URL(fileURLWithPath: override) }
        return packaged ? FileManager.default.urls(for: .musicDirectory, in: .userDomainMask)[0].appendingPathComponent("YuE Studio")
                 : repoRoot.appendingPathComponent("outputs/app")
    }
    static var imports: URL { output.appendingPathComponent("Imports", isDirectory: true) }
    static var coverSupport: URL { support.appendingPathComponent("cover-runtime", isDirectory: true) }
    static var coverAnalyses: URL { coverSupport.appendingPathComponent("analyses", isDirectory: true) }
    static var coverPython: URL { coverSupport.appendingPathComponent("env/bin/python") }
    static var coverWhisperPython: URL { coverSupport.appendingPathComponent("whisper-env/bin/python") }
    static var coverRequirements: URL { packaged ? src.appendingPathComponent("tools/sheetsage-requirements.txt") : repoRoot.appendingPathComponent("tools/sheetsage-requirements.txt") }
    static var coverWhisperRequirements: URL { packaged ? src.appendingPathComponent("tools/whisper-requirements.txt") : repoRoot.appendingPathComponent("tools/whisper-requirements.txt") }
    static var coverSeparationModel: URL { coverSupport.appendingPathComponent("torch/torchaudio/models/hdemucs_high_musdbhq_only.pt") }
    static var installedMarker: URL { support.appendingPathComponent("installed.json") }
    static var bundledVersion: String { (try? String(contentsOf: payload!.appendingPathComponent("version.txt"), encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "dev" }
    static var workerEnvironment: [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"; env["TQDM_DISABLE"] = "1"
        env["YUE2_OUTPUT_DIR"] = output.path; env["YUE2_ANE_CACHE"] = aneCache.path
        env["HF_HOME"] = models.path; env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        env["PYTHONPATH"] = (packaged ? src : repoRoot).appendingPathComponent("src").path
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        return env
    }
}

// MARK: - Installer

@MainActor
final class Installer: ObservableObject {
    static let runtimeSchema = 1
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

    private var mainModelURL: URL { Paths.models.appendingPathComponent("hub/models--m-a-p--YuE2-3B") }

    static func installationIsUsable(pythonPresent: Bool, modelsPresent: Bool, installedSchema: Int?) -> Bool {
        pythonPresent && modelsPresent && (installedSchema == nil || installedSchema == runtimeSchema)
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
        let modelsPresent = fm.fileExists(atPath: mainModelURL.path)
        let schema = info["runtime_schema"] as? Int
        guard Self.installationIsUsable(pythonPresent: pythonPresent, modelsPresent: modelsPresent, installedSchema: schema) else {
            state = .needed; return
        }
        if info["version"] as? String != Paths.bundledVersion || schema == nil {
            do {
                try refreshBundledSource()
                try writeInstalledMarker()
                append("Existing models and runtime reused for this app update")
            } catch {
                append("Could not refresh app resources: \(error.localizedDescription)")
                state = .needed; return
            }
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
                    if FileManager.default.fileExists(atPath: self.mainModelURL.path) {
                        self.detail = "Existing YuE2 model found · reusing downloaded files"
                        self.append("YuE2 model already downloaded; skipping model download")
                    } else {
                        // The download script reports byte progress from the Hub client's own callbacks.
                        try await self.run(Paths.python.path, [Paths.src.appendingPathComponent("tools/download_models.py").path], env) { [weak self] line in
                            guard line.hasPrefix("{"), let d = line.data(using: .utf8),
                                  let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                                  let bytes = o["bytes"] as? Double, let total = o["total"] as? Double, total > 0 else { return }
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
                }
                try await step(5) {
                    try? FileManager.default.removeItem(at: support.appendingPathComponent("uv-cache"))   // ~750 MB, not needed after install
                    try self.writeInstalledMarker()
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

struct LogLine: Identifiable { let id = UUID(); let time: String; let message: String }
struct Song: Identifiable, Equatable {
    /// In-flight stages come from the worker; ready/stalled/failed are settled states.
    enum Status: Equatable { case queued, planning, tokens, synth, decode, ready, stalled, failed }
    var id: String { path }                      // the audio file path: stable across restarts
    let run: String                              // output folder name (timestamp)
    let index: Int
    let path: String
    var score: String
    var seconds: Double
    let seed: Int
    var truncated: Bool
    var status: Status
    var quality = "full"                         // "draft" (8-step MLX preview) or "full" (32 steps)
    var detail = ""                              // what the current stage is doing, or the failure
    var fraction: Double? = nil                  // stage progress, when known
    var gflops: Double? = nil                    // rough throughput of the current stage, GFLOP/s
    var engine = ""                              // synthesis engine (ane / mlx / torch)
    var priority = 0                             // scheduling order (1 = first added); 0 = unknown
    var title = ""                               // user-facing title, persisted in metadata.json when available
    var style = ""
    var kind = "GENERATED"                       // GENERATED, UPLOAD or COVER
    var sourcePath: String? = nil                // original audio for a cover
    var lyrics = ""
    var createdAt: Date = .distantPast
    var canRender: Bool { kind != "UPLOAD" && !inFlight && FileManager.default.fileExists(atPath: directory.appendingPathComponent("semantic.npy").path) && FileManager.default.fileExists(atPath: directory.appendingPathComponent("plan_manifest.json").path) }
    var directory: URL { URL(fileURLWithPath: path).deletingLastPathComponent() }
    var inFlight: Bool { [.queued, .planning, .tokens, .synth, .decode].contains(status) }
    var runLabel: String {
        let f = DateFormatter(); f.dateFormat = "yyyyMMdd-HHmmss"
        guard let d = f.date(from: String(run.prefix(15))) else { return run }
        return d.formatted(date: .abbreviated, time: .standard)
    }
    var engineLabel: String {
        switch engine { case "ane": return "Neural Engine"; case "mlx": return "GPU (MLX)"; case "torch": return "GPU (PyTorch)"; case "mlx+ane": return "GPU, then Neural Engine"; default: return "" }
    }
    /// Position on the stage track: completed stages plus progress within the current one (0 queued ... 4 done).
    var trackProgress: Double {
        let within = min(1, max(0, fraction ?? 0))
        switch status {
        case .queued: return 0
        case .planning: return within
        case .tokens: return 1 + within
        case .synth: return 2 + within
        case .decode: return 3 + within
        case .ready: return 4
        case .stalled: return 2
        case .failed: return 0
        }
    }
    var throughput: String {
        guard let g = gflops, g > 0 else { return "" }
        return g >= 1000 ? String(format: " · %.2f TFLOP/s", g / 1000) : String(format: " · %.0f GFLOP/s", g)
    }
    var statusLine: String {
        switch status {
        case .queued: return "Queued" + (detail.isEmpty ? "" : " · \(detail)")
        case .planning: return "Planning the score on GPU" + (detail.isEmpty ? "" : " · \(detail)") + throughput
        case .tokens: return "Tokenizing on GPU" + (detail.isEmpty ? "" : " · \(detail)") + throughput
        case .synth: return "Synthing" + (engineLabel.isEmpty ? "" : " using \(engineLabel)") + (detail.isEmpty ? "" : " · \(detail)") + throughput
        case .decode: return "Rendering on GPU" + (detail.isEmpty ? "" : " · \(detail)") + throughput
        case .ready: return "Done"
        case .stalled: return "Tokens saved · not synthesized yet"
        case .failed: return "Failed · \(detail)"
        }
    }

    /// Songs on disk under the output folder: <run>/song<N>/ with audio.flac (ready) or only saved
    /// tokens (stalled: can be synthesized later) plus their sidecar files.
    nonisolated static func scan(_ root: URL) -> [Song] {
        let fm = FileManager.default
        guard let runs = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else { return [] }
        var songs: [Song] = []
        for run in runs {
            guard let dirs = try? fm.contentsOfDirectory(at: run, includingPropertiesForKeys: nil) else { continue }
            for dir in dirs where dir.lastPathComponent.hasPrefix("song") {
                let audio = dir.appendingPathComponent("audio.flac")
                let hasAudio = fm.fileExists(atPath: audio.path)
                let hasTokens = fm.fileExists(atPath: dir.appendingPathComponent("semantic.npy").path) && fm.fileExists(atPath: dir.appendingPathComponent("plan_manifest.json").path)
                guard hasAudio || hasTokens else { continue }
                let json = { (name: String) -> [String: Any]? in (try? JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent(name)))) as? [String: Any] }
                let request = json("request.json"), result = hasAudio ? json("result.json") : nil, tokens = json("tokens.json"), metadata = json("metadata.json")
                let truncated = (result?["truncated"] as? [String: Bool])?.values.contains(true) ?? (tokens?["truncated"] as? Bool ?? false)
                songs.append(Song(run: run.lastPathComponent, index: Int(dir.lastPathComponent.dropFirst(4)) ?? 0, path: audio.path,
                                  score: (try? String(contentsOf: dir.appendingPathComponent("score.abc"), encoding: .utf8)) ?? (try? String(contentsOf: dir.appendingPathComponent("cover-score.abc"), encoding: .utf8)) ?? "",
                                  seconds: result?["audio_seconds"] as? Double ?? ((tokens?["frames"] as? Double ?? 0) / 25),
                                  seed: request?["seed"] as? Int ?? 0, truncated: truncated, status: hasAudio ? .ready : .stalled,
                                  quality: result?["quality"] as? String ?? "full",
                                  detail: hasAudio ? "" : "tokens saved · not synthesized yet",
                                  title: metadata?["title"] as? String ?? "Song \(Int(dir.lastPathComponent.dropFirst(4)) ?? 0)",
                                  style: metadata?["style"] as? String ?? request?["style"] as? String ?? "",
                                  kind: metadata?["kind"] as? String ?? "GENERATED",
                                  sourcePath: metadata?["source_path"] as? String,
                                  lyrics: request?["lyrics"] as? String ?? "",
                                  createdAt: (try? dir.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast))
            }
        }
        let imports = root.appendingPathComponent("Imports", isDirectory: true)
        if let importedFiles = try? fm.contentsOfDirectory(at: imports, includingPropertiesForKeys: nil) {
            let audioExtensions = Set(["mp3", "wav", "aiff", "aif", "m4a", "flac", "caf"])
            for audio in importedFiles where audioExtensions.contains(audio.pathExtension.lowercased()) {
                let metadataURL = audio.deletingPathExtension().appendingPathExtension("json")
                let metadata = (try? JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL)) as? [String: Any]) ?? [:]
                let title = metadata["title"] as? String ?? audio.deletingPathExtension().lastPathComponent
                let seconds = (try? AVAudioFile(forReading: audio)).map { Double($0.length) / $0.processingFormat.sampleRate } ?? 0
                songs.append(Song(run: "Imports", index: songs.count + 1, path: audio.path, score: metadata["score"] as? String ?? "",
                                  seconds: seconds, seed: 0, truncated: false, status: .ready, quality: "full",
                                  title: title, style: metadata["style"] as? String ?? "", kind: "UPLOAD",
                                  lyrics: metadata["lyrics"] as? String ?? "",
                                  createdAt: (try? audio.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast))
            }
        }
        return songs
    }
}

struct AudioSource: Identifiable, Equatable {
    let id: String
    let path: String
    var title: String
    var seconds: Double
    var score: String = ""
    var lyrics: String = ""
    var style: String = ""
    var kind: String = "UPLOAD"
}

struct CoverAnalysis {
    let score: String
    let lyrics: String
    let genre: String
    var warnings: [String] = []
}

@MainActor
final class Backend: ObservableObject {
    @Published var log: [LogLine] = []
    @Published var songs: [Song] = []
    @Published var audioSources: [AudioSource] = []
    @Published var busy = false                  // anything queued or in a stage
    @Published var connected = false
    @Published var coverLyricsReady = CoverComponent.lyrics.isInstalled
    @Published var coverStyleReady = CoverComponent.style.isInstalled
    @Published var coverReady = CoverComponent.melody.isInstalled
    @Published var coverMlxReady = CoverComponent.mlxWhisper.isInstalled && FileManager.default.isExecutableFile(atPath: Paths.coverWhisperPython.path)
    @Published var coverSeparationReady = CoverComponent.vocalActivity.isInstalled
    @Published var coverComponents: [CoverComponent: Bool] = Dictionary(uniqueKeysWithValues: CoverComponent.allCases.map { ($0, $0.isInstalled) })
    @Published var coverBusy = false
    @Published var coverStatus = ""
    @Published var errorMessage: String?
    var coverTask: Task<Void, Never>?
    var coverProcess: Process?
    var shuttingDown = false
    private var pending: Set<String> = []
    private var pendingRenders: [String: String] = [:]

    var coverRuntimeReady: Bool { coverReady && FileManager.default.isExecutableFile(atPath: Paths.coverPython.path) }

    func coverComponentReady(_ component: CoverComponent) -> Bool {
        coverComponents[component] ?? component.isInstalled
    }

    func refreshCoverInstallationState() {
        coverComponents = Dictionary(uniqueKeysWithValues: CoverComponent.allCases.map { ($0, $0.isInstalled) })
        coverReady = coverComponents[.melody] == true
        coverStyleReady = coverComponents[.style] == true
        coverLyricsReady = coverComponents[.lyrics] == true
        coverMlxReady = coverComponents[.mlxWhisper] == true && FileManager.default.isExecutableFile(atPath: Paths.coverWhisperPython.path)
        coverSeparationReady = coverComponents[.vocalActivity] == true
    }

    var process: Process?
    private var stdin: FileHandle?
    private var buffer = Data()

    func start() {
        guard process == nil, !shuttingDown else { return }
        buffer.removeAll()
        let p = Process()
        p.executableURL = Paths.python
        p.arguments = ["-u", Paths.worker.path]
        p.currentDirectoryURL = Paths.packaged ? Paths.src : Paths.repoRoot
        p.environment = Paths.workerEnvironment
        let inPipe = Pipe(), outPipe = Pipe(), errPipe = Pipe()
        p.standardInput = inPipe; p.standardOutput = outPipe; p.standardError = errPipe
        outPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            Task { @MainActor in self?.consume(data) }
        }
        errPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard let text = String(data: data, encoding: .utf8), !text.isEmpty else { return }
            Task { @MainActor in
                for line in text.split(separator: "\n") where !line.contains("Warning") && !line.contains("warn") && !line.contains("Running MIL") && !line.contains("passes/s") && !line.contains("torch_dtype") && !line.contains("Fetching") && !line.contains("coremltools") && !line.contains("has not been tested") && !line.isEmpty {
                    self?.append("stderr: \(line)")
                }
            }
        }
        p.terminationHandler = { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.connected = false; self.process = nil; self.stdin = nil
                self.pending.removeAll()
                self.pendingRenders.removeAll()
                for i in self.songs.indices where self.songs[i].inFlight {
                    self.songs[i].status = .failed; self.songs[i].detail = "Worker exited; saved tokens can be recovered"
                }
                self.append("Worker exited"); self.rescan()
            }
        }
        do {
            try p.run()
            process = p; stdin = inPipe.fileHandleForWriting
            append("Worker started: \(p.executableURL!.path)")
        } catch {
            report("Could not start worker: \(error.localizedDescription)")
        }
    }

    func consume(_ data: Data) {
        buffer.append(data)
        while let range = buffer.range(of: Data([0x0A])) {
            let line = buffer.subdata(in: buffer.startIndex..<range.lowerBound)
            buffer.removeSubrange(buffer.startIndex...range.lowerBound)
            guard let obj = try? JSONSerialization.jsonObject(with: line) as? [String: Any], let event = obj["event"] as? String else { continue }
            let path = obj["path"] as? String ?? ""
            switch event {
            case "ready": connected = true; append("Worker ready")
            case "log": append(obj["message"] as? String ?? "")
            case "started":
                if let id = obj["request_id"] as? String { pending.remove(id); pendingRenders.removeValue(forKey: id) }
                // Placeholders for the queued songs appear at once; a song already listed (a render of a
                // draft, or a stalled song) goes back into the pipeline.
                songs.removeAll { $0.status == .failed }
                let run = URL(fileURLWithPath: obj["output"] as? String ?? "").lastPathComponent
                for entry in obj["songs"] as? [[String: Any]] ?? [] {
                    let path = entry["path"] as? String ?? ""
                    let priority = entry["priority"] as? Int ?? 0
                    let title = entry["title"] as? String ?? "Song \(entry["index"] as? Int ?? 0)"
                    let kind = entry["kind"] as? String ?? "GENERATED"
                    let sourcePath = entry["source_path"] as? String
                    if let i = songs.firstIndex(where: { $0.path == path }) {
                        songs[i].status = .queued; songs[i].detail = "queued"; songs[i].fraction = nil; songs[i].priority = priority
                        songs[i].title = title; songs[i].kind = kind; songs[i].sourcePath = sourcePath
                        songs[i].style = entry["style"] as? String ?? ""; songs[i].lyrics = entry["lyrics"] as? String ?? ""
                        songs[i].quality = entry["quality"] as? String ?? "full"
                    } else {
                        songs.append(Song(run: run, index: entry["index"] as? Int ?? 0, path: path, score: "", seconds: 0,
                                          seed: entry["seed"] as? Int ?? 0, truncated: false, status: .queued, quality: entry["quality"] as? String ?? "full", detail: "queued", priority: priority,
                                          title: title, style: entry["style"] as? String ?? "", kind: kind, sourcePath: sourcePath,
                                          lyrics: entry["lyrics"] as? String ?? "", createdAt: Date()))
                    }
                }
                sortSongs(); updateBusy()
            case "stage":
                guard let i = songs.firstIndex(where: { $0.path == path }) else { break }
                let detail = obj["detail"] as? String ?? ""
                if let engine = obj["engine"] as? String { songs[i].engine = engine }
                if let p = obj["priority"] as? Int { songs[i].priority = p }
                let stages: [String: Song.Status] = ["queued": .queued, "planning": .planning, "tokens": .tokens, "synth": .synth,
                                                     "decode": .decode, "ready": .ready, "failed": .failed]
                let stage = obj["stage"] as? String ?? ""
                if stage == "cancelled" { songs.remove(at: i); rescan(); updateBusy(); break }   // back to whatever is on disk
                guard let status = stages[stage] else { break }
                if status != songs[i].status { songs[i].fraction = nil; songs[i].gflops = nil }    // a new stage starts from zero
                songs[i].status = status
                songs[i].detail = status == .ready ? "" : detail
                updateBusy()
            case "progress":
                guard let i = songs.firstIndex(where: { $0.path == path }) else { break }
                songs[i].fraction = obj["fraction"] as? Double
                songs[i].detail = obj["detail"] as? String ?? songs[i].detail
                songs[i].gflops = obj["gflops"] as? Double              // absent = no rate to show (e.g. a finished row)
            case "song":
                let song = Song(run: URL(fileURLWithPath: path).deletingLastPathComponent().deletingLastPathComponent().lastPathComponent,
                                index: obj["index"] as? Int ?? 0, path: path, score: obj["score"] as? String ?? "",
                                seconds: obj["seconds"] as? Double ?? 0, seed: obj["seed"] as? Int ?? 0,
                                truncated: obj["truncated"] as? Bool ?? false, status: .ready, quality: obj["quality"] as? String ?? "full",
                                engine: obj["engine"] as? String ?? "", title: obj["title"] as? String ?? "",
                                style: obj["style"] as? String ?? "", kind: obj["kind"] as? String ?? "GENERATED",
                                sourcePath: obj["source_path"] as? String, lyrics: obj["lyrics"] as? String ?? "", createdAt: Date())
                if let i = songs.firstIndex(where: { $0.path == path }) { songs[i] = song } else { songs.append(song) }
                sortSongs(); updateBusy()
            case "failed":
                if let i = songs.firstIndex(where: { $0.path == path }) { songs[i].status = .failed; songs[i].detail = obj["message"] as? String ?? "failed" }
                updateBusy()
            case "idle": updateBusy(); rescan()
            case "error":
                if let id = obj["request_id"] as? String {
                    pending.remove(id)
                    if let path = pendingRenders.removeValue(forKey: id), let i = songs.firstIndex(where: { $0.path == path }) {
                        songs[i].status = .failed; songs[i].detail = obj["message"] as? String ?? "Render failed"
                        rescan()
                    }
                }
                report("Worker: \(obj["message"] as? String ?? "error")"); updateBusy()
            default: break
            }
        }
    }

    func append(_ message: String) {
        let f = DateFormatter(); f.dateFormat = "HH:mm:ss"
        log.append(LogLine(time: f.string(from: Date()), message: message))
        if log.count > 2000 { log.removeFirst(log.count - 2000) }
    }

    func report(_ message: String) { append(message); errorMessage = message }
    private func updateBusy() { busy = !pending.isEmpty || songs.contains { $0.inFlight } }

    /// Newest run first, songs in order within a run.
    private func sortSongs() {
        songs.sort { $0.createdAt != $1.createdAt ? $0.createdAt > $1.createdAt : $0.index < $1.index }
    }

    /// Reconcile with the songs folder: everything on disk is listed (finished songs, and songs whose
    /// tokens were saved but never synthesized), and entries whose files are gone disappear. Songs
    /// the worker is still working on, and failures of this session, are kept as they are.
    func rescan() {
        let kept = songs.filter { $0.inFlight }
        let onDisk = Song.scan(Paths.output)
        let keptPaths = Set(kept.map(\.path))
        let diskPaths = Set(onDisk.map(\.path))
        songs = onDisk.filter { !keptPaths.contains($0.path) } + kept + songs.filter { $0.status == .failed && !diskPaths.contains($0.path) }
        sortSongs(); updateBusy()
    }

    @discardableResult func send(_ obj: [String: Any]) -> Bool {
        guard let stdin, process?.isRunning == true else { report("Worker is disconnected. Use Reconnect and try again."); return false }
        var request = obj
        let tracked = ["generate", "render"].contains(obj["cmd"] as? String ?? "")
        if tracked && (!connected || coverBusy) { report("Wait until the local engine is ready."); return false }
        let id = UUID().uuidString
        if tracked { request["request_id"] = id }
        do {
            var data = try JSONSerialization.data(withJSONObject: request); data.append(0x0A)
            try stdin.write(contentsOf: data)
            if tracked {
                pending.insert(id)
                if obj["cmd"] as? String == "render", let path = obj["path"] as? String { pendingRenders[id] = path }
                updateBusy()
            }
            return true
        } catch { report("Could not send command: \(error.localizedDescription)"); return false }
    }

    /// Queue a run; the worker announces its songs with a "started" event.
    func generate(style: String, lyrics: String, cot: String, seed: Int, randomSeed: Bool, batch: Int, maxTokens: Int, engine: String, abc: String, quality: String, instrumental: Bool) {
        send(["cmd": "generate", "style": style, "lyrics": lyrics, "cot": cot, "seed": seed, "random_seed": randomSeed,
              "batch": batch, "max_tokens": maxTokens, "engine": engine, "abc": abc, "quality": quality, "instrumental": instrumental])
    }

    func generate(title: String, style: String, lyrics: String, cot: String, seed: Int, randomSeed: Bool, batch: Int,
                  maxTokens: Int, quality: String, instrumental: Bool, abc: String, kind: String, sourcePath: String?,
                  promptFidelity: Double, styleFidelity: Double, sourceFidelity: Double, targetSeconds: Double?) {
        var request: [String: Any] = ["cmd": "generate", "title": title, "style": style, "lyrics": lyrics,
                                       "cot": cot, "seed": seed, "random_seed": randomSeed, "batch": batch,
                                       "max_tokens": maxTokens, "quality": quality, "instrumental": instrumental,
                                       "abc": abc, "kind": kind, "prompt_fidelity": promptFidelity,
                                       "style_fidelity": styleFidelity, "source_fidelity": sourceFidelity]
        if let sourcePath { request["source_path"] = sourcePath }
        if let targetSeconds { request["target_seconds"] = targetSeconds }
        send(request)
    }

    func importAudio() -> AudioSource? {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio, .mpeg4Audio, .mp3, .wav, .aiff]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.prompt = "Import Audio"
        guard panel.runModal() == .OK, let selected = panel.url else { return nil }
        do {
            try FileManager.default.createDirectory(at: Paths.imports, withIntermediateDirectories: true)
            let ext = selected.pathExtension.isEmpty ? "audio" : selected.pathExtension
            let destination = Paths.imports.appendingPathComponent(UUID().uuidString).appendingPathExtension(ext)
            let file = try AVAudioFile(forReading: selected)
            guard file.length > 0, file.processingFormat.sampleRate > 0 else { throw CocoaError(.fileReadCorruptFile) }
            let seconds = Double(file.length) / file.processingFormat.sampleRate
            try FileManager.default.copyItem(at: selected, to: destination)
            let source = AudioSource(id: destination.path, path: destination.path, title: selected.deletingPathExtension().lastPathComponent,
                                     seconds: seconds)
            let metadata: [String: Any] = ["title": source.title, "kind": "UPLOAD", "style": "", "score": ""]
            let metadataURL = destination.deletingPathExtension().appendingPathExtension("json")
            try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted]).write(to: metadataURL)
            audioSources.append(source)
            rescan()
            append("Imported audio: \(selected.lastPathComponent)")
            return source
        } catch {
            report("Audio import failed: \(error.localizedDescription)")
            return nil
        }
    }

    func source(for song: Song) -> AudioSource {
        AudioSource(id: song.path, path: song.path, title: song.title.isEmpty ? "Song \(song.index)" : song.title,
                    seconds: song.seconds, score: song.score, lyrics: song.lyrics, style: song.style, kind: song.kind)
    }

    func delete(_ song: Song) {
        guard !song.inFlight, !coverBusy, song.path.hasPrefix(Paths.output.path + "/") else { return }
        do {
            let target = song.kind == "UPLOAD" ? URL(fileURLWithPath: song.path) : song.directory
            try FileManager.default.trashItem(at: target, resultingItemURL: nil)
            if song.kind == "UPLOAD" { try? FileManager.default.trashItem(at: target.deletingPathExtension().appendingPathExtension("json"), resultingItemURL: nil) }
            songs.removeAll { $0.id == song.id }
            append("Moved \(song.title.isEmpty ? "song \(song.index)" : song.title) to Trash")
        } catch { report("Could not move song to Trash: \(error.localizedDescription)") }
    }

    func rename(_ song: Song) {
        guard !song.inFlight else { return }
        let alert = NSAlert(); alert.messageText = "Rename song"; alert.informativeText = "Choose a title for this local song."
        let field = NSTextField(string: song.title.isEmpty ? "Song \(song.index)" : song.title)
        field.frame = NSRect(x: 0, y: 0, width: 320, height: 24); alert.accessoryView = field
        alert.addButton(withTitle: "Rename"); alert.addButton(withTitle: "Cancel")
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        let title = field.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else { return }
        do {
            let metadataURL = song.kind == "UPLOAD" ? URL(fileURLWithPath: song.path).deletingPathExtension().appendingPathExtension("json") : song.directory.appendingPathComponent("metadata.json")
            var metadata = (try? JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL)) as? [String: Any]) ?? [:]
            metadata["title"] = title
            try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted]).write(to: metadataURL)
            if let i = songs.firstIndex(where: { $0.id == song.id }) { songs[i].title = title }
        } catch { report("Could not rename song: \(error.localizedDescription)") }
    }

    func export(_ song: Song) {
        let original = URL(fileURLWithPath: song.path)
        let panel = NSSavePanel(); panel.nameFieldStringValue = (song.title.isEmpty ? "YuE Song" : song.title) + "." + original.pathExtension
        panel.allowedContentTypes = [UTType(filenameExtension: original.pathExtension) ?? .audio]
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        do {
            guard original.standardizedFileURL != destination.standardizedFileURL else { return }
            try Data(contentsOf: original).write(to: destination, options: .atomic)
            append("Exported audio to \(destination.lastPathComponent)")
        } catch { report("Could not export audio: \(error.localizedDescription)") }
    }

    /// Synthesize a song from its saved tokens: a full-quality render of a draft, or a stalled song.
    func render(_ song: Song, engine: String, quality: String) {
        guard song.canRender, let i = songs.firstIndex(where: { $0.id == song.id }), !songs[i].inFlight else { return }
        guard send(["cmd": "render", "path": song.path, "engine": engine, "quality": quality]) else { return }
        songs[i].status = .queued; songs[i].detail = "queued"; songs[i].fraction = nil
        updateBusy()
    }

    func cancel(_ song: Song) { send(["cmd": "cancel", "path": song.path]) }
    func stop() { cancelCover(); if process != nil { send(["cmd": "stop"]); append("Stop sent") } }
    func quit() { shuttingDown = true; cancelCover(); if process != nil { send(["cmd": "quit"]); process?.terminate() } }

    /// Remove reusable evidence only. The source audio, generated songs and
    /// imported metadata remain in place; the next cover analysis recomputes
    /// every stage from the melody pass.
    func clearCoverAnalysisCache() {
        guard !busy, !coverBusy else { report("Wait until generation and audio analysis have finished."); return }
        let cache = Paths.coverAnalyses
        guard FileManager.default.fileExists(atPath: cache.path) else {
            append("No previous audio analyses to clear")
            return
        }
        do {
            try FileManager.default.trashItem(at: cache, resultingItemURL: nil)
            append("Previous audio analysis data moved to Trash")
        } catch {
            report("Could not clear previous audio analyses: \(error.localizedDescription)")
        }
    }
}

// MARK: - Audio

// MARK: - Views

/// The song's journey: Planning | Tokenizing | Synthing | Rendering, from queued (empty) to done (full).
/// Fills whatever width it is given.
struct StageTrack: View {
    let progress: Double                         // 0 (queued) ... 4 (done): whole stages plus progress within the current one
    static let segments = ["Planning", "Tokenizing", "Synthing", "Rendering"]
    static let total = Double(segments.count)
    var body: some View {
        GeometryReader { g in
            let w = max(1, g.size.width - 8)
            let x = { (units: Double) in w * units / Self.total }
            let filled = x(max(0, min(Self.total, progress)))
            ZStack(alignment: .topLeading) {
                ForEach(Self.segments.indices, id: \.self) { i in
                    Text(Self.segments[i]).font(.caption2).foregroundStyle(.secondary).lineLimit(1).minimumScaleFactor(0.7)
                        .frame(width: x(1)).position(x: x(Double(i) + 0.5), y: 7)
                }
                Capsule().fill(Color.secondary.opacity(0.18)).frame(width: w, height: 8).position(x: w / 2, y: 24)
                Capsule().fill(Color.green).frame(width: filled, height: 8).position(x: filled / 2, y: 24)
                ForEach(0...Self.segments.count, id: \.self) { i in
                    Rectangle().fill(Color.primary.opacity(0.7)).frame(width: 2, height: 18).position(x: x(Double(i)), y: 24)
                }
            }
        }
        .frame(maxWidth: .infinity).frame(height: 36)
    }
}

struct ContentView: View {
    @EnvironmentObject var backend: Backend
    @StateObject private var players = Players()
    @AppStorage("style") private var style = "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums"
    @AppStorage("lyrics") private var lyrics = ""
    @AppStorage("cot") private var cot = "full"
    @AppStorage("seed") private var seed = 831001
    @AppStorage("randomSeed") private var randomSeed = false
    @AppStorage("batch") private var batch = 2
    @AppStorage("maxSeconds") private var maxSeconds = 120.0
    @AppStorage("quality") private var quality = "draft"
    @AppStorage("instrumental") private var instrumental = false
    @AppStorage("promptFidelity") private var promptFidelity = 0.75
    @AppStorage("styleFidelity") private var styleFidelity = 0.75
    @AppStorage("sourceFidelity") private var sourceFidelity = 1.0
    @State private var title = ""
    @State private var abc = ""
    @State private var mode = "new"
    @State private var selectedTab = "Create"
    @State private var source: AudioSource?
    @State private var search = ""
    @State private var filter = "All"
    @State private var sortNewest = true
    @State private var confirmClearAnalysis = false
    @Environment(\.openSettings) private var openSettings
    @State private var showOptions = false
    @State private var lyricsExpanded = true
    @State private var stylesExpanded = true
    @State private var showScore: Song?
    @State private var transcribing = false
    @State private var transcriptionError = ""
    @State private var suggestedGenre = ""
    @State private var analysisWarnings: [String] = []
    @State private var installingCoverRuntime = false
    @State private var collapsedSidebar = false
    @State private var showLyricsEditor = false
    @State private var showLogs = false
    @State private var undoLyrics: [String] = []
    @State private var redoLyrics: [String] = []
    @State private var restoringLyrics = false

    private let dark = Color(red: 0.055, green: 0.055, blue: 0.065)
    private let card = Color(red: 0.105, green: 0.105, blue: 0.12)

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                sidebar.frame(width: collapsedSidebar ? 74 : 205)
                Divider().overlay(.white.opacity(0.08))
                VStack(spacing: 0) {
                    topBar
                    Divider().overlay(.white.opacity(0.08))
                    HStack(spacing: 0) {
                        if selectedTab == "Create" {
                            createPane.frame(minWidth: 430, idealWidth: 500)
                            Divider().overlay(.white.opacity(0.08))
                        }
                        libraryPane.frame(minWidth: 430, idealWidth: 560)
                    }
                }
            }
            playerBar
        }
        .background(dark)
        .preferredColorScheme(.dark)
        .frame(minWidth: 1260, minHeight: 760)
        .onAppear { backend.rescan(); if backend.process == nil { backend.start() } }
        .onChange(of: lyrics) { old, _ in
            if restoringLyrics { restoringLyrics = false }
            else { undoLyrics.append(old); if undoLyrics.count > 100 { undoLyrics.removeFirst() }; redoLyrics.removeAll() }
        }
        .onChange(of: players.error) { _, message in if let message { backend.report(message) } }
        .onChange(of: backend.songs) { _, songs in
            if let path = source?.path, let updated = songs.first(where: { $0.path == path }) { source?.title = updated.title }
            if let playing = players.currentSong, let updated = songs.first(where: { $0.id == playing.id }) { players.updateMetadata(updated) }
        }
        .alert("YuE Studio", isPresented: Binding(get: { backend.errorMessage != nil }, set: { if !$0 { backend.errorMessage = nil } })) {
            Button("OK") { backend.errorMessage = nil }
        } message: { Text(backend.errorMessage ?? "") }
        .alert("Clear previous audio analyses?", isPresented: $confirmClearAnalysis) {
            Button("Cancel", role: .cancel) { }
            Button("Move to Trash", role: .destructive) { backend.clearCoverAnalysisCache() }
        } message: {
            Text("Reusable melody, lyric, genre and style evidence will be removed. The next analysis will start from zero; songs and source audio stay in the library.")
        }
        .sheet(isPresented: $showLyricsEditor) {
            VStack(alignment: .leading, spacing: 14) {
                Text("Lyrics").font(.title2).bold()
                TextEditor(text: $lyrics).font(.system(size: 17))
                HStack { lyricsTools; Spacer(); Button("Done") { showLyricsEditor = false }.keyboardShortcut(.cancelAction) }
            }.padding(24).frame(width: 720, height: 580)
        }
        .sheet(isPresented: $showLogs) {
            VStack {
                HStack { Text("Process Log").font(.title2); Spacer(); Button("Clear") { backend.log.removeAll() }; Button("Close") { showLogs = false }.keyboardShortcut(.cancelAction) }
                LogTextView(lines: backend.log)
            }.padding(20).frame(width: 840, height: 560)
        }
        .sheet(item: $showScore) { song in
            VStack(alignment: .leading, spacing: 12) {
                Text(song.title.isEmpty ? "Score" : song.title).font(.title3).bold()
                ScrollView { Text(song.score).font(.system(.body, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
                HStack { Spacer(); Button("Close") { showScore = nil }.keyboardShortcut(.cancelAction) }
            }.padding().frame(width: 720, height: 520)
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack { if !collapsedSidebar { Text("YuE Studio").font(.system(size: 25, weight: .bold)); Spacer() }; Button { collapsedSidebar.toggle() } label: { Image(systemName: "sidebar.left") }.buttonStyle(.plain).help("Toggle sidebar") }
                .padding(.horizontal, 22).padding(.top, 22).padding(.bottom, 26)
            sideButton("music.note", "Create", "Create")
            sideButton("waveform", "Library", "Library")
            Spacer()
            Button { openSettings() } label: {
                Label(collapsedSidebar ? "" : "Settings", systemImage: "gearshape")
                    .frame(maxWidth: .infinity, alignment: .leading).padding(12).contentShape(Rectangle())
            }.buttonStyle(.plain).padding(.horizontal, 10).help("Settings").accessibilityLabel("Settings")
            Button { showLogs = true } label: { Label(collapsedSidebar ? "" : "Process Log", systemImage: "terminal") }.buttonStyle(.plain).padding(.horizontal, 22).help("Process Log").accessibilityLabel("Process Log")
            if !backend.connected && !backend.coverBusy {
                Button("Reconnect") { backend.start() }.padding(.horizontal, 12)
            }
            if backend.busy || backend.coverBusy {
                Button("Stop All", role: .destructive) { backend.stop() }.padding(.horizontal, 12)
            }
            Text(collapsedSidebar ? (backend.connected ? "Ready" : "Offline") : "Local generation · \(backend.connected ? "Ready" : "Offline")")
                .font(.caption).foregroundStyle(backend.connected ? .green : .orange).padding(.horizontal, 22).padding(.bottom, 16)
        }.background(dark)
    }

    private func sideButton(_ icon: String, _ label: String, _ value: String) -> some View {
        Button { selectedTab = value } label: {
            Label(collapsedSidebar ? "" : label, systemImage: icon).font(.system(size: 15, weight: selectedTab == value ? .semibold : .regular))
                .foregroundStyle(selectedTab == value ? .white : .secondary)
                .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 22).padding(.vertical, 11)
                .background(selectedTab == value ? Color.white.opacity(0.13) : .clear, in: Capsule()).contentShape(Capsule())
        }.buttonStyle(.plain).padding(.horizontal, 10)
        .help(label)
    }

    private var topBar: some View {
        HStack(spacing: 18) {
            Picker("", selection: $mode) { Text("New song").tag("new"); Text("Cover").tag("cover") }
                .pickerStyle(.segmented).frame(width: 190)
                .disabled(backend.coverBusy)
            Picker("Quality", selection: $quality) { Text("Draft").tag("draft"); Text("Full").tag("full") }
                .pickerStyle(.segmented).frame(width: 135)
            Spacer()
            Label("YuE2", systemImage: "waveform").foregroundStyle(.secondary)
            Button { openOutput() } label: { Image(systemName: "folder") }
                .buttonStyle(.plain).help("Open songs folder")
        }.padding(.horizontal, 24).padding(.vertical, 15)
    }

    private var createPane: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    HStack(spacing: 0) {
                        createSourceButton("Audio", systemImage: "plus") { importSource() }.disabled(backend.coverBusy)
                    }.background(card, in: Capsule())
                    if let source { sourceCard(source) }
                    editorCard(title: "Lyrics", expanded: $lyricsExpanded) {
                        TextEditor(text: $lyrics).scrollContentBackground(.hidden).font(.system(size: 15)).frame(minHeight: 210)
                        HStack { Text("\(lyrics.count) characters").font(.caption).foregroundStyle(.secondary); Spacer(); Toggle("Instrumental", isOn: $instrumental).toggleStyle(.switch).controlSize(.small) }
                    }
                    editorCard(title: "Styles", expanded: $stylesExpanded) {
                        TextEditor(text: $style).scrollContentBackground(.hidden).font(.system(size: 15)).frame(minHeight: 100)
                        HStack(spacing: 8) {
                            styleChip("pop"); styleChip("acoustic"); styleChip("cinematic"); styleChip("electronic")
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }
                    editorCard(title: "More Options", expanded: $showOptions) {
                        VStack(alignment: .leading, spacing: 12) {
                            if mode == "cover" { Text("Planning: preserve source melody").font(.caption).foregroundStyle(.secondary) }
                            else { Picker("Planning", selection: $cot) { Text("Full score").tag("full"); Text("Melody").tag("melody"); Text("Direct").tag("off") }.pickerStyle(.segmented) }
                            Stepper("Variants: \(batch)", value: $batch, in: 1...8)
                            if mode == "cover", let source {
                                HStack {
                                    Text("Cover length"); Spacer()
                                    Text(timeLabel(min(source.seconds, 360))).monospacedDigit()
                                    Text(source.seconds <= 360 ? "matches source" : "YuE max · source \(timeLabel(source.seconds))").foregroundStyle(.secondary)
                                }
                            } else {
                                VStack(alignment: .leading, spacing: 6) { HStack { Text("Max length"); Spacer(); Text("\(Int(maxSeconds))s").monospacedDigit() }; Slider(value: $maxSeconds, in: 30...360, step: 10).accessibilityLabel("Max length").accessibilityValue("\(Int(maxSeconds)) seconds") }
                            }
                            fidelitySlider("Prompt fidelity", value: $promptFidelity)
                            fidelitySlider("Style fidelity", value: $styleFidelity)
                            if mode == "cover" { fidelitySlider("Source audio fidelity", value: $sourceFidelity) }
                            HStack { TextField("Seed", value: $seed, format: .number).disabled(randomSeed); Toggle("Random", isOn: $randomSeed) }
                            HStack { Text("ABC score").font(.caption); Spacer(); Button("Import ABC") { importScore() }.disabled(backend.coverBusy) }
                            TextEditor(text: $abc).font(.system(.caption, design: .monospaced)).frame(height: 90).overlay(RoundedRectangle(cornerRadius: 8).stroke(.white.opacity(0.1)))
                        }.padding(.top, 10)
                    }.font(.system(size: 13)).tint(.accentColor)
                    VStack(alignment: .leading, spacing: 10) {
                        Label("Song title", systemImage: "music.note").foregroundStyle(.secondary)
                        TextField("Optional title", text: $title).textFieldStyle(.plain).font(.system(size: 16)).padding(12).background(card, in: RoundedRectangle(cornerRadius: 12))
                        Button { openOutput() } label: { Label(Paths.output.path, systemImage: "folder").font(.caption).lineLimit(1).truncationMode(.middle) }.buttonStyle(.plain).foregroundStyle(.secondary)
                    }
                }.padding(24)
            }
            Divider().overlay(.white.opacity(0.08))
            if let reason = createUnavailableReason {
                Text(reason).font(.caption).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 18).padding(.top, 8)
            }
            HStack(spacing: 12) {
                Button { resetForm() } label: { Image(systemName: "trash").frame(width: 46, height: 42) }.buttonStyle(.bordered).tint(.secondary).help("Reset form").disabled(backend.coverBusy)
                Button { create() } label: { Label(mode == "cover" ? "Create Cover" : "Create", systemImage: mode == "cover" ? "arrow.triangle.2.circlepath" : "music.note")
                        .frame(maxWidth: .infinity).frame(height: 42) }
                    .buttonStyle(.borderedProminent).tint(.blue).disabled(!canCreate)
            }.padding(18)
        }
    }

    private func createSourceButton(_ title: String, systemImage: String, action: @escaping () -> Void) -> some View {
        Button(action: action) { Label(title, systemImage: systemImage).frame(maxWidth: .infinity).padding(.vertical, 16).background(card, in: Capsule()).contentShape(Capsule()) }.buttonStyle(.plain)
    }

    private func editorCard<Content: View>(title: String, expanded: Binding<Bool>, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Button { expanded.wrappedValue.toggle() } label: { HStack { Image(systemName: expanded.wrappedValue ? "chevron.down" : "chevron.right"); Text(title).font(.system(size: 17, weight: .semibold)); Spacer(minLength: 0) }.padding(.vertical, 6).contentShape(Rectangle()) }.buttonStyle(.plain).accessibilityValue(expanded.wrappedValue ? "Expanded" : "Collapsed")
                if title == "More Options" {
                    Button { resetAdvancedOptions() } label: {
                        Image(systemName: "arrow.counterclockwise").frame(width: 28, height: 28).contentShape(Rectangle())
                    }.buttonStyle(.plain).help("Reset advanced options").accessibilityLabel("Reset advanced options").disabled(backend.coverBusy)
                }
                if title == "Styles" {
                    Button { style = "" } label: {
                        Image(systemName: "trash").frame(width: 28, height: 28).contentShape(Rectangle())
                    }.buttonStyle(.plain).help("Clear styles").accessibilityLabel("Clear styles").disabled(style.isEmpty || backend.coverBusy)
                }
                if title == "Lyrics" {
                    lyricsTools
                    Button { showLyricsEditor = true } label: { Image(systemName: "arrow.up.left.and.arrow.down.right") }.buttonStyle(.plain).help("Expand lyrics editor")
                }
            }
            if expanded.wrappedValue { content() }
        }.padding(18).background(card, in: RoundedRectangle(cornerRadius: 18))
    }

    private func styleChip(_ value: String) -> some View {
        Button(value) {
            var tags = style.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            if let i = tags.firstIndex(where: { $0.caseInsensitiveCompare(value) == .orderedSame }) { tags.remove(at: i) } else { tags.append(value) }
            style = tags.joined(separator: ", ")
        }
            .buttonStyle(.bordered).tint(.secondary).controlSize(.small)
    }

    private func sourceCard(_ source: AudioSource) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button { if let song = backend.songs.first(where: { $0.path == source.path }) { players.toggle(song) } } label: {
                    ZStack { WaveformView(path: source.path).frame(width: 70, height: 46); Image(systemName: players.currentSong?.path == source.path && players.isPlaying ? "pause.fill" : "play.fill") }
                }.buttonStyle(.plain).help("Preview source")
                VStack(alignment: .leading) { Text(source.title).bold(); Text(String(format: "%.1f seconds · %@", source.seconds, source.kind)).font(.caption).foregroundStyle(.secondary) }
                Spacer()
                Button("Replace") { importSource() }.disabled(backend.coverBusy)
                Button { self.source = nil; abc = ""; mode = "new" } label: { Image(systemName: "xmark") }.buttonStyle(.plain).help("Remove source").disabled(backend.coverBusy)
            }
            if mode == "cover" {
                HStack {
                    if backend.coverBusy { ProgressView().controlSize(.small); Text(backend.coverStatus).font(.caption); Button("Cancel") { backend.cancelCover() } }
                    else if !backend.coverRuntimeReady {
                        if installingCoverRuntime { ProgressView().controlSize(.small); Text("Installing cover engine…") }
                        else { Button("Install Cover Engine") { installCoverRuntime() }.buttonStyle(.borderedProminent).controlSize(.small) }
                    } else { Button("Analyze audio") { transcribe(source) }.buttonStyle(.borderedProminent).controlSize(.small) }
                    if !transcriptionError.isEmpty { Text(transcriptionError).foregroundStyle(.red).font(.caption).lineLimit(2) }
                }
            }
            ForEach(analysisWarnings, id: \.self) { Text(friendlyAnalysisWarning($0)).font(.caption).foregroundStyle(.secondary) }
            if !suggestedGenre.isEmpty {
                HStack { Label("Suggested style: \(suggestedGenre)", systemImage: "sparkles").font(.caption).foregroundStyle(.secondary); Spacer(); Button("Use") { style = suggestedGenre }.controlSize(.small) }
            }
        }.padding(14).background(card, in: RoundedRectangle(cornerRadius: 16))
    }

    private func friendlyAnalysisWarning(_ warning: String) -> String {
        if warning.localizedCaseInsensitiveContains("Word alignment") {
            return "Lyric timing is approximate; the recognized text was kept."
        }
        if warning.localizedCaseInsensitiveContains("sections are inferred") {
            return "Section labels are suggestions based on repetition and position; review them before generating."
        }
        if warning.localizedCaseInsensitiveContains("digital silence") {
            return "Very quiet timestamps were omitted; the raw transcription remains available in the import metadata."
        }
        if warning.localizedCaseInsensitiveContains("style model unavailable") {
            return "Detailed style analysis is unavailable until the cover engine is repaired."
        }
        return warning
    }

    private var libraryPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack { Text("My Library").font(.title2).bold(); Spacer(); Button { backend.rescan() } label: { Image(systemName: "arrow.clockwise") }.buttonStyle(.plain) }.padding(.horizontal, 24).padding(.top, 24)
            HStack(spacing: 10) {
                TextField("Search", text: $search).textFieldStyle(.roundedBorder)
                Menu { Picker("Filter", selection: $filter) { Text("All").tag("All"); Text("Covers").tag("COVER"); Text("Uploads").tag("UPLOAD"); Text("Drafts").tag("DRAFT") } } label: { Label(filter, systemImage: "line.3.horizontal.decrease.circle") }.buttonStyle(.bordered)
                Menu { Button("Newest") { sortNewest = true }; Button("Title") { sortNewest = false } } label: { Label(sortNewest ? "New" : "Title", systemImage: "arrow.up.arrow.down") }.buttonStyle(.bordered)
            }.padding(24)
            ScrollView {
                LazyVStack(spacing: 2) { ForEach(filteredSongs) { song in libraryRow(song) } }
            }.overlay { if filteredSongs.isEmpty { Text(backend.songs.isEmpty ? "Your songs will appear here" : "No songs match this search or filter").foregroundStyle(.secondary) } }
            HStack(spacing: 10) {
                Image(systemName: "arrow.counterclockwise")
                    .foregroundStyle(.secondary)
                Text("Need a clean re-analysis? Clear the previous audio-analysis data below.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                Spacer()
                Button("Clear analysis data…") { confirmClearAnalysis = true }
                    .buttonStyle(.bordered)
                    .disabled(backend.busy || backend.coverBusy)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(card.opacity(0.75))
        }
    }

    private var filteredSongs: [Song] {
        let songs = backend.songs.filter { song in
            let matchesSearch = search.isEmpty || song.title.localizedCaseInsensitiveContains(search) || song.style.localizedCaseInsensitiveContains(search)
            let matchesFilter = filter == "All" || (filter == "DRAFT" ? song.quality == "draft" : song.kind == filter)
            return matchesSearch && matchesFilter
        }
        return songs.sorted { sortNewest ? $0.createdAt > $1.createdAt : $0.title.localizedCaseInsensitiveCompare($1.title) == .orderedAscending }
    }

    private func libraryRow(_ song: Song) -> some View {
        HStack(spacing: 12) {
            Button { players.toggle(song) } label: { ZStack { WaveformView(path: song.path).frame(width: 72, height: 72); Image(systemName: players.currentSong?.id == song.id && players.isPlaying ? "pause.fill" : "play.fill").padding(10).background(.black.opacity(0.55), in: Circle()) } }.buttonStyle(.plain).disabled(song.status != .ready)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 7) { Text(song.title.isEmpty ? "Song \(song.index)" : song.title).font(.system(size: 15, weight: .semibold)).lineLimit(1); Text(song.kind).font(.caption2).foregroundStyle(song.kind == "COVER" ? .pink : .secondary) }
                Text(song.style.isEmpty ? "YuE2 local generation" : song.style).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                HStack { Text(song.status == .ready ? String(format: "%.1f s", song.seconds) : song.statusLine).font(.caption2).foregroundStyle(.secondary); if song.quality == "draft" { Text("DRAFT").font(.caption2).foregroundStyle(.orange) } }
                if song.inFlight { StageTrack(progress: song.trackProgress).frame(height: 30) }
            }
            Spacer()
            Menu {
                songActions(song)
            } label: { Image(systemName: "ellipsis").frame(width: 36, height: 36).background(card, in: Circle()) }.menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize().help("Song actions")
        }.padding(.horizontal, 24).padding(.vertical, 10)
            .contentShape(Rectangle())
            .contextMenu { songActions(song) }
    }

    @ViewBuilder private func songActions(_ song: Song) -> some View {
                Button("Cover", systemImage: "arrow.triangle.2.circlepath") { beginCover(song) }
                    .disabled(backend.coverBusy || song.inFlight || (song.status != .ready && song.score.isEmpty))
                Button("Reuse Prompt & Lyrics", systemImage: "doc.on.doc") { reusePrompt(song) }
                    .disabled(backend.coverBusy || song.inFlight || (song.style.isEmpty && song.lyrics.isEmpty))
                if song.canRender && (song.quality == "draft" || song.status != .ready) { Button(song.status == .ready ? "Render full quality" : "Recover synthesis") { players.forget(song); backend.render(song, engine: "auto", quality: "full") }.disabled(!backend.connected || backend.coverBusy) }
                if !song.score.isEmpty { Button("View Score") { showScore = song } }
                if song.inFlight { Button("Cancel") { backend.cancel(song) } }
                if song.status == .ready { Button("Export Audio") { backend.export(song) }; Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: song.path)]) } }
                Button("Rename") { backend.rename(song) }.disabled(song.inFlight)
                Divider(); Button("Move to Trash", role: .destructive) { players.forget(song); backend.delete(song); if source?.path == song.path { source = nil; abc = "" } }.disabled(song.inFlight || backend.coverBusy)
    }

    private var playerBar: some View {
        HStack(spacing: 22) {
            if let song = players.currentSong { HStack { WaveformView(path: song.path).frame(width: 42, height: 34); VStack(alignment: .leading) { Text(song.title.isEmpty ? "Song \(song.index)" : song.title).font(.caption).bold(); Text("YuE2 local player").font(.caption2).foregroundStyle(.secondary) } }.frame(width: 250, alignment: .leading) }
            else { Text("Nothing playing").font(.caption).foregroundStyle(.secondary).frame(width: 250, alignment: .leading) }
            Spacer()
            Button { if let song = players.currentSong { players.toggle(song) } } label: { Image(systemName: players.isPlaying ? "pause.fill" : "play.fill").font(.title3) }.buttonStyle(.plain).disabled(players.currentSong == nil).help("Play / Pause")
            Text(timeLabel(players.position)).font(.caption).monospacedDigit()
            Slider(value: Binding(get: { min(players.position, max(1, players.duration)) }, set: { players.seek($0) }), in: 0...max(1, players.duration)).disabled(players.duration <= 0).help("Playback position")
            Text(timeLabel(players.duration)).font(.caption).monospacedDigit()
            Image(systemName: "speaker.wave.2.fill").foregroundStyle(.secondary)
            Slider(value: $players.volume, in: 0...1).frame(width: 95).help("Volume")
        }.padding(.horizontal, 24).frame(height: 58).background(Color(red: 0.09, green: 0.09, blue: 0.1))
    }

    private var canCreate: Bool { backend.connected && !backend.coverBusy && (mode == "cover" || !style.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) && (instrumental || !lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) && (mode == "new" || !abc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) }

    private var createUnavailableReason: String? {
        if backend.coverBusy { return backend.coverStatus }
        if !backend.connected { return "Waiting for the local engine. Open Process Log for details." }
        if mode == "new" && style.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Enter a style to create a song." }
        if !instrumental && lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Enter lyrics or enable Instrumental." }
        if mode == "cover" && abc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Transcribe the source melody or import an ABC score in More Options." }
        return nil
    }

    private func create() {
        let finalTitle = title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? (mode == "cover" ? "Cover" : "New Song") : title
        if mode == "cover" {
            let seconds = min(source?.seconds ?? maxSeconds, 360)
            backend.generate(title: finalTitle, style: style, lyrics: lyrics, cot: "melody", seed: seed, randomSeed: randomSeed, batch: batch, maxTokens: Int(seconds * 25), quality: quality, instrumental: instrumental, abc: abc, kind: "COVER", sourcePath: source?.path, promptFidelity: promptFidelity, styleFidelity: styleFidelity, sourceFidelity: sourceFidelity, targetSeconds: seconds)
        } else {
            backend.generate(title: finalTitle, style: style, lyrics: lyrics, cot: cot, seed: seed, randomSeed: randomSeed, batch: batch, maxTokens: Int(maxSeconds * 25), quality: quality, instrumental: instrumental, abc: abc, kind: "GENERATED", sourcePath: nil, promptFidelity: promptFidelity, styleFidelity: styleFidelity, sourceFidelity: 0, targetSeconds: nil)
        }
    }

    private func importSource() {
        if let imported = backend.importAudio() {
            source = imported; abc = imported.score; mode = "cover"; selectedTab = "Create"; transcriptionError = ""; suggestedGenre = ""; analysisWarnings = []
            maxSeconds = min(imported.seconds, 360)
            if backend.coverRuntimeReady { transcribe(imported) }
        }
    }

    private func transcribe(_ source: AudioSource) {
        transcribing = true; transcriptionError = ""; analysisWarnings = []
        backend.transcribe(source) { result in
            transcribing = false
            guard self.source?.id == source.id else { return }
            switch result {
            case .success(let analysis):
                self.source?.score = analysis.score; abc = analysis.score
                if !analysis.lyrics.isEmpty && lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    lyrics = analysis.lyrics; self.source?.lyrics = analysis.lyrics
                }
                analysisWarnings = analysis.warnings
                suggestedGenre = analysis.genre
                if style.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !analysis.genre.isEmpty { style = analysis.genre }
            case .failure(let error): transcriptionError = error.localizedDescription
            }
        }
    }

    private func installCoverRuntime() {
        installingCoverRuntime = true; transcriptionError = ""
        backend.installCoverRuntime { result in
            installingCoverRuntime = false
            if case .failure(let error) = result { transcriptionError = error.localizedDescription }
            else if let source { transcribe(source) }
        }
    }

    private func beginCover(_ song: Song) { source = backend.source(for: song); abc = song.score; mode = "cover"; selectedTab = "Create"; style = song.style; lyrics = song.lyrics; maxSeconds = min(song.seconds, 360); suggestedGenre = ""; analysisWarnings = []; transcriptionError = ""; title = (song.title.isEmpty ? "Song \(song.index)" : song.title) + " Cover" }
    private func reusePrompt(_ song: Song) {
        mode = "new"; selectedTab = "Create"; source = nil; abc = ""
        style = song.style; lyrics = song.lyrics
        instrumental = song.lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        title = ""; transcriptionError = ""; suggestedGenre = ""; analysisWarnings = []
        lyricsExpanded = true; stylesExpanded = true
    }

    private func resetAdvancedOptions() {
        cot = "full"; seed = 831001; randomSeed = false; batch = 2
        maxSeconds = mode == "cover" ? min(source?.seconds ?? 120, 360) : 120
        promptFidelity = 0.75; styleFidelity = 0.75; sourceFidelity = 1.0
        abc = mode == "cover" ? (source?.score ?? "") : ""
    }
    private func resetForm() { title = ""; style = ""; lyrics = ""; abc = ""; source = nil; mode = "new"; suggestedGenre = ""; transcriptionError = "" }

    private func fidelitySlider(_ label: String, value: Binding<Double>) -> some View {
        VStack(alignment: .leading, spacing: 6) { HStack { Text(label); Spacer(); Text("\(Int(value.wrappedValue * 100))%").monospacedDigit() }; Slider(value: value, in: 0...1, step: 0.05).accessibilityLabel(label).accessibilityValue("\(Int(value.wrappedValue * 100)) percent") }
    }

    private var lyricsTools: some View {
        HStack(spacing: 12) {
            Button { lyrics = "" } label: {
                Image(systemName: "trash").frame(width: 28, height: 28).contentShape(Rectangle())
            }.disabled(lyrics.isEmpty || backend.coverBusy).help("Clear lyrics").accessibilityLabel("Clear lyrics")
            Button { guard let previous = undoLyrics.popLast() else { return }; redoLyrics.append(lyrics); restoringLyrics = true; lyrics = previous } label: { Image(systemName: "arrow.uturn.backward") }.disabled(undoLyrics.isEmpty).help("Undo lyrics change")
            Button { guard let next = redoLyrics.popLast() else { return }; undoLyrics.append(lyrics); restoringLyrics = true; lyrics = next } label: { Image(systemName: "arrow.uturn.forward") }.disabled(redoLyrics.isEmpty).help("Redo lyrics change")
        }.buttonStyle(.plain)
    }
    private func timeLabel(_ seconds: Double) -> String { String(format: "%d:%02d", Int(max(0, seconds)) / 60, Int(max(0, seconds)) % 60) }
    private func openOutput() { do { try FileManager.default.createDirectory(at: Paths.output, withIntermediateDirectories: true); NSWorkspace.shared.open(Paths.output) } catch { backend.report(error.localizedDescription) } }
    private func importScore() {
        let panel = NSOpenPanel(); panel.allowsMultipleSelection = false; panel.canChooseDirectories = false
        panel.allowedContentTypes = [UTType(filenameExtension: "abc") ?? .plainText, .plainText]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { abc = try String(contentsOf: url, encoding: .utf8) } catch { backend.report("Could not read score: \(error.localizedDescription)") }
    }
}

/// Native, selectable, read-only text view: drag to select across lines, Cmd-C to copy, Cmd-F to find.
struct LogTextView: NSViewRepresentable {
    let lines: [LogLine]

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView()
        let text = NSTextView()
        text.isEditable = false; text.isSelectable = true; text.isRichText = false
        text.usesFindBar = true; text.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        text.textContainerInset = NSSize(width: 6, height: 6)
        text.autoresizingMask = [.width]; text.isVerticallyResizable = true; text.isHorizontallyResizable = false
        text.textContainer?.widthTracksTextView = true
        scroll.documentView = text; scroll.hasVerticalScroller = true; scroll.borderType = .noBorder
        context.coordinator.rendered = 0
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let text = scroll.documentView as? NSTextView, let storage = text.textStorage else { return }
        let coordinator = context.coordinator
        if lines.count < coordinator.rendered || (coordinator.rendered > 0 && lines.prefix(coordinator.rendered).last?.id != coordinator.lastID) {
            storage.setAttributedString(NSAttributedString(string: "")); coordinator.rendered = 0
        }
        guard lines.count > coordinator.rendered else { return }
        let attrs: [NSAttributedString.Key: Any] = [.font: text.font!, .foregroundColor: NSColor.textColor]
        let chunk = lines[coordinator.rendered...].map { "\($0.time)  \($0.message)" }.joined(separator: "\n") + "\n"
        let atBottom = scroll.contentView.bounds.maxY >= (text.bounds.height - 40)
        storage.append(NSAttributedString(string: chunk, attributes: attrs))
        coordinator.rendered = lines.count
        coordinator.lastID = lines.last?.id
        if atBottom || text.selectedRange().length == 0 { text.scrollToEndOfDocument(nil) }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator { var rendered = 0; var lastID: UUID? }
}

struct SetupView: View {
    @EnvironmentObject var installer: Installer
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Setting up YuE Studio").font(.title2).bold()
            Text("First run only: this installs a private Python runtime, the YuE music model code, and downloads the model weights. It needs about 10 GB of disk space and an internet connection.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            ForEach(installer.steps) { step in
                HStack {
                    Image(systemName: step.done ? "checkmark.circle.fill" : (installer.current == step.id && installer.state == .running ? "arrow.triangle.2.circlepath" : "circle"))
                        .foregroundStyle(step.done ? .green : .secondary)
                    Text(step.title).bold(installer.current == step.id && installer.state == .running)
                    if installer.current == step.id && installer.state == .running && !installer.detail.isEmpty { Text("· " + installer.detail).foregroundStyle(.secondary) }
                }
            }
            ProgressView(value: installer.progress)
            LogTextView(lines: installer.log).frame(minHeight: 160)
            HStack {
                if case .failed(let message) = installer.state {
                    Text(message).foregroundStyle(.red); Spacer()
                    Button("Retry") { installer.install() }.buttonStyle(.borderedProminent)
                } else if installer.state == .needed {
                    Spacer(); Button("Install") { installer.install() }.buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                } else { Spacer(); ProgressView().controlSize(.small) }
            }
        }
        .padding(24).frame(minWidth: 640, minHeight: 520)
    }
}

struct RootView: View {
    @EnvironmentObject var installer: Installer
    @EnvironmentObject var backend: Backend
    var body: some View {
        Group {
            switch installer.state {
            case .ready: ContentView()
            case .checking: ProgressView("Checking installation").frame(minWidth: 400, minHeight: 200)
            default: SetupView()
            }
        }
        .onAppear { installer.check(); NSApp.setActivationPolicy(.regular); NSApp.activate(ignoringOtherApps: true) }
        .onChange(of: installer.state) { _, new in if new == .ready { backend.start() } }
    }
}

@main
struct YuEStudioApp: App {
    @StateObject private var backend = Backend()
    @StateObject private var installer = Installer()
    var body: some Scene {
        WindowGroup("YuE Studio") { RootView().environmentObject(backend).environmentObject(installer) }
            .commands {
                CommandGroup(replacing: .appTermination) { Button("Quit YuE Studio") { installer.cancel(); backend.quit(); NSApp.terminate(nil) }.keyboardShortcut("q") }
                CommandGroup(after: .appSettings) {
                    Button("Open Songs Folder") { NSWorkspace.shared.open(Paths.output) }
                    Button("Repair Installation…") { backend.quit(); backend.shuttingDown = false; installer.repair() }.disabled(!Paths.packaged || backend.busy || backend.coverBusy)
                }
            }
        Settings {
            StorageSettingsView()
                .environmentObject(backend)
                .environmentObject(installer)
        }
    }
}
