import SwiftUI
import AVFoundation
import AppKit

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
    static var python: URL { packaged ? support.appendingPathComponent("env/bin/python") : repoRoot.appendingPathComponent(".venv/bin/python") }
    static var worker: URL {
        if let o = ProcessInfo.processInfo.environment["YUE_STUDIO_WORKER"] { return URL(fileURLWithPath: o) }   // tests
        return packaged ? support.appendingPathComponent("src/tools/yue2_worker.py") : repoRoot.appendingPathComponent("tools/yue2_worker.py")
    }
    static var src: URL { support.appendingPathComponent("src") }
    static var models: URL { ProcessInfo.processInfo.environment["YUE_STUDIO_HF_HOME"].map { URL(fileURLWithPath: $0) } ?? support.appendingPathComponent("models") }
    static var aneCache: URL { support.appendingPathComponent("ane-cache") }
    static var output: URL {
        packaged ? FileManager.default.urls(for: .musicDirectory, in: .userDomainMask)[0].appendingPathComponent("YuE Studio")
                 : repoRoot.appendingPathComponent("outputs/app")
    }
    static var installedMarker: URL { support.appendingPathComponent("installed.json") }
    static var bundledVersion: String { (try? String(contentsOf: payload!.appendingPathComponent("version.txt"), encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "dev" }
    static var workerEnvironment: [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"; env["TQDM_DISABLE"] = "1"
        env["YUE2_OUTPUT_DIR"] = output.path; env["YUE2_ANE_CACHE"] = aneCache.path
        if packaged { env["HF_HOME"] = models.path; env["HF_HUB_DISABLE_TELEMETRY"] = "1" }
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        return env
    }
}

// MARK: - Installer

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
        try p.run()
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in p.terminationHandler = { _ in c.resume() } }
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
                let request = json("request.json"), result = hasAudio ? json("result.json") : nil, tokens = json("tokens.json")
                let truncated = (result?["truncated"] as? [String: Bool])?.values.contains(true) ?? (tokens?["truncated"] as? Bool ?? false)
                songs.append(Song(run: run.lastPathComponent, index: Int(dir.lastPathComponent.dropFirst(4)) ?? 0, path: audio.path,
                                  score: (try? String(contentsOf: dir.appendingPathComponent("score.abc"), encoding: .utf8)) ?? "",
                                  seconds: result?["audio_seconds"] as? Double ?? ((tokens?["frames"] as? Double ?? 0) / 25),
                                  seed: request?["seed"] as? Int ?? 0, truncated: truncated, status: hasAudio ? .ready : .stalled,
                                  quality: result?["quality"] as? String ?? "full",
                                  detail: hasAudio ? "" : "tokens saved · not synthesized yet"))
            }
        }
        return songs
    }
}

@MainActor
final class Backend: ObservableObject {
    @Published var log: [LogLine] = []
    @Published var songs: [Song] = []
    @Published var busy = false                  // anything queued or in a stage
    @Published var connected = false

    var process: Process?
    private var stdin: FileHandle?
    private var buffer = Data()

    func start() {
        guard process == nil else { return }
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
            Task { @MainActor in self?.connected = false; self?.process = nil; self?.append("Worker exited"); self?.rescan() }
        }
        do {
            try p.run()
            process = p; stdin = inPipe.fileHandleForWriting
            append("Worker started: \(p.executableURL!.path)")
        } catch {
            append("Could not start worker: \(error.localizedDescription)")
        }
    }

    private func consume(_ data: Data) {
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
                // Placeholders for the queued songs appear at once; a song already listed (a render of a
                // draft, or a stalled song) goes back into the pipeline.
                songs.removeAll { $0.status == .failed }
                let run = URL(fileURLWithPath: obj["output"] as? String ?? "").lastPathComponent
                for entry in obj["songs"] as? [[String: Any]] ?? [] {
                    let path = entry["path"] as? String ?? ""
                    let priority = entry["priority"] as? Int ?? 0
                    if let i = songs.firstIndex(where: { $0.path == path }) {
                        songs[i].status = .queued; songs[i].detail = "queued"; songs[i].fraction = nil; songs[i].priority = priority
                    } else {
                        songs.append(Song(run: run, index: entry["index"] as? Int ?? 0, path: path, score: "", seconds: 0,
                                          seed: entry["seed"] as? Int ?? 0, truncated: false, status: .queued, detail: "queued", priority: priority))
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
                                engine: obj["engine"] as? String ?? "")
                if let i = songs.firstIndex(where: { $0.path == path }) { songs[i] = song } else { songs.append(song) }
                sortSongs(); updateBusy()
            case "failed":
                if let i = songs.firstIndex(where: { $0.path == path }) { songs[i].status = .failed; songs[i].detail = obj["message"] as? String ?? "failed" }
                updateBusy()
            case "idle": updateBusy(); rescan()
            case "error": append("Worker: \(obj["message"] as? String ?? "error")")
            default: break
            }
        }
    }

    func append(_ message: String) {
        let f = DateFormatter(); f.dateFormat = "HH:mm:ss"
        log.append(LogLine(time: f.string(from: Date()), message: message))
        if log.count > 2000 { log.removeFirst(log.count - 2000) }
    }

    private func updateBusy() { busy = songs.contains { $0.inFlight } }

    /// Newest run first, songs in order within a run.
    private func sortSongs() {
        songs.sort { $0.run != $1.run ? $0.run > $1.run : $0.index < $1.index }
    }

    /// Reconcile with the songs folder: everything on disk is listed (finished songs, and songs whose
    /// tokens were saved but never synthesized), and entries whose files are gone disappear. Songs
    /// the worker is still working on, and failures of this session, are kept as they are.
    func rescan() {
        let kept = songs.filter { $0.inFlight || $0.status == .failed }
        let onDisk = Song.scan(Paths.output)
        let known = Dictionary(songs.map { ($0.path, $0) }, uniquingKeysWith: { (a: Song, _: Song) in a })
        let keptPaths = Set(kept.map(\.path))
        songs = onDisk.filter { !keptPaths.contains($0.path) }.map { (disk: Song) -> Song in
            // A ready song we already know keeps its in-memory copy (score text etc.); stalled ones come from disk.
            if let k = known[disk.path], k.status == .ready, disk.status == .ready { return k }
            return disk
        } + kept
        sortSongs(); updateBusy()
    }

    func send(_ obj: [String: Any]) {
        guard let stdin, let data = try? JSONSerialization.data(withJSONObject: obj) else { return }
        stdin.write(data); stdin.write(Data([0x0A]))
    }

    /// Queue a run; the worker announces its songs with a "started" event.
    func generate(style: String, lyrics: String, cot: String, seed: Int, randomSeed: Bool, batch: Int, maxTokens: Int, engine: String, abc: String, quality: String, instrumental: Bool) {
        send(["cmd": "generate", "style": style, "lyrics": lyrics, "cot": cot, "seed": seed, "random_seed": randomSeed,
              "batch": batch, "max_tokens": maxTokens, "engine": engine, "abc": abc, "quality": quality, "instrumental": instrumental])
    }

    /// Synthesize a song from its saved tokens: a full-quality render of a draft, or a stalled song.
    func render(_ song: Song, engine: String, quality: String) {
        guard let i = songs.firstIndex(where: { $0.id == song.id }), !songs[i].inFlight else { return }
        songs[i].status = .queued; songs[i].detail = "queued"; songs[i].fraction = nil
        updateBusy()
        send(["cmd": "render", "path": song.path, "engine": engine, "quality": quality])
    }

    func cancel(_ song: Song) { send(["cmd": "cancel", "path": song.path]) }
    func stop() { send(["cmd": "stop"]); append("Stop sent") }
    func quit() { send(["cmd": "quit"]); process?.terminate() }
}

// MARK: - Audio

@MainActor
final class Players: ObservableObject {
    private var players: [String: AVPlayer] = [:]
    @Published var playing: String?
    func toggle(_ song: Song) {
        guard song.status == .ready else { return }
        if playing == song.id { players[song.id]?.pause(); playing = nil; return }
        if let current = playing { players[current]?.pause() }
        let player = players[song.id] ?? AVPlayer(url: URL(fileURLWithPath: song.path))
        players[song.id] = player
        player.seek(to: .zero); player.play(); playing = song.id
    }
    /// A re-rendered file must not be served from the old player.
    func forget(_ song: Song) { if playing == song.id { players[song.id]?.pause(); playing = nil }; players[song.id] = nil }
}

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
    @AppStorage("style") private var style = "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM"
    @AppStorage("lyrics") private var lyrics = "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\nFold the night and leave it here\nMorning has a sky to clear\n\n[Chorus]\nLet the day come into view\nEvery road begins with you\nHold a little room for light\nWe will sing beyond the night"
    @AppStorage("cot") private var cot = "full"
    @AppStorage("seed") private var seed = 831001
    @AppStorage("randomSeed") private var randomSeed = false
    @AppStorage("batch") private var batch = 2
    @AppStorage("maxSeconds") private var maxSeconds = 120.0
    @AppStorage("quality") private var quality = "draft"
    @AppStorage("instrumental") private var instrumental = false
    @State private var abc = ""
    @State private var showScore: Song?
    @AppStorage("logPanelHeight") private var logPanelHeight = 130.0
    @State private var logDragStart: Double? = nil

    var body: some View {
        HSplitView {
            form.frame(minWidth: 380, idealWidth: 440)
            VStack(spacing: 0) {
                results.frame(minHeight: 160)
                logSplitter
                logView.frame(height: max(64, min(logPanelHeight, 600)))
            }.frame(minWidth: 480)
        }
        .frame(minWidth: 960, minHeight: 640)
        .onAppear { backend.rescan(); if backend.process == nil { backend.start() } }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in backend.rescan() }
    }

    private var form: some View {
        Form {
            Section("Song") {
                TextField("Style", text: $style, axis: .vertical).lineLimit(2...4).multilineTextAlignment(.leading)
                Text("Lyrics").font(.caption).foregroundStyle(.secondary)
                TextEditor(text: $lyrics).font(.system(.body, design: .monospaced)).frame(minHeight: 220)
            }
            Section("Run") {
                Stepper("Songs per run: \(batch)", value: $batch, in: 1...8)
                Picker("Planning", selection: $cot) { Text("full").tag("full"); Text("melody").tag("melody"); Text("off").tag("off") }.pickerStyle(.segmented)
                Picker("Quality", selection: $quality) { Text("Draft (fast preview)").tag("draft"); Text("Full").tag("full") }.pickerStyle(.segmented)
                Text(quality == "draft" ? "Drafts synthesize in 8 steps on the GPU: same song, rougher sound, no Neural Engine compile. Render any draft at full quality from its row."
                                        : "Full quality synthesizes in 32 steps; long songs can take an hour or more.").font(.caption).foregroundStyle(.secondary)
                Toggle("Instrumental (no vocals)", isOn: $instrumental)
                if instrumental {
                    Text("Adds no-vocal tags, keeps only the section markers of the lyrics, and silences the vocal voice in the planned score before the song is tokenized. Planning is used even if set to off.").font(.caption).foregroundStyle(.secondary)
                }
                HStack { Text("Max length"); Slider(value: $maxSeconds, in: 30...360, step: 10); Text("\(Int(maxSeconds)) s").monospacedDigit().frame(width: 44) }
                HStack {
                    TextField("Base seed", value: $seed, format: .number).disabled(randomSeed)
                    Toggle("Random", isOn: $randomSeed)
                }
                DisclosureGroup("ABC score (optional)") {
                    TextEditor(text: $abc).font(.system(.caption, design: .monospaced)).frame(height: 100)
                }
            }
            Section {
                HStack {
                    Button(action: {
                        backend.generate(style: style, lyrics: lyrics, cot: cot, seed: seed, randomSeed: randomSeed, batch: batch,
                                         maxTokens: Int(maxSeconds * 25), engine: "auto", abc: abc, quality: quality, instrumental: instrumental)
                    }) { Label(backend.busy ? "Add to queue" : "Generate", systemImage: backend.busy ? "plus" : "play.fill").frame(maxWidth: .infinity) }
                        .buttonStyle(.borderedProminent).keyboardShortcut(.return, modifiers: .command).disabled(!backend.connected)
                    Button(action: { backend.stop() }) { Label("Stop all", systemImage: "stop.fill") }
                        .disabled(!backend.busy)
                }
                pipelineStatus
                if !backend.connected { Text("Worker not connected").foregroundStyle(.red).font(.caption) }
            }
        }
        .formStyle(.grouped)
    }

    /// One line per stage: what it is working on right now.
    private var pipelineStatus: some View {
        let queued = backend.songs.filter { $0.status == .queued }.count
        return VStack(alignment: .leading, spacing: 3) {
            stageLine("Planning · GPU", backend.songs.filter { $0.status == .planning })
            stageLine("Tokenizing · GPU", backend.songs.filter { $0.status == .tokens })
            stageLine("Synthing · " + (backend.songs.first { $0.status == .synth }?.engineLabel ?? "Neural Engine"), backend.songs.filter { $0.status == .synth })
            stageLine("Rendering · GPU", backend.songs.filter { $0.status == .decode })
            HStack { Text("Queued").bold(); Spacer(); Text(queued == 0 ? "—" : "\(queued) song\(queued == 1 ? "" : "s")").foregroundStyle(.secondary) }.font(.caption)
        }
    }

    private func stageLine(_ title: String, _ songs: [Song]) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title).bold()
            Spacer()
            if let first = songs.first {
                let names = songs.map { "song \($0.index)" }.joined(separator: ", ")
                Text("\(names) · \(first.runLabel)" + (first.detail.isEmpty ? "" : " · \(first.detail)")).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            } else {
                Text("idle").foregroundStyle(.secondary)
            }
        }.font(.caption)
    }

    private var runs: [String] { Array(NSOrderedSet(array: backend.songs.filter { !$0.inFlight }.map(\.run))) as? [String] ?? [] }

    /// In-flight songs are grouped by the stage they are in (in pipeline order), then finished runs, newest first.
    private static let stages: [(Song.Status, String)] = [(.queued, "Queued"), (.planning, "Planning · GPU"), (.tokens, "Tokenizing · GPU"), (.synth, "Synthing"), (.decode, "Rendering · GPU")]

    private var results: some View {
        List {
            ForEach(Self.stages, id: \.1) { status, title in
                let inStage = backend.songs.filter { $0.status == status }
                if !inStage.isEmpty {
                    Section(status == .synth ? "Synthing · " + (inStage.first { !$0.engineLabel.isEmpty }?.engineLabel ?? "Neural Engine / GPU") : title) {
                        ForEach(inStage) { song in row(song) }
                    }
                }
            }
            ForEach(runs, id: \.self) { run in
                Section(backend.songs.first { $0.run == run }?.runLabel ?? run) {
                    ForEach(backend.songs.filter { $0.run == run && !$0.inFlight }) { song in row(song) }
                }
            }
        }
        .overlay { if backend.songs.isEmpty { Text("Songs appear here").foregroundStyle(.secondary) } }
        .safeAreaInset(edge: .top) { HStack { Text("Songs").font(.caption).bold(); Spacer(); Button("Open songs folder") { NSWorkspace.shared.open(Paths.output) }.font(.caption) }.padding(.horizontal, 8).padding(.vertical, 4).background(.bar) }
        .sheet(item: $showScore) { song in
            VStack(alignment: .leading) {
                Text("Score for song \(song.index)").font(.headline)
                ScrollView { Text(song.score).font(.system(.body, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
                HStack { Spacer(); Button("Close") { showScore = nil }.keyboardShortcut(.cancelAction) }
            }.padding().frame(width: 640, height: 480)
        }
    }

    private func row(_ song: Song) -> some View {
        HStack {
            Button(action: { players.toggle(song) }) {
                switch song.status {
                case .ready: Image(systemName: players.playing == song.id ? "pause.circle.fill" : "play.circle.fill").font(.title)
                case .stalled: Image(systemName: "doc.text").font(.title).foregroundStyle(.secondary).frame(width: 28)
                case .failed: Image(systemName: "exclamationmark.triangle.fill").font(.title).foregroundStyle(.red).frame(width: 28)
                case .queued: Image(systemName: "clock").font(.title).foregroundStyle(.secondary).frame(width: 28)
                default: ProgressView().controlSize(.small).frame(width: 28)
                }
            }.buttonStyle(.plain).disabled(song.status != .ready)
            VStack(alignment: .leading, spacing: 2) {
                switch song.status {
                case .ready:
                    Text("Song \(song.index) · seed \(song.seed) · \(String(format: "%.1f", song.seconds)) s" + (song.truncated ? " · truncated" : "")
                         + (song.quality == "draft" ? " · draft" : "")).bold()
                case .stalled:
                    Text("Song \(song.index) · seed \(song.seed) · about \(Int(song.seconds)) s · tokens only").bold().foregroundStyle(.secondary)
                case .failed:
                    Text("Song \(song.index) · seed \(song.seed) · failed").bold().foregroundStyle(.red)
                default:
                    Text((song.priority > 0 ? "#\(song.priority) · " : "") + "Song \(song.index) · seed \(song.seed) · \(song.runLabel)" + (song.quality == "draft" ? " · draft" : "")).bold().foregroundStyle(.secondary)
                }
                if song.inFlight {
                    StageTrack(progress: song.trackProgress)
                    Text("Status: " + song.statusLine).font(.caption).lineLimit(1)
                } else if song.status == .failed || song.status == .stalled {
                    Text(song.statusLine).font(.caption).foregroundStyle(song.status == .failed ? .red : .secondary).lineLimit(2)
                } else {
                    Text(song.path).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                }
            }
            Spacer()
            if song.inFlight {
                Button("Cancel") { backend.cancel(song) }.disabled(!backend.connected)
            } else if song.status == .stalled || (song.status == .failed && FileManager.default.fileExists(atPath: song.directory.appendingPathComponent("semantic.npy").path)) {
                Button(quality == "draft" ? "Synthesize draft" : "Synthesize full") { backend.render(song, engine: "auto", quality: quality) }.disabled(!backend.connected)
            } else if song.quality == "draft" && song.status == .ready {
                Button("Render full quality") { players.forget(song); backend.render(song, engine: "auto", quality: "full") }.disabled(!backend.connected)
            }
            if !song.score.isEmpty { Button("Score") { showScore = song } }
            Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: song.path)]) }.disabled(song.status != .ready)
        }.padding(.vertical, 2)
    }

    /// The divider above the log: drag up or down to resize it; the height persists.
    private var logSplitter: some View {
        ZStack { Divider() }
            .frame(height: 9)
            .contentShape(Rectangle())
            .onHover { inside in
                if inside { NSCursor.resizeUpDown.push() } else { NSCursor.pop() }
            }
            .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .global)
                .onChanged { v in
                    let start = logDragStart ?? logPanelHeight
                    logDragStart = start
                    logPanelHeight = min(600, max(64, start - v.translation.height))
                }
                .onEnded { _ in logDragStart = nil })
    }

    private var logView: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Log").font(.caption).bold(); Spacer()
                Button("Copy all") {
                    let text = backend.log.map { "\($0.time)  \($0.message)" }.joined(separator: "\n")
                    NSPasteboard.general.clearContents(); NSPasteboard.general.setString(text, forType: .string)
                }.font(.caption)
                Button("Clear") { backend.log.removeAll() }.font(.caption)
            }.padding(.horizontal, 8).padding(.vertical, 4)
            LogTextView(lines: backend.log)
        }
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
        if lines.count < coordinator.rendered {                       // cleared
            storage.setAttributedString(NSAttributedString(string: "")); coordinator.rendered = 0
        }
        guard lines.count > coordinator.rendered else { return }
        let attrs: [NSAttributedString.Key: Any] = [.font: text.font!, .foregroundColor: NSColor.textColor]
        let chunk = lines[coordinator.rendered...].map { "\($0.time)  \($0.message)" }.joined(separator: "\n") + "\n"
        let atBottom = scroll.contentView.bounds.maxY >= (text.bounds.height - 40)
        storage.append(NSAttributedString(string: chunk, attributes: attrs))
        coordinator.rendered = lines.count
        if atBottom || text.selectedRange().length == 0 { text.scrollToEndOfDocument(nil) }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator { var rendered = 0 }
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
                    Button("Repair Installation…") { backend.quit(); installer.repair() }.disabled(!Paths.packaged)
                }
            }
    }
}
