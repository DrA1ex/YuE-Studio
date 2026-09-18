import Foundation
import Combine

@MainActor
final class Backend: ObservableObject {
    @Published var log: [LogLine] = []
    @Published var songs: [Song] = []
    @Published var busy = false                  // anything queued or in a stage
    @Published var connected = false
    @Published var remoteStatus = ""             // what the worker says about the iPhone
    private var remoteSent: (String, Int)?       // host/port last handed to the worker

    enum TranscribeState: Equatable { case idle, transcribing, review, failed(String, code: String) }
    @Published var transcribe: TranscribeState = .idle
    @Published var transcribeDetail = ""
    @Published var transcribeFraction: Double?   // nil = indeterminate
    @Published var transcribeABC = ""
    @Published var transcribeWarnings: [String] = []
    @Published var transcribeOutput = ""
    private var transcribeID = ""

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
            Task { @MainActor in
                guard let self else { return }
                self.connected = false; self.process = nil; self.append("Worker exited"); self.rescan()
                if self.transcribe == .transcribing { self.transcribe = .failed("the worker exited", code: "worker") }
            }
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
                let title = obj["title"] as? String ?? ""
                for entry in obj["songs"] as? [[String: Any]] ?? [] {
                    let path = entry["path"] as? String ?? ""
                    let priority = entry["priority"] as? Int ?? 0
                    if let i = songs.firstIndex(where: { $0.path == path }) {
                        songs[i].status = .queued; songs[i].detail = "queued"; songs[i].fraction = nil; songs[i].priority = priority
                        if !title.isEmpty { songs[i].title = title }
                    } else {
                        songs.append(Song(run: run, index: entry["index"] as? Int ?? 0, path: path, score: "", seconds: 0,
                                          seed: entry["seed"] as? Int ?? 0, truncated: false, status: .queued, detail: "queued", priority: priority, title: title))
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
                                engine: obj["engine"] as? String ?? "", title: obj["title"] as? String ?? "")
                if let i = songs.firstIndex(where: { $0.path == path }) { songs[i] = song } else { songs.append(song) }
                sortSongs(); updateBusy()
            case "failed":
                if let i = songs.firstIndex(where: { $0.path == path }) { songs[i].status = .failed; songs[i].detail = obj["message"] as? String ?? "failed" }
                updateBusy()
            case "idle": updateBusy(); rescan()
            case "remote":
                let name = obj["name"] as? String ?? "iPhone", detail = obj["detail"] as? String ?? ""
                switch obj["state"] as? String ?? "" {
                case "connected": remoteStatus = "\(name) ready" + (detail.isEmpty ? "" : " · \(detail)")
                case "gone": remoteStatus = "\(name) disconnected"; remoteSent = nil; scheduleRemoteRetry()
                default: remoteStatus = "\(name): \(detail)"; remoteSent = nil; scheduleRemoteRetry()
                }
            case "error": append("Worker: \(obj["message"] as? String ?? "error")")
            case "transcribe":
                guard obj["id"] as? String == transcribeID else { break }   // stale run
                switch obj["stage"] as? String ?? "" {
                case "starting", "progress":
                    transcribeDetail = obj["detail"] as? String ?? transcribeDetail
                    transcribeFraction = obj["fraction"] as? Double ?? transcribeFraction
                case "done":
                    transcribeABC = obj["abc"] as? String ?? ""
                    transcribeWarnings = obj["warnings"] as? [String] ?? []
                    transcribeOutput = obj["output"] as? String ?? ""
                    transcribe = .review
                case "failed":
                    transcribe = .failed(obj["message"] as? String ?? "transcription failed",
                                         code: obj["code"] as? String ?? "")
                case "cancelled": transcribe = .idle
                default: break
                }
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
    func generate(title: String, style: String, lyrics: String, cot: String, seed: Int, randomSeed: Bool, batch: Int, maxTokens: Int, engine: String, abc: String, abcOpen: Bool, quality: String, engines: String, instrumental: Bool) {
        send(["cmd": "generate", "title": title, "style": style, "lyrics": lyrics, "cot": cot, "seed": seed, "random_seed": randomSeed,
              "batch": batch, "max_tokens": maxTokens, "engine": engine, "abc": abc, "abc_open": abcOpen, "quality": quality, "engines": engines, "instrumental": instrumental])
    }

    /// Synthesize a song from its saved tokens: a full-quality render of a draft, or a stalled song.
    func render(_ song: Song, engine: String, quality: String, engines: String) {
        guard let i = songs.firstIndex(where: { $0.id == song.id }), !songs[i].inFlight else { return }
        songs[i].status = .queued; songs[i].detail = "queued"; songs[i].fraction = nil
        updateBusy()
        send(["cmd": "render", "path": song.path, "engine": engine, "quality": quality, "engines": engines])
    }

    func cancel(_ song: Song) { send(["cmd": "cancel", "path": song.path]) }
    var remoteRetry: (() -> Void)?             // set by the view: re-offers the phone after a refusal
    private func scheduleRemoteRetry() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 15) { [weak self] in self?.remoteRetry?() }
    }

    /// Hand the worker the phone to use (or nil to stop using one).
    func useRemote(_ phone: RemoteBrowser.Phone?) {
        if let phone {
            guard remoteSent?.0 != phone.host || remoteSent?.1 != phone.port else { return }
            remoteSent = (phone.host, phone.port)
            send(["cmd": "remote", "host": phone.host, "port": phone.port, "name": phone.name])
        } else if remoteSent != nil {
            remoteSent = nil
            send(["cmd": "remote", "host": NSNull()])
        }
    }

    func startTranscription(audio: URL, task: String) {
        transcribeID = UUID().uuidString
        transcribe = .transcribing; transcribeFraction = nil; transcribeDetail = "starting"
        // Offline in packaged mode: the installer already downloaded the snapshot into HF_HOME.
        send(["cmd": "transcribe", "id": transcribeID, "audio": audio.path, "task": task, "offline": Paths.packaged])
    }
    func cancelTranscription() { send(["cmd": "transcribe_cancel", "id": transcribeID]) }
    func stop() { send(["cmd": "stop"]); append("Stop sent") }
    func quit() { send(["cmd": "quit"]); process?.terminate() }
}
