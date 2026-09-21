import SwiftUI
import AVFoundation
import AppKit
import UniformTypeIdentifiers

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

    @Published var remoteStatus = ""
    private var remoteSent: (String, Int)?

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
                self.remoteSent = nil; self.remoteStatus = ""
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
            case "remote":
                let name = obj["name"] as? String ?? "iPhone", detail = obj["detail"] as? String ?? ""
                switch obj["state"] as? String ?? "" {
                case "connected": remoteStatus = "\(name) ready" + (detail.isEmpty ? "" : " · \(detail)")
                case "gone": remoteStatus = "\(name) disconnected"; remoteSent = nil; scheduleRemoteRetry()
                default: remoteStatus = "\(name): \(detail)"; remoteSent = nil; scheduleRemoteRetry()
                }
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
                  promptFidelity: Double, styleFidelity: Double, sourceFidelity: Double, targetSeconds: Double?, engines: String = "gpu+ane") {
        var request: [String: Any] = ["cmd": "generate", "title": title, "style": style, "lyrics": lyrics,
                                       "cot": cot, "seed": seed, "random_seed": randomSeed, "batch": batch,
                                       "max_tokens": maxTokens, "quality": quality, "engines": engines, "instrumental": instrumental,
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
    func render(_ song: Song, engine: String, quality: String, engines: String = "gpu+ane") {
        guard song.canRender, let i = songs.firstIndex(where: { $0.id == song.id }), !songs[i].inFlight else { return }
        guard send(["cmd": "render", "path": song.path, "engine": engine, "quality": quality, "engines": engines]) else { return }
        songs[i].status = .queued; songs[i].detail = "queued"; songs[i].fraction = nil
        updateBusy()
    }

    var remoteRetry: (() -> Void)?             // set by the view: re-offers the phone after a refusal
    private func scheduleRemoteRetry() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 15) { [weak self] in self?.remoteRetry?() }
    }

    /// Hand the worker the phone to use (or nil to stop using one).
    func useRemote(_ phone: RemoteBrowser.Phone?) {
        guard connected else { remoteSent = nil; return }
        if let phone {
            guard remoteSent?.0 != phone.host || remoteSent?.1 != phone.port else { return }
            if send(["cmd": "remote", "host": phone.host, "port": phone.port, "name": phone.name]) {
                remoteSent = (phone.host, phone.port)
            }
        } else if remoteSent != nil {
            remoteSent = nil
            send(["cmd": "remote", "host": NSNull()])
        }
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
