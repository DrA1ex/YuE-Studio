import Foundation
import AVFoundation

@MainActor
extension Backend {
    static func transcriptionArguments(
        audio: URL,
        output: URL,
        hum: Bool,
        mlxPython: URL?,
        lyrics: Bool = true,
        genre: Bool = true,
        style: Bool = true,
        vocalActivity: Bool = true
    ) -> [String] {
        var args = ["-u", Paths.transcriber.path, audio.path, "--output", output.path,
                    "--cache-dir", Paths.coverAnalyses.path, "--device", "cpu", "--dtype", "fp32"]
        if hum {
            args.append("--hum")
            return args
        }
        if !lyrics { args.append("--skip-lyrics") }
        if !genre { args.append("--skip-genre") }
        if !style { args.append("--skip-style") }
        if !vocalActivity || !lyrics { args.append("--disable-separation") }
        if lyrics, let mlxPython { args += ["--mlx-python", mlxPython.path] }
        return args
    }

    func cancelCover() {
        coverTask?.cancel()
        if coverProcess?.isRunning == true { coverProcess?.terminate() }
    }

    private var coverEnvironment: [String: String] {
        Paths.workerEnvironment.merging([
            "HF_HOME": Paths.coverSupport.appendingPathComponent("models").path,
            "HF_HUB_DISABLE_XET": "1",
            "UV_CACHE_DIR": Paths.coverSupport.appendingPathComponent("uv-cache").path,
            "TORCH_HOME": Paths.coverSupport.appendingPathComponent("torch").path
        ]) { _, new in new }
    }

    /// Drain the existing queue before releasing YuE; never run both models together.
    func transcribe(_ source: AudioSource, hum: Bool = false, completion: @escaping (Result<CoverAnalysis, Error>) -> Void) {
        guard !coverBusy, coverRuntimeReady else {
            completion(.failure(coverError("Install the cover engine before transcribing."))); return
        }
        coverBusy = true; coverStatus = "Waiting for queued songs…"
        coverTask = Task {
            let output = FileManager.default.temporaryDirectory.appendingPathComponent("yue-cover-" + UUID().uuidString)
            var stoppedWorker = false
            defer {
                try? FileManager.default.removeItem(at: output)
                coverBusy = false; coverTask = nil; coverProcess = nil; coverStatus = ""
                if stoppedWorker && !shuttingDown { start() }
            }
            do {
                while busy { try Task.checkCancellation(); try await Task.sleep(nanoseconds: 200_000_000) }
                try Task.checkCancellation()
                coverStatus = "Releasing YuE memory…"
                if process != nil {
                    send(["cmd": "quit"]); connected = false; stoppedWorker = true
                    while process != nil { try Task.checkCancellation(); try await Task.sleep(nanoseconds: 200_000_000) }
                }
                try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
                let normalized = output.appendingPathComponent("input.wav")
                coverStatus = "Decoding source audio…"
                try await Task.detached(priority: .userInitiated) { try AudioPreparation.writeAnalysisAudio(source: URL(fileURLWithPath: source.path), destination: normalized) }.value
                try Task.checkCancellation()
                let analyzeLyrics = hum ? false : CoverAnalysisPreferences.lyricsEnabled
                let analyzeGenre = hum ? false : CoverAnalysisPreferences.genreEnabled
                let analyzeStyle = hum ? false : CoverAnalysisPreferences.styleEnabled
                let analyzeVocalActivity = hum ? false : CoverAnalysisPreferences.vocalActivityEnabled
                let lyricsBackend = CoverAnalysisPreferences.lyricsBackend
                let useMLX = analyzeLyrics && lyricsBackend != .transformers &&
                    FileManager.default.isExecutableFile(atPath: Paths.coverWhisperPython.path)
                coverStatus = hum ? "Transcribing melody…" : "Analyzing enabled cover stages…"
                let arguments = Self.transcriptionArguments(
                    audio: normalized,
                    output: output,
                    hum: hum,
                    mlxPython: useMLX ? Paths.coverWhisperPython : nil,
                    lyrics: analyzeLyrics,
                    genre: analyzeGenre,
                    style: analyzeStyle,
                    vocalActivity: analyzeVocalActivity
                )
                try await runCoverCommand(Paths.coverPython.path, arguments)
                try Task.checkCancellation()
                let score = try String(contentsOf: output.appendingPathComponent("score.abc"), encoding: .utf8)
                guard !score.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw coverError("The transcriber returned an empty score.") }
                let analysisURL = output.appendingPathComponent("cover_analysis.json")
                let analysisJSON = (try? JSONSerialization.jsonObject(with: Data(contentsOf: analysisURL)) as? [String: Any]) ?? [:]
                let warnings = analysisJSON["warnings"] as? [String] ?? []
                let lyrics = analysisJSON["lyrics"] as? String ?? ""
                let genre = analysisJSON["style"] as? String ?? analysisJSON["genre"] as? String ?? ""
                if source.kind == "UPLOAD" {
                    let metadataURL = URL(fileURLWithPath: source.path).deletingPathExtension().appendingPathExtension("json")
                    var metadata = (try? JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL)) as? [String: Any]) ?? [:]
                    metadata["score"] = score
                    metadata["lyrics"] = lyrics
                    metadata["suggested_genre"] = genre
                    metadata["style"] = genre
                    metadata["analysis"] = analysisJSON
                    try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted]).write(to: metadataURL, options: .atomic)
                } else if !hum {
                    try score.write(to: URL(fileURLWithPath: source.path).deletingLastPathComponent().appendingPathComponent("cover-score.abc"), atomically: true, encoding: .utf8)
                }
                rescan(); append("Audio analysis complete"); completion(.success(CoverAnalysis(score: score, lyrics: lyrics, genre: genre, warnings: warnings)))
            } catch {
                let message = Task.isCancelled ? "Transcription cancelled" : "Transcription failed: \(error.localizedDescription)"
                append(message); completion(.failure(coverError(message)))
                // A cancellation during shutdown must still wait for the old worker before restarting.
                if stoppedWorker && process != nil {
                    process?.terminate()
                    while process != nil { await Task.detached { try? await Task.sleep(nanoseconds: 100_000_000) }.value }
                }
            }
        }
    }

    /// Install only the components required by the enabled cover-analysis stages.
    func installCoverRuntime(completion: @escaping (Result<Void, Error>) -> Void) {
        var components: [CoverComponent] = [.melody]
        if CoverAnalysisPreferences.lyricsEnabled {
            switch CoverAnalysisPreferences.lyricsBackend {
            case .automatic, .mlx:
                components.append(.mlxWhisper)
            case .transformers:
                components.append(.lyrics)
            }
            if CoverAnalysisPreferences.vocalActivityEnabled { components.append(.vocalActivity) }
        }
        if CoverAnalysisPreferences.genreEnabled { components.append(.genre) }
        if CoverAnalysisPreferences.styleEnabled { components.append(.style) }

        startCoverInstallation(completion: completion) {
            let uv = try self.coverUV()
            for component in components {
                try await self.installCoverComponentProcess(component, uv: uv)
                try self.markCoverComponentInstalled(component)
            }
            self.append("Enabled cover components installed")
        }
    }

    func installCoverComponent(_ component: CoverComponent, completion: @escaping (Result<Void, Error>) -> Void) {
        startCoverInstallation(completion: completion) {
            let uv = try self.coverUV()
            try await self.installCoverComponentProcess(component, uv: uv)
            try self.markCoverComponentInstalled(component)
            self.append("Installed \(component.title)")
        }
    }

    func removeCoverComponent(_ component: CoverComponent, completion: @escaping (Result<Void, Error>) -> Void) {
        guard !coverBusy, !busy else {
            completion(.failure(coverError("Audio analysis or generation is busy."))); return
        }
        do {
            let fm = FileManager.default
            for path in component.modelPaths where fm.fileExists(atPath: path.path) {
                try fm.trashItem(at: path, resultingItemURL: nil)
            }
            for marker in [component.marker] + component.legacyMarkers where fm.fileExists(atPath: marker.path) {
                try fm.trashItem(at: marker, resultingItemURL: nil)
            }
            try Data("removed\n".utf8).write(to: component.disabledMarker, options: .atomic)
            refreshCoverInstallationState()
            append("Removed \(component.title) — moved component files to Trash")
            completion(.success(()))
        } catch {
            completion(.failure(coverError("Could not remove \(component.title): \(error.localizedDescription)")))
        }
    }

    func clearCoverComponentMarkers() {
        let fm = FileManager.default
        let markers = CoverComponent.allCases.flatMap { [$0.marker, $0.disabledMarker] + $0.legacyMarkers }
            + [Paths.coverSupport.appendingPathComponent("installed-v3"), Paths.coverSupport.appendingPathComponent("installed-v4"), Paths.coverSupport.appendingPathComponent("installed-v5"), Paths.coverSupport.appendingPathComponent("installed-v6")]
        for marker in markers where fm.fileExists(atPath: marker.path) { try? fm.removeItem(at: marker) }
        refreshCoverInstallationState()
    }

    private func startCoverInstallation(completion: @escaping (Result<Void, Error>) -> Void, operation: @escaping () async throws -> Void) {
        guard !coverBusy else { completion(.failure(coverError("Audio analysis engine is busy."))); return }
        coverBusy = true; coverStatus = "Preparing component installation…"
        coverTask = Task {
            do {
                try await operation()
                try Task.checkCancellation()
                refreshCoverInstallationState()
                finishCoverInstallation(.success(()), completion: completion)
            } catch {
                let message = Task.isCancelled ? "Installation cancelled; it can be resumed." : error.localizedDescription
                append(message)
                finishCoverInstallation(.failure(coverError(message)), completion: completion)
            }
        }
    }

    private func coverUV() throws -> String {
        var candidates = ["/opt/homebrew/bin/uv", "/usr/local/bin/uv", NSHomeDirectory() + "/.local/bin/uv"]
        if let bundled = Paths.payload?.appendingPathComponent("uv").path { candidates.insert(bundled, at: 0) }
        guard let uv = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            throw coverError("uv is required to install local cover components.")
        }
        return uv
    }

    private func installCoverComponentProcess(_ component: CoverComponent, uv: String) async throws {
        try FileManager.default.createDirectory(at: Paths.coverSupport, withIntermediateDirectories: true)
        if component == .mlxWhisper {
            if !FileManager.default.isExecutableFile(atPath: Paths.coverWhisperPython.path) {
                coverStatus = "Installing the isolated MLX Whisper environment…"
                try await runCoverCommand(uv, ["venv", Paths.coverSupport.appendingPathComponent("whisper-env").path, "--python", "3.11"])
            }
            coverStatus = "Installing MLX Whisper…"
            try await runCoverCommand(uv, ["pip", "install", "--python", Paths.coverWhisperPython.path, "-r", Paths.coverWhisperRequirements.path])
            coverStatus = "Downloading MLX Whisper weights…"
            try await runCoverCommand(Paths.coverWhisperPython.path, ["-u", Paths.transcriber.path, "--install-component", component.installName])
            return
        }

        if !FileManager.default.isExecutableFile(atPath: Paths.coverPython.path) {
            coverStatus = "Installing the cover Python environment…"
            try await runCoverCommand(uv, ["venv", Paths.coverSupport.appendingPathComponent("env").path, "--python", "3.11"])
        }
        coverStatus = "Checking cover analysis dependencies…"
        try await runCoverCommand(uv, ["pip", "install", "--python", Paths.coverPython.path, "-r", Paths.coverRequirements.path])
        coverStatus = "Downloading \(component.title)…"
        try await runCoverCommand(Paths.coverPython.path, ["-u", Paths.transcriber.path, "--install-component", component.installName])
    }

    private func markCoverComponentInstalled(_ component: CoverComponent) throws {
        try? FileManager.default.removeItem(at: component.disabledMarker)
        try Data("ready\n".utf8).write(to: component.marker, options: .atomic)
    }

    /// Release state before calling client code, which may immediately start analysis.
    func finishCoverInstallation(_ result: Result<Void, Error>, completion: (Result<Void, Error>) -> Void) {
        coverBusy = false; coverTask = nil; coverProcess = nil; coverStatus = ""
        completion(result)
    }

    private func runCoverCommand(_ executable: String, _ arguments: [String]) async throws {
        try Task.checkCancellation()
        let p = Process(); p.executableURL = URL(fileURLWithPath: executable); p.arguments = arguments
        p.environment = coverEnvironment
        let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
        let lines = ProcessLineBuffer()
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            let complete = lines.append(data)
            Task { @MainActor in
                for line in complete {
                    if let data = line.data(using: .utf8), let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       event["event"] as? String == "transcription", let message = event["message"] as? String {
                        self?.coverStatus = message; self?.append(message)
                    } else { self?.append(line) }
                }
            }
        }
        coverProcess = p
        let status: Int32 = try await withCheckedThrowingContinuation { continuation in
            p.terminationHandler = { process in continuation.resume(returning: process.terminationStatus) }
            do { try p.run() } catch { continuation.resume(throwing: error) }
        }
        pipe.fileHandleForReading.readabilityHandler = nil
        coverProcess = nil
        try Task.checkCancellation()
        guard status == 0 else { throw coverError("Process exited with status \(status). Open Process Log for details.") }
    }

    private func coverError(_ message: String) -> NSError {
        NSError(domain: "YuEStudio.Cover", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}

private final class ProcessLineBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = Data()
    func append(_ data: Data) -> [String] {
        lock.lock(); defer { lock.unlock() }
        pending.append(data)
        var result: [String] = []
        while let boundary = pending.firstIndex(where: { $0 == 10 || $0 == 13 }) {
            let line = String(decoding: pending[..<boundary], as: UTF8.self)
            pending.removeSubrange(...boundary)
            if !line.isEmpty { result.append(line) }
        }
        if data.isEmpty && !pending.isEmpty { result.append(String(decoding: pending, as: UTF8.self)); pending.removeAll() }
        return result
    }
}

enum AudioPreparation {
    /// Preserve the original channels for source separation. The Python
    /// analyzer downmixes a separate mono view for SheetSage and Whisper.
    static func writeAnalysisAudio(source: URL, destination: URL) throws {
        let input = try AVAudioFile(forReading: source)
        let format = input.processingFormat
        guard input.length > 0, format.sampleRate > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 16384) else { throw CocoaError(.fileReadCorruptFile) }
        let output = try AVAudioFile(forWriting: destination, settings: format.settings)
        while input.framePosition < input.length {
            try input.read(into: buffer)
            guard buffer.frameLength > 0, let channels = buffer.floatChannelData else { throw CocoaError(.fileReadCorruptFile) }
            for channel in 0..<Int(format.channelCount) {
                for frame in 0..<Int(buffer.frameLength) {
                    guard channels[channel][frame].isFinite else { throw CocoaError(.fileReadCorruptFile) }
                }
            }
            try output.write(from: buffer)
        }
    }

    /// AVFoundation handles local formats; the transcriber receives finite mono PCM samples.
    static func writeMono(source: URL, destination: URL) throws {
        let input = try AVAudioFile(forReading: source)
        let format = input.processingFormat
        guard input.length > 0, format.sampleRate > 0,
              let mono = AVAudioFormat(standardFormatWithSampleRate: format.sampleRate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 16384),
              let outputBuffer = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity: 16384) else { throw CocoaError(.fileReadCorruptFile) }
        let output = try AVAudioFile(forWriting: destination, settings: mono.settings)
        while input.framePosition < input.length {
            try input.read(into: buffer)
            guard buffer.frameLength > 0, let channels = buffer.floatChannelData, let target = outputBuffer.floatChannelData?[0] else { throw CocoaError(.fileReadCorruptFile) }
            outputBuffer.frameLength = buffer.frameLength
            for frame in 0..<Int(buffer.frameLength) {
                var sample: Float = 0
                for channel in 0..<Int(format.channelCount) { sample += channels[channel][frame] / Float(format.channelCount) }
                guard sample.isFinite else { throw CocoaError(.fileReadCorruptFile) }
                target[frame] = min(1, max(-1, sample))
            }
            try output.write(from: outputBuffer)
        }
    }
}
