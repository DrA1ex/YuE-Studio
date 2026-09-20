import Foundation
import AVFoundation

@MainActor
extension Backend {
    func cancelCover() {
        coverTask?.cancel()
        if coverProcess?.isRunning == true { coverProcess?.terminate() }
    }

    private var coverEnvironment: [String: String] {
        Paths.workerEnvironment.merging([
            "HF_HOME": Paths.coverSupport.appendingPathComponent("models").path,
            "HF_HUB_DISABLE_XET": "1",
            "UV_CACHE_DIR": Paths.coverSupport.appendingPathComponent("uv-cache").path
        ]) { _, new in new }
    }

    /// Drain the existing queue before releasing YuE; never run both models together.
    func transcribe(_ source: AudioSource, completion: @escaping (Result<CoverAnalysis, Error>) -> Void) {
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
                try await Task.detached(priority: .userInitiated) { try AudioPreparation.writeMono(source: URL(fileURLWithPath: source.path), destination: normalized) }.value
                try Task.checkCancellation()
                coverStatus = "Analyzing melody, lyrics and genre…"
                try await runCoverCommand(Paths.coverPython.path, ["-u", Paths.transcriber.path, normalized.path, "--output", output.path, "--device", "cpu", "--dtype", "fp32"])
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
                } else {
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

    func installCoverRuntime(completion: @escaping (Result<Void, Error>) -> Void) {
        guard !coverBusy else { completion(.failure(coverError("Audio analysis engine is busy."))); return }
        var candidates = ["/opt/homebrew/bin/uv", "/usr/local/bin/uv", NSHomeDirectory() + "/.local/bin/uv"]
        if let bundled = Paths.payload?.appendingPathComponent("uv").path { candidates.insert(bundled, at: 0) }
        guard let uv = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            completion(.failure(coverError("uv is required to install the local cover engine."))); return
        }
        coverBusy = true; coverStatus = "Installing Python 3.11…"
        coverTask = Task {
            do {
                try FileManager.default.createDirectory(at: Paths.coverSupport, withIntermediateDirectories: true)
                if !FileManager.default.isExecutableFile(atPath: Paths.coverPython.path) {
                    try await runCoverCommand(uv, ["venv", Paths.coverSupport.appendingPathComponent("env").path, "--python", "3.11"])
                }
                coverStatus = "Installing transcription dependencies…"
                try await runCoverCommand(uv, ["pip", "install", "--python", Paths.coverPython.path, "-r", Paths.coverRequirements.path])
                coverStatus = "Downloading SheetSage2 and MERT2…"
                try await runCoverCommand(Paths.coverPython.path, ["-u", Paths.transcriber.path, "--install"])
                try Task.checkCancellation()
                try Data("ready\n".utf8).write(to: Paths.coverSupport.appendingPathComponent("installed-v3"), options: .atomic)
                try Data("ready\n".utf8).write(to: Paths.coverSupport.appendingPathComponent("installed-v4"), options: .atomic)
                try Data("ready\n".utf8).write(to: Paths.coverSupport.appendingPathComponent("installed-v5"), options: .atomic)
                coverReady = true; coverStyleReady = true; coverLyricsReady = true
                append("Cover engine installed")
                finishCoverInstallation(.success(()), completion: completion)
            } catch {
                let message = Task.isCancelled ? "Installation cancelled; it can be resumed." : "Cover engine installation failed: \(error.localizedDescription)"
                append(message)
                finishCoverInstallation(.failure(coverError(message)), completion: completion)
            }
        }
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
