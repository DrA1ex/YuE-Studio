import XCTest
import AVFoundation
@testable import YuEStudio

final class StudioTests: XCTestCase {
    func fixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("yue-tests-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }

    func writeAudio(_ url: URL, frames: UInt32 = 24000, channels: UInt32 = 1) throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: channels)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
        buffer.frameLength = frames
        for channel in 0..<Int(channels) {
            for i in 0..<Int(frames) { buffer.floatChannelData![channel][i] = Float(sin(Double(i) * 0.1)) * (channel == 0 ? 0.7 : 0.3) }
        }
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
    }

    func testWaveformVeryShortAudioAndMissingFile() throws {
        let root = try fixture(), file = root.appendingPathComponent("short.wav")
        try writeAudio(file, frames: 3)
        let bars = WaveformSamples.load(file.path)
        XCTAssertEqual(bars.count, 32)
        XCTAssertTrue(bars.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
        XCTAssertEqual(WaveformSamples.load(root.appendingPathComponent("missing.wav").path), [])
    }

    func testAudioLevelMeterMapsAndClampsSyntheticLevels() {
        XCTAssertEqual(AudioLevelMeter.normalized(decibels: -60), 0, accuracy: 0.0001)
        XCTAssertEqual(AudioLevelMeter.normalized(decibels: -3), 1, accuracy: 0.0001)
        XCTAssertEqual(AudioLevelMeter.normalized(decibels: -100), 0, accuracy: 0.0001)
        XCTAssertEqual(AudioLevelMeter.normalized(decibels: 4), 1, accuracy: 0.0001)
        XCTAssertEqual(AudioLevelMeter.normalized(decibels: .nan), 0, accuracy: 0.0001)
    }

    @MainActor func testVoiceMemoStorePersistsOnlyItsOwnAudio() throws {
        let root = try fixture(), source = root.appendingPathComponent("idea.wav")
        try writeAudio(source)
        let store = VoiceMemoStore(root: root.appendingPathComponent("Voice Memos"))
        let memo = try XCTUnwrap(store.importRecording(source, title: "Test idea"))
        XCTAssertEqual(store.memos.first?.title, "Test idea")
        XCTAssertTrue(FileManager.default.fileExists(atPath: store.url(for: memo).path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: source.path))
        store.delete(memo)
        XCTAssertTrue(store.memos.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.url(for: memo).path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: source.path))
    }

    func testNormalizationUsesAllChannelsAndPreservesDuration() throws {
        let root = try fixture(), source = root.appendingPathComponent("stereo.wav"), dest = root.appendingPathComponent("mono.wav")
        try writeAudio(source, channels: 2)
        try AudioPreparation.writeMono(source: source, destination: dest)
        let audio = try AVAudioFile(forReading: dest)
        XCTAssertEqual(audio.length, 24000); XCTAssertEqual(audio.processingFormat.channelCount, 1)
        let buffer = AVAudioPCMBuffer(pcmFormat: audio.processingFormat, frameCapacity: 24000)!
        try audio.read(into: buffer)
        XCTAssertEqual(buffer.floatChannelData![0][10], Float(sin(1.0)) * 0.5, accuracy: 0.0001)
    }

    func testAnalysisAudioPreservesStereoForSeparation() throws {
        let root = try fixture(), source = root.appendingPathComponent("stereo.wav"), dest = root.appendingPathComponent("analysis.wav")
        try writeAudio(source, channels: 2)
        try AudioPreparation.writeAnalysisAudio(source: source, destination: dest)
        let audio = try AVAudioFile(forReading: dest)
        XCTAssertEqual(audio.length, 24000); XCTAssertEqual(audio.processingFormat.channelCount, 2)
    }

    func testImportedMetadataSurvivesRescan() throws {
        let root = try fixture(), imports = root.appendingPathComponent("Imports")
        try FileManager.default.createDirectory(at: imports, withIntermediateDirectories: true)
        try writeAudio(imports.appendingPathComponent("source.wav"))
        let metadata = imports.appendingPathComponent("source.json")
        try JSONSerialization.data(withJSONObject: ["title": "Original", "style": "Piano", "lyrics": "Hello", "score": "ABC"]).write(to: metadata)
        var song = try XCTUnwrap(Song.scan(root).first)
        XCTAssertEqual(song.kind, "UPLOAD"); XCTAssertEqual(song.seconds, 1, accuracy: 0.001)
        XCTAssertEqual(song.lyrics, "Hello"); XCTAssertEqual(song.score, "ABC")
        XCTAssertFalse(song.canRender)
        try JSONSerialization.data(withJSONObject: ["title": "Renamed", "score": "New ABC"]).write(to: metadata)
        song = try XCTUnwrap(Song.scan(root).first)
        XCTAssertEqual(song.title, "Renamed"); XCTAssertEqual(song.score, "New ABC")
    }

    func testSavedTokensAreRecoverableAndLegacySongsRemainReadable() throws {
        let root = try fixture(), directory = root.appendingPathComponent("20260920-100000/song1")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data().write(to: directory.appendingPathComponent("semantic.npy"))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("plan_manifest.json"))
        let song = try XCTUnwrap(Song.scan(root).first)
        XCTAssertEqual(song.status, .stalled); XCTAssertTrue(song.canRender)
        XCTAssertEqual(song.title, "Song 1")
    }

    @MainActor func testWorkerMetadataAndSplitEvents() throws {
        let backend = Backend()
        let started: [String: Any] = ["event": "started", "output": "/tmp/run", "songs": [["path": "/tmp/run/song1/audio.flac", "index": 1, "title": "A Cover", "style": "Jazz", "lyrics": "Words", "quality": "draft", "kind": "COVER", "source_path": "/tmp/original.wav"]]]
        var data = try JSONSerialization.data(withJSONObject: started); data.append(10)
        backend.consume(data.prefix(20)); XCTAssertTrue(backend.songs.isEmpty)
        backend.consume(data.dropFirst(20))
        XCTAssertTrue(backend.busy)
        let source = backend.source(for: try XCTUnwrap(backend.songs.first))
        XCTAssertEqual(source.lyrics, "Words"); XCTAssertEqual(source.style, "Jazz")
        XCTAssertEqual(backend.songs.first?.quality, "draft")
        let finished: [String: Any] = ["event": "song", "path": source.path, "title": "A Cover", "kind": "COVER", "style": "Jazz", "lyrics": "Words", "source_path": "/tmp/original.wav", "quality": "draft"]
        var final = try JSONSerialization.data(withJSONObject: finished); final.append(10); backend.consume(final)
        XCTAssertFalse(backend.busy); XCTAssertEqual(backend.songs.first?.kind, "COVER")
        XCTAssertEqual(backend.songs.first?.sourcePath, "/tmp/original.wav")
    }

    @MainActor func testDisconnectedCommandReportsErrorWithoutQueueing() {
        let backend = Backend()
        XCTAssertFalse(backend.send(["cmd": "generate"]))
        XCTAssertNotNil(backend.errorMessage); XCTAssertFalse(backend.busy)
    }

    @MainActor func testExistingRuntimeSurvivesOrdinaryAppVersionChanges() {
        XCTAssertTrue(Installer.installationIsUsable(pythonPresent: true, modelsPresent: true, installedSchema: nil))
        XCTAssertTrue(Installer.installationIsUsable(pythonPresent: true, modelsPresent: true, installedSchema: Installer.runtimeSchema))
        XCTAssertFalse(Installer.installationIsUsable(pythonPresent: false, modelsPresent: true, installedSchema: Installer.runtimeSchema))
        XCTAssertFalse(Installer.installationIsUsable(pythonPresent: true, modelsPresent: false, installedSchema: Installer.runtimeSchema))
        XCTAssertFalse(Installer.installationIsUsable(pythonPresent: true, modelsPresent: true, installedSchema: Installer.runtimeSchema + 1))
    }

    func testCoverComponentsHaveIndependentInstallableRoles() {
        XCTAssertEqual(CoverComponent.allCases.count, 6)
        XCTAssertEqual(CoverComponent.allCases.filter(\.required).map(\.rawValue), ["melody"])
        XCTAssertTrue(CoverComponent.mlxWhisper.detail.contains("falls back"))
        XCTAssertTrue(CoverComponent.genre.detail.contains("excerpts"))
        XCTAssertNotEqual(CoverComponent.melody.modelPaths, CoverComponent.lyrics.modelPaths)
        XCTAssertNotEqual(CoverComponent.mlxWhisper.marker, CoverComponent.style.marker)
    }

    @MainActor func testInstallCompletionAllowsImmediateAnalysisWithoutClearingNewState() {
        let backend = Backend()
        backend.coverBusy = true
        backend.coverStatus = "Installing"
        backend.finishCoverInstallation(.success(())) { result in
            XCTAssertFalse(backend.coverBusy)
            XCTAssertNil(backend.coverTask)
            XCTAssertEqual(backend.coverStatus, "")
            backend.coverBusy = true
            backend.coverStatus = "Analyzing"
        }
        XCTAssertTrue(backend.coverBusy)
        XCTAssertEqual(backend.coverStatus, "Analyzing")
    }

    @MainActor func testPlayerPauseSeekAndForget() async throws {
        let root = try fixture(), file = root.appendingPathComponent("play.wav")
        try writeAudio(file, frames: 240000)
        let song = Song(run: "test", index: 1, path: file.path, score: "", seconds: 10, seed: 0, truncated: false, status: .ready)
        let players = Players(); players.volume = 0 // exercise transport without playing test sound aloud
        players.toggle(song)
        for _ in 0..<30 where !players.isPlaying { try await Task.sleep(nanoseconds: 100_000_000) }
        XCTAssertTrue(players.isPlaying)
        players.toggle(song)
        try await Task.sleep(nanoseconds: 200_000_000)
        XCTAssertFalse(players.isPlaying); XCTAssertEqual(players.currentSong?.id, song.id)
        players.seek(4)
        try await Task.sleep(nanoseconds: 400_000_000)
        XCTAssertEqual(players.position, 4, accuracy: 0.4)
        players.forget(song); XCTAssertNil(players.currentSong); XCTAssertEqual(players.position, 0)
    }
    func testTitleFallbackSkipsSectionsAndSupportsInstrumentals() {
        XCTAssertEqual(TitleSuggester.fallback(lyrics: "[Verse]\n\nWalking home under autumn skies tonight", style: "", instrumental: false), "Walking Home Under Autumn Skies")
        XCTAssertEqual(TitleSuggester.fallback(lyrics: "Old song words", style: "warm acoustic piano ballad with strings", instrumental: true), "Warm Acoustic Piano Ballad")
        XCTAssertEqual(TitleSuggester.fallback(lyrics: "[Outro]", style: "cinematic ambient", instrumental: false), "Cinematic Ambient")
    }

    func testUpstreamResultMetadataAndForkRenamePrecedence() throws {
        let root = try fixture(), directory = root.appendingPathComponent("20260921-120000-Title/song1")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data().write(to: directory.appendingPathComponent("audio.flac"))
        try JSONSerialization.data(withJSONObject: ["title": "Upstream title", "nar_engine": "mlx+remote", "quality": "draft"]).write(to: directory.appendingPathComponent("result.json"))
        var song = try XCTUnwrap(Song.scan(root).first)
        XCTAssertEqual(song.title, "Upstream title")
        XCTAssertEqual(song.engineLabel, "GPU, then iPhone")
        XCTAssertEqual(song.quality, "draft")
        try JSONSerialization.data(withJSONObject: ["title": "My rename", "kind": "COVER", "source_path": "/source.wav"]).write(to: directory.appendingPathComponent("metadata.json"))
        song = try XCTUnwrap(Song.scan(root).first)
        XCTAssertEqual(song.title, "My rename")
        XCTAssertEqual(song.kind, "COVER")
        XCTAssertEqual(song.sourcePath, "/source.wav")
    }

    @MainActor func testRemoteEventsAndMigrationKeepSongMetadata() throws {
        let backend = Backend()
        func event(_ object: [String: Any]) throws {
            var data = try JSONSerialization.data(withJSONObject: object); data.append(10)
            backend.consume(data)
        }
        try event(["event": "remote", "state": "connected", "name": "Test iPhone", "detail": "weights cached"])
        XCTAssertEqual(backend.remoteStatus, "Test iPhone ready · weights cached")
        backend.songs = [Song(run: "test", index: 1, path: "/test/audio.flac", score: "ABC", seconds: 1, seed: 1, truncated: false, status: .synth, fraction: 0.5, title: "Cover", kind: "COVER")]
        try event(["event": "stage", "path": "/test/audio.flac", "stage": "synth", "engine": "mlx+remote"])
        XCTAssertEqual(backend.songs[0].engineLabel, "GPU, then iPhone")
        XCTAssertEqual(backend.songs[0].fraction, 0.5)
        XCTAssertEqual(backend.songs[0].kind, "COVER")
        try event(["event": "remote", "state": "gone", "name": "Test iPhone"])
        XCTAssertEqual(backend.remoteStatus, "Test iPhone disconnected")
    }

    @MainActor func testRemoteOfferBeforeWorkerReadyDoesNotReportError() {
        let backend = Backend()
        backend.useRemote(RemoteBrowser.Phone(name: "Test", host: "127.0.0.1", port: 1234))
        XCTAssertNil(backend.errorMessage)
        XCTAssertFalse(backend.busy)
    }

    @MainActor func testHumTranscriptionKeepsCPUCacheAndSkipsWhisper() {
        let audio = URL(fileURLWithPath: "/tmp/hum.wav"), output = URL(fileURLWithPath: "/tmp/hum-out")
        let mlx = URL(fileURLWithPath: "/tmp/whisper/python")
        let hum = Backend.transcriptionArguments(audio: audio, output: output, hum: true, mlxPython: mlx)
        XCTAssertTrue(hum.contains("--hum"))
        XCTAssertFalse(hum.contains("--mlx-python"))
        XCTAssertTrue(hum.contains(Paths.coverAnalyses.path))
        XCTAssertTrue(hum.contains("fp32")); XCTAssertTrue(hum.contains("cpu"))
        let cover = Backend.transcriptionArguments(audio: audio, output: output, hum: false, mlxPython: mlx)
        XCTAssertFalse(cover.contains("--hum"))
        XCTAssertEqual(Array(cover.suffix(2)), ["--mlx-python", mlx.path])
    }

    @MainActor func testDiscardHumRemovesOnlyItsRecording() throws {
        let root = try fixture(), recording = root.appendingPathComponent("hum.wav"), other = root.appendingPathComponent("source.wav")
        try writeAudio(recording); try writeAudio(other)
        let recorder = HumRecorder(); recorder.state = .done(recording)
        recorder.deactivate() // leaving review must preserve the selected recording
        XCTAssertTrue(FileManager.default.fileExists(atPath: recording.path))
        recorder.discard()
        XCTAssertEqual(recorder.state, .idle)
        XCTAssertFalse(FileManager.default.fileExists(atPath: recording.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: other.path))
    }

}
