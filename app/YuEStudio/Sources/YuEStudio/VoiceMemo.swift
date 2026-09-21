import Foundation
import AVFoundation
import AppKit
import CoreAudio

/// Shared, testable audio-level mapping used by the microphone UI.
enum AudioLevelMeter {
    static func normalized(decibels: Float, floor: Float = -60, ceiling: Float = -3) -> Float {
        guard decibels.isFinite else { return 0 }
        return min(1, max(0, (decibels - floor) / (ceiling - floor)))
    }
}

struct MicrophoneDevice: Identifiable, Equatable {
    let id: AudioDeviceID
    let name: String
    var isDefault: Bool = false
}

enum MicrophoneCatalog {
    static func devices() -> [MicrophoneDevice] {
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices,
                                                  mScope: kAudioObjectPropertyScopeGlobal,
                                                  mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr else { return [] }
        let count = Int(size) / MemoryLayout<AudioDeviceID>.stride
        guard count > 0 else { return [] }
        var ids = [AudioDeviceID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids) == noErr else { return [] }
        let current = defaultInput()
        return ids.compactMap { id in
            guard hasInput(id), let name = name(for: id) else { return nil }
            return MicrophoneDevice(id: id, name: name, isDefault: id == current)
        }.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    static func defaultInput() -> AudioDeviceID {
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultInputDevice,
                                                  mScope: kAudioObjectPropertyScopeGlobal,
                                                  mElement: kAudioObjectPropertyElementMain)
        var id = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        _ = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &id)
        return id
    }

    @discardableResult
    static func select(_ device: MicrophoneDevice) -> Bool {
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultInputDevice,
                                                  mScope: kAudioObjectPropertyScopeGlobal,
                                                  mElement: kAudioObjectPropertyElementMain)
        var id = device.id
        let size = UInt32(MemoryLayout<AudioDeviceID>.size)
        return AudioObjectSetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, size, &id) == noErr
    }

    private static func hasInput(_ id: AudioDeviceID) -> Bool {
        var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreamConfiguration,
                                                  mScope: kAudioDevicePropertyScopeInput,
                                                  mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size) == noErr, size > 0 else { return false }
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, raw) == noErr else { return false }
        let list = raw.assumingMemoryBound(to: AudioBufferList.self)
        return list.pointee.mNumberBuffers > 0
    }

    private static func name(for id: AudioDeviceID) -> String? {
        var address = AudioObjectPropertyAddress(mSelector: kAudioObjectPropertyName,
                                                  mScope: kAudioObjectPropertyScopeGlobal,
                                                  mElement: kAudioObjectPropertyElementMain)
        var value: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, &value) == noErr,
              let value else { return nil }
        return value.takeUnretainedValue() as String
    }
}

struct VoiceMemo: Identifiable, Codable, Equatable {
    let id: String
    var title: String
    let createdAt: Date
    let seconds: Double
    let fileName: String
    var source: String = "Mac"
}

@MainActor
final class VoiceMemoStore: ObservableObject {
    @Published private(set) var memos: [VoiceMemo] = []
    let root: URL
    private var indexURL: URL { root.appendingPathComponent("index.json") }

    init(root: URL = Paths.output.appendingPathComponent("Voice Memos", isDirectory: true)) {
        self.root = root
        load()
    }

    func importRecording(_ url: URL, title: String = "Vocal idea", source: String = "Mac") -> VoiceMemo? {
        do {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let id = UUID().uuidString
            let fileName = "\(id).wav"
            let destination = root.appendingPathComponent(fileName)
            try FileManager.default.copyItem(at: url, to: destination)
            let seconds = (try? AVAudioFile(forReading: destination)).map { Double($0.length) / $0.processingFormat.sampleRate } ?? 0
            let memo = VoiceMemo(id: id, title: title, createdAt: Date(), seconds: seconds, fileName: fileName, source: source)
            memos.insert(memo, at: 0)
            save()
            return memo
        } catch { return nil }
    }

    func addTransferred(_ memo: VoiceMemo, data: Data) -> VoiceMemo? {
        do {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let destination = root.appendingPathComponent(memo.fileName)
            try data.write(to: destination, options: .atomic)
            let local = VoiceMemo(id: memo.id, title: memo.title, createdAt: memo.createdAt,
                                  seconds: memo.seconds, fileName: memo.fileName, source: memo.source)
            memos.removeAll { $0.id == local.id }
            memos.insert(local, at: 0)
            save()
            return local
        } catch { return nil }
    }

    func url(for memo: VoiceMemo) -> URL { root.appendingPathComponent(memo.fileName) }

    func delete(_ memo: VoiceMemo) {
        try? FileManager.default.removeItem(at: url(for: memo))
        memos.removeAll { $0.id == memo.id }
        save()
    }

    private func load() {
        guard let data = try? Data(contentsOf: indexURL),
              let value = try? JSONDecoder().decode([VoiceMemo].self, from: data) else { return }
        memos = value.filter { FileManager.default.fileExists(atPath: url(for: $0).path) }
    }

    private func save() {
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        guard let data = try? JSONEncoder().encode(memos) else { return }
        try? data.write(to: indexURL, options: .atomic)
    }
}
