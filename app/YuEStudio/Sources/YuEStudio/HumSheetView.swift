import SwiftUI
import AVFoundation

/// Records a hummed melody from the selected microphone and keeps the WAV until the
/// user either sends it to transcription or explicitly discards it.
@MainActor
final class HumRecorder: NSObject, ObservableObject {
    enum State: Equatable { case idle, denied, recording, done(URL), failed(String) }
    static let maxSeconds = 30.0
    @Published var state: State = .idle
    @Published var seconds = 0.0
    @Published var level: Float = 0
    @Published private(set) var devices: [MicrophoneDevice] = []
    @Published var selectedDeviceID: AudioDeviceID?
    private var active = true
    private var recorder: AVAudioRecorder?
    private var timer: Timer?
    private var previousDefault: AudioDeviceID?

    override init() {
        super.init()
        refreshDevices()
    }

    func refreshDevices() {
        devices = MicrophoneCatalog.devices()
        selectedDeviceID = devices.first(where: { $0.isDefault })?.id ?? devices.first?.id
    }

    func start() {
        active = true
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            Task { @MainActor in
                guard self.active else { return }
                granted ? self.begin() : (self.state = .denied)
            }
        }
    }

    private func begin() {
        if let id = selectedDeviceID, let device = devices.first(where: { $0.id == id }) {
            previousDefault = MicrophoneCatalog.defaultInput()
            _ = MicrophoneCatalog.select(device)
        }
        let dir = Paths.output.appendingPathComponent("hums", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let f = DateFormatter(); f.dateFormat = "yyyyMMdd-HHmmss"
        let url = dir.appendingPathComponent("hum-\(f.string(from: Date()))-\(UUID().uuidString).wav")
        let settings: [String: Any] = [AVFormatIDKey: Int(kAudioFormatLinearPCM), AVSampleRateKey: 44100.0,
                                       AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
                                       AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false]
        do {
            let r = try AVAudioRecorder(url: url, settings: settings)
            r.isMeteringEnabled = true
            guard r.record() else { state = .failed("The microphone could not start."); restoreDefault(); return }
            recorder = r; seconds = 0; level = 0; state = .recording
            timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
                Task { @MainActor in self?.tick() }
            }
        } catch { state = .failed(error.localizedDescription); restoreDefault() }
    }

    private func tick() {
        guard let r = recorder else { return }
        r.updateMeters()
        seconds = r.currentTime
        level = AudioLevelMeter.normalized(decibels: r.averagePower(forChannel: 0))
        if seconds >= Self.maxSeconds { stop() }
    }

    func stop() {
        timer?.invalidate(); timer = nil
        guard let r = recorder else { return }
        let url = r.url
        r.stop(); recorder = nil
        restoreDefault()
        if seconds < 2 {
            try? FileManager.default.removeItem(at: url)
            state = .failed("Too short: hum for at least two seconds.")
        } else {
            state = .done(url)
        }
    }

    func deactivate() {
        active = false
        if state == .recording { stop() }
        restoreDefault()
    }

    func discard() {
        if case .done(let url) = state { try? FileManager.default.removeItem(at: url) }
        state = .idle; seconds = 0; level = 0
    }

    private func restoreDefault() {
        guard let id = previousDefault else { return }
        _ = MicrophoneCatalog.select(MicrophoneDevice(id: id, name: "previous input"))
        previousDefault = nil
    }
}

struct MicLevelMeter: View {
    let level: Float
    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Capsule().fill(.white.opacity(0.09))
                Capsule().fill(level > 0.88 ? .red : (level > 0.68 ? .orange : .green))
                    .frame(width: proxy.size.width * CGFloat(min(1, max(0, level))))
            }
        }.frame(height: 10)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Microphone level")
        .accessibilityValue("\(Int(level * 100)) percent")
    }
}

struct HumSheetView: View {
    let onRecorded: (URL) -> Void
    @ObservedObject private var memoStore: VoiceMemoStore
    @ObservedObject private var memoRemote: VoiceMemoRemote
    @Environment(\.dismiss) private var dismiss
    @StateObject private var rec = HumRecorder()

    init(memoStore: VoiceMemoStore, memoRemote: VoiceMemoRemote, onRecorded: @escaping (URL) -> Void) {
        self.onRecorded = onRecorded
        self._memoStore = ObservedObject(wrappedValue: memoStore)
        self._memoRemote = ObservedObject(wrappedValue: memoRemote)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label("Hum a melody", systemImage: "waveform.and.mic").font(.title3.bold())
                Spacer(); Text("Melody input").font(.caption).foregroundStyle(.secondary)
            }
            Text("Record a clean 10–30 second vocal idea. YuE Studio converts it into an editable melody and can continue from your opening.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            micPicker
            Group {
                switch rec.state {
                case .idle: recordButton
                case .denied:
                    Text("Microphone access was refused. Allow YuE Studio in System Settings → Privacy & Security → Microphone, then try again.").foregroundStyle(.red)
                case .recording: recordingPanel
                case .done:
                    HStack {
                        Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                        Text(String(format: "Recorded %.1f s", rec.seconds)).monospacedDigit(); Spacer()
                        Button("Record again") { rec.discard(); rec.start() }
                    }
                case .failed(let why):
                    HStack { Text(why).foregroundStyle(.red); Spacer(); Button("Try again") { rec.discard(); rec.start() } }
                }
            }
            if !memoStore.memos.isEmpty {
                Divider(); Text("Saved vocal ideas").font(.caption.bold()).foregroundStyle(.secondary)
                ScrollView {
                    VStack(spacing: 6) {
                        ForEach(memoStore.memos.prefix(4)) { memo in
                            HStack {
                                Image(systemName: "waveform").foregroundStyle(Color.accentColor)
                                VStack(alignment: .leading) {
                                    Text(memo.title).lineLimit(1)
                                    Text("\(memo.source) · \(memo.seconds, format: .number.precision(.fractionLength(0))) s").font(.caption2).foregroundStyle(.secondary)
                                }
                                Spacer()
                                Button("Use") { onRecorded(memoStore.url(for: memo)); dismiss() }.buttonStyle(.bordered)
                            }.padding(8).background(.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
                        }
                    }
                }.frame(maxHeight: 105)
            }
            if !memoRemote.memos.isEmpty || memoRemote.phoneName != nil {
                Divider()
                HStack {
                    Label("iPhone vocal ideas", systemImage: "iphone").font(.caption.bold())
                    Spacer()
                    Button("Refresh") { memoRemote.refresh() }.font(.caption)
                }
                Text(memoRemote.detail).font(.caption2).foregroundStyle(.secondary)
                ForEach(memoRemote.memos.prefix(3)) { memo in
                    HStack {
                        Text(memo.title).lineLimit(1); Spacer()
                        Button("Import & use") {
                            memoRemote.importMemo(memo, into: memoStore) { url in
                                guard let url else { return }
                                onRecorded(url); dismiss()
                            }
                        }.buttonStyle(.bordered)
                    }.font(.caption)
                }
            }
            Spacer(minLength: 0)
            HStack {
                Spacer()
                Button("Cancel") { if rec.state == .recording { rec.stop() }; rec.discard(); dismiss() }.keyboardShortcut(.cancelAction)
                if case .done(let url) = rec.state {
                    Button("Save & transcribe") { _ = memoStore.importRecording(url); dismiss(); onRecorded(url) }
                        .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                }
            }
        }
        .padding(22).frame(width: 560, height: 430)
        .background(Color(red: 0.055, green: 0.055, blue: 0.065))
        .preferredColorScheme(.dark)
        .interactiveDismissDisabled(rec.state == .recording)
        .onDisappear { rec.deactivate() }
    }

    @ViewBuilder private var micPicker: some View {
        if rec.devices.isEmpty {
            Label("No input device detected", systemImage: "mic.slash").font(.caption).foregroundStyle(.secondary)
        } else {
            HStack {
                Image(systemName: "mic").foregroundStyle(.secondary)
                Picker("Microphone", selection: Binding(get: { rec.selectedDeviceID ?? rec.devices[0].id }, set: { rec.selectedDeviceID = $0 })) {
                    ForEach(rec.devices) { device in Text(device.name + (device.isDefault ? " · default" : "")).tag(device.id) }
                }.labelsHidden().frame(maxWidth: .infinity, alignment: .leading)
                Button { rec.refreshDevices() } label: { Image(systemName: "arrow.clockwise") }.buttonStyle(.plain)
            }
        }
    }

    private var recordButton: some View {
        Button { rec.start() } label: { Label("Record vocal idea", systemImage: "record.circle") }
            .buttonStyle(.borderedProminent).controlSize(.large)
    }

    private var recordingPanel: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Circle().fill(.red).frame(width: 10, height: 10)
                Text(String(format: "Recording  %.1f s", rec.seconds)).monospacedDigit(); Spacer()
                Button("Stop") { rec.stop() }.buttonStyle(.borderedProminent)
            }
            MicLevelMeter(level: rec.level)
            HStack { Text("Live input").font(.caption2).foregroundStyle(.secondary); Spacer(); Text("\(Int(rec.level * 100))%").font(.caption2.monospaced()).foregroundStyle(.secondary) }
            ProgressView(value: rec.seconds, total: HumRecorder.maxSeconds).tint(.secondary)
        }.padding(12).background(.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 10))
    }
}
