import SwiftUI
import AVFoundation

struct VoiceMemo: Identifiable, Codable, Equatable {
    let id: String
    var title: String
    let createdAt: Date
    let seconds: Double
    let fileName: String
    var source: String = "iPhone"
}

@MainActor
final class VoiceMemoStore: ObservableObject {
    @Published private(set) var memos: [VoiceMemo] = []
    let root: URL
    private var indexURL: URL { root.appendingPathComponent("index.json") }

    init(root: URL? = nil) {
        self.root = root ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("Voice Memos", isDirectory: true)
        load()
    }

    func add(_ source: URL, title: String = "Vocal idea") -> VoiceMemo? {
        do {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let id = UUID().uuidString
            let fileName = "\(id).m4a"
            let destination = root.appendingPathComponent(fileName)
            try FileManager.default.copyItem(at: source, to: destination)
            let seconds = (try? AVAudioFile(forReading: destination)).map { Double($0.length) / $0.processingFormat.sampleRate } ?? 0
            let memo = VoiceMemo(id: id, title: title, createdAt: Date(), seconds: seconds, fileName: fileName)
            memos.insert(memo, at: 0); save(); return memo
        } catch { return nil }
    }

    func url(for memo: VoiceMemo) -> URL { root.appendingPathComponent(memo.fileName) }

    func delete(_ memo: VoiceMemo) {
        try? FileManager.default.removeItem(at: url(for: memo))
        memos.removeAll { $0.id == memo.id }; save()
    }

    private func load() {
        guard let data = try? Data(contentsOf: indexURL), let value = try? JSONDecoder().decode([VoiceMemo].self, from: data) else { return }
        memos = value.filter { FileManager.default.fileExists(atPath: url(for: $0).path) }
    }

    private func save() {
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        guard let data = try? JSONEncoder().encode(memos) else { return }
        try? data.write(to: indexURL, options: .atomic)
    }
}

@MainActor
final class VoiceMemoRecorder: NSObject, ObservableObject {
    enum State: Equatable { case idle, denied, recording, failed(String) }
    @Published var state: State = .idle
    @Published var seconds = 0.0
    @Published var level: Float = 0
    @Published private(set) var inputs: [AVAudioSessionPortDescription] = []
    @Published var selectedInputUID: String?
    private let store: VoiceMemoStore
    private let session = AVAudioSession.sharedInstance()
    private var recorder: AVAudioRecorder?
    private var timer: Timer?

    init(store: VoiceMemoStore) {
        self.store = store
        super.init()
        refreshInputs()
    }

    func refreshInputs() {
        inputs = session.availableInputs ?? []
        if selectedInputUID == nil { selectedInputUID = inputs.first?.uid }
    }

    func start() {
        AVAudioApplication.requestRecordPermission { [weak self] granted in
            Task { @MainActor in
                guard let self else { return }
                granted ? self.begin() : (self.state = .denied)
            }
        }
    }

    private func begin() {
        do {
            try session.setCategory(.record, mode: .measurement, options: [.allowBluetoothHFP])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
            if let uid = selectedInputUID, let input = inputs.first(where: { $0.uid == uid }) { try session.setPreferredInput(input) }
            let dir = FileManager.default.temporaryDirectory.appendingPathComponent("yue-voice-memos", isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let url = dir.appendingPathComponent("memo-\(UUID().uuidString).m4a")
            let settings: [String: Any] = [AVFormatIDKey: Int(kAudioFormatMPEG4AAC), AVSampleRateKey: 44100.0,
                                           AVNumberOfChannelsKey: 1, AVEncoderBitRateKey: 96000]
            let r = try AVAudioRecorder(url: url, settings: settings)
            r.isMeteringEnabled = true
            guard r.record() else { throw CocoaError(.fileWriteUnknown) }
            recorder = r; seconds = 0; level = 0; state = .recording
            timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in Task { @MainActor in self?.tick() } }
        } catch { state = .failed(error.localizedDescription); try? session.setActive(false) }
    }

    private func tick() {
        guard let recorder else { return }
        recorder.updateMeters(); seconds = recorder.currentTime
        let db = recorder.averagePower(forChannel: 0)
        level = min(1, max(0, (db + 60) / 57))
    }

    func stop() {
        timer?.invalidate(); timer = nil
        guard let recorder else { return }
        let url = recorder.url; recorder.stop(); self.recorder = nil
        try? session.setActive(false, options: .notifyOthersOnDeactivation)
        guard seconds >= 0.4, store.add(url) != nil else {
            try? FileManager.default.removeItem(at: url); state = .failed("Record a little longer so the idea can be saved."); return
        }
        try? FileManager.default.removeItem(at: url); state = .idle
    }

    func cancel() {
        timer?.invalidate(); timer = nil
        if let recorder { let url = recorder.url; recorder.stop(); try? FileManager.default.removeItem(at: url) }
        self.recorder = nil; try? session.setActive(false, options: .notifyOthersOnDeactivation)
        state = .idle; seconds = 0; level = 0
    }
}

struct VoiceMemosView: View {
    @ObservedObject var store: VoiceMemoStore
    @ObservedObject var server: MemoServer
    @StateObject private var recorder: VoiceMemoRecorder

    init(store: VoiceMemoStore, server: MemoServer) {
        self.store = store; self.server = server
        _recorder = StateObject(wrappedValue: VoiceMemoRecorder(store: store))
    }

    var body: some View {
        List {
            Section {
                HStack {
                    Image(systemName: recorder.state == .recording ? "record.circle.fill" : "waveform.and.mic").foregroundStyle(recorder.state == .recording ? Color.red : Color.accentColor)
                    VStack(alignment: .leading) {
                        Text(recorder.state == .recording ? "Recording \(recorder.seconds, format: .number.precision(.fractionLength(1))) s" : "Capture a vocal idea")
                        Text(server.state).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if recorder.state == .recording { Button("Stop") { recorder.stop() }.buttonStyle(.borderedProminent) }
                    else { Button("Record") { recorder.start() }.buttonStyle(.borderedProminent) }
                }
                if recorder.state == .recording {
                    ProgressView(value: Double(recorder.level)).tint(recorder.level > 0.88 ? .red : .green)
                    Text("Input level \(Int(recorder.level * 100))%").font(.caption2).foregroundStyle(.secondary)
                }
                if case .denied = recorder.state { Text("Enable Microphone in Settings to record vocal ideas.").font(.caption).foregroundStyle(.red) }
                if case .failed(let message) = recorder.state { Text(message).font(.caption).foregroundStyle(.red) }
                if !recorder.inputs.isEmpty {
                    Picker("Microphone", selection: Binding(get: { recorder.selectedInputUID ?? recorder.inputs[0].uid }, set: { recorder.selectedInputUID = $0 })) {
                        ForEach(recorder.inputs, id: \.uid) { input in Text(input.portName).tag(input.uid) }
                    }.pickerStyle(.menu)
                    Button("Refresh inputs") { recorder.refreshInputs() }.font(.caption)
                }
            } header: { Text("New memo") }

            Section("Saved ideas") {
                if store.memos.isEmpty { Text("No vocal ideas yet.").foregroundStyle(.secondary) }
                ForEach(store.memos) { memo in
                    HStack {
                        Image(systemName: "waveform").foregroundStyle(Color.accentColor)
                        VStack(alignment: .leading) { Text(memo.title); Text("\(memo.seconds, format: .number.precision(.fractionLength(1))) s · \(memo.createdAt.formatted(date: .abbreviated, time: .shortened))").font(.caption).foregroundStyle(.secondary) }
                        Spacer(); Button(role: .destructive) { store.delete(memo) } label: { Image(systemName: "trash") }.buttonStyle(.borderless)
                    }
                }
            }
        }
        .navigationTitle("Voice ideas")
        .onDisappear { if recorder.state == .recording { recorder.cancel() } }
    }
}
