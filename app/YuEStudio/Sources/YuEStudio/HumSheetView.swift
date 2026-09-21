import SwiftUI
import AVFoundation

/// Records a hummed melody from the microphone (mono WAV under the output folder's "hums"),
/// then hands it to the transcription flow in hum mode.
@MainActor
final class HumRecorder: NSObject, ObservableObject {
    enum State: Equatable { case idle, denied, recording, done(URL), failed(String) }
    static let maxSeconds = 30.0
    @Published var state: State = .idle
    @Published var seconds = 0.0
    @Published var level: Float = 0            // 0…1
    private var active = true
    private var recorder: AVAudioRecorder?
    private var timer: Timer?

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
        let dir = Paths.output.appendingPathComponent("hums")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let f = DateFormatter(); f.dateFormat = "yyyyMMdd-HHmmss"
        let url = dir.appendingPathComponent("hum-\(f.string(from: Date()))-\(UUID().uuidString).wav")
        let settings: [String: Any] = [AVFormatIDKey: Int(kAudioFormatLinearPCM), AVSampleRateKey: 44100.0, AVNumberOfChannelsKey: 1,
                                       AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false]
        do {
            let r = try AVAudioRecorder(url: url, settings: settings)
            r.isMeteringEnabled = true
            guard r.record() else { state = .failed("The microphone could not start."); return }
            recorder = r; seconds = 0; level = 0; state = .recording
            timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in Task { @MainActor in self?.tick() } }
        } catch { state = .failed(error.localizedDescription) }
    }

    private func tick() {
        guard let r = recorder else { return }
        r.updateMeters()
        seconds = r.currentTime
        level = max(0, min(1, (r.averagePower(forChannel: 0) + 50) / 50))
        if seconds >= Self.maxSeconds { stop() }
    }

    func stop() {
        timer?.invalidate(); timer = nil
        guard let r = recorder else { return }
        let url = r.url
        r.stop(); recorder = nil
        if seconds < 2 {
            try? FileManager.default.removeItem(at: url)
            state = .failed("Too short: hum for at least a few seconds.")
        } else {
            state = .done(url)
        }
    }

    func deactivate() {
        active = false
        if state == .recording { stop() }
    }

    func discard() {
        if case .done(let url) = state { try? FileManager.default.removeItem(at: url) }
        state = .idle; seconds = 0; level = 0
    }
}

struct HumSheetView: View {
    let onRecorded: (URL) -> Void
    @Environment(\.dismiss) private var dismiss
    @StateObject private var rec = HumRecorder()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Hum a melody").font(.headline)
            Text("Hum or sing 10 to 30 seconds of tune on its own, no backing. SheetSage2 transcribes it into a melody score, and the song is built around it: either exactly your tune, or your tune as the opening that the planner continues.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Group {
                switch rec.state {
                case .idle:
                    Button { rec.start() } label: { Label("Record", systemImage: "mic.fill") }.buttonStyle(.borderedProminent)
                case .denied:
                    Text("Microphone access was refused. Allow YuE Studio in System Settings → Privacy & Security → Microphone, then try again.").foregroundStyle(.red)
                case .recording:
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Circle().fill(.red).frame(width: 10, height: 10)
                            Text(String(format: "Recording  %.0f s", rec.seconds)).monospacedDigit()
                            Spacer()
                            Button("Stop") { rec.stop() }.buttonStyle(.borderedProminent)
                        }
                        ProgressView(value: Double(rec.level))
                        ProgressView(value: rec.seconds, total: HumRecorder.maxSeconds).tint(.secondary)
                    }
                case .done:
                    HStack {
                        Text(String(format: "Recorded %.0f s", rec.seconds)).monospacedDigit()
                        Button("Record again") { rec.discard(); rec.start() }
                    }
                case .failed(let why):
                    HStack { Text(why).foregroundStyle(.red); Button("Try again") { rec.discard(); rec.start() } }
                }
            }
            Spacer()
            HStack {
                Spacer()
                Button("Cancel") { if rec.state == .recording { rec.stop() }; rec.discard(); dismiss() }.keyboardShortcut(.cancelAction)
                if case .done(let url) = rec.state {
                    Button("Transcribe") { dismiss(); onRecorded(url) }.buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                }
            }
        }
        .padding().frame(width: 520, height: 300)
        .preferredColorScheme(.dark)
        .interactiveDismissDisabled(rec.state == .recording)
        .onDisappear { rec.deactivate() }
    }
}
