import SwiftUI
import AVFoundation

struct WaveformView: View {
    let path: String
    @State private var bars: [CGFloat] = []
    var body: some View {
        Canvas { context, size in
            guard !bars.isEmpty else { return }
            let stride = size.width / CGFloat(bars.count)
            for (i, value) in bars.enumerated() {
                let height = max(1, size.height * value)
                let rect = CGRect(x: CGFloat(i) * stride, y: (size.height - height) / 2, width: max(0.5, stride * 0.65), height: height)
                context.fill(Path(roundedRect: rect, cornerRadius: 1), with: .color(.white.opacity(0.7)))
            }
        }
        .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
        .task(id: path) {
            let samples = await Task.detached(priority: .utility) { WaveformSamples.load(path) }.value
            guard !Task.isCancelled else { return }
            bars = samples
        }
    }
}

enum WaveformSamples {
    /// Sample windows across the entire song, including very short imported files.
    static func load(_ path: String, count: Int = 32) -> [CGFloat] {
        guard count > 0, let file = try? AVAudioFile(forReading: URL(fileURLWithPath: path)), file.length > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 2048) else { return [] }
        var peaks: [CGFloat] = []
        for i in 0..<count {
            file.framePosition = min(file.length - 1, AVAudioFramePosition(Double(file.length) * Double(i) / Double(count)))
            guard (try? file.read(into: buffer, frameCount: AVAudioFrameCount(min(2048, file.length - file.framePosition)))) != nil,
                  let channels = buffer.floatChannelData else { return [] }
            var peak: Float = 0
            for channel in 0..<Int(file.processingFormat.channelCount) {
                for frame in 0..<Int(buffer.frameLength) { let value = abs(channels[channel][frame]); if value.isFinite { peak = max(peak, value) } }
            }
            peaks.append(CGFloat(min(1, peak)))
        }
        return peaks
    }
}
