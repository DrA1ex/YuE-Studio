import SwiftUI
import AVFoundation
import AppKit
import UniformTypeIdentifiers

struct StageTrack: View {
    let progress: Double                         // 0 (queued) ... 4 (done): whole stages plus progress within the current one
    static let segments = ["Planning", "Tokenizing", "Synthing", "Rendering"]
    static let total = Double(segments.count)
    var body: some View {
        GeometryReader { g in
            let w = max(1, g.size.width - 8)
            let x = { (units: Double) in w * units / Self.total }
            let filled = x(max(0, min(Self.total, progress)))
            ZStack(alignment: .topLeading) {
                ForEach(Self.segments.indices, id: \.self) { i in
                    Text(Self.segments[i]).font(.caption2).foregroundStyle(.secondary).lineLimit(1).minimumScaleFactor(0.7)
                        .frame(width: x(1)).position(x: x(Double(i) + 0.5), y: 7)
                }
                Capsule().fill(Color.secondary.opacity(0.18)).frame(width: w, height: 8).position(x: w / 2, y: 24)
                Capsule().fill(Color.green).frame(width: filled, height: 8).position(x: filled / 2, y: 24)
                ForEach(0...Self.segments.count, id: \.self) { i in
                    Rectangle().fill(Color.primary.opacity(0.7)).frame(width: 2, height: 18).position(x: x(Double(i)), y: 24)
                }
            }
        }
        .frame(maxWidth: .infinity).frame(height: 36)
    }
}
