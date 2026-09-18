import SwiftUI

/// Transport bar for the loaded song: back to start, skip, play/pause, scrub with times.
struct TransportBar: View {
    @ObservedObject var players: Players
    var body: some View {
        let song = players.current
        let loaded = song != nil
            HStack(spacing: 10) {
                Button(action: { players.rewind() }) { Image(systemName: "backward.end.fill") }.help("Back to start")
                Button(action: { players.skip(-10) }) { Image(systemName: "gobackward.10") }.help("Back 10 seconds")
                Button(action: { players.playing != nil ? players.pause() : players.resume() }) {
                    Image(systemName: players.playing != nil ? "pause.fill" : "play.fill").frame(width: 16)
                }.keyboardShortcut(.space, modifiers: []).help("Play or pause")
                Button(action: { players.skip(10) }) { Image(systemName: "goforward.10") }.help("Forward 10 seconds")
                Text(clock(players.currentTime)).monospacedDigit().font(.caption)
                Slider(value: Binding(get: { players.currentTime }, set: { players.currentTime = $0 }),
                       in: 0...max(players.duration, 0.1),
                       onEditingChanged: { editing in
                           players.isScrubbing = editing
                           if !editing { players.seek(to: players.currentTime) }
                       })
                Text(clock(players.duration)).monospacedDigit().font(.caption)
                Text(song.map { $0.rowName + ($0.quality == "draft" ? " · draft" : "") } ?? "Press a song's play button to load it")
                    .font(.caption).foregroundStyle(loaded ? .primary : .secondary).lineLimit(1).truncationMode(.middle).frame(maxWidth: 260, alignment: .trailing)
            }
            .buttonStyle(.borderless)
            .disabled(!loaded)
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(.bar)
    }
    private func clock(_ t: Double) -> String {
        let s = Int(t.rounded(.down)); return String(format: "%d:%02d", s / 60, s % 60)
    }
}
