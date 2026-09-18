import Foundation
import AVFoundation
import Combine

@MainActor
final class Players: ObservableObject {
    /// One player at a time, with transport state for the bar under the song list.
    private var player: AVPlayer?
    private var timeObserver: Any?
    private var endObserver: NSObjectProtocol?
    @Published var playing: String?              // song id when playing
    @Published var current: Song?                // the song loaded in the player (playing or paused)
    @Published var currentTime = 0.0
    @Published var duration = 0.0
    @Published var isScrubbing = false
    var isPaused: Bool { current != nil && playing == nil }

    func toggle(_ song: Song) {
        guard song.status == .ready else { return }
        if current?.id == song.id {
            if playing != nil { pause() } else { resume() }
            return
        }
        load(song); resume()
    }
    func pause() { player?.pause(); playing = nil }
    func resume() {
        guard let player, let current else { return }
        if duration > 0 && currentTime >= duration - 0.05 { player.seek(to: .zero) }
        player.play(); playing = current.id
    }
    func seek(to seconds: Double) {
        guard let player else { return }
        currentTime = max(0, min(seconds, duration))
        player.seek(to: CMTime(seconds: currentTime, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
    }
    func skip(_ delta: Double) { seek(to: currentTime + delta) }
    func rewind() { seek(to: 0) }
    /// A re-rendered file must not be served from the old player.
    func forget(_ song: Song) { if current?.id == song.id { unload() } }

    private func load(_ song: Song) {
        unload()
        let item = AVPlayerItem(url: URL(fileURLWithPath: song.path))
        let p = AVPlayer(playerItem: item)
        player = p; current = song; currentTime = 0
        duration = song.seconds
        timeObserver = p.addPeriodicTimeObserver(forInterval: CMTime(seconds: 0.25, preferredTimescale: 600), queue: .main) { [weak self] t in
            guard let self, !self.isScrubbing else { return }
            self.currentTime = t.seconds
            let d = item.duration.seconds
            if d.isFinite && d > 0 { self.duration = d }
        }
        endObserver = NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime, object: item, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.playing = nil; self?.currentTime = self?.duration ?? 0 }
        }
    }
    private func unload() {
        if let timeObserver, let player { player.removeTimeObserver(timeObserver) }
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        timeObserver = nil; endObserver = nil
        player?.pause(); player = nil; current = nil; playing = nil; currentTime = 0; duration = 0
    }
}
