import SwiftUI
import AVFoundation

@MainActor
final class Players: ObservableObject {
    private let player = AVPlayer()
    private var timer: Any?
    private var observation: NSKeyValueObservation?
    private var itemObservation: NSKeyValueObservation?
    private var ending: NSObjectProtocol?
    @Published private(set) var currentSong: Song?
    @Published private(set) var isPlaying = false
    @Published private(set) var position = 0.0
    @Published private(set) var duration = 0.0
    @Published var error: String?
    @Published var volume = 0.8 { didSet { player.volume = Float(volume) } }

    init() {
        player.volume = Float(volume)
        observation = player.observe(\.timeControlStatus, options: [.new]) { [weak self] player, _ in
            let playing = player.timeControlStatus == .playing
            Task { @MainActor in self?.isPlaying = playing }
        }
        timer = player.addPeriodicTimeObserver(forInterval: CMTime(seconds: 0.2, preferredTimescale: 600), queue: .main) { [weak self] time in
            Task { @MainActor in
                guard let self else { return }
                if time.seconds.isFinite { self.position = max(0, time.seconds) }
                if let length = self.player.currentItem?.duration.seconds, length.isFinite { self.duration = max(0, length) }
            }
        }
        ending = NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime, object: nil, queue: .main) { [weak self] note in
            guard let item = note.object as? AVPlayerItem else { return }
            Task { @MainActor in
                guard let self, item === self.player.currentItem else { return }
                self.isPlaying = false
                self.position = self.duration
            }
        }
    }

    func toggle(_ song: Song) {
        guard song.status == .ready else { return }
        if currentSong?.id != song.id {
            player.pause()
            currentSong = song; position = 0; duration = song.seconds; error = nil
            let item = AVPlayerItem(url: URL(fileURLWithPath: song.path))
            itemObservation = item.observe(\.status, options: [.new]) { [weak self] item, _ in
                if item.status == .failed {
                    let message = item.error?.localizedDescription ?? "This audio could not be played."
                    Task { @MainActor in self?.error = message; self?.isPlaying = false }
                }
            }
            player.replaceCurrentItem(with: item)
            player.play()
        } else if isPlaying || player.timeControlStatus == .waitingToPlayAtSpecifiedRate {
            player.pause(); isPlaying = false
        } else {
            if duration > 0 && position >= duration - 0.05 { seek(0) }
            player.play()
        }
    }

    func seek(_ seconds: Double) {
        guard currentSong != nil, seconds.isFinite else { return }
        position = min(duration, max(0, seconds))
        player.seek(to: CMTime(seconds: position, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
    }

    func updateMetadata(_ song: Song) { if currentSong?.id == song.id { currentSong = song } }

    func forget(_ song: Song) {
        guard currentSong?.id == song.id else { return }
        player.pause(); player.replaceCurrentItem(with: nil)
        itemObservation = nil; currentSong = nil; isPlaying = false; position = 0; duration = 0
    }

    deinit {
        if let timer { player.removeTimeObserver(timer) }
        if let ending { NotificationCenter.default.removeObserver(ending) }
    }
}
