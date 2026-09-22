import Foundation

/// Optional pieces of the local cover-analysis engine.
///
/// The melody component is the only required piece for creating a cover. All
/// other components improve the analysis and can be installed independently.
enum CoverComponent: String, CaseIterable, Identifiable, Hashable {
    case melody
    case lyrics
    case genre
    case style
    case mlxWhisper
    case vocalActivity

    var id: String { rawValue }

    var title: String {
        switch self {
        case .melody: "Melody and score"
        case .lyrics: "Lyrics transcription"
        case .genre: "Genre classifier"
        case .style: "Detailed style analysis"
        case .mlxWhisper: "MLX Whisper acceleration"
        case .vocalActivity: "Vocal activity detection"
        }
    }

    var detail: String {
        switch self {
        case .melody: "SheetSage2 and MERT identify the source melody and create the ABC score required for covers."
        case .lyrics: "Whisper large-v3-turbo transcribes lyrics with timestamped segments and section formatting."
        case .genre: "Classifies several audio excerpts to suggest a more reliable primary or hybrid genre."
        case .style: "CLAP compares excerpts with instruments, vocals, mood, production and tempo descriptors."
        case .mlxWhisper: "Optional Apple Silicon Whisper backend. It is preferred automatically and falls back to Transformers."
        case .vocalActivity: "Hybrid Demucs finds likely singing windows so Whisper spends less time on instrumental passages. Download is about 319 MB."
        }
    }

    var models: [String] {
        switch self {
        case .melody:
            ["m-a-p/SheetSage2", "m-a-p/MERT-v2-FullSong"]
        case .lyrics:
            ["openai/whisper-large-v3-turbo", "openai/whisper-small (legacy, if present)"]
        case .genre:
            ["dima806/music_genres_classification"]
        case .style:
            ["laion/clap-htsat-unfused"]
        case .mlxWhisper:
            ["mlx-community/whisper-large-v3-turbo"]
        case .vocalActivity:
            ["torchaudio Hybrid Demucs · HDEMUCS_HIGH_MUSDB"]
        }
    }

    var relationship: String {
        switch self {
        case .melody:
            "SheetSage2 and MERT2 are used together; both are required for source-audio covers."
        case .lyrics:
            "CPU/Transformers lyrics backend. This is an alternative to MLX Whisper, not an additional requirement."
        case .genre:
            "Independent optional stage. It is not required by melody, lyrics, or detailed style analysis."
        case .style:
            "Independent optional CLAP stage for instrumentation, vocals, mood, production, and pace descriptors."
        case .mlxWhisper:
            "Apple Silicon lyrics backend. This is an alternative to the Transformers Whisper model."
        case .vocalActivity:
            "Optional pre-pass used only by lyric transcription to focus Whisper on likely singing regions."
        }
    }

    var required: Bool { self == .melody }

    var installName: String { rawValue }

    var marker: URL {
        Paths.coverSupport.appendingPathComponent("component-\(rawValue).ready")
    }

    var disabledMarker: URL {
        Paths.coverSupport.appendingPathComponent("component-\(rawValue).disabled")
    }

    /// Legacy markers are retained for already installed engines. A disabled
    /// marker wins over a legacy marker after the user removes one component.
    var legacyMarkers: [URL] {
        let marker: String?
        switch self {
        case .melody, .genre: marker = "installed-v3"
        case .style: marker = "installed-v4"
        case .lyrics: marker = "installed-v5"
        case .mlxWhisper: marker = "installed-v6"
        case .vocalActivity: marker = nil
        }
        return marker.map { [Paths.coverSupport.appendingPathComponent($0)] } ?? []
    }

    var modelPaths: [URL] {
        let hub = Paths.coverSupport.appendingPathComponent("models/hub", isDirectory: true)
        switch self {
        case .melody:
            return [hub.appendingPathComponent("models--m-a-p--SheetSage2", isDirectory: true),
                    hub.appendingPathComponent("models--m-a-p--MERT-v2-FullSong", isDirectory: true)]
        case .lyrics:
            return [hub.appendingPathComponent("models--openai--whisper-large-v3-turbo", isDirectory: true),
                    hub.appendingPathComponent("models--openai--whisper-small", isDirectory: true)]
        case .genre:
            return [hub.appendingPathComponent("models--dima806--music_genres_classification", isDirectory: true)]
        case .style:
            return [hub.appendingPathComponent("models--laion--clap-htsat-unfused", isDirectory: true)]
        case .mlxWhisper:
            return [hub.appendingPathComponent("models--mlx-community--whisper-large-v3-turbo", isDirectory: true),
                    Paths.coverSupport.appendingPathComponent("whisper-env", isDirectory: true)]
        case .vocalActivity:
            return [Paths.coverSeparationModel]
        }
    }

    var isInstalled: Bool {
        let fm = FileManager.default
        if fm.fileExists(atPath: disabledMarker.path) { return false }
        if self == .vocalActivity {
            let size = (try? Paths.coverSeparationModel.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return size >= 100 * 1024 * 1024
        }
        if fm.fileExists(atPath: marker.path) || legacyMarkers.contains(where: { fm.fileExists(atPath: $0.path) }) {
            return true
        }
        return false
    }
}
