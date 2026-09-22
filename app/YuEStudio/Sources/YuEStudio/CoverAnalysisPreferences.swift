import Foundation

enum CoverLyricsBackend: String, CaseIterable, Identifiable {
    case automatic
    case mlx
    case transformers

    var id: String { rawValue }

    var title: String {
        switch self {
        case .automatic: "Automatic"
        case .mlx: "MLX Whisper"
        case .transformers: "Transformers Whisper"
        }
    }
}

enum CoverAnalysisPreferences {
    static let lyricsKey = "cover.analysis.lyrics"
    static let genreKey = "cover.analysis.genre"
    static let styleKey = "cover.analysis.style"
    static let vocalActivityKey = "cover.analysis.vocalActivity"
    static let lyricsBackendKey = "cover.analysis.lyricsBackend"

    static func registerDefaults() {
        UserDefaults.standard.register(defaults: [
            lyricsKey: true,
            genreKey: true,
            styleKey: true,
            vocalActivityKey: true,
            lyricsBackendKey: CoverLyricsBackend.automatic.rawValue,
        ])
    }

    static var lyricsEnabled: Bool {
        registerDefaults()
        return UserDefaults.standard.bool(forKey: lyricsKey)
    }

    static var genreEnabled: Bool {
        registerDefaults()
        return UserDefaults.standard.bool(forKey: genreKey)
    }

    static var styleEnabled: Bool {
        registerDefaults()
        return UserDefaults.standard.bool(forKey: styleKey)
    }

    static var vocalActivityEnabled: Bool {
        registerDefaults()
        return lyricsEnabled && UserDefaults.standard.bool(forKey: vocalActivityKey)
    }

    static var lyricsBackend: CoverLyricsBackend {
        registerDefaults()
        let raw = UserDefaults.standard.string(forKey: lyricsBackendKey) ?? CoverLyricsBackend.automatic.rawValue
        return CoverLyricsBackend(rawValue: raw) ?? .automatic
    }
}
