import SwiftUI
import AVFoundation
import AppKit
import UniformTypeIdentifiers

struct Paths {
    static let support: URL = {
        if let o = ProcessInfo.processInfo.environment["YUE_STUDIO_SUPPORT"] { return URL(fileURLWithPath: o) }
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("YuE Studio")
    }()
    static let payload: URL? = {
        guard let r = Bundle.main.resourceURL?.appendingPathComponent("payload"),
              FileManager.default.fileExists(atPath: r.appendingPathComponent("uv").path) else { return nil }
        return r
    }()
    static var packaged: Bool { payload != nil }
    static let repoRoot: URL = { var u = URL(fileURLWithPath: #filePath); for _ in 0..<5 { u.deleteLastPathComponent() }; return u }()
    static var python: URL {
        if let override = ProcessInfo.processInfo.environment["YUE_STUDIO_PYTHON"] { return URL(fileURLWithPath: override) }
        let local = repoRoot.appendingPathComponent(".venv/bin/python")
        return !packaged && FileManager.default.isExecutableFile(atPath: local.path) ? local : support.appendingPathComponent("env/bin/python")
    }
    static var worker: URL {
        if let o = ProcessInfo.processInfo.environment["YUE_STUDIO_WORKER"] { return URL(fileURLWithPath: o) }   // tests
        return packaged ? support.appendingPathComponent("src/tools/yue2_worker.py") : repoRoot.appendingPathComponent("tools/yue2_worker.py")
    }
    static var transcriber: URL {
        if let o = ProcessInfo.processInfo.environment["YUE_STUDIO_TRANSCRIBER"] { return URL(fileURLWithPath: o) }
        return packaged ? support.appendingPathComponent("src/tools/transcribe_cover.py") : repoRoot.appendingPathComponent("tools/transcribe_cover.py")
    }
    static var src: URL { support.appendingPathComponent("src") }
    static var models: URL { ProcessInfo.processInfo.environment["YUE_STUDIO_HF_HOME"].map { URL(fileURLWithPath: $0) } ?? support.appendingPathComponent("models") }
    static var aneCache: URL { support.appendingPathComponent("ane-cache") }
    static var output: URL {
        if let override = ProcessInfo.processInfo.environment["YUE_STUDIO_OUTPUT"] { return URL(fileURLWithPath: override) }
        return packaged ? FileManager.default.urls(for: .musicDirectory, in: .userDomainMask)[0].appendingPathComponent("YuE Studio")
                 : repoRoot.appendingPathComponent("outputs/app")
    }
    static var imports: URL { output.appendingPathComponent("Imports", isDirectory: true) }
    static var coverSupport: URL { support.appendingPathComponent("cover-runtime", isDirectory: true) }
    static var coverAnalyses: URL { coverSupport.appendingPathComponent("analyses", isDirectory: true) }
    static var coverPython: URL { coverSupport.appendingPathComponent("env/bin/python") }
    static var coverWhisperPython: URL { coverSupport.appendingPathComponent("whisper-env/bin/python") }
    static var coverRequirements: URL { packaged ? src.appendingPathComponent("tools/sheetsage-requirements.txt") : repoRoot.appendingPathComponent("tools/sheetsage-requirements.txt") }
    static var coverWhisperRequirements: URL { packaged ? src.appendingPathComponent("tools/whisper-requirements.txt") : repoRoot.appendingPathComponent("tools/whisper-requirements.txt") }
    static var coverSeparationModel: URL { coverSupport.appendingPathComponent("torch/torchaudio/models/hdemucs_high_musdbhq_only.pt") }
    static var installedMarker: URL { support.appendingPathComponent("installed.json") }
    static var bundledVersion: String { (try? String(contentsOf: payload!.appendingPathComponent("version.txt"), encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "dev" }
    static var workerEnvironment: [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"; env["TQDM_DISABLE"] = "1"
        env["YUE2_OUTPUT_DIR"] = output.path; env["YUE2_ANE_CACHE"] = aneCache.path
        env["HF_HOME"] = models.path; env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        env["PYTHONPATH"] = (packaged ? src : repoRoot).appendingPathComponent("src").path
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        return env
    }
}

// MARK: - Installer
