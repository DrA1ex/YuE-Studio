import Foundation

enum AudioTranscoder {
    static let mp3Bitrate = "320k"

    static func importArguments(source: URL, destination: URL) -> [String] {
        ["-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-i", source.path, "-vn", "-c:a", "pcm_s16le", "-ar", "48000", destination.path]
    }

    static func exportArguments(source: URL, destination: URL) -> [String] {
        ["-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-i", source.path, "-vn", "-c:a", "libmp3lame", "-b:a", mp3Bitrate,
         "-map_metadata", "0", destination.path]
    }

    static func importToWAV(source: URL, destination: URL) throws {
        try run(importArguments(source: source, destination: destination))
    }

    static func exportMP3(source: URL, destination: URL) throws {
        try run(exportArguments(source: source, destination: destination))
    }

    private static func run(_ arguments: [String]) throws {
        guard let ffmpeg = Paths.ffmpeg else {
            throw NSError(
                domain: "YuEStudio.Audio",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "FFmpeg is not installed. Run YuE Studio setup again to install the missing audio dependency."]
            )
        }

        let process = Process()
        let pipe = Pipe()
        process.executableURL = ffmpeg
        process.arguments = arguments
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let message = String(decoding: data, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw NSError(
                domain: "YuEStudio.Audio",
                code: Int(process.terminationStatus),
                userInfo: [NSLocalizedDescriptionKey:
                    message.isEmpty ? "FFmpeg exited with status \(process.terminationStatus)." : message]
            )
        }
    }
}
