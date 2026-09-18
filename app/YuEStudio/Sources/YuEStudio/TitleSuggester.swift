import Foundation
#if canImport(FoundationModels)
import FoundationModels
#endif

#if canImport(FoundationModels)
@available(macOS 26.0, *)
@Generable(description: "Original song lyrics")
struct WrittenLyrics {
    @Guide(description: "Verse 1: four to six lines that set a scene with concrete images; no line repeated", .count(4...6))
    var verse1: [String]
    @Guide(description: "Chorus: four lines with the hook, memorable and singable; different from the verses", .count(3...5))
    var chorus: [String]
    @Guide(description: "Verse 2: four to six lines that move the story on with new images, not those of verse 1", .count(4...6))
    var verse2: [String]
    @Guide(description: "Bridge: three or four lines with a turn or a new perspective", .count(2...4))
    var bridge: [String]
    @Guide(description: "Outro: two or three complete closing lines that end the song", .count(2...3))
    var outro: [String]
    // The small model fumbles the closing of its last field (brackets leak into the text), so the
    // outro is not last: this throwaway field takes the damage.
    @Guide(description: "One word for the mood of the song")
    var mood: String
}
#endif

/// Names a run from its lyrics with the on-device language model (macOS 26 and later, Apple
/// Intelligence enabled); otherwise from the first line of the lyrics.
enum TitleSuggester {
    /// The on-device model can be used right now (macOS 26, Apple Intelligence on, model downloaded).
    static var modelAvailable: Bool {
        #if canImport(FoundationModels)
        guard #available(macOS 26.0, *) else { return false }
        if case .available = SystemLanguageModel.default.availability { return true }
        return false
        #else
        return false
        #endif
    }

    /// Lyrics for a song from its style and title, in the generator's format; nil if the model
    /// is unavailable. Guided generation (one field per section, each described) keeps the small
    /// on-device model from repeating itself, which free-form prompting did.
    static func writeLyrics(style: String, title: String, about: String) async -> Result<String, Error>? {
        #if canImport(FoundationModels)
        guard #available(macOS 26.0, *), modelAvailable else { return nil }
        let session = LanguageModelSession(instructions:
            "You write original, vivid song lyrics with varied imagery and no repeated lines except the chorus. Plain words, one phrase per line, no chord names, no markdown.")
        let request = "Write lyrics for a song.\nStyle: \(style.isEmpty ? "a popular song" : style)"
            + (title.isEmpty ? "" : "\nTitle: \(title)") + (about.isEmpty ? "" : "\nThe song is about: \(about)")
        do {
            let l = try await session.respond(to: request, generating: WrittenLyrics.self,
                                              options: GenerationOptions(temperature: 0.9, maximumResponseTokens: 1500)).content
            let clean: ([String]) -> [String] = { $0.map(scrubLine).filter { !$0.isEmpty } }
            let sections: [[String]] = [
                ["[Verse]"], clean(l.verse1), ["", "[Chorus]"], clean(l.chorus), ["", "[Verse]"], clean(l.verse2),
                ["", "[Chorus]"], clean(l.chorus), ["", "[Bridge]"], clean(l.bridge), ["", "[Outro]"], clean(l.outro),
            ]
            return .success(sections.flatMap { $0 }.joined(separator: "\n"))
        } catch {
            return .failure(error)
        }
        #else
        return nil
        #endif
    }

    /// Strip structured-output debris (stray brackets, quotes, commas) from the ends of a line.
    private static func scrubLine(_ line: String) -> String {
        var t = line.trimmingCharacters(in: .whitespacesAndNewlines)
        while let last = t.last, "]}\",".contains(last) { t.removeLast() }
        while let first = t.first, "[{\"".contains(first) { t.removeFirst() }
        return t.trimmingCharacters(in: .whitespaces)
    }

    static func suggest(lyrics: String, style: String, instrumental: Bool) async -> String {
        if let title = await fromModel(lyrics: lyrics, style: style, instrumental: instrumental), !title.isEmpty { return title }
        return fallback(lyrics: lyrics, style: style, instrumental: instrumental)
    }

    private static func fromModel(lyrics: String, style: String, instrumental: Bool) async -> String? {
        #if canImport(FoundationModels)
        guard #available(macOS 26.0, *) else { return nil }
        guard case .available = SystemLanguageModel.default.availability else { return nil }
        let words = lyrics.split(whereSeparator: { $0.isWhitespace }).filter { !$0.hasPrefix("[") }
        let body = instrumental || words.count < 3
            ? "An instrumental piece in this style: \(style)"
            : "Lyrics:\n" + String(lyrics.prefix(3000))
        let session = LanguageModelSession(instructions:
            "You name songs. Given lyrics or a style description, reply with one evocative title of two to five words. Title only: no quotes, no punctuation at the end, no explanation.")
        do {
            let reply = try await session.respond(to: body).content
            return clean(reply)
        } catch {
            return nil
        }
        #else
        return nil
        #endif
    }

    private static func clean(_ text: String) -> String {
        var t = text.components(separatedBy: .newlines).first?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        t = t.trimmingCharacters(in: CharacterSet(charactersIn: "\"'“”‘’.!,:; "))
        if t.lowercased().hasPrefix("title:") { t = String(t.dropFirst(6)).trimmingCharacters(in: .whitespaces) }
        let capped = t.split(separator: " ").prefix(8).joined(separator: " ")
        return String(capped.prefix(60))
    }

    /// The first line of lyrics with words in it, up to five words; or the style's first words.
    static func fallback(lyrics: String, style: String, instrumental: Bool) -> String {
        let lines = lyrics.components(separatedBy: .newlines).map { $0.trimmingCharacters(in: .whitespaces) }
        if !instrumental, let line = lines.first(where: { !$0.isEmpty && !$0.hasPrefix("[") }) {
            return clean(line.split(separator: " ").prefix(5).joined(separator: " ").capitalized)
        }
        return clean(style.split(separator: " ").prefix(4).joined(separator: " ").capitalized)
    }
}
