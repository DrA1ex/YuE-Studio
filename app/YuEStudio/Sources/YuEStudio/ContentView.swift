import SwiftUI
import AVFoundation
import AppKit
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var backend: Backend
    @StateObject private var players = Players()
    @AppStorage("style") private var style = "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums"
    @AppStorage("lyrics") private var lyrics = ""
    @AppStorage("cot") private var cot = "full"
    @AppStorage("seed") private var seed = 831001
    @AppStorage("randomSeed") private var randomSeed = false
    @AppStorage("batch") private var batch = 2
    @AppStorage("maxSeconds") private var maxSeconds = 120.0
    @AppStorage("quality") private var quality = "draft"
    @AppStorage("instrumental") private var instrumental = false
    @AppStorage("promptFidelity") private var promptFidelity = 0.75
    @AppStorage("styleFidelity") private var styleFidelity = 0.75
    @AppStorage("sourceFidelity") private var sourceFidelity = 1.0
    @AppStorage("engines") private var engines = "gpu+ane"
    @AppStorage("useRemote") private var useRemote = false
    @StateObject private var remote = RemoteBrowser()
    @State private var writingLyrics = false
    @State private var naming = false
    @State private var askAbout = false
    @State private var lyricsAbout = ""
    @State private var lyricsVersion = 0
    @State private var titleAuto = ""
    @State private var title = ""
    @State private var abc = ""
    @State private var abcOpen = false
    @State private var humming = false
    @State private var humReview = false
    @State private var humScore = ""
    @State private var humContinue = true
    @State private var humError = ""
    @State private var mode = "new"
    @State private var selectedTab = "Create"
    @State private var source: AudioSource?
    @State private var search = ""
    @State private var filter = "All"
    @State private var sortNewest = true
    @State private var confirmClearAnalysis = false
    @Environment(\.openSettings) private var openSettings
    @State private var showOptions = false
    @State private var lyricsExpanded = true
    @State private var stylesExpanded = true
    @State private var showScore: Song?
    @State private var transcribing = false
    @State private var transcriptionError = ""
    @State private var suggestedGenre = ""
    @State private var analysisWarnings: [String] = []
    @State private var installingCoverRuntime = false
    @State private var collapsedSidebar = false
    @State private var showLyricsEditor = false
    @State private var showLogs = false
    @State private var undoLyrics: [String] = []
    @State private var redoLyrics: [String] = []
    @State private var restoringLyrics = false

    private let dark = Color(red: 0.055, green: 0.055, blue: 0.065)
    private let card = Color(red: 0.105, green: 0.105, blue: 0.12)

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                sidebar.frame(width: collapsedSidebar ? 74 : 205)
                Divider().overlay(.white.opacity(0.08))
                VStack(spacing: 0) {
                    topBar.disabled(writingLyrics || naming)
                    Divider().overlay(.white.opacity(0.08))
                    HStack(spacing: 0) {
                        if selectedTab == "Create" {
                            createPane.frame(minWidth: 430, idealWidth: 500).disabled(writingLyrics || naming)
                            Divider().overlay(.white.opacity(0.08))
                        }
                        libraryPane.frame(minWidth: 430, idealWidth: 560).disabled(writingLyrics || naming)
                    }
                }
            }
            playerBar
        }
        .background(dark)
        .preferredColorScheme(.dark)
        .frame(minWidth: 1260, minHeight: 760)
        .onAppear {
            backend.rescan(); if backend.process == nil { backend.start() }
            if useRemote { remote.start() }
            backend.remoteRetry = { if useRemote && engines == "gpu+ane" { backend.useRemote(remote.phone) } }
        }
        .onChange(of: remote.phone) { _, _ in syncRemote() }
        .onChange(of: useRemote) { _, on in if on { remote.start() } else { remote.stop() }; syncRemote() }
        .onChange(of: engines) { _, _ in syncRemote() }
        .onChange(of: backend.connected) { _, up in if up { syncRemote() } }
        .sheet(isPresented: $askAbout) {
            VStack(alignment: .leading, spacing: 16) {
                Text("What is the song about?").font(.title2)
                TextField("Theme, story or mood", text: $lyricsAbout).textFieldStyle(.roundedBorder)
                HStack {
                    Button("Cancel") { askAbout = false }
                    Spacer()
                    Button("Write lyrics") { askAbout = false; Task { await writeLyrics() } }
                        .disabled(backend.busy || backend.coverBusy).buttonStyle(.borderedProminent)
                }
            }.padding(24).frame(width: 480)
        }
        .onChange(of: abc) { _, text in if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { abcOpen = false } }
        .onChange(of: lyrics) { old, _ in
            if restoringLyrics { restoringLyrics = false }
            else { undoLyrics.append(old); if undoLyrics.count > 100 { undoLyrics.removeFirst() }; redoLyrics.removeAll() }
        }
        .onChange(of: players.error) { _, message in if let message { backend.report(message) } }
        .onChange(of: backend.songs) { _, songs in
            if let path = source?.path, let updated = songs.first(where: { $0.path == path }) { source?.title = updated.title }
            if let playing = players.currentSong, let updated = songs.first(where: { $0.id == playing.id }) { players.updateMetadata(updated) }
        }
        .alert("YuE Studio", isPresented: Binding(get: { backend.errorMessage != nil }, set: { if !$0 { backend.errorMessage = nil } })) {
            Button("OK") { backend.errorMessage = nil }
        } message: { Text(backend.errorMessage ?? "") }
        .alert("Clear previous audio analyses?", isPresented: $confirmClearAnalysis) {
            Button("Cancel", role: .cancel) { }
            Button("Move to Trash", role: .destructive) { backend.clearCoverAnalysisCache() }
        } message: {
            Text("Reusable melody, lyric, genre and style evidence will be removed. The next analysis will start from zero; songs and source audio stay in the library.")
        }
        .sheet(isPresented: $humming) {
            HumSheetView { url in
                humming = false
                humScore = ""; humError = ""; humContinue = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    humReview = true
                    let recording = AudioSource(id: url.path, path: url.path, title: "Hummed melody", seconds: 30, kind: "HUM")
                    backend.transcribe(recording, hum: true) { result in
                        switch result {
                        case .success(let analysis): humScore = analysis.score
                        case .failure(let error): humError = error.localizedDescription
                        }
                    }
                }
            }
        }
        .sheet(isPresented: $humReview) {
            VStack(alignment: .leading, spacing: 16) {
                Text("Your melody").font(.title2).bold()
                if backend.coverBusy {
                    ProgressView(backend.coverStatus)
                } else if !humError.isEmpty {
                    Text(humError).foregroundStyle(.red)
                } else {
                    Text("Review the transcribed notes, then choose how to use them.").foregroundStyle(.secondary)
                    TextEditor(text: $humScore).font(.system(.body, design: .monospaced))
                    Picker("Use melody", selection: $humContinue) {
                        Text("Continue from my melody").tag(true)
                        Text("Use the complete melody").tag(false)
                    }.pickerStyle(.segmented)
                }
                HStack {
                    Button("Cancel") { backend.cancelCover(); humReview = false }
                    Spacer()
                    Button("Use melody") {
                        abc = humScore; abcOpen = humContinue; cot = "melody"
                        source = nil; mode = "new"; selectedTab = "Create"; showOptions = true
                        transcriptionError = ""; suggestedGenre = ""; analysisWarnings = []
                        humReview = false
                    }.buttonStyle(.borderedProminent)
                        .disabled(backend.coverBusy || !humError.isEmpty || humScore.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }.padding(24).frame(width: 660, height: 460).interactiveDismissDisabled(backend.coverBusy)
        }
        .sheet(isPresented: $showLyricsEditor) {
            VStack(alignment: .leading, spacing: 14) {
                Text("Lyrics").font(.title2).bold()
                TextEditor(text: $lyrics).id(lyricsVersion).disabled(writingLyrics).font(.system(size: 17))
                HStack { lyricsTools; Spacer(); Button("Done") { showLyricsEditor = false }.keyboardShortcut(.cancelAction) }
            }.padding(24).frame(width: 720, height: 580)
        }
        .sheet(isPresented: $showLogs) {
            VStack {
                HStack { Text("Process Log").font(.title2); Spacer(); Button("Clear") { backend.log.removeAll() }; Button("Close") { showLogs = false }.keyboardShortcut(.cancelAction) }
                LogTextView(lines: backend.log)
            }.padding(20).frame(width: 840, height: 560)
        }
        .sheet(item: $showScore) { song in
            VStack(alignment: .leading, spacing: 12) {
                Text(song.title.isEmpty ? "Score" : song.title).font(.title3).bold()
                ScrollView { Text(song.score).font(.system(.body, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
                HStack { Spacer(); Button("Close") { showScore = nil }.keyboardShortcut(.cancelAction) }
            }.padding().frame(width: 720, height: 520)
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack { if !collapsedSidebar { Text("YuE Studio").font(.system(size: 25, weight: .bold)); Spacer() }; Button { collapsedSidebar.toggle() } label: { Image(systemName: "sidebar.left") }.buttonStyle(.plain).help("Toggle sidebar") }
                .padding(.horizontal, 22).padding(.top, 22).padding(.bottom, 26)
            sideButton("music.note", "Create", "Create")
            sideButton("waveform", "Library", "Library")
            Spacer()
            Button { openSettings() } label: {
                Label(collapsedSidebar ? "" : "Settings", systemImage: "gearshape")
                    .frame(maxWidth: .infinity, alignment: .leading).padding(12).contentShape(Rectangle())
            }.buttonStyle(.plain).padding(.horizontal, 10).help("Settings").accessibilityLabel("Settings")
            Button { showLogs = true } label: { Label(collapsedSidebar ? "" : "Process Log", systemImage: "terminal") }.buttonStyle(.plain).padding(.horizontal, 22).help("Process Log").accessibilityLabel("Process Log")
            if !backend.connected && !backend.coverBusy {
                Button("Reconnect") { backend.start() }.padding(.horizontal, 12)
            }
            if backend.busy || backend.coverBusy {
                Button("Stop All", role: .destructive) { backend.stop() }.padding(.horizontal, 12)
            }
            Text(collapsedSidebar ? (backend.connected ? "Ready" : "Offline") : "Local generation · \(backend.connected ? "Ready" : "Offline")")
                .font(.caption).foregroundStyle(backend.connected ? .green : .orange).padding(.horizontal, 22).padding(.bottom, 16)
        }.background(dark)
    }

    private func sideButton(_ icon: String, _ label: String, _ value: String) -> some View {
        Button { selectedTab = value } label: {
            Label(collapsedSidebar ? "" : label, systemImage: icon).font(.system(size: 15, weight: selectedTab == value ? .semibold : .regular))
                .foregroundStyle(selectedTab == value ? .white : .secondary)
                .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 22).padding(.vertical, 11)
                .background(selectedTab == value ? Color.white.opacity(0.13) : .clear, in: Capsule()).contentShape(Capsule())
        }.buttonStyle(.plain).padding(.horizontal, 10)
        .help(label)
    }

    private var topBar: some View {
        HStack(spacing: 18) {
            Picker("", selection: $mode) { Text("New song").tag("new"); Text("Cover").tag("cover") }
                .pickerStyle(.segmented).frame(width: 190)
                .disabled(backend.coverBusy)
            Picker("Quality", selection: $quality) { Text("Draft").tag("draft"); Text("Full").tag("full") }
                .pickerStyle(.segmented).frame(width: 135)
            Spacer()
            Label("YuE2", systemImage: "waveform").foregroundStyle(.secondary)
            Button { openOutput() } label: { Image(systemName: "folder") }
                .buttonStyle(.plain).help("Open songs folder")
        }.padding(.horizontal, 24).padding(.vertical, 15)
    }

    private var createPane: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    HStack(spacing: 0) {
                        createSourceButton("Audio", systemImage: "plus") { importSource() }.disabled(backend.coverBusy)
                        createSourceButton("Hum a melody", systemImage: "mic") {
                            if backend.coverRuntimeReady { humming = true }
                            else { backend.report("Install the Melody component in Settings before recording a melody.") }
                        }.disabled(backend.coverBusy || backend.busy)
                    }.background(card, in: Capsule())
                    if let source { sourceCard(source) }
                    editorCard(title: "Lyrics", expanded: $lyricsExpanded) {
                        TextEditor(text: $lyrics).id(lyricsVersion).disabled(writingLyrics).scrollContentBackground(.hidden).font(.system(size: 15)).frame(minHeight: 210)
                        HStack { Text("\(lyrics.count) characters").font(.caption).foregroundStyle(.secondary); Spacer(); Toggle("Instrumental", isOn: $instrumental).toggleStyle(.switch).controlSize(.small) }
                    }
                    editorCard(title: "Styles", expanded: $stylesExpanded) {
                        TextEditor(text: $style).scrollContentBackground(.hidden).font(.system(size: 15)).frame(minHeight: 100)
                        HStack(spacing: 8) {
                            styleChip("pop"); styleChip("acoustic"); styleChip("cinematic"); styleChip("electronic")
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }
                    editorCard(title: "More Options", expanded: $showOptions) {
                        VStack(alignment: .leading, spacing: 12) {
                            if mode == "cover" { Text("Planning: preserve source melody").font(.caption).foregroundStyle(.secondary) }
                            else { Picker("Planning", selection: $cot) { Text("Full score").tag("full"); Text("Melody").tag("melody"); Text("Direct").tag("off") }.pickerStyle(.segmented) }
                            Picker("Synthesis", selection: $engines) {
                                Text("GPU only").tag("gpu")
                                Text("GPU + Neural Engine").tag("gpu+ane")
                            }
                            Toggle("Use an iPhone’s Neural Engine", isOn: $useRemote)
                            if useRemote {
                                Text(engines == "gpu" ? "Select GPU + Neural Engine to use the iPhone." : (backend.remoteStatus.isEmpty ? remote.detail : backend.remoteStatus))
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            Stepper("Variants: \(batch)", value: $batch, in: 1...8)
                            if mode == "cover", !abcOpen, let source {
                                HStack {
                                    Text("Cover length"); Spacer()
                                    Text(timeLabel(min(source.seconds, 360))).monospacedDigit()
                                    Text(source.seconds <= 360 ? "matches source" : "YuE max · source \(timeLabel(source.seconds))").foregroundStyle(.secondary)
                                }
                            } else {
                                VStack(alignment: .leading, spacing: 6) { HStack { Text("Max length"); Spacer(); Text("\(Int(maxSeconds))s").monospacedDigit() }; Slider(value: $maxSeconds, in: 30...360, step: 10).accessibilityLabel("Max length").accessibilityValue("\(Int(maxSeconds)) seconds") }
                            }
                            fidelitySlider("Prompt fidelity", value: $promptFidelity)
                            fidelitySlider("Style fidelity", value: $styleFidelity)
                            if mode == "cover" { fidelitySlider("Source audio fidelity", value: $sourceFidelity) }
                            HStack { TextField("Seed", value: $seed, format: .number).disabled(randomSeed); Toggle("Random", isOn: $randomSeed) }
                            HStack { Text("ABC score").font(.caption); Spacer(); Button("Import ABC") { importScore() }.disabled(backend.coverBusy) }
                            if !abc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                Picker("Use score as", selection: $abcOpen) {
                                    Text("Complete melody").tag(false)
                                    Text("Opening to continue").tag(true)
                                }
                                Text(abcOpen ? "The planner continues your opening to the selected maximum length." : "The supplied score defines the melody.")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            TextEditor(text: $abc).font(.system(.caption, design: .monospaced)).frame(height: 90).overlay(RoundedRectangle(cornerRadius: 8).stroke(.white.opacity(0.1)))
                        }.padding(.top, 10)
                    }.font(.system(size: 13)).tint(.accentColor)
                    VStack(alignment: .leading, spacing: 10) {
                        Label("Song title", systemImage: "music.note").foregroundStyle(.secondary)
                        TextField("Optional title", text: $title).textFieldStyle(.plain).font(.system(size: 16)).padding(12).background(card, in: RoundedRectangle(cornerRadius: 12))
                        Button { openOutput() } label: { Label(Paths.output.path, systemImage: "folder").font(.caption).lineLimit(1).truncationMode(.middle) }.buttonStyle(.plain).foregroundStyle(.secondary)
                    }
                }.padding(24)
            }
            Divider().overlay(.white.opacity(0.08))
            if let reason = createUnavailableReason {
                Text(reason).font(.caption).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 18).padding(.top, 8)
            }
            HStack(spacing: 12) {
                Button { resetForm() } label: { Image(systemName: "trash").frame(width: 46, height: 42) }.buttonStyle(.bordered).tint(.secondary).help("Reset form").disabled(backend.coverBusy)
                Button { Task { await create() } } label: { Label(mode == "cover" ? "Create Cover" : "Create", systemImage: mode == "cover" ? "arrow.triangle.2.circlepath" : "music.note")
                        .frame(maxWidth: .infinity).frame(height: 42) }
                    .buttonStyle(.borderedProminent).tint(.blue).disabled(!canCreate)
            }.padding(18)
        }
    }

    private func createSourceButton(_ title: String, systemImage: String, action: @escaping () -> Void) -> some View {
        Button(action: action) { Label(title, systemImage: systemImage).frame(maxWidth: .infinity).padding(.vertical, 16).background(card, in: Capsule()).contentShape(Capsule()) }.buttonStyle(.plain)
    }

    private func editorCard<Content: View>(title: String, expanded: Binding<Bool>, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Button { expanded.wrappedValue.toggle() } label: { HStack { Image(systemName: expanded.wrappedValue ? "chevron.down" : "chevron.right"); Text(title).font(.system(size: 17, weight: .semibold)); Spacer(minLength: 0) }.padding(.vertical, 6).contentShape(Rectangle()) }.buttonStyle(.plain).accessibilityValue(expanded.wrappedValue ? "Expanded" : "Collapsed")
                if title == "More Options" {
                    Button { resetAdvancedOptions() } label: {
                        Image(systemName: "arrow.counterclockwise").frame(width: 28, height: 28).contentShape(Rectangle())
                    }.buttonStyle(.plain).help("Reset advanced options").accessibilityLabel("Reset advanced options").disabled(backend.coverBusy)
                }
                if title == "Styles" {
                    Button { style = "" } label: {
                        Image(systemName: "trash").frame(width: 28, height: 28).contentShape(Rectangle())
                    }.buttonStyle(.plain).help("Clear styles").accessibilityLabel("Clear styles").disabled(style.isEmpty || backend.coverBusy)
                }
                if title == "Lyrics" {
                    lyricsTools
                    Button { showLyricsEditor = true } label: { Image(systemName: "arrow.up.left.and.arrow.down.right") }.buttonStyle(.plain).help("Expand lyrics editor")
                }
            }
            if expanded.wrappedValue { content() }
        }.padding(18).background(card, in: RoundedRectangle(cornerRadius: 18))
    }

    private func styleChip(_ value: String) -> some View {
        Button(value) {
            var tags = style.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            if let i = tags.firstIndex(where: { $0.caseInsensitiveCompare(value) == .orderedSame }) { tags.remove(at: i) } else { tags.append(value) }
            style = tags.joined(separator: ", ")
        }
            .buttonStyle(.bordered).tint(.secondary).controlSize(.small)
    }

    private func sourceCard(_ source: AudioSource) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button { if let song = backend.songs.first(where: { $0.path == source.path }) { players.toggle(song) } } label: {
                    ZStack { WaveformView(path: source.path).frame(width: 70, height: 46); Image(systemName: players.currentSong?.path == source.path && players.isPlaying ? "pause.fill" : "play.fill") }
                }.buttonStyle(.plain).help("Preview source")
                VStack(alignment: .leading) { Text(source.title).bold(); Text(String(format: "%.1f seconds · %@", source.seconds, source.kind)).font(.caption).foregroundStyle(.secondary) }
                Spacer()
                Button("Replace") { importSource() }.disabled(backend.coverBusy)
                Button { self.source = nil; abc = ""; mode = "new" } label: { Image(systemName: "xmark") }.buttonStyle(.plain).help("Remove source").disabled(backend.coverBusy)
            }
            if mode == "cover" {
                HStack {
                    if backend.coverBusy { ProgressView().controlSize(.small); Text(backend.coverStatus).font(.caption); Button("Cancel") { backend.cancelCover() } }
                    else if !backend.coverRuntimeReady {
                        if installingCoverRuntime { ProgressView().controlSize(.small); Text("Installing cover engine…") }
                        else { Button("Install Cover Engine") { installCoverRuntime() }.buttonStyle(.borderedProminent).controlSize(.small) }
                    } else { Button("Analyze audio") { transcribe(source) }.buttonStyle(.borderedProminent).controlSize(.small) }
                    if !transcriptionError.isEmpty { Text(transcriptionError).foregroundStyle(.red).font(.caption).lineLimit(2) }
                }
            }
            ForEach(analysisWarnings, id: \.self) { Text(friendlyAnalysisWarning($0)).font(.caption).foregroundStyle(.secondary) }
            if !suggestedGenre.isEmpty {
                HStack { Label("Suggested style: \(suggestedGenre)", systemImage: "sparkles").font(.caption).foregroundStyle(.secondary); Spacer(); Button("Use") { style = suggestedGenre }.controlSize(.small) }
            }
        }.padding(14).background(card, in: RoundedRectangle(cornerRadius: 16))
    }

    private func friendlyAnalysisWarning(_ warning: String) -> String {
        if warning.localizedCaseInsensitiveContains("Word alignment") {
            return "Lyric timing is approximate; the recognized text was kept."
        }
        if warning.localizedCaseInsensitiveContains("sections are inferred") {
            return "Section labels are suggestions based on repetition and position; review them before generating."
        }
        if warning.localizedCaseInsensitiveContains("digital silence") {
            return "Very quiet timestamps were omitted; the raw transcription remains available in the import metadata."
        }
        if warning.localizedCaseInsensitiveContains("style model unavailable") {
            return "Detailed style analysis is unavailable until the cover engine is repaired."
        }
        return warning
    }

    private var libraryPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack { Text("My Library").font(.title2).bold(); Spacer(); Button { backend.rescan() } label: { Image(systemName: "arrow.clockwise") }.buttonStyle(.plain) }.padding(.horizontal, 24).padding(.top, 24)
            HStack(spacing: 10) {
                TextField("Search", text: $search).textFieldStyle(.roundedBorder)
                Menu { Picker("Filter", selection: $filter) { Text("All").tag("All"); Text("Covers").tag("COVER"); Text("Uploads").tag("UPLOAD"); Text("Drafts").tag("DRAFT") } } label: { Label(filter, systemImage: "line.3.horizontal.decrease.circle") }.buttonStyle(.bordered)
                Menu { Button("Newest") { sortNewest = true }; Button("Title") { sortNewest = false } } label: { Label(sortNewest ? "New" : "Title", systemImage: "arrow.up.arrow.down") }.buttonStyle(.bordered)
            }.padding(24)
            ScrollView {
                LazyVStack(spacing: 2) { ForEach(filteredSongs) { song in libraryRow(song) } }
            }.overlay { if filteredSongs.isEmpty { Text(backend.songs.isEmpty ? "Your songs will appear here" : "No songs match this search or filter").foregroundStyle(.secondary) } }
            HStack(spacing: 10) {
                Image(systemName: "arrow.counterclockwise")
                    .foregroundStyle(.secondary)
                Text("Need a clean re-analysis? Clear the previous audio-analysis data below.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                Spacer()
                Button("Clear analysis data…") { confirmClearAnalysis = true }
                    .buttonStyle(.bordered)
                    .disabled(backend.busy || backend.coverBusy)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(card.opacity(0.75))
        }
    }

    private var filteredSongs: [Song] {
        let songs = backend.songs.filter { song in
            let matchesSearch = search.isEmpty || song.title.localizedCaseInsensitiveContains(search) || song.style.localizedCaseInsensitiveContains(search)
            let matchesFilter = filter == "All" || (filter == "DRAFT" ? song.quality == "draft" : song.kind == filter)
            return matchesSearch && matchesFilter
        }
        return songs.sorted { sortNewest ? $0.createdAt > $1.createdAt : $0.title.localizedCaseInsensitiveCompare($1.title) == .orderedAscending }
    }

    private func libraryRow(_ song: Song) -> some View {
        HStack(spacing: 12) {
            Button { players.toggle(song) } label: { ZStack { WaveformView(path: song.path).frame(width: 72, height: 72); Image(systemName: players.currentSong?.id == song.id && players.isPlaying ? "pause.fill" : "play.fill").padding(10).background(.black.opacity(0.55), in: Circle()) } }.buttonStyle(.plain).disabled(song.status != .ready)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 7) { Text(song.title.isEmpty ? "Song \(song.index)" : song.title).font(.system(size: 15, weight: .semibold)).lineLimit(1); Text(song.kind).font(.caption2).foregroundStyle(song.kind == "COVER" ? .pink : .secondary) }
                Text(song.style.isEmpty ? "YuE2 local generation" : song.style).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                HStack { Text(song.status == .ready ? String(format: "%.1f s", song.seconds) : song.statusLine).font(.caption2).foregroundStyle(.secondary); if song.quality == "draft" { Text("DRAFT").font(.caption2).foregroundStyle(.orange) } }
                if song.inFlight { StageTrack(progress: song.trackProgress).frame(height: 30) }
            }
            Spacer()
            Menu {
                songActions(song)
            } label: { Image(systemName: "ellipsis").frame(width: 36, height: 36).background(card, in: Circle()) }.menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize().help("Song actions")
        }.padding(.horizontal, 24).padding(.vertical, 10)
            .contentShape(Rectangle())
            .contextMenu { songActions(song) }
    }

    @ViewBuilder private func songActions(_ song: Song) -> some View {
                Button("Cover", systemImage: "arrow.triangle.2.circlepath") { beginCover(song) }
                    .disabled(backend.coverBusy || song.inFlight || (song.status != .ready && song.score.isEmpty))
                Button("Reuse Prompt & Lyrics", systemImage: "doc.on.doc") { reusePrompt(song) }
                    .disabled(backend.coverBusy || song.inFlight || (song.style.isEmpty && song.lyrics.isEmpty))
                if song.canRender && (song.quality == "draft" || song.status != .ready) { Button(song.status == .ready ? "Render full quality" : "Recover synthesis") { players.forget(song); backend.render(song, engine: "auto", quality: "full", engines: engines) }.disabled(!backend.connected || backend.coverBusy) }
                if !song.score.isEmpty { Button("View Score") { showScore = song } }
                if song.inFlight { Button("Cancel") { backend.cancel(song) } }
                if song.status == .ready { Button("Export Audio") { backend.export(song) }; Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: song.path)]) } }
                Button("Rename") { backend.rename(song) }.disabled(song.inFlight)
                Divider(); Button("Move to Trash", role: .destructive) { players.forget(song); backend.delete(song); if source?.path == song.path { source = nil; abc = "" } }.disabled(song.inFlight || backend.coverBusy)
    }

    private var playerBar: some View {
        HStack(spacing: 22) {
            if let song = players.currentSong { HStack { WaveformView(path: song.path).frame(width: 42, height: 34); VStack(alignment: .leading) { Text(song.title.isEmpty ? "Song \(song.index)" : song.title).font(.caption).bold(); Text("YuE2 local player").font(.caption2).foregroundStyle(.secondary) } }.frame(width: 250, alignment: .leading) }
            else { Text("Nothing playing").font(.caption).foregroundStyle(.secondary).frame(width: 250, alignment: .leading) }
            Spacer()
            Button { if let song = players.currentSong { players.toggle(song) } } label: { Image(systemName: players.isPlaying ? "pause.fill" : "play.fill").font(.title3) }.buttonStyle(.plain).disabled(players.currentSong == nil).help("Play / Pause")
            Text(timeLabel(players.position)).font(.caption).monospacedDigit()
            Slider(value: Binding(get: { min(players.position, max(1, players.duration)) }, set: { players.seek($0) }), in: 0...max(1, players.duration)).disabled(players.duration <= 0).help("Playback position")
            Text(timeLabel(players.duration)).font(.caption).monospacedDigit()
            Image(systemName: "speaker.wave.2.fill").foregroundStyle(.secondary)
            Slider(value: $players.volume, in: 0...1).frame(width: 95).help("Volume")
        }.padding(.horizontal, 24).frame(height: 58).background(Color(red: 0.09, green: 0.09, blue: 0.1))
    }

    private var canCreate: Bool { !writingLyrics && !naming && backend.connected && !backend.coverBusy && (mode == "cover" || !style.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) && (instrumental || !lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) && (mode == "new" || !abc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) }

    private var createUnavailableReason: String? {
        if writingLyrics { return "Writing lyrics…" }
        if naming { return "Choosing a title…" }
        if backend.coverBusy { return backend.coverStatus }
        if !backend.connected { return "Waiting for the local engine. Open Process Log for details." }
        if mode == "new" && style.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Enter a style to create a song." }
        if !instrumental && lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Enter lyrics or enable Instrumental." }
        if mode == "cover" && abc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Transcribe the source melody or import an ABC score in More Options." }
        return nil
    }

    private func create() async {
        guard canCreate else { return }
        naming = true
        defer { naming = false }
        let typed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if typed.isEmpty || typed == titleAuto {
            // Do not compete with synthesis or analysis for the on-device language model.
            let suggested = backend.busy || backend.coverBusy
                ? TitleSuggester.fallback(lyrics: lyrics, style: style, instrumental: instrumental)
                : await TitleSuggester.suggest(lyrics: lyrics, style: style, instrumental: instrumental)
            title = suggested; titleAuto = suggested
        }
        guard backend.connected, !backend.coverBusy else { return }
        let finalTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if mode == "cover" {
            let seconds = abcOpen ? maxSeconds : min(source?.seconds ?? maxSeconds, 360)
            backend.generate(title: finalTitle, style: style, lyrics: lyrics, cot: "melody", seed: seed, randomSeed: randomSeed, batch: batch, maxTokens: Int(seconds * 25), quality: quality, instrumental: instrumental, abc: abc, kind: "COVER", sourcePath: source?.path, promptFidelity: promptFidelity, styleFidelity: styleFidelity, sourceFidelity: sourceFidelity, targetSeconds: abcOpen ? nil : seconds, engines: engines, abcOpen: abcOpen)
        } else {
            backend.generate(title: finalTitle, style: style, lyrics: lyrics, cot: abc.isEmpty ? cot : "melody", seed: seed, randomSeed: randomSeed, batch: batch, maxTokens: Int(maxSeconds * 25), quality: quality, instrumental: instrumental, abc: abc, kind: "GENERATED", sourcePath: nil, promptFidelity: promptFidelity, styleFidelity: styleFidelity, sourceFidelity: 0, targetSeconds: nil, engines: engines, abcOpen: abcOpen)
        }
    }

    private func importSource() {
        if let imported = backend.importAudio() {
            source = imported; abc = imported.score; abcOpen = false; mode = "cover"; selectedTab = "Create"; transcriptionError = ""; suggestedGenre = ""; analysisWarnings = []
            maxSeconds = min(imported.seconds, 360)
            if backend.coverRuntimeReady { transcribe(imported) }
        }
    }

    private func transcribe(_ source: AudioSource) {
        transcribing = true; transcriptionError = ""; analysisWarnings = []
        backend.transcribe(source) { result in
            transcribing = false
            guard self.source?.id == source.id else { return }
            switch result {
            case .success(let analysis):
                self.source?.score = analysis.score; abc = analysis.score; abcOpen = false
                if !analysis.lyrics.isEmpty && lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    lyrics = analysis.lyrics; self.source?.lyrics = analysis.lyrics
                }
                analysisWarnings = analysis.warnings
                suggestedGenre = analysis.genre
                if style.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !analysis.genre.isEmpty { style = analysis.genre }
            case .failure(let error): transcriptionError = error.localizedDescription
            }
        }
    }

    private func installCoverRuntime() {
        installingCoverRuntime = true; transcriptionError = ""
        backend.installCoverRuntime { result in
            installingCoverRuntime = false
            if case .failure(let error) = result { transcriptionError = error.localizedDescription }
            else if let source { transcribe(source) }
        }
    }

    private func beginCover(_ song: Song) { abcOpen = false; source = backend.source(for: song); abc = song.score; mode = "cover"; selectedTab = "Create"; style = song.style; lyrics = song.lyrics; maxSeconds = min(song.seconds, 360); suggestedGenre = ""; analysisWarnings = []; transcriptionError = ""; title = (song.title.isEmpty ? "Song \(song.index)" : song.title) + " Cover" }
    private func reusePrompt(_ song: Song) {
        abcOpen = false
        mode = "new"; selectedTab = "Create"; source = nil; abc = ""
        style = song.style; lyrics = song.lyrics
        instrumental = song.lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        title = ""; transcriptionError = ""; suggestedGenre = ""; analysisWarnings = []
        lyricsExpanded = true; stylesExpanded = true
    }

    private func resetAdvancedOptions() {
        abcOpen = false
        cot = "full"; seed = 831001; randomSeed = false; batch = 2
        maxSeconds = mode == "cover" ? min(source?.seconds ?? 120, 360) : 120
        promptFidelity = 0.75; styleFidelity = 0.75; sourceFidelity = 1.0
        abc = mode == "cover" ? (source?.score ?? "") : ""
    }
    private func resetForm() { abcOpen = false; title = ""; style = ""; lyrics = ""; abc = ""; source = nil; mode = "new"; suggestedGenre = ""; transcriptionError = "" }

    private func fidelitySlider(_ label: String, value: Binding<Double>) -> some View {
        VStack(alignment: .leading, spacing: 6) { HStack { Text(label); Spacer(); Text("\(Int(value.wrappedValue * 100))%").monospacedDigit() }; Slider(value: value, in: 0...1, step: 0.05).accessibilityLabel(label).accessibilityValue("\(Int(value.wrappedValue * 100)) percent") }
    }

    private func syncRemote() {
        backend.useRemote(useRemote && engines == "gpu+ane" ? remote.phone : nil)
    }

    private func writeLyrics() async {
        guard !backend.busy, !backend.coverBusy, !writingLyrics, !naming else { return }
        writingLyrics = true
        defer { writingLyrics = false }
        let typed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        let userTitle = typed == titleAuto ? "" : typed
        if userTitle.isEmpty { title = ""; titleAuto = "" }
        switch await TitleSuggester.writeLyrics(style: style, title: userTitle, about: lyricsAbout) {
        case .success(let text)?: lyrics = text; lyricsVersion += 1
        case .failure(let error)?: backend.report("Could not write lyrics: \(error.localizedDescription)")
        case nil: backend.report("Enable Apple Intelligence on macOS 26 or later to write lyrics.")
        }
    }

    private var lyricsTools: some View {
        HStack(spacing: 12) {
            if writingLyrics { ProgressView().controlSize(.small) }
            Button { lyricsAbout = ""; askAbout = true } label: { Image(systemName: "sparkles") }
                .disabled(!TitleSuggester.modelAvailable || backend.busy || backend.coverBusy || writingLyrics || naming || showLyricsEditor)
                .help("Write lyrics with Apple Intelligence (macOS 26+)").accessibilityLabel("Write lyrics")
            Button { lyrics = "" } label: {
                Image(systemName: "trash").frame(width: 28, height: 28).contentShape(Rectangle())
            }.disabled(lyrics.isEmpty || backend.coverBusy).help("Clear lyrics").accessibilityLabel("Clear lyrics")
            Button { guard let previous = undoLyrics.popLast() else { return }; redoLyrics.append(lyrics); restoringLyrics = true; lyrics = previous } label: { Image(systemName: "arrow.uturn.backward") }.disabled(undoLyrics.isEmpty).help("Undo lyrics change")
            Button { guard let next = redoLyrics.popLast() else { return }; undoLyrics.append(lyrics); restoringLyrics = true; lyrics = next } label: { Image(systemName: "arrow.uturn.forward") }.disabled(redoLyrics.isEmpty).help("Redo lyrics change")
        }.buttonStyle(.plain).disabled(writingLyrics)
    }
    private func timeLabel(_ seconds: Double) -> String { String(format: "%d:%02d", Int(max(0, seconds)) / 60, Int(max(0, seconds)) % 60) }
    private func openOutput() { do { try FileManager.default.createDirectory(at: Paths.output, withIntermediateDirectories: true); NSWorkspace.shared.open(Paths.output) } catch { backend.report(error.localizedDescription) } }
    private func importScore() {
        let panel = NSOpenPanel(); panel.allowsMultipleSelection = false; panel.canChooseDirectories = false
        panel.allowedContentTypes = [UTType(filenameExtension: "abc") ?? .plainText, .plainText]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { abc = try String(contentsOf: url, encoding: .utf8); abcOpen = false; cot = "melody" } catch { backend.report("Could not read score: \(error.localizedDescription)") }
    }
}

/// Native, selectable, read-only text view: drag to select across lines, Cmd-C to copy, Cmd-F to find.
