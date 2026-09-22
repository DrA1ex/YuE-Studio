import SwiftUI
import AppKit

enum StorageTarget: String, CaseIterable, Identifiable {
    case mainModels, coverModels, coverAnalysisCache, aneCache, coverInstallCache, installCache, sourceCaches

    var id: String { rawValue }

    var title: String {
        switch self {
        case .mainModels: "YuE2 model"
        case .coverModels: "All cover model files"
        case .coverAnalysisCache: "Cover analysis cache"
        case .aneCache: "Neural Engine cache"
        case .coverInstallCache: "Cover installer cache"
        case .installCache: "Installer cache"
        case .sourceCaches: "Python / build caches"
        }
    }

    var detail: String {
        switch self {
        case .mainModels: "Main generation model. If removed, YuE Studio will ask to download it again before generating."
        case .coverModels: "Bulk cleanup for every cover-analysis model and the isolated MLX Whisper environment. Individual model groups can be removed below."
        case .coverAnalysisCache: "Resumable melody, lyric, genre and style evidence keyed by source audio."
        case .aneCache: "Compiled Neural Engine programs. They are regenerated automatically when needed."
        case .coverInstallCache: "Temporary package files used by the cover engine installer."
        case .installCache: "Temporary package files used by the main installer."
        case .sourceCaches: "Generated Python bytecode and local build folders. Safe to recreate."
        }
    }

    var paths: [URL] {
        switch self {
        case .mainModels:
            [Paths.models]
        case .coverModels:
            [Paths.coverSupport.appendingPathComponent("models"), Paths.coverSupport.appendingPathComponent("torch"), Paths.coverSupport.appendingPathComponent("whisper-env")]
        case .coverAnalysisCache:
            [Paths.coverAnalyses]
        case .aneCache:
            [Paths.aneCache]
        case .coverInstallCache:
            [Paths.coverSupport.appendingPathComponent("uv-cache")]
        case .installCache:
            [Paths.support.appendingPathComponent("uv-cache")]
        case .sourceCaches:
            StorageScanner.generatedCaches(under: Paths.src)
        }
    }
}

enum StorageScanner {
    static func generatedCaches(under root: URL) -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: []
        ) else { return [] }

        var result: [URL] = []
        for case let url as URL in enumerator {
            let name = url.lastPathComponent
            if ["__pycache__", ".pytest_cache", "build"].contains(name),
               (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                result.append(url)
                enumerator.skipDescendants()
            }
        }
        return result
    }

    static func allocatedSize(of paths: [URL]) -> Int64 {
        let fm = FileManager.default
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .totalFileAllocatedSizeKey, .fileAllocatedSizeKey]
        var total: Int64 = 0

        for path in paths where fm.fileExists(atPath: path.path) {
            if let values = try? path.resourceValues(forKeys: keys), values.isRegularFile == true {
                total += Int64(values.totalFileAllocatedSize ?? values.fileAllocatedSize ?? 0)
                continue
            }

            guard let enumerator = fm.enumerator(
                at: path,
                includingPropertiesForKeys: Array(keys),
                options: [.skipsPackageDescendants]
            ) else { continue }

            for case let file as URL in enumerator {
                guard let values = try? file.resourceValues(forKeys: keys), values.isRegularFile == true else { continue }
                total += Int64(values.totalFileAllocatedSize ?? values.fileAllocatedSize ?? 0)
            }
        }
        return total
    }
}

@MainActor
final class StorageSettingsModel: ObservableObject {
    @Published var sizes: [StorageTarget: Int64] = [:]
    @Published var componentSizes: [CoverComponent: Int64] = [:]
    @Published var scanning = false
    @Published var deleting: StorageTarget?
    @Published var error: String?

    func refresh() {
        guard !scanning else { return }
        scanning = true
        let targets = StorageTarget.allCases.map { ($0, $0.paths) }
        let components = CoverComponent.allCases.map { ($0, $0.modelPaths) }
        Task {
            let measured = await Task.detached(priority: .utility) {
                let targetSizes = Dictionary(uniqueKeysWithValues: targets.map { ($0.0, StorageScanner.allocatedSize(of: $0.1)) })
                let componentSizes = Dictionary(uniqueKeysWithValues: components.map { ($0.0, StorageScanner.allocatedSize(of: $0.1)) })
                return (targetSizes, componentSizes)
            }.value
            sizes = measured.0
            componentSizes = measured.1
            scanning = false
        }
    }

    func moveToTrash(_ target: StorageTarget, installer: Installer, backend: Backend) {
        guard deleting == nil else { return }
        deleting = target
        error = nil

        if target == .mainModels { backend.quit() }

        do {
            for path in target.paths where FileManager.default.fileExists(atPath: path.path) {
                try FileManager.default.trashItem(at: path, resultingItemURL: nil)
            }

            if target == .coverModels {
                backend.clearCoverComponentMarkers()
                backend.refreshCoverInstallationState()
            }

            if target == .mainModels {
                backend.shuttingDown = false
                installer.check()
            }
        } catch {
            self.error = error.localizedDescription
        }

        if target == .mainModels { backend.shuttingDown = false }

        deleting = nil
        refresh()
    }
}

struct StorageSettingsView: View {
    @EnvironmentObject var installer: Installer
    @EnvironmentObject var backend: Backend
    @StateObject private var storage = StorageSettingsModel()
    @State private var pendingDelete: StorageTarget?
    @State private var pendingComponent: CoverComponent?
    @AppStorage(CoverAnalysisPreferences.lyricsKey) private var analyzeLyrics = true
    @AppStorage(CoverAnalysisPreferences.genreKey) private var analyzeGenre = true
    @AppStorage(CoverAnalysisPreferences.styleKey) private var analyzeStyle = true
    @AppStorage(CoverAnalysisPreferences.vocalActivityKey) private var analyzeVocalActivity = true
    @AppStorage(CoverAnalysisPreferences.lyricsBackendKey) private var lyricsBackend = CoverLyricsBackend.automatic.rawValue

    var body: some View {
        Form {
            Section("Audio analysis engine") {
                Text(backend.coverRuntimeReady ? "Installed · ready for analysis" : "Engine installation required")
                Text("Only Melody and score is required. Configure optional analysis stages below; the model manager shows the exact downloads, purpose, and disk usage for each component.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("Install enabled components") {
                    backend.installCoverRuntime { result in
                        if case .failure(let error) = result { storage.error = error.localizedDescription }
                        storage.refresh()
                    }
                }.disabled(backend.coverBusy || backend.busy)
                if backend.coverBusy {
                    ProgressView()
                    Text(backend.coverStatus).font(.caption)
                }
            }
            Section("Cover analysis") {
                Text("Melody and score always runs because the ABC melody is required for source-audio covers. Everything below is optional.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Toggle("Extract lyrics", isOn: $analyzeLyrics)
                if analyzeLyrics {
                    Picker("Lyrics backend", selection: $lyricsBackend) {
                        ForEach(CoverLyricsBackend.allCases) { backend in
                            Text(backend.title).tag(backend.rawValue)
                        }
                    }
                    Toggle("Detect vocal regions first", isOn: $analyzeVocalActivity)
                    Text("Hybrid Demucs is only a preprocessing step for Whisper. Disable it to transcribe the original mix directly.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Toggle("Detect genre", isOn: $analyzeGenre)
                Toggle("Analyze detailed style", isOn: $analyzeStyle)
                Text("Genre classification and CLAP style analysis are independent stages; neither is required for lyrics or melody extraction.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section("Cover model manager") {
                ForEach(CoverComponent.allCases) { component in
                    componentRow(component)
                }
            }
            Section("Downloaded models") {
                storageRow(.mainModels)
                storageRow(.coverModels)
            }

            Section("Generated caches") {
                storageRow(.aneCache)
                storageRow(.coverInstallCache)
                storageRow(.installCache)
                storageRow(.sourceCaches)
            }

            if backend.busy || backend.coverBusy {
                Text("Storage changes are disabled while generation or transcription is running.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let error = storage.error {
                Text(error).font(.caption).foregroundStyle(.red)
            }

            HStack {
                Text("Data folder").foregroundStyle(.secondary)
                Spacer()
                Button("Open") { NSWorkspace.shared.open(Paths.support) }
                Button("Refresh") { storage.refresh() }.disabled(storage.scanning)
            }
        }
        .formStyle(.grouped)
        .padding(8)
        .frame(width: 720, height: 760)
        .onAppear { backend.refreshCoverInstallationState(); storage.refresh() }
        .alert(
            "Delete \(pendingDelete?.title ?? "data")?",
            isPresented: Binding(
                get: { pendingDelete != nil },
                set: { if !$0 { pendingDelete = nil } }
            )
        ) {
            Button("Cancel", role: .cancel) { pendingDelete = nil }
            Button("Move to Trash", role: .destructive) {
                if let target = pendingDelete {
                    storage.moveToTrash(target, installer: installer, backend: backend)
                }
                pendingDelete = nil
            }
        } message: {
            Text(pendingDelete?.detail ?? "")
        }
        .alert(
            "Remove \(pendingComponent?.title ?? "component")?",
            isPresented: Binding(
                get: { pendingComponent != nil },
                set: { if !$0 { pendingComponent = nil } }
            )
        ) {
            Button("Cancel", role: .cancel) { pendingComponent = nil }
            Button("Move to Trash", role: .destructive) {
                if let component = pendingComponent {
                    backend.removeCoverComponent(component) { result in
                        if case .failure(let error) = result { storage.error = error.localizedDescription }
                        storage.refresh()
                    }
                }
                pendingComponent = nil
            }
        } message: {
            Text(pendingComponent?.detail ?? "The component files will be moved to the macOS Trash.")
        }
    }

    @ViewBuilder
    private func storageRow(_ target: StorageTarget) -> some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(target.title)
                Text(target.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 16)
            if storage.scanning && storage.sizes[target] == nil {
                ProgressView().controlSize(.small)
            } else {
                Text(sizeText(storage.sizes[target] ?? 0))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            Button("Delete…") { pendingDelete = target }
                .disabled((storage.sizes[target] ?? 0) == 0 || storage.deleting != nil || backend.busy || backend.coverBusy)
        }
        .padding(.vertical, 4)
    }

    private func sizeText(_ bytes: Int64) -> String {
        guard bytes > 0 else { return "—" }
        return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    @ViewBuilder
    private func componentRow(_ component: CoverComponent) -> some View {
        let installed = backend.coverComponentReady(component)
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: installed ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(installed ? .green : .secondary)
                .padding(.top, 2)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(component.title)
                    if component.required { Text("Required").font(.caption2).foregroundStyle(.secondary) }
                    if component == .mlxWhisper { Text("Optional").font(.caption2).foregroundStyle(.secondary) }
                }
                    Text(component.detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(component.models.joined(separator: " · "))
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(component.relationship)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            if storage.scanning && storage.componentSizes[component] == nil {
                ProgressView().controlSize(.small)
            } else {
                Text(sizeText(storage.componentSizes[component] ?? 0))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            if installed {
                Button("Remove…") { pendingComponent = component }
                    .buttonStyle(.bordered)
                    .disabled(backend.busy || backend.coverBusy)
            } else {
                Button("Install") {
                    storage.error = nil
                    backend.installCoverComponent(component) { result in
                        if case .failure(let error) = result { storage.error = error.localizedDescription }
                        storage.refresh()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(backend.busy || backend.coverBusy)
            }
        }
        .padding(.vertical, 3)
    }
}
