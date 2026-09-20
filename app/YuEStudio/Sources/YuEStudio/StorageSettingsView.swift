import SwiftUI
import AppKit

enum StorageTarget: String, CaseIterable, Identifiable {
    case mainModels, coverModels, aneCache, coverInstallCache, installCache, sourceCaches

    var id: String { rawValue }

    var title: String {
        switch self {
        case .mainModels: "YuE2 model"
        case .coverModels: "Cover transcription models"
        case .aneCache: "Neural Engine cache"
        case .coverInstallCache: "Cover installer cache"
        case .installCache: "Installer cache"
        case .sourceCaches: "Python / build caches"
        }
    }

    var detail: String {
        switch self {
        case .mainModels: "Main generation model. If removed, YuE Studio will ask to download it again before generating."
        case .coverModels: "SheetSage2, MERT2, Whisper, genre classification and CLAP style analysis."
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
            [Paths.coverSupport.appendingPathComponent("models")]
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
    @Published var scanning = false
    @Published var deleting: StorageTarget?
    @Published var error: String?

    func refresh() {
        guard !scanning else { return }
        scanning = true
        let targets = StorageTarget.allCases.map { ($0, $0.paths) }
        Task {
            let measured = await Task.detached(priority: .utility) {
                Dictionary(uniqueKeysWithValues: targets.map { ($0.0, StorageScanner.allocatedSize(of: $0.1)) })
            }.value
            sizes = measured
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
                let marker = Paths.coverSupport.appendingPathComponent("installed-v3")
                if FileManager.default.fileExists(atPath: marker.path) {
                    try FileManager.default.trashItem(at: marker, resultingItemURL: nil)
                }
                let styleMarker = Paths.coverSupport.appendingPathComponent("installed-v4")
                if FileManager.default.fileExists(atPath: styleMarker.path) { try FileManager.default.trashItem(at: styleMarker, resultingItemURL: nil) }
                let lyricsMarker = Paths.coverSupport.appendingPathComponent("installed-v5")
                if FileManager.default.fileExists(atPath: lyricsMarker.path) { try FileManager.default.trashItem(at: lyricsMarker, resultingItemURL: nil) }
                backend.coverReady = false; backend.coverStyleReady = false; backend.coverLyricsReady = false
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

    var body: some View {
        Form {
            Section("Audio analysis engine") {
                Text(backend.coverRuntimeReady ? "Installed · ready for analysis" : "Engine installation required")
                if backend.coverRuntimeReady && !backend.coverLyricsReady { Text("Update to Whisper large-v3-turbo for the new lyric transcriber. Additional model download required.").font(.caption) }
                if backend.coverRuntimeReady && !backend.coverStyleReady { Text("Update the engine to download detailed style analysis.").font(.caption) }
                Button(backend.coverRuntimeReady ? "Update / Repair Engine" : "Install Engine") {
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
        .frame(width: 620, height: 520)
        .onAppear { storage.refresh() }
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
}
