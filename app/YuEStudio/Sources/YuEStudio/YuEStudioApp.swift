import SwiftUI
import AVFoundation
import AppKit
import UniformTypeIdentifiers

@main
struct YuEStudioApp: App {
    @StateObject private var backend = Backend()
    @StateObject private var installer = Installer()
    var body: some Scene {
        WindowGroup("YuE Studio") { RootView().environmentObject(backend).environmentObject(installer) }
            .commands {
                CommandGroup(replacing: .appTermination) { Button("Quit YuE Studio") { installer.cancel(); backend.quit(); NSApp.terminate(nil) }.keyboardShortcut("q") }
                CommandGroup(after: .appSettings) {
                    Button("Open Songs Folder") { NSWorkspace.shared.open(Paths.output) }
                    Button("Repair Installation…") { backend.quit(); backend.shuttingDown = false; installer.repair() }.disabled(!Paths.packaged || backend.busy || backend.coverBusy)
                }
            }
        Settings {
            StorageSettingsView()
                .environmentObject(backend)
                .environmentObject(installer)
        }
    }
}
