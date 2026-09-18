import SwiftUI
import AppKit

struct RootView: View {
    @EnvironmentObject var installer: Installer
    @EnvironmentObject var backend: Backend
    var body: some View {
        Group {
            switch installer.state {
            case .ready: ContentView()
            case .checking: ProgressView("Checking installation").frame(minWidth: 400, minHeight: 200)
            default: SetupView()
            }
        }
        .onAppear { installer.check(); NSApp.setActivationPolicy(.regular); NSApp.activate(ignoringOtherApps: true) }
        .onChange(of: installer.state) { _, new in if new == .ready { backend.start() } }
    }
}
