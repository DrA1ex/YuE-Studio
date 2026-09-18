// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "YuEStudio",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "YuEStudio",
            exclude: ["Resources/AppIcon.icon"],
            swiftSettings: [.unsafeFlags(["-parse-as-library"])],
            // Foundation Models (the on-device language model) exists from macOS 26; weak-link it so
            // the app still launches on macOS 14 and 15, where the title suggester falls back.
            linkerSettings: [.unsafeFlags(["-Xlinker", "-weak_framework", "-Xlinker", "FoundationModels"])]
        )
    ],
    swiftLanguageVersions: [.v5]
)
