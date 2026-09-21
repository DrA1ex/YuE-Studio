// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "YuEStudio",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "YuEStudio", path: "Sources/YuEStudio", exclude: ["Resources"],
                          swiftSettings: [.unsafeFlags(["-parse-as-library"])],
                          linkerSettings: [.unsafeFlags(["-Xlinker", "-weak_framework", "-Xlinker", "FoundationModels"])]),
        .testTarget(name: "YuEStudioTests", dependencies: ["YuEStudio"])
    ]
)
