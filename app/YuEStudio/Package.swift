// swift-tools-version:5.9
import PackageDescription
import Foundation
let microphonePlist = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Info.plist").path

let package = Package(
    name: "YuEStudio",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "YuEStudio", path: "Sources/YuEStudio", exclude: ["Resources"],
                          swiftSettings: [.unsafeFlags(["-parse-as-library"])],
                          linkerSettings: [.unsafeFlags(["-Xlinker", "-weak_framework", "-Xlinker", "FoundationModels", "-Xlinker", "-sectcreate", "-Xlinker", "__TEXT", "-Xlinker", "__info_plist", "-Xlinker", microphonePlist])]),
        .testTarget(name: "YuEStudioTests", dependencies: ["YuEStudio"])
    ]
)
