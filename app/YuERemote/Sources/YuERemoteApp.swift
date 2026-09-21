import SwiftUI

@main
struct YuERemoteApp: App {
    var body: some Scene { WindowGroup { RootView() } }
}

struct RootView: View {
    @StateObject private var server = Server()

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 10) {
                Tachometer(value: server.tflops, running: server.running,
                           caption: server.running ? server.session : "waiting for a song", odometer: server.odometer)
                Group {
                    row("Status", server.state)
                    row("Mac", server.peer)
                    row("Weights", server.weights)
                    row("Session", server.session)
                    if !server.passLine.isEmpty { row("Last pass", server.passLine) }
                }
                Divider()
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 2) {
                            ForEach(Array(server.lines.enumerated()), id: \.offset) { i, line in
                                Text(line).font(.caption.monospaced()).id(i).textSelection(.enabled)
                            }
                        }
                    }
                    .onChange(of: server.lines.count) { _, n in if n > 0 { proxy.scrollTo(n - 1) } }
                }
            }
            .padding()
            .navigationTitle("YuE Remote")
            .toolbar { NavigationLink("Bench") { BenchView() } }
        }
        .onAppear { server.start() }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(alignment: .top) {
            Text(label).font(.caption.bold()).frame(width: 70, alignment: .leading)
            Text(value).font(.caption.monospaced())
        }
    }
}

struct BenchView: View {
    @StateObject private var bench = Bench()
    @State private var bucket = Bench.buckets[0]
    @State private var layersToHold = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(bench.deviceLine).font(.caption.monospaced())
            Picker("Bucket", selection: $bucket) {
                ForEach(Bench.buckets) { b in Text(b.label).tag(b) }
            }.pickerStyle(.menu)
            Stepper("Hold \(layersToHold) layers of weights", value: $layersToHold, in: 0...28)
            HStack {
                Button("Run") { Task { await bench.run(bucket: bucket, hold: layersToHold) } }
                    .buttonStyle(.borderedProminent).disabled(bench.busy)
                Button("Run all from here") { Task { await bench.runAll(from: bucket, hold: layersToHold) } }
                    .buttonStyle(.bordered).disabled(bench.busy)
                Button("Clear") { bench.log.removeAll() }.disabled(bench.busy)
            }
            Text(bench.memoryLine).font(.caption.monospaced())
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(bench.log.enumerated()), id: \.offset) { i, line in
                            Text(line).font(.caption.monospaced()).id(i).textSelection(.enabled)
                        }
                    }
                }
                .onChange(of: bench.log.count) { _, n in if n > 0 { proxy.scrollTo(n - 1) } }
            }
        }
        .padding()
        .navigationTitle("Neural Engine bench")
        .onAppear { bench.start() }
    }
}
