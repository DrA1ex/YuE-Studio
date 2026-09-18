import Foundation
import Network

/// Finds the YuE Remote iPhone app on the local network (Bonjour, `_yuestudio._tcp`) and
/// resolves it to a host and port the Python worker can connect to.
@MainActor
final class RemoteBrowser: ObservableObject {
    struct Phone: Equatable { let name: String; let host: String; let port: Int }
    @Published var phone: Phone?
    @Published var detail = "looking for an iPhone running YuE Remote…"
    private var browser: NWBrowser?
    private var resolving: NWConnection?

    func start() {
        guard browser == nil else { return }
        let params = NWParameters.tcp
        params.includePeerToPeer = true
        let b = NWBrowser(for: .bonjour(type: "_yuestudio._tcp", domain: nil), using: params)
        b.stateUpdateHandler = { [weak self] state in
            Task { @MainActor in
                if case .failed(let e) = state { self?.detail = "Bonjour failed: \(e.localizedDescription)" }
            }
        }
        b.browseResultsChangedHandler = { [weak self] results, _ in
            Task { @MainActor in self?.update(results) }
        }
        b.start(queue: .main)
        browser = b
    }

    private func update(_ results: Set<NWBrowser.Result>) {
        guard let result = results.first else {
            phone = nil; detail = "no iPhone running YuE Remote on this network"; return
        }
        var name = "iPhone"
        if case .service(let n, _, _, _) = result.endpoint { name = n }
        detail = "resolving \(name)…"
        resolving?.cancel()
        let conn = NWConnection(to: result.endpoint, using: .tcp)
        resolving = conn
        conn.stateUpdateHandler = { [weak self] state in
            Task { @MainActor in
                guard let self else { return }
                switch state {
                case .ready:
                    if case .hostPort(let host, let port)? = conn.currentPath?.remoteEndpoint {
                        let h = "\(host)".components(separatedBy: "%")[0]      // strip the interface scope of link-local addresses
                        self.phone = Phone(name: name, host: h, port: Int(port.rawValue))
                        self.detail = "\(name) at \(h):\(port.rawValue)"
                    }
                    conn.cancel()
                case .failed(let e):
                    self.detail = "cannot reach \(name): \(e.localizedDescription)"; conn.cancel()
                default: break
                }
            }
        }
        conn.start(queue: .main)
    }
}
