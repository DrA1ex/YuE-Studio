import SwiftUI

/// A tachometer for the Neural Engine: TFLOP/s of the last pass, with a redline near the
/// chip's practical ceiling. Digits are monospaced so the needle and readout move together.
struct Tachometer: View {
    var value: Double          // TFLOP/s
    var maximum: Double = 8
    var redline: Double = 6
    var running: Bool
    var caption: String
    var odometer: Double       // total TFLOP done

    private let start = Angle.degrees(135), sweep = 270.0

    var body: some View {
        VStack(spacing: 6) {
            ZStack {
                Canvas { ctx, size in
                    let c = CGPoint(x: size.width / 2, y: size.height / 2)
                    let r = min(size.width, size.height) / 2 - 12
                    // Track.
                    var track = Path()
                    track.addArc(center: c, radius: r, startAngle: start, endAngle: start + .degrees(sweep), clockwise: false)
                    ctx.stroke(track, with: .color(.white.opacity(0.12)), style: StrokeStyle(lineWidth: 14, lineCap: .round))
                    // Redline.
                    var red = Path()
                    red.addArc(center: c, radius: r, startAngle: start + .degrees(sweep * redline / maximum), endAngle: start + .degrees(sweep), clockwise: false)
                    ctx.stroke(red, with: .color(.red.opacity(0.55)), style: StrokeStyle(lineWidth: 14, lineCap: .round))
                    // Lit arc.
                    let frac = max(0, min(1, value / maximum))
                    if frac > 0 {
                        var lit = Path()
                        lit.addArc(center: c, radius: r, startAngle: start, endAngle: start + .degrees(sweep * frac), clockwise: false)
                        let colors: [Color] = [.cyan, .green, .yellow, .orange, .red]
                        ctx.stroke(lit, with: .conicGradient(Gradient(colors: colors), center: c, angle: start), style: StrokeStyle(lineWidth: 14, lineCap: .round))
                    }
                    // Ticks and labels.
                    for i in 0...Int(maximum) {
                        let a = start + .degrees(sweep * Double(i) / maximum)
                        let major = true
                        let inner = CGPoint(x: c.x + cos(a.radians) * (r - 16), y: c.y + sin(a.radians) * (r - 16))
                        let outer = CGPoint(x: c.x + cos(a.radians) * (r - (major ? 26 : 22)), y: c.y + sin(a.radians) * (r - (major ? 26 : 22)))
                        var tick = Path(); tick.move(to: inner); tick.addLine(to: outer)
                        ctx.stroke(tick, with: .color(.white.opacity(0.6)), lineWidth: 2)
                        let lp = CGPoint(x: c.x + cos(a.radians) * (r - 40), y: c.y + sin(a.radians) * (r - 40))
                        ctx.draw(Text("\(i)").font(.system(size: 12, weight: .semibold, design: .rounded)).foregroundColor(.white.opacity(0.8)), at: lp)
                    }
                    // Needle.
                    let na = start + .degrees(sweep * frac)
                    let tip = CGPoint(x: c.x + cos(na.radians) * (r - 30), y: c.y + sin(na.radians) * (r - 30))
                    let tail = CGPoint(x: c.x - cos(na.radians) * 14, y: c.y - sin(na.radians) * 14)
                    var needle = Path(); needle.move(to: tail); needle.addLine(to: tip)
                    ctx.stroke(needle, with: .color(.white), style: StrokeStyle(lineWidth: 3, lineCap: .round))
                    ctx.fill(Path(ellipseIn: CGRect(x: c.x - 7, y: c.y - 7, width: 14, height: 14)), with: .color(.white))
                    ctx.fill(Path(ellipseIn: CGRect(x: c.x - 3, y: c.y - 3, width: 6, height: 6)), with: .color(.black))
                }
                .animation(.spring(response: 0.6, dampingFraction: 0.6), value: value)
                VStack(spacing: 2) {
                    Spacer()
                    Text(String(format: "%.2f", value)).font(.system(size: 34, weight: .bold, design: .rounded).monospacedDigit()).foregroundStyle(.white)
                    Text("TFLOP/s").font(.caption.bold()).foregroundStyle(.white.opacity(0.7))
                    Text(caption).font(.caption2.monospacedDigit()).foregroundStyle(.white.opacity(0.6)).lineLimit(1)
                }
                .padding(.bottom, 28)
            }
            .frame(height: 230)
            HStack {
                Circle().fill(running ? Color.green : Color.gray).frame(width: 8, height: 8)
                    .shadow(color: running ? .green : .clear, radius: 6)
                Text(running ? "NEURAL ENGINE ENGAGED" : "IDLE").font(.caption2.bold()).foregroundStyle(.white.opacity(0.8))
                Spacer()
                Text(String(format: "%08.1f TFLOP", odometer)).font(.caption.monospacedDigit()).foregroundStyle(.white.opacity(0.8))
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 18).fill(LinearGradient(colors: [Color(red: 0.08, green: 0.09, blue: 0.14), Color.black], startPoint: .top, endPoint: .bottom)))
        .overlay(RoundedRectangle(cornerRadius: 18).stroke(.white.opacity(0.1)))
    }
}
