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
            gauge
            statusBar
        }
        .padding(14)
        .background(background)
        .overlay(RoundedRectangle(cornerRadius: 18).stroke(.white.opacity(0.1)))
    }

    private var gauge: some View {
        ZStack {
            Canvas { context, size in
                drawGauge(context: &context, size: size)
            }
            .animation(.spring(response: 0.6, dampingFraction: 0.6), value: value)

            VStack(spacing: 2) {
                Spacer()
                Text(String(format: "%.2f", value))
                    .font(.system(size: 34, weight: .bold, design: .rounded).monospacedDigit())
                    .foregroundStyle(.white)
                Text("TFLOP/s")
                    .font(.caption.bold())
                    .foregroundStyle(.white.opacity(0.7))
                Text(caption)
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.white.opacity(0.6))
                    .lineLimit(1)
            }
            .padding(.bottom, 28)
        }
        .frame(height: 230)
    }

    private var statusBar: some View {
        HStack {
            Circle()
                .fill(running ? Color.green : Color.gray)
                .frame(width: 8, height: 8)
                .shadow(color: running ? .green : .clear, radius: 6)
            Text(running ? "NEURAL ENGINE ENGAGED" : "IDLE")
                .font(.caption2.bold())
                .foregroundStyle(.white.opacity(0.8))
            Spacer()
            Text(String(format: "%08.1f TFLOP", odometer))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.white.opacity(0.8))
        }
    }

    private var background: some ShapeStyle {
        LinearGradient(
            colors: [Color(red: 0.08, green: 0.09, blue: 0.14), Color.black],
            startPoint: .top,
            endPoint: .bottom
        )
    }

    private func drawGauge(context: inout GraphicsContext, size: CGSize) {
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        let radius = min(size.width, size.height) / 2 - 12
        let fraction = max(0, min(1, value / maximum))

        drawTrack(context: &context, center: center, radius: radius)
        drawRedline(context: &context, center: center, radius: radius)
        drawLitArc(context: &context, center: center, radius: radius, fraction: fraction)
        drawTicks(context: &context, center: center, radius: radius)
        drawNeedle(context: &context, center: center, radius: radius, fraction: fraction)
    }

    private func drawTrack(context: inout GraphicsContext, center: CGPoint, radius: Double) {
        var path = Path()
        path.addArc(center: center, radius: radius, startAngle: start,
                    endAngle: start + .degrees(sweep), clockwise: false)
        context.stroke(path, with: .color(.white.opacity(0.12)),
                       style: StrokeStyle(lineWidth: 14, lineCap: .round))
    }

    private func drawRedline(context: inout GraphicsContext, center: CGPoint, radius: Double) {
        var path = Path()
        path.addArc(center: center, radius: radius,
                    startAngle: start + .degrees(sweep * redline / maximum),
                    endAngle: start + .degrees(sweep), clockwise: false)
        context.stroke(path, with: .color(.red.opacity(0.55)),
                       style: StrokeStyle(lineWidth: 14, lineCap: .round))
    }

    private func drawLitArc(context: inout GraphicsContext, center: CGPoint, radius: Double, fraction: Double) {
        guard fraction > 0 else { return }
        var path = Path()
        path.addArc(center: center, radius: radius, startAngle: start,
                    endAngle: start + .degrees(sweep * fraction), clockwise: false)
        let gradient = Gradient(colors: [.cyan, .green, .yellow, .orange, .red])
        context.stroke(path, with: .conicGradient(gradient, center: center, angle: start),
                       style: StrokeStyle(lineWidth: 14, lineCap: .round))
    }

    private func drawTicks(context: inout GraphicsContext, center: CGPoint, radius: Double) {
        for i in 0...Int(maximum) {
            let angle = start + .degrees(sweep * Double(i) / maximum)
            let inner = point(center: center, angle: angle, radius: radius - 16)
            let outer = point(center: center, angle: angle, radius: radius - 26)
            var tick = Path()
            tick.move(to: inner)
            tick.addLine(to: outer)
            context.stroke(tick, with: .color(.white.opacity(0.6)), lineWidth: 2)

            let labelPoint = point(center: center, angle: angle, radius: radius - 40)
            let label = Text("\(i)")
                .font(.system(size: 12, weight: .semibold, design: .rounded))
                .foregroundColor(.white.opacity(0.8))
            context.draw(label, at: labelPoint)
        }
    }

    private func drawNeedle(context: inout GraphicsContext, center: CGPoint, radius: Double, fraction: Double) {
        let angle = start + .degrees(sweep * fraction)
        let tip = point(center: center, angle: angle, radius: radius - 30)
        let tail = point(center: center, angle: angle + .degrees(180), radius: 14)

        var needle = Path()
        needle.move(to: tail)
        needle.addLine(to: tip)
        context.stroke(needle, with: .color(.white),
                       style: StrokeStyle(lineWidth: 3, lineCap: .round))
        context.fill(Path(ellipseIn: CGRect(x: center.x - 7, y: center.y - 7, width: 14, height: 14)),
                     with: .color(.white))
        context.fill(Path(ellipseIn: CGRect(x: center.x - 3, y: center.y - 3, width: 6, height: 6)),
                     with: .color(.black))
    }

    private func point(center: CGPoint, angle: Angle, radius: Double) -> CGPoint {
        CGPoint(x: center.x + cos(angle.radians) * radius,
                y: center.y + sin(angle.radians) * radius)
    }
}
