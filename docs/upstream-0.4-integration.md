# Fork alignment with YuE Studio 0.4.0

Date: 2026-09-21. Upstream: `tonywestonuk/YuE-Studio`, tag `v0.4.0`,
commit `93ed8e8`. Fork starting point: `d4f79dd` (`0.2.0-ui.2`).
The integration merges the release tag, not upstream's later main-branch commits.
App packaging now identifies the Mac app as **0.4.0**; the Python inference package
keeps upstream's independent version, 0.1.6.

## Feature mapping

| Upstream change | Fork integration |
| --- | --- |
| iPhone synthesis and GPU-to-phone migration | Remote client, MIL generation, scheduler lane, Bonjour discovery, status and engine labels integrated. Enable in **More Options**, select **GPU + Neural Engine**, then enable the iPhone toggle. |
| Separate quality and engine choices | Existing Draft/Full control retained; GPU-only or GPU + Neural Engine selectable for either quality. Render/recovery also forwards the engine choice. |
| Automatic song titles | Empty/previously automatic titles are suggested from lyrics; deterministic first-line/style fallback when the model is unavailable or generation is busy. Unique run folders include a sanitized title. |
| Lyric writing and outro/theme fixes | Sparkles button in Lyrics uses the upstream guided writer, including the trailing disposable field, response limit, scrubbing, fresh session and automatic-title reset. macOS 26 + available Apple Intelligence required. Disabled during generation/cover analysis. |
| Saved-token rendering and submit failures | Existing fork recovery/correlation retained; missing audio paths resolve to saved-token directories. Cover metadata survives. |
| Neural Engine compiler reliability | Background compilation without loading and transient inference retry integrated; local native library rebuilt with the new entry point. |
| Optional MLX import | Upstream import fix integrated. |
| Source split | Customized app split into Paths, Installer, Song, Backend, ContentView and other focused files. Existing player and cover coordinator retained. |
| SheetSage transcription UI | Fork's broader cover workflow retained (melody, lyrics, genre/style, VAD, cache and recovery). Upstream worker transcription commands/tool remain available. |
| Icon and transport | Fork's existing Icon Composer icon, waveform player, seeking and volume retained. |

## Integration corrections

- Upstream's iOS project referenced six untracked benchmark `.mlpackage` files.
  Removed these mandatory build inputs and the original author's development team;
  set your own team when installing on a phone. Production synthesis receives its
  programs from the Mac. The optional Bench screen still requires separately
  supplied benchmark models; these were not generated or bundled.
- Phone loss before synthesis, and refusal/disconnection after GPU migration,
  return the job to local synthesis. Mid-migration retry restarts the solve from
  the saved tokens/seed; it does not resume the lost remote solver state.
- Preserve IPv6 interface scopes during Bonjour resolution and cancel obsolete
  discovery/resolution when the toggle is disabled.
- Ignore phone offers until the worker is ready; clear remembered offers on worker
  termination, so reconnection can offer the phone again.
- Weak-link FoundationModels. Verified `LC_LOAD_WEAK_DYLIB` in the release binary;
  the app's minimum target remains macOS 14.
- Packaging includes Bonjour/local-network declarations and hashes the new remote
  Python sources in the payload version, preserving existing runtime/model reuse.

## Verification

All tests use synthetic fixtures/tiny random models or mocks; no music checkpoint
was downloaded and no full song was generated for this integration.

- Python: **235 passed, 34 subtests passed, 11 skipped**. Includes 21 new integration
  cases covering engine selection, metadata, unique concurrent run folders,
  scheduler assignment/migration, phone capacity/fallback, binary wire replies,
  saved-token recovery and unloaded background ANE compilation.
- Skips: 8 CUDA-capture cases, 1 optional vLLM case, 1 NVIDIA GPU case and 1
  original-model comparison without its reference checkpoint. 15 existing PyTorch
  weight_norm deprecation warnings.
- Swift: **15 passed**, including four additional tests for fallback titles,
  upstream/fork metadata precedence, phone events/migration and pre-ready offers.
- macOS release build: passed.
- iOS device-target Debug build with signing disabled: passed.
- Native ANE dylib: clang build passed; `ane_program_compile` exported.
- Actual worker subprocess: ready, ping/pong, correlated invalid-request rejection
  and clean quit passed, with no output artifacts created.
- Package scripts, iOS plist/project syntax, and Git whitespace checks passed.

Commands (Python test dependencies are temporary uv additions, not installed into
or pinned over the user's application runtime):

```sh
uv run --no-project --with pytest --with scipy --with numpy==2.2.6 \
  --python "$HOME/Library/Application Support/YuE Studio/env/bin/python" \
  python -m pytest -q -rs
swift test --package-path app/YuEStudio
swift build --package-path app/YuEStudio -c release
xcodebuild -project app/YuERemote/YuERemote.xcodeproj -scheme YuERemote \
  -configuration Debug -sdk iphoneos -destination 'generic/platform=iOS' \
  -derivedDataPath /tmp/yue-remote-0.4-build CODE_SIGNING_ALLOWED=NO build
```

Not verified here: real iPhone discovery/transfer/compilation/synthesis, numerical
parity on physical engines, Apple Intelligence generation, manual GUI interaction,
launch on macOS 14/15, or a packaged DMG upgrade. Upstream limits phone synthesis to
4096 padded rows (roughly 2m40s); actual acceptance depends on the device. Source
and build alignment does not certify those hardware-dependent behaviors.

The initial integration exported no DMG and did not replace the installed
app/runtime/models. The subsequent authorized export is documented in
[release 0.4.0-ui.1](release-notes/v0.4.0-ui.1.md), including DMG/IPA integrity checks.
The previous exported Mac app and DMG were preserved locally.

## Post-release alignment (0.4.0-ui.2)

The subsequent merge includes upstream main through `69e85c7`, including
`6163abf` (hum recording/open-score continuation) and `69e85c7` (explicit ANE
compilation phase). The earlier release-only boundary above describes ui.1.
The hum feature is adapted to the fork's Create pane, dark score review and
existing analysis coordinator; full cover analysis is retained. See the
[ui.2 release notes](release-notes/v0.4.0-ui.2.md) for UI paths and validation.
