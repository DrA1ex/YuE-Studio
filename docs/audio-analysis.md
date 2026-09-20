# Audio analysis improvements

Whisper uses multilingual `large-v3-turbo` in the isolated cover environment. Settings exposes six independent components with short descriptions, per-component download sizes, install/remove actions and a separate "Install recommended components" action. The MLX Whisper 0.4.3 environment and converted `large-v3-turbo` weights are optional on Apple Silicon: MLX is preferred automatically when available and Transformers remains the fallback. Removing lyrics, genre or style no longer aborts the rest of the analysis; the result records a clear warning and continues with the installed stages. The melody/score component remains the only required component for creating a cover. Complete legacy `small` snapshots remain usable offline with an explicit upgrade message; partial snapshots are skipped. Segment timestamps, single-beam decoding, temperature fallback, silence/log-probability thresholds and disabled previous-window conditioning are used for the primary result. Word alignment remains a guarded recovery path in the adapter, so a timestamp indexing failure cannot discard the transcript. Only segments with valid timestamps wholly inside near-digital silence are removed, retaining the original transcript. These settings do not establish transcription accuracy on singing.

Lyrics retain recognized words and repetitions. Line layout uses word pauses and punctuation, with a 12-word upper bound. Repeated whole passages are aligned using normalized token anchors, tolerating small transcription differences without rewriting words. The dominant recurrent passage suggests `[chorus]`; repeated lead-ins suggest `[pre-chorus]`, and short surrounding passages can suggest `[intro]`, `[bridge]` or `[outro]` from position. Unclassified text remains `[verse]` without arbitrary four-line section splits. These are editable heuristics, not acoustic section detection, and cannot guarantee reference labels. Matching is bounded to 3,000 meaningful tokens; longer input keeps its text and line layout without section inference. Raw text, timestamps and warnings are retained in imported audio metadata.

Genre classification averages three bounded excerpts and preserves the top five candidates plus per-excerpt rankings. The displayed suggestion reports close candidates as a hybrid and abstains when evidence is weak; scores are never presented as calibrated confidence. CLAP reuses text embeddings and adds independently supported descriptors across instrumentation, vocal delivery, mood, production and pace, retaining per-excerpt support. Scores are cosine similarities, not confidence probabilities. Ambiguous categories are omitted. A missing CLAP installation leaves genre analysis available and displays an upgrade message. Existing users download the additional pinned model through Settings → Update / Repair Engine.

Each analysis is keyed by the normalized audio file's SHA-256 and stores melody, acoustic structure, vocal activity, lyrics, genre and style stages atomically under `cover-runtime/analyses`. A completed stage records its model revision, options and timing; a later run reuses only a matching stage. The cache preserves segment diagnostics and failed-stage evidence, and it can be removed independently from downloaded models in Settings. The final import metadata contains the cache key, stage status, segment timing diagnostics, acoustic repetition/tempo evidence, vocal windows, genre candidates and excerpt rankings.

When the torchaudio Hybrid Demucs MUSDB model is installed, the app preserves the source channels, extracts vocal energy in bounded chunks and uses that energy only to select transcription windows. Installation downloads the asset directly and verifies its minimum size before marking it ready; it does not load the 319 MB model into memory during download. Whisper continues to decode the original mix, which avoids promoting separator artefacts to lyrics. If the separator is unavailable, the full original mix remains the fallback. The acoustic structure fallback computes spectral repetition and cautious tempo candidates; it uses repetition only to choose between lyric-derived section candidates and does not invent semantic labels without lyric evidence.

Models run sequentially with inference mode and explicit release between stages. This bounds concurrent model memory. CLAP uses the existing Transformers/Torch runtime instead of adding an audio language model runtime. CPU FP32 remains the compatibility baseline; no MPS speedup or benchmark superiority is claimed without measurement. Installation limits downloads to runtime assets and inference resolves local snapshots.

The install completion callback now runs after busy/task/process state is released. It can start analysis immediately without being rejected or having its task cleared by the installer. Settings are accessible from the sidebar. The Lyrics and Styles headers have independent clear buttons (lyrics clearing participates in undo). Advanced options have a reset action restoring the original planning, seed, variants, duration and fidelity defaults; cover mode retains the source score and duration. The library shares one action menu between the ellipsis and right-click, including direct Cover and Reuse Prompt & Lyrics actions. A button below the library moves the reusable audio-analysis cache to the macOS Trash so the next analysis starts from zero while songs and source audio remain. Button hit regions include their padded labels; advanced options share the editor card design and sliders have separate labels, values and accessibility names.

## Technical references

- Upgraded Whisper checkpoint and decoding options: https://huggingface.co/openai/whisper-large-v3-turbo
- Whisper version-matched API and optimization options: https://huggingface.co/docs/transformers/v4.45.2/en/model_doc/whisper
- CLAP architecture and intended similarity use: https://github.com/LAION-AI/CLAP
- Pinned CLAP checkpoint: https://huggingface.co/laion/clap-htsat-unfused/tree/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a
- SwiftUI button label composition: https://developer.apple.com/documentation/swiftui/button

Validation uses generated PCM, installed Whisper preprocessing without weights, descriptor ambiguity cases, lyric preservation and repeated sections, native state/metadata tests, compilation and package checks. No UI automation, playback checks or real-song inference is used. Musical accuracy, actual click behavior and performance on real songs remain unmeasured.

Packaging for this update uses `bash app/package.sh --app-only`, producing `releases/YuE Studio.app` directly without a DMG.

## Whisper alignment recovery

The original ASR pipeline can raise `IndexError: list index out of range` when
word timestamp arrays are shorter than decoded token sequences. A synthetic
vocabulary/tokenizer test reproduces the original failure and verifies that the
adapter retains the same recognized words with segment timestamps. This matches
a failure mechanism, not a proven stack trace of the user's original recording
(the old CLI logged only the exception message).

`RobustWhisperPipeline` uses `input_features` and the real mask explicitly, adapts
legacy/empty decoder caches to `EncoderDecoderCache`, and validates timestamp
length, finiteness and ordering before word decoding. If generation itself raises
an IndexError during the word pass, it retries once without word alignment, beam
search or cache. If only text alignment fails, it reuses already generated tokens
without rerunning the model. Empty long-form segments yield empty text. Other
errors and failed recovery still propagate with full tracebacks; no endless
retry or fabricated transcript is introduced. Recovery warnings and the actual
timestamp mode are included in saved analysis and shown by the app.

Only SciPy's specific optional non-data WAV chunk notice is ignored for
AVFoundation output; malformed/truncated WAV warnings remain enabled.

Validation includes the real pinned Transformers pipeline with four small,
random-weight, four-decoder-layer models and synthetic 31-second PCM, plus
deterministic recovery tests. No real-song inference or UI testing was performed.
Existing model downloads remain valid; this update needs no model reinstall.
