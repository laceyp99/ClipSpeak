# ClipSpeak

A small Windows tray app that reads copied text aloud using local Piper speech
synthesis. Copy text in any app, press **Ctrl+Alt+A**, and ClipSpeak adds it to
the reading queue.

- Local Ryan English voice, kept loaded for repeated readings.
- CUDA acceleration with CPU fallback.
- Pitch-preserving playback speed from **1x to 2x**, default **1.5x**.
- Adjustable volume, default **22.5%**.
- Cleanup of common Markdown headings, emphasis, lists, links, and code delimiters.
- Copied text and generated audio stay in memory. No LLM or HTTP service.

## Setup

Requires **Windows x64**, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and Git. The project uses Python 3.12; uv manages the project environment.

Run these commands in PowerShell:

```powershell
git clone https://github.com/laceyp99/ClipSpeak.git
cd ClipSpeak
uv sync --locked
```

Download the Ryan voice [model](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx)
and [configuration](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx.json).
Create `%LOCALAPPDATA%\ClipSpeak\voices` and place both files there, keeping their names:

```text
en_US-ryan-high.onnx
en_US-ryan-high.onnx.json
```

Dependencies and voice files require an internet connection during setup.
Routine reading works offline afterward.

## Start the app

From the repository folder:

```powershell
uv run --locked clipspeak --tray
```

For CPU processing explicitly:

```powershell
uv run --locked clipspeak --tray --cpu
```

For CUDA, supply an existing directory containing compatible **CUDA 12 and
cuDNN 9 DLLs**:

```powershell
uv run --locked clipspeak --tray --cuda-dll-dir 'C:\path\to\runtime-dlls'
```

Replace the example path with your runtime directory. CUDA is optional; see
[development notes](docs/development.md) for the tested configuration and fallback
details. Quit the running instance before restarting to load code updates.

## Use the tray controls

Copy text normally, then press **Ctrl+Alt+A**. Repeated presses submit separate
items, including identical text. New items wait for the current reading.

Right-click the speaker tray icon to open its menu. Left-click opens Settings.

| Action | Behavior |
| --- | --- |
| Pause | Hold the reading position and queue. |
| Resume | Continue playback; after an error, retry the active item from its beginning. |
| Stop | Cancel the current reading and clear pending items. |
| Clear Queue | Remove pending items while preserving the current reading. |
| Settings | Change the read shortcut, speed, volume, and startup preference. |
| Quit | Stop reading and exit. |

Settings changes apply to future submissions. Active and queued items keep their
original speed and volume. Preferences are saved in
`%LOCALAPPDATA%\ClipSpeak\settings.json`; saved values override the defaults.

## Current limits

- Windows startup integration is pending. The startup checkbox currently saves
  a preference only and does not enable sign-in launch.
- Reads plain Unicode clipboard text, without modifying the clipboard. Images
  and scanned documents require a separate text extraction step.
- Accepts up to 100,000 characters per item, 20 pending items, and 500,000 total
  queued characters including the active item. Excess submissions are rejected.
- Plays through the default audio output. Live device switching still needs
  verification.
- Markdown cleanup handles common syntax; tables and HTML are not converted.

## Development

```powershell
uv run --locked pytest -q
uv run --locked clipspeak
```

The second command prints runtime diagnostics. Detailed setup, audio measurements,
and verification notes are in [docs/development.md](docs/development.md).

## License

ClipSpeak source is covered by the [MIT license](LICENSE). Dependencies and voice
assets have their own terms. The [Ryan model card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/ryan/high/MODEL_CARD)
lists its training dataset license as CC BY-NC-SA 4.0.
