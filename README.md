# Audio Transcription with Speaker Diarization

Transcribe audio files locally or on Google Colab, with optional speaker labelling. Free, no API key required for transcription.

---

## Files

| File | Description |
|------|-------------|
| `transcribe_v1_colab.ipynb` | Transcription only — no speaker labels |
| `transcribe_v2_colab.ipynb` | Transcription + speaker diarization |

---

## Quick Start

### Prerequisites
- Google account (for Colab + Drive)
- Hugging Face account (v2 only — free)

### v1 — Transcription only
1. Open `transcribe_v1_colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (`Runtime → Change runtime type`)
3. Upload audio to Google Drive
4. Update `AUDIO_FILE` path in Cell 3
5. Run all cells

### v2 — With speaker labels
1. Open `transcribe_v2_colab.ipynb` in Google Colab
2. Set runtime to **T4 GPU**
3. Upload audio to Google Drive
4. Accept gated model terms on Hugging Face (see below)
5. Update `AUDIO_FILE` and `HF_TOKEN` in Cells 3 and 6
6. Run all cells

#### Hugging Face setup (one-time)
1. Sign up at [huggingface.co](https://huggingface.co)
2. Accept terms at both URLs:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. `Settings → Access Tokens → New token (read)` → copy it

---

## Model Size Guide

| Model | Size | CPU Speed | GPU Speed | Notes |
|-------|------|-----------|-----------|-------|
| `tiny` | 75 MB | ~15 min / 30min audio | ~1 min | Rougher accuracy |
| `base` | 145 MB | ~45 min / 30min audio | ~2 min | Good default |
| `small` | 466 MB | ~90 min / 30min audio | ~4 min | Better accuracy |
| `medium` | 1.5 GB | Not recommended | ~8 min | Near-production |
| `large-v3` | 3 GB | Not recommended | ~15 min | Best accuracy |

---

## What Was Tried and Why It Changed

### Approach 1 — Local Python script (`openai-whisper`)
**Status: Discarded**

Started with a standalone `.py` script using the original `openai-whisper` library.

Issues encountered:
- **WinError 2 (FileNotFoundError):** Whisper shells out to `ffmpeg` via subprocess. On Windows, ffmpeg was not on the system PATH even after `imageio[ffmpeg]` injection via `os.environ["PATH"]` — the subprocess did not inherit it reliably.
- Fix attempted: monkey-patching `whisper.audio.load_audio` to call ffmpeg by its full absolute path. This worked but added fragility.

### Approach 2 — Local Jupyter notebook (`openai-whisper`)
**Status: Discarded**

Moved to a Jupyter notebook for easier step-by-step execution.

Issues encountered:
- ffmpeg PATH issue persisted on Windows, required the monkey-patch workaround
- **Kernel kept dying** mid-transcription on a laptop with Intel Iris Xe GPU — insufficient CPU/RAM to hold the model and process a 70MB audio file

### Approach 3 — Local Jupyter notebook (`faster-whisper`)
**Status: Discarded**

Switched to `faster-whisper` (CTranslate2 backend, ~4x faster on CPU, uses `int8` quantisation).

Issues encountered:
- Kernel still dying — the bottleneck was system RAM, not just compute. `faster-whisper` is lighter but a 30-minute audio file still overwhelmed the machine.
- Intel Iris Xe is integrated graphics — CUDA is NVIDIA-only, so no GPU acceleration possible locally.

### Approach 4 — Google Colab + `faster-whisper` ✅
**Status: Current solution (v1)**

Moved to Google Colab with a free T4 GPU. 30-minute file transcribed in ~2 minutes.

Issues encountered:
- **File truncation:** Direct browser upload to Colab silently truncated the 70MB file, resulting in only ~30 seconds of transcript. Fixed by uploading to Google Drive and mounting it in Colab.

### Approach 5 — Adding speaker diarization (`pyannote.audio`) ✅
**Status: Current solution (v2)**

Added `pyannote.audio` on top of v1 to label segments by speaker.

Issues encountered:

1. **`use_auth_token` deprecated:** `Pipeline.from_pretrained()` no longer accepts `use_auth_token`. Renamed to `token` in newer versions.

2. **Multiple gated models:** `pyannote/speaker-diarization-3.1` depends on `pyannote/segmentation-3.0`, which is separately gated. Both require individual Terms of Service acceptance on Hugging Face — the error message only surfaces the sub-model, making it non-obvious.

3. **`torch` not imported:** `pipeline.to(torch.device("cuda"))` requires an explicit `import torch`.

4. **Audio decoder error (`Invalid data found when processing input`):** pyannote is stricter than Whisper about audio format. The mp3 had a malformed header. Fixed by converting to 16kHz mono WAV using ffmpeg before passing to pyannote:
   ```
   ffmpeg -err_detect ignore_err -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
   ```

5. **`DiarizeOutput` API change:** Newer versions of pyannote wrap the result in a `DiarizeOutput` dataclass. The annotation is no longer directly iterable — it lives at `diarization.speaker_diarization`. Calling `.itertracks()` directly on the outer object raises `AttributeError`.

---

## Output Format

### v1
```
[00:00 → 00:04]  So the main thing we wanted to cover today was...
[00:04 → 00:09]  Right, and I think the key issue is the timeline.
```

### v2
```
[00:00 → 00:04]  SPEAKER_00: So the main thing we wanted to cover today was...
[00:04 → 00:09]  SPEAKER_01: Right, and I think the key issue is the timeline.
```

Speaker labels are anonymous (`SPEAKER_00`, `SPEAKER_01`, etc.) — pyannote does not identify who is who, only that they are different people.

---

## Dependencies

### v1
```
faster-whisper
```

### v2
```
faster-whisper
pyannote.audio
torch (pre-installed on Colab)
```
