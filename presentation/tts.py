"""Synthesize narration with Kokoro (open-source, runs locally, free).

Called by make_video.py inside its own virtualenv (presentation/.tts) so the heavy TTS dependencies
stay out of the project's requirements.

Usage: presentation/.tts/bin/python presentation/tts.py jobs.json
jobs.json: [{"text": "...", "out": "/path/beat.wav"}, ...]
"""

import json
import sys

import numpy as np
import soundfile as sf
from kokoro import KPipeline

VOICE = "af_heart"
SPEED = 0.82
SAMPLE_RATE = 24000


def main() -> None:
    """Render every job to a 24 kHz mono WAV."""
    jobs = json.loads(open(sys.argv[1], encoding="utf-8").read())
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    for job in jobs:
        chunks = [audio for _, _, audio in pipeline(job["text"], voice=VOICE, speed=SPEED)]
        audio = np.concatenate([np.asarray(c) for c in chunks]) if chunks else np.zeros(SAMPLE_RATE // 2)
        sf.write(job["out"], audio, SAMPLE_RATE)
        print(job["out"], flush=True)


if __name__ == "__main__":
    main()
