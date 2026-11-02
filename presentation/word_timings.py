"""Word timings for each narration beat, from Kokoro's alignment (runs in the presentation/.tts virtualenv).

Kokoro is deterministic, so re-running the same text gives exactly the timing of the audio already in the video.

Usage: presentation/.tts/bin/python presentation/word_timings.py jobs.json out.json
jobs.json: [{"key": "...", "text": "..."}]   ->   out.json: {key: [[word, start_s, end_s], ...]}
"""

import json
import sys

from kokoro import KPipeline

from tts import SAMPLE_RATE, SPEED, VOICE


def main() -> None:
    """Align every job and write the timings."""
    jobs = json.loads(open(sys.argv[1], encoding="utf-8").read())
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    timings = {}
    for job in jobs:
        words, offset = [], 0.0
        for result in pipeline(job["text"], voice=VOICE, speed=SPEED):
            for token in result.tokens:
                if token.start_ts is not None and token.text.strip():
                    words.append([token.text, offset + token.start_ts, offset + (token.end_ts or token.start_ts)])
            offset += len(result.audio) / SAMPLE_RATE
        timings[job["key"]] = words
        print(job["key"], len(words), flush=True)
    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        json.dump(timings, handle)


if __name__ == "__main__":
    main()
