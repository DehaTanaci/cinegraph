"""Teleprompter version of the demo video: the narration scrolls under the picture, karaoke style.

The current line lights up word by word at the exact pace of the generated voice; the previous line fades
above and the next line waits below. A strip above the text shows the scene name and the clock.

Workflow for a human voice-over:
  1. record your voice in iMovie over CineGraph_prompteur.mp4 (it has the same timing as the clean video);
  2. export it from iMovie;
  3. python presentation/prompter.py --merge <your export>  ->  CineGraph_final.mp4
     (takes your voice from the export and puts it on the clean video, without the text band).

Build: .venv/bin/python presentation/prompter.py   (needs make_video.py to have been run)
"""

import json
import re
import subprocess
import sys
import textwrap
from bisect import bisect_right
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from make_video import BUILD, OUT_SILENT, PRES, SCENES, TTS_PYTHON, spoken  # noqa: E402  pylint: disable=wrong-import-position

PROMPTER = PRES / "CineGraph_prompteur.mp4"
FINAL = PRES / "CineGraph_final.mp4"
# Playback speed of the prompter and of the clean video your voice is merged onto (1.0 = as recorded).
SPEED = 0.88
CLEAN = PRES / "CineGraph_demo_silent_slow.mp4"
BW, BH, FPS = 1920, 430, 30
LINE_CHARS = 58
FONT = "/System/Library/Fonts/SFNS.ttf"
LEAD = 0.15  # light each line up a little before its first word, seconds


def run(cmd, **kwargs):
    """Run a command and fail loudly."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    return result.stdout


def word_timings():
    """Kokoro word timings for every beat, keyed 'scene|beat' (cached)."""
    cache = BUILD / "word_timings.json"
    jobs = [{"key": f"{name}|{k}", "text": spoken(text)}
            for name, _, beats in SCENES for k, (text, _) in enumerate(beats)]
    if cache.exists():
        cached = json.loads(cache.read_text())
        if all(job["key"] in cached for job in jobs):
            return cached
    job_file = BUILD / "timing_jobs.json"
    job_file.write_text(json.dumps(jobs))
    print("Aligning words with Kokoro...", flush=True)
    run([str(TTS_PYTHON), str(PRES / "word_timings.py"), str(job_file), str(cache)], cwd=PRES)
    return json.loads(cache.read_text())


def time_mapper(spoken_text, words):
    """Map a character fraction of the text to seconds, interpolating between aligned words."""
    points, cursor = [(0.0, words[0][1] if words else 0.0)], 0
    for word, start, _ in words:
        found = spoken_text.find(word, cursor)
        if found >= 0:
            points.append((found / max(1, len(spoken_text)), start))
            cursor = found + len(word)
    points.append((1.0, words[-1][2] if words else 0.0))
    points.sort()
    xs = [p[0] for p in points]

    def at(fraction):
        i = max(1, min(len(points) - 1, bisect_right(xs, fraction)))
        (x0, t0), (x1, t1) = points[i - 1], points[i]
        return t0 if x1 == x0 else t0 + (t1 - t0) * (fraction - x0) / (x1 - x0)
    return at


MAX_LINE = 52        # characters per lyric line
BREAK_WORDS = ("and", "so", "which", "but", "then", "because", "with", "against", "using", "for", "that", "whose",
               "like", "without", "only")
# Never end a line on one of these: the phrase obviously continues.
WEAK_ENDINGS = ("the", "a", "an", "it", "must", "next", "best", "of", "to", "into", "on", "is", "are", "be",
                "hundred", "thousand", "its", "their", "my", "this", "that", "your", "has", "have", "can", "would",
                "in", "from", "at", "by", "every", "each", "no", "not", "so", "and", "but", "which", "what",
                "whether", "will", "extra", "sci-fi", "shows", "one", "two", "three", "four", "five", "twenty",
                "thirty", "forty", "most", "with", "writes")


def phrase_lines(text):
    """Split narration into short lines at natural phrase boundaries, like song lyrics."""
    sentences = re.split(r"(?<=[.!?])\s+", text)  # a full stop only ends a sentence before a space
    lines = []
    for sentence in (s.strip() for s in sentences):
        if sentence:
            lines.extend(_split_phrase(sentence))
    return lines


def _split_phrase(phrase):
    if len(phrase) <= MAX_LINE:
        return [phrase]
    words = phrase.split(" ")
    best, best_score = None, None
    for i in range(1, len(words)):
        left = " ".join(words[:i])
        if len(left) < 12 or len(phrase) - len(left) < 12:
            continue
        score = abs(len(left) - len(phrase) / 2)
        last = words[i - 1].lower().strip(",.:;'\"")
        if left[-1] in ",:;":
            score -= 40          # strongly prefer breaking after punctuation
        else:
            if words[i].lower().strip(",") in BREAK_WORDS and last not in ("hundred", "thousand"):
                score -= 18      # or before a joining word
            if last in WEAK_ENDINGS:
                score += 45      # never on a word that obviously continues
        if len(left) > MAX_LINE + 6:
            score += 60
        if best_score is None or score < best_score:
            best, best_score = i, score
    if best is None:
        return textwrap.wrap(phrase, MAX_LINE)
    return _split_phrase(" ".join(words[:best])) + _split_phrase(" ".join(words[best:]))


def build_lines():
    """Every lyric line with its scene and start time on the video's clock."""
    manifest = json.loads((BUILD / "manifest.json").read_text())
    timings = word_timings()
    lines, clock = [], 0.0
    for i, (name, _, beats) in enumerate(SCENES, 1):
        entry = manifest[str(i)]
        beat_start = clock
        for k, (text, _) in enumerate(beats):
            at = time_mapper(spoken(text), timings[f"{name}|{k}"])
            position = 0
            for line in phrase_lines(text):
                position = text.find(line, position)
                lines.append({"scene": f"{i}. {name.replace('-', ' ').capitalize()}",
                              "start": beat_start + at(position / len(text)), "text": line})
                position += len(line)
            beat_start += entry["beats"][k]
        clock += entry["seconds"]
    return lines, clock


def _ease(x):
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


class Band:
    """Spotify-style lyrics: past line faded above, current line bright, upcoming lines below; smooth scroll."""

    LINE_H = 68
    CURRENT_Y = 112     # where the current line sits inside the band
    LEFT = 150

    def __init__(self, lines):
        self.lines = lines
        self.starts = [line["start"] - LEAD for line in lines]
        self.font = ImageFont.truetype(FONT, 50)
        self.meta = ImageFont.truetype(FONT, 22)
        try:
            self.bold = ImageFont.truetype(FONT, 50, index=0)
            self.bold.set_variation_by_name("Bold")
        except (OSError, AttributeError, ValueError):
            self.bold = self.font

    def frame(self, t):
        """RGB bytes of the band at time t."""
        image = Image.new("RGB", (BW, BH), (12, 14, 19))
        draw = ImageDraw.Draw(image)
        draw.line([(0, 0), (BW, 0)], fill=(42, 49, 66), width=2)
        index = bisect_right(self.starts, t) - 1
        scene = self.lines[max(index, 0)]["scene"]
        draw.text((30, 16), scene, font=self.meta, fill=(255, 122, 89))
        clock = f"{int(t // 60)}:{int(t % 60):02d}"
        draw.text((BW - 30 - draw.textlength(clock, font=self.meta), 16), clock, font=self.meta, fill=(110, 118, 135))
        # Continuous scroll position: glides from the previous line to the current one over 0.45 s.
        if index < 0:
            scroll = -1 + _ease((t - (self.starts[0] - 0.45)) / 0.45) if t > self.starts[0] - 0.45 else -1.0
        else:
            scroll = index - 1 + _ease((t - self.starts[index]) / 0.45)
        for k in range(max(0, index - 2), min(len(self.lines), index + 6)):
            y = self.CURRENT_Y + (k - scroll) * self.LINE_H
            if y < 40 or y + 62 > BH:  # only whole lines inside the band
                continue
            if k == index:
                font, color = self.bold, (255, 255, 255)
            elif k < index:
                font, color = self.font, (88, 96, 114)
            else:
                fade = min(k - index, 4)
                font, color = self.font, (200 - 22 * fade, 206 - 22 * fade, 218 - 20 * fade)
            draw.text((self.LEFT, y), self.lines[k]["text"], font=font, fill=color)
        return image.tobytes()


def build():
    """Render the band and stack it under the silent video."""
    lines, total = build_lines()
    band = Band(lines)
    frames = int(total * FPS) + FPS
    print(f"Rendering {frames} frames of lyrics...", flush=True)
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(OUT_SILENT),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{BW}x{BH}", "-r", str(FPS), "-i", "-",
         "-filter_complex",
         f"[0:v]fps=30,scale=-2:{1080 - BH},pad=1920:{1080 - BH}:(ow-iw)/2:0:color=0x0c0e13[v0];"
         "[v0][1:v]vstack=inputs=2:shortest=1,format=yuv420p[v]",
         "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-movflags", "+faststart",
         str(BUILD / "prompter_1x.mp4")],
        stdin=subprocess.PIPE)
    try:
        for n in range(frames):
            encoder.stdin.write(band.frame(n / FPS))
            if n % (FPS * 60) == 0:
                print(f"  {n // FPS // 60} min", flush=True)
        encoder.stdin.close()
    except BrokenPipeError:
        pass  # ffmpeg stops reading once the video ends (vstack shortest=1)
    if encoder.wait():
        raise RuntimeError("ffmpeg failed")
    slow_down()
    print(f"Prompter: {PROMPTER}")


def slow_down():
    """Stretch the prompter and the clean video by the same factor so the voice-over lines up."""
    for source, target in ((BUILD / "prompter_1x.mp4", PROMPTER), (OUT_SILENT, CLEAN)):
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-an", "-filter:v",
             f"setpts=PTS/{SPEED},fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-movflags", "+faststart", str(target)])
    # A practice copy with the AI voice slowed the same way (pitch kept).
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(PRES / "CineGraph_demo.mp4"),
         "-filter_complex", f"[0:v]setpts=PTS/{SPEED},fps=30[v];[0:a]atempo={SPEED}[a]", "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-movflags", "+faststart",
         str(PRES / "CineGraph_demo_slow.mp4")])
    script = PRES / "narration.md"
    text = re.sub(r"\*\*(\d+):(\d+)\*\*",
                  lambda m: "**{}:{:02d}**".format(*divmod(round((int(m[1]) * 60 + int(m[2])) / SPEED), 60)),
                  script.read_text(encoding="utf-8"))
    text = re.sub(r"_Total length: (\d+):(\d+)_",
                  lambda m: "_Total length: {}:{:02d}_".format(*divmod(round((int(m[1]) * 60 + int(m[2])) / SPEED), 60)),
                  text)
    script.with_name("narration_slow.md").write_text(text.replace(
        "`CineGraph_demo_silent.mp4`", "`CineGraph_prompteur.mp4`"), encoding="utf-8")


def merge(voice_file):
    """Put the voice from an iMovie export onto the clean video."""
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(CLEAN), "-i", voice_file, "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(FINAL)])
    print(f"Final video: {FINAL}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--merge":
        merge(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "--slow-only":
        slow_down()
    else:
        build()
