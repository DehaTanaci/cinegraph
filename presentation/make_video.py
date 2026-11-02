"""Build the CineGraph demo video: live recordings of the running project + AI voice-over. No slides.

Everything on screen is real: the demo page, Neo4j Browser, Swagger, a terminal running the real
`make` commands, and the public GitHub repository (README, code, pull requests, Actions) and Docker Hub.

Requirements: the Compose stack running and seeded (`make docker-run seed gds train`), Ollama running,
ffmpeg, macOS `say`, and playwright (`pip install playwright && playwright install chromium`).

Each scene is a list of beats. A beat = one narration sentence (spoken by `say`) + an optional on-screen
action that starts with the sentence. The recording waits for the longer of the two and the audio is
padded with silence to match, so voice and screen stay in sync.

Run from the project root:  .venv/bin/python presentation/make_video.py [scene-number ...]
"""

import html
import json
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "presentation" / "build"
OUT = ROOT / "presentation" / "CineGraph_demo.mp4"
APP = "http://localhost:8080"
REPO = "https://github.com/DehaTanaci/cinegraph"
W, H = 1600, 900
VOICE, RATE = "Samantha", 172
GAP = 0.35  # silence after each beat, seconds

# Words the TTS voice mispronounces -> spoken spelling (the written narration keeps the real names).
SPOKEN = {
    "Qwen": "Kwen", "FastRP": "Fast R P", "Nginx": "engine X", "AIDAMS": "aidams", "GDS": "G D S",
    "MovieLens": "Movie Lens", "GroupLens": "Group Lens", "IMDb": "I M D B", "LangGraph": "Lang Graph",
    "AUC": "A U C", "README": "read me", "Pylint": "Pie lint", "0.83": "zero point eight three",
    "0.82": "zero point eight two", "9.8": "nine point eight", "EXPLAIN": "explain", "PROFILE": "profile",
    "FastAPI": "Fast A P I", "Swagger": "Swagger", "Pydantic": "Pydantic",
}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# ---------------------------------------------------------------------------------------------------------------
# Terminal page: commands are typed visibly and their REAL output is streamed in.
# ---------------------------------------------------------------------------------------------------------------

TERMINAL = """<html><head><style>
body{margin:0;background:#0b0e13;color:#d6dde8;font:19px/1.45 Menlo,ui-monospace,monospace}
.bar{height:38px;background:#1b1f27;display:flex;align-items:center;padding:0 16px;gap:8px;color:#8a93a6;font:14px -apple-system}
.dot{width:13px;height:13px;border-radius:50%} .t{margin-left:auto;margin-right:auto}
#out{padding:18px 24px;height:822px;overflow:hidden;white-space:pre-wrap;word-break:break-all}
.p{color:#3ccf91} .c{color:#fff;font-weight:600} .cur{display:inline-block;width:10px;background:#d6dde8;animation:b 1s step-end infinite}
@keyframes b{50%{opacity:0}}
</style></head><body><div class='bar'><span class='dot' style='background:#ff5f57'></span><span class='dot'
style='background:#febc2e'></span><span class='dot' style='background:#28c840'></span><span class='t'>cinegraph — zsh</span></div>
<div id='out'></div><script>
const out=document.getElementById('out');
const PROMPT="<span class='p'>deha@mac cinegraph %</span> ";
function scroll(){out.scrollTop=out.scrollHeight}
window.prompt_=()=>{out.insertAdjacentHTML('beforeend',PROMPT+"<span class='c'></span><span class='cur'>&nbsp;</span>");scroll()};
window.typeCmd=async(t)=>{const c=out.querySelectorAll('.c');const el=c[c.length-1];
  for(const ch of t){el.textContent+=ch;await new Promise(r=>setTimeout(r,45))}};
window.feed=async(lines,delay)=>{out.querySelector('.cur')?.remove();out.insertAdjacentText('beforeend','\\n');
  for(const l of lines){out.insertAdjacentText('beforeend',l+'\\n');scroll();await new Promise(r=>setTimeout(r,delay))}
  window.prompt_()};
window.prompt_();
</script></body></html>"""


def terminal_setup(page):
    """Open the terminal page."""
    page.goto("about:blank")  # leave any site whose security policy would block the inline script
    page.set_content(TERMINAL)
    page.wait_for_timeout(300)


def term(command, delay_ms=25, shell_cmd=None, keep_last=None):
    """Action: type `command`, run it for real (or `shell_cmd`), stream its output."""
    def action(page):
        page.evaluate("t=>typeCmd(t)", command)
        page.wait_for_timeout(len(command) * 45 + 200)
        result = subprocess.run(shell_cmd or command, shell=True, cwd=ROOT, capture_output=True, text=True,
                                check=False, stdin=subprocess.DEVNULL, timeout=300)
        lines = ANSI.sub("", result.stdout + result.stderr).rstrip("\n").splitlines()
        if keep_last:
            lines = lines[-keep_last:]
        page.evaluate("([l,d])=>feed(l,d)", [lines, delay_ms])
        page.wait_for_timeout(len(lines) * delay_ms + 300)
    return action


# ---------------------------------------------------------------------------------------------------------------
# Live actions
# ---------------------------------------------------------------------------------------------------------------

def slow_type(page, selector, text, delay=35):
    """Clear an input and type into it visibly."""
    page.fill(selector, "")
    page.type(selector, text, delay=delay)


def scroll_to(page, selector):
    """Smooth-scroll an element into view."""
    page.evaluate("s=>document.querySelector(s).scrollIntoView({behavior:'smooth',block:'start'})", selector)


def smooth_scroll(page, pixels, steps=12):
    """Scroll gradually so the motion is readable on video."""
    for _ in range(steps):
        page.mouse.wheel(0, pixels / steps)
        page.wait_for_timeout(60)


def github(path):
    """Setup/action: open a page of the public repository."""
    def action(page):
        page.goto(f"{REPO}{path}", wait_until="domcontentloaded", timeout=90000)
        page.wait_for_timeout(2500)
    return action


def readme_section(anchor):
    """Action: scroll the README to a section heading."""
    def action(page):
        page.evaluate(
            "a=>{const e=document.getElementById('user-content-'+a)||document.getElementById(a);"
            "if(e)e.scrollIntoView({behavior:'smooth',block:'start'})}", anchor)
        page.wait_for_timeout(900)
    return action


def neo4j_setup(page):
    """Log into Neo4j Browser (local dev password from .env) before the scene starts."""
    password = re.search(r"^NEO4J_PASSWORD=(.*)$", (ROOT / ".env").read_text(), re.M).group(1).strip()
    page.goto("http://localhost:7474/browser/")
    page.wait_for_selector("input[type=password]", timeout=30000)
    page.fill("input[type=password]", password)
    page.keyboard.press("Enter")
    page.wait_for_timeout(4000)
    page.keyboard.press("Escape")
    neo4j_query(":clear")(page)  # remove the :welcome frame
    try:
        page.get_by_role("button", name="Collapse Sidebar").click(timeout=1500)
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def neo4j_query(query):
    """Action: type a query in the Neo4j Browser editor and run it."""
    def action(page):
        editor = page.get_by_role("textbox", name="Cypher Editor")
        editor.click()
        page.keyboard.press("Meta+A")
        page.keyboard.press("Backspace")
        page.keyboard.type(query, delay=18)
        page.keyboard.press("Escape")  # dismiss autocomplete
        page.keyboard.press("Meta+Enter")
        page.wait_for_timeout(1500)
    return action


def demo_setup(tab=None):
    """Setup: open the demo page on a tab."""
    def setup(page):
        page.goto(f"{APP}/demo")
        page.wait_for_load_state("networkidle")
        if tab:
            page.click(f"nav button[data-tab={tab}]")
    return setup


def swagger_try(page):
    """Open the recommendations operation in Swagger and execute it."""
    op = page.locator("#operations-users-recommendations_api_users__user_id__recommendations_get")
    op.locator(".opblock-summary").click()
    page.wait_for_timeout(500)
    op.get_by_role("button", name="Try it out").click()
    op.locator("input[placeholder='user_id']").fill("1")
    op.locator("input[placeholder='limit']").fill("3")
    page.wait_for_timeout(400)
    op.get_by_role("button", name="Execute").click()
    page.wait_for_timeout(1500)
    op.locator(".responses-wrapper").scroll_into_view_if_needed()


# ---------------------------------------------------------------------------------------------------------------
# Scenes: (name, setup(page), [(narration, action(page) | None), ...])
# ---------------------------------------------------------------------------------------------------------------

SCENES = [
    ("intro", demo_setup("foryou"), [
        ("Hi, I'm Deha Tanaci. This is CineGraph, my knowledge graph project for the AIDAMS GraphDB course: a movie "
         "recommender built on Neo4j, with graph algorithms, a machine learning model, and a guarded natural-language "
         "assistant running on a local language model.", None),
        ("Everything in this video is the real, running project. Recommendation is a question about connections: "
         "people who liked what you liked also liked something you haven't seen. In a relational database that's "
         "three self-joins on one ratings table; in a graph it's a single pattern, and its cost depends on your "
         "neighbourhood, not on the size of the whole database.", None),
    ]),
    ("terminal-stack", terminal_setup, [
        ("Let's start from the terminal. The Makefile wraps every command of the project.", term("make help", 20)),
        ("The whole system runs with Docker Compose: three services, all healthy. Neo4j 5 with the Graph Data Science "
         "plugin, the FastAPI application, and an Nginx reverse proxy, which is the only public entry point.",
         term("docker compose ps --format 'table {{.Name}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}'", 60)),
        ("The language model, Qwen 2.5 Coder with seven billion parameters, runs locally in Ollama: no paid API, and "
         "no data leaves the machine.", term("ollama show qwen2.5-coder:7b", 60)),
    ]),
    ("neo4j-schema", neo4j_setup, [
        ("This is the database itself, in the Neo4j Browser. Here is the schema of the graph.",
         neo4j_query("CALL db.schema.visualization()")),
        ("Four node labels: User, Movie, Genre and Tag. Users rate movies; movies belong to genres and carry tags; and "
         "movies are linked to each other by SIMILAR, which is computed by a graph algorithm.", None),
        ("These were deliberate choices. A rating is a relationship, not a node: it always connects exactly one user "
         "and one movie, and nothing else points at it, so a node would only add a hop to every traversal.", None),
        ("Genres are stored twice: as nodes, so we can walk from a movie to its genre to other movies, and as a list "
         "on each movie, so a filter like 'only sci-fi' needs no extra hop. A test checks the two never disagree. "
         "And 'liked' means a rating of four or more, applied at query time.", None),
        ("Here is Toy Story and its most similar movies.",
         neo4j_query("MATCH p = (m:Movie {title: 'Toy Story'})-[:SIMILAR]->(:Movie) RETURN p")),
        ("Toy Story 2, the Star Wars trilogy, Jurassic Park, The Lion King. No genre data was used: these links come "
         "purely from who liked what.", None),
    ]),
    ("readme-data", github(""), [
        ("The README documents the whole project. The data is MovieLens, from GroupLens: real ratings, about a hundred "
         "thousand of them, from six hundred and ten users on almost ten thousand movies.", readme_section("5-dataset")),
        ("The loading script cleans it on the way in: it parses the year out of titles, restores articles like 'Matrix, "
         "The', normalises tags, and drops the fake 'no genres listed' genre.", None),
        ("I also asked an AI assistant to design a schema for the same project, and compared it with mine. It invented "
         "user names, emails and ages that the dataset doesn't contain, and stored every rating twice. This appendix "
         "explains each difference.", readme_section("appendix--our-schema-vs-an-ai-generated-schema")),
    ]),
    ("demo-foryou", demo_setup("foryou"), [
        ("Now the demo interface. I pick user one.", lambda p: p.click("#loadUser")),
        ("The profile comes from the graph: two hundred and thirty-two ratings, and a clear taste for action and "
         "adventure.", None),
        ("The recommendations come from collaborative filtering: find the thirty users whose likes overlap most with "
         "yours, then collect what they loved that you haven't seen. Terminator 2 comes first.", None),
        ("And the key question: why was this recommended?", lambda p: p.click("[data-why]")),
        ("The explanation is built only from real paths in the graph: similar users who loved it, its similarity to "
         "The Terminator, which this user rated five stars, and their taste for action. No language model writes "
         "this, so it can't invent a reason.", None),
    ]),
    ("demo-movies", demo_setup("foryou"), [
        ("On the right, discovery. Louvain community detection groups users and movies into taste communities.",
         lambda p: p.click("#loadUser")),
        ("These are movies from communities this user rarely explores, but that are popular in their own community.",
         None),
        ("In the Movies tab, I search for The Matrix.",
         lambda p: (p.click("nav button[data-tab=movies]"), slow_type(p, "#q", "The Matrix"), p.click("#search"))),
        ("On the left, movies liked by the same people, computed by node similarity: Fight Club, Saving Private Ryan, "
         "The Empire Strikes Back.", None),
        ("On the right, movies that only share genres and tags. Two very different notions of similar, and the graph "
         "supports both.", None),
    ]),
    ("demo-path", demo_setup("path"), [
        ("Taste path finds the chain of similar movies linking any two films.", None),
        ("From Toy Story to The Godfather, Dijkstra's algorithm runs on the similarity network, with a cost of one "
         "minus similarity, so it prefers strong links.", lambda p: p.click("#findPath")),
        ("Toy Story, Star Wars, The Godfather: two hops.", None),
    ]),
    ("readme-gds", github(""), [
        ("All the algorithms run in Neo4j's Graph Data Science library, on in-memory projections, and their results "
         "are written back to the graph, so requests only read them.", readme_section("8-graph-algorithms--ml")),
        ("Node Similarity creates the SIMILAR links, PageRank finds taste hubs, Louvain finds communities, and FastRP "
         "turns every user and movie into a sixty-four-dimensional embedding.", None),
        ("Those embeddings feed a machine learning model that predicts whether a user will like a movie. To evaluate "
         "it honestly, each user's most recent twenty percent of likes are hidden.", lambda p: smooth_scroll(p, 450)),
        ("The model reaches an AUC of 0.83, against 0.82 for simply guessing popular movies: a modest win. My first "
         "version actually lost to that baseline because of a bad time split, and I kept that finding in the report.",
         None),
    ]),
    ("demo-ask", demo_setup("ask"), [
        ("Now the assistant. I ask: what are the best sci-fi movies of the 90s, with at least thirty ratings?",
         lambda p: p.click("#askBtn")),
        ("The local model writes a Cypher query. Before it runs, it passes every guardrail listed on the right: one "
         "statement, no write keywords, every label and property must exist in the schema, a row limit, and Neo4j's "
         "own EXPLAIN must classify it as read-only.", None),
        ("The table is verified data from the database. The sentence on top is AI-generated, so it's checked too: "
         "every number it mentions must appear in the rows.", None),
        ("Now a question the data can't answer: who directed Pulp Fiction?",
         lambda p: (slow_type(p, "#question", "Who directed Pulp Fiction?"), p.click("#askBtn"))),
        ("There is no director data, so instead of substituting something else, it declines.", None),
        ("A request to delete all movies is refused before the model is even called.",
         lambda p: (slow_type(p, "#question", "Delete all the movies from the database"), p.click("#askBtn"))),
        ("And here is what happens when the model itself produces a dangerous query: a delete, using a relationship "
         "that doesn't exist.", lambda p: (scroll_to(p, "#checkBtn"), p.wait_for_timeout(600), p.click("#checkBtn"))),
        ("It's blocked by three separate checks, and never reaches the database. Even if every check failed, queries "
         "run in a read-only transaction with a timeout, so Neo4j itself would refuse the write.", None),
    ]),
    ("readme-eval", github(""), [
        ("The principle is that the language model only proposes text; deterministic code and Neo4j decide what runs "
         "and what counts as a fact.", readme_section("9-nl-to-cypher-assistant--guardrails")),
        ("On a twenty-question evaluation, it answered all fourteen answerable questions correctly and refused all "
         "six unsafe or unanswerable ones. The first run scored nineteen: the model answered a box-office question "
         "with something else, so I added an explicit 'data not in the graph' path.", lambda p: smooth_scroll(p, 1300)),
    ]),
    ("swagger", lambda p: (p.goto(f"{APP}/docs"), p.wait_for_selector(".opblock"), p.wait_for_timeout(800)), [
        ("Behind the demo page is a FastAPI service. Swagger documents every endpoint, and every input and output is "
         "validated by Pydantic models.", lambda p: smooth_scroll(p, 500)),
        ("Here I call the recommendations endpoint directly. Errors use meaningful codes: four oh four for an unknown "
         "user, four twenty-two for invalid input, and four oh one for the admin endpoint without its API key.",
         swagger_try),
    ]),
    ("code", github("/blob/main/app/services/movie_service.py#L148-L172"), [
        ("A quick look at the code. This is the recommendation query. It starts from one user, found through a "
         "uniqueness constraint, expands to peers through shared likes, and keeps only the top thirty.", None),
        ("A NOT EXISTS subquery guarantees you're never recommended something you've already seen, the genre filter "
         "uses the list property, and every value is a parameter.", None),
        ("And this is the LangGraph workflow: screen the question, generate, validate, execute, and summarise, with "
         "failed validations fed back to the model for up to three attempts.",
         github("/blob/main/app/agent/graph.py#L183-L194")),
    ]),
    ("readme-perf", github(""), [
        ("Every optimisation was measured with PROFILE, before and after.", readme_section("10-performance")),
        ("The index on rating timestamps cuts the work for recent trending by thirty-four times. Limiting "
         "recommendations to the top thirty peers cuts it by almost four, from about two hundred milliseconds to "
         "fifty. Precomputing average ratings makes the top-rated list almost three hundred times cheaper.", None),
    ]),
    ("terminal-quality", terminal_setup, [
        ("Code quality: Pylint with the course configuration, above the required nine point five.",
         term("make lint", 4, keep_last=40)),
        ("And the test suite: ninety-six tests, from unit tests of every guardrail to Cypher tests against the real "
         "database and health checks of the running containers. Coverage is eighty-three percent.",
         term("make test", 12)),
    ]),
    ("github", github("/pulls?q=is%3Apr+is%3Aclosed"), [
        ("Each part of the project was merged through a reviewed pull request, and the reviews caught real bugs that "
         "were fixed before merging.", None),
        ("GitHub Actions runs the linter and the full test suite against a real Neo4j on every push and pull request.",
         github("/actions")),
        ("And the API image is published on Docker Hub, for both Intel and ARM machines.",
         lambda p: (terminal_setup(p), term("docker buildx imagetools inspect dehatanaci06/cinegraph-api:1.0",
                                            40)(p))),
    ]),
    ("outro", demo_setup("foryou"), [
        ("To wrap up, the limitations: new users with no ratings only get popularity-based lists, and the local model "
         "is much weaker than a hosted one, which is exactly why nothing it writes is trusted.",
         lambda p: p.click("#loadUser")),
        ("Everything is public on GitHub and reproducible with a handful of make commands. Thanks for watching.", None),
    ]),
]

# ---------------------------------------------------------------------------------------------------------------
# Audio + recording + assembly
# ---------------------------------------------------------------------------------------------------------------


def run(cmd):
    """Run a command, raising with its output on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{cmd[0]} failed: {result.stderr[-800:]}")
    return result.stdout


def spoken(text):
    """Apply pronunciation fixes."""
    for word, say_as in SPOKEN.items():
        text = re.sub(rf"(?<![\w.]){re.escape(word)}(?![\w])", say_as, text)
    return text


def tts(text, path):
    """Synthesize one beat to a 44.1 kHz mono WAV; return its duration."""
    aiff = path.with_suffix(".aiff")
    run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(aiff), spoken(text)])
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", "44100", "-ac", "1", str(path)])
    aiff.unlink()
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def build_audio(beat_wavs, actual_lengths, out):
    """Concatenate beat audio, padding each beat with silence to its on-screen length."""
    with wave.open(str(out), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(44100)
        for wav_path, length in zip(beat_wavs, actual_lengths):
            with wave.open(str(wav_path)) as src:
                frames = src.readframes(src.getnframes())
            dst.writeframes(frames)
            pad = max(0, int(round(length * 44100)) - len(frames) // 2)
            dst.writeframes(b"\x00\x00" * pad)


def record_scene(browser, folder, setup, beats):
    """Record one scene; return (video path, trim offset, beat lengths, beat wavs)."""
    for old in folder.glob("*.webm"):
        old.unlink()
    wavs, durations = [], []
    for k, (text, _) in enumerate(beats):
        wav = folder / f"beat{k}.wav"
        durations.append(tts(text, wav) + GAP)
        wavs.append(wav)
    context = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(folder),
                                  record_video_size={"width": W, "height": H}, color_scheme="dark")
    t_create = time.monotonic()
    page = context.new_page()
    page.set_default_timeout(60000)
    setup(page)
    page.wait_for_timeout(500)
    t_start = time.monotonic()
    lengths = []
    for k, ((_, action), target) in enumerate(zip(beats, durations)):
        print(f"      beat {k + 1}/{len(beats)}", flush=True)
        began = time.monotonic()
        if action is not None:
            action(page)
        remaining = target - (time.monotonic() - began)
        if remaining > 0:
            page.wait_for_timeout(int(remaining * 1000))
        lengths.append(time.monotonic() - began)
    page.wait_for_timeout(300)
    video = Path(page.video.path())
    context.close()
    return video, t_start - t_create, lengths, wavs


def assemble_scene(folder, video, offset, lengths, wavs):
    """Trim the recording, attach the voice track, normalise to 1080p30 H.264."""
    audio = folder / "voice.wav"
    build_audio(wavs, lengths, audio)
    total = sum(lengths)
    out = folder / "scene.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{offset:.3f}", "-i", str(video), "-i", str(audio),
         "-filter_complex", "[0:v]scale=1920:1080:flags=lanczos,fps=30,format=yuv420p,"
         "tpad=stop_mode=clone:stop_duration=10[v]",
         "-map", "[v]", "-map", "1:a", "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-c:a", "aac", "-b:a", "160k", "-ar", "44100", str(out)])
    return out, total


def write_script(manifest):
    """Narration as Markdown with timestamps, for recording a human voice-over later."""
    lines = ["# CineGraph demo video — narration script\n",
             "Read each section while the matching footage plays. Timestamps are where each scene starts in "
             "`CineGraph_demo.mp4`.\n"]
    start = 0.0
    for i, (name, _, beats) in enumerate(SCENES, 1):
        seconds = manifest.get(str(i), {}).get("seconds", 0)
        lines.append(f"## {i}. {name.replace('-', ' ').title()} — {int(start // 60)}:{int(start % 60):02d}\n")
        for text, action in beats:
            cue = " *(on screen: action)*" if action is not None else ""
            lines.append(f"- {html.unescape(text)}{cue}")
        lines.append("")
        start += seconds
    (ROOT / "presentation" / "narration.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    """Record selected scenes (default all), then concatenate every recorded scene in order."""
    wanted = {int(a) for a in sys.argv[1:]} or set(range(1, len(SCENES) + 1))
    BUILD.mkdir(parents=True, exist_ok=True)
    manifest_path = BUILD / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for i, (name, setup, beats) in enumerate(SCENES, 1):
            if i not in wanted:
                continue
            print(f"[{i}/{len(SCENES)}] {name} ...", flush=True)
            folder = BUILD / f"{i:02d}-{name}"
            folder.mkdir(parents=True, exist_ok=True)
            video, offset, lengths, wavs = record_scene(browser, folder, setup, beats)
            out, total = assemble_scene(folder, video, offset, lengths, wavs)
            manifest[str(i)] = {"name": name, "file": str(out), "seconds": round(total, 1)}
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"      {total:.1f}s", flush=True)
        browser.close()
    write_script(manifest)
    parts = [manifest[str(i)]["file"] for i in range(1, len(SCENES) + 1)
             if str(i) in manifest and manifest[str(i)]["name"] == SCENES[i - 1][0]]
    (BUILD / "concat.txt").write_text("".join(f"file '{p}'\n" for p in parts))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(BUILD / "concat.txt"),
         "-c", "copy", "-movflags", "+faststart", str(OUT)])
    total = sum(manifest[str(i)]["seconds"] for i in range(1, len(SCENES) + 1) if str(i) in manifest)
    print(f"Video: {OUT} ({total / 60:.1f} min, {len(parts)} scenes)")


if __name__ == "__main__":
    main()
