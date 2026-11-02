"""Build the CineGraph demo video: a narrated walkthrough of the running project. No slides.

Everything on screen is real: the demo page, Neo4j Browser, a terminal running the real `make` commands,
the actual source files, Swagger, and the public GitHub repository. A visible mouse moves to each control
before clicking, like a person giving a live demo.

Narration is synthesized with Kokoro (free, local; see presentation/tts.py). Each scene is a list of beats:
one or two spoken sentences plus the on-screen action that starts with them. The recording waits for the
longer of the two, and the audio is padded with silence to match, so voice and screen stay in sync.

Outputs (presentation/):
  CineGraph_demo.mp4         video with the AI voice
  CineGraph_demo_silent.mp4  same video, no audio, for recording your own voice-over
  narration.md               the script, with a timestamp for every sentence

Requirements: the Compose stack running and seeded (`make docker-run seed gds train`), Ollama running,
ffmpeg, playwright, and the TTS virtualenv:
  python3 -m venv presentation/.tts && presentation/.tts/bin/pip install "kokoro>=0.9.4" soundfile

Run from the project root:  .venv/bin/python presentation/make_video.py [scene-number ...]
"""

import hashlib
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
PRES = ROOT / "presentation"
BUILD = PRES / "build"
OUT = PRES / "CineGraph_demo.mp4"
OUT_SILENT = PRES / "CineGraph_demo_silent.mp4"
TTS_PYTHON = PRES / ".tts" / "bin" / "python"
APP = "http://localhost:8080"
REPO = "https://github.com/DehaTanaci/cinegraph"
W, H = 1440, 810
GAP = 0.6  # breathing pause after each beat, seconds
RATE = 44100

# Spoken spellings for words the voice would mispronounce (the written script keeps the real names).
SPOKEN = {
    "CineGraph": "Sinegraph", "Nginx": "engine X", "FastAPI": "Fast A.P.I.", "AUC": "A.U.C.", "Pylint": "Pie-lint",
    "LangGraph": "Lang Graph", "MovieLens": "Movie Lens", "Neo4j": "Neo four J", "0.83": "zero point eight three",
    "0.82": "zero point eight two", "9.5": "nine point five", "ARM": "arm", "Ollama": "Oh-lama",
}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# A visible cursor that follows real mouse events, with a ripple on click.
CURSOR_JS = """
(() => {
  if (window.__cursor) return; window.__cursor = true;
  const add = () => {
    const c = document.createElement('div');
    c.id = '__cursor';
    c.style.cssText = 'position:fixed;left:-50px;top:-50px;width:22px;height:22px;z-index:2147483647;pointer-events:none;'
      + 'transition:transform .08s;background:url("data:image/svg+xml;utf8,<svg xmlns=%22http://www.w3.org/2000/svg%22 '
      + 'width=%2222%22 height=%2222%22 viewBox=%220 0 22 22%22><path d=%22M2 1 L2 18 L6.5 13.8 L9.6 20.5 L12.4 19.2 '
      + 'L9.4 12.7 L15.5 12.7 Z%22 fill=%22white%22 stroke=%22black%22 stroke-width=%221.4%22/></svg>") no-repeat;';
    document.documentElement.appendChild(c);
    document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
    document.addEventListener('mousedown', e => {
      c.style.transform = 'scale(.85)';
      const r = document.createElement('div');
      r.style.cssText = `position:fixed;left:${e.clientX - 16}px;top:${e.clientY - 16}px;width:32px;height:32px;`
        + 'border-radius:50%;border:2px solid rgba(255,122,89,.9);z-index:2147483646;pointer-events:none;'
        + 'transition:all .45s ease-out;opacity:1';
      document.documentElement.appendChild(r);
      requestAnimationFrame(() => { r.style.transform = 'scale(1.8)'; r.style.opacity = '0'; });
      setTimeout(() => r.remove(), 500);
    }, true);
    document.addEventListener('mouseup', () => { c.style.transform = 'scale(1)'; }, true);
  };
  if (document.documentElement) add(); else document.addEventListener('DOMContentLoaded', add);
})();
"""

# ---------------------------------------------------------------------------------------------------------------
# Human-like interaction helpers
# ---------------------------------------------------------------------------------------------------------------

_MOUSE = {"x": W * 0.55, "y": H * 0.45}


def move_to(page, locator, pause=250):
    """Glide the mouse to the centre of an element."""
    locator.scroll_into_view_if_needed()
    box = locator.bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    distance = ((x - _MOUSE["x"]) ** 2 + (y - _MOUSE["y"]) ** 2) ** 0.5
    page.mouse.move(x, y, steps=max(12, int(distance / 18)))
    _MOUSE.update(x=x, y=y)
    page.wait_for_timeout(pause)


def click(page, target, pause=250):
    """Move to an element (selector or locator) and click it."""
    locator = page.locator(target).first if isinstance(target, str) else target
    move_to(page, locator, pause)
    page.mouse.down()
    page.wait_for_timeout(70)
    page.mouse.up()


def type_into(page, selector, text, delay=55):
    """Click a field, clear it and type like a person."""
    click(page, selector)
    page.keyboard.press("Meta+A")
    page.keyboard.press("Backspace")
    page.keyboard.type(text, delay=delay)


def wheel(page, pixels, steps=14):
    """Scroll gradually so the motion is easy to follow."""
    for _ in range(steps):
        page.mouse.wheel(0, pixels / steps)
        page.wait_for_timeout(55)


def seq(*actions):
    """Combine actions into one."""
    def action(page):
        for step in actions:
            step(page)
    return action


def wait(ms):
    """Action that only waits."""
    return lambda page: page.wait_for_timeout(ms)


# ---------------------------------------------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------------------------------------------

TERMINAL = """<html><head><style>
body{margin:0;background:#0b0e13;color:#d6dde8;font:17px/1.45 Menlo,ui-monospace,monospace}
.bar{height:34px;background:#1b1f27;display:flex;align-items:center;padding:0 14px;gap:8px;color:#8a93a6;font:13px -apple-system}
.dot{width:12px;height:12px;border-radius:50%} .t{margin-left:auto;margin-right:auto}
#out{padding:16px 22px;height:740px;overflow:hidden;white-space:pre-wrap;word-break:break-all}
.p{color:#3ccf91} .c{color:#fff;font-weight:600} .cur{display:inline-block;width:9px;background:#d6dde8;animation:b 1s step-end infinite}
@keyframes b{50%{opacity:0}}
</style></head><body><div class='bar'><span class='dot' style='background:#ff5f57'></span><span class='dot'
style='background:#febc2e'></span><span class='dot' style='background:#28c840'></span><span class='t'>cinegraph — zsh</span></div>
<div id='out'></div><script>
const out=document.getElementById('out');
const PROMPT="<span class='p'>deha@mac cinegraph %</span> ";
function scroll(){out.scrollTop=out.scrollHeight}
window.prompt_=()=>{out.insertAdjacentHTML('beforeend',PROMPT+"<span class='c'></span><span class='cur'>&nbsp;</span>");scroll()};
window.typeCmd=async(t)=>{const c=out.querySelectorAll('.c');const el=c[c.length-1];
  for(const ch of t){el.textContent+=ch;await new Promise(r=>setTimeout(r,45+Math.random()*40))}};
window.feed=async(lines,delay)=>{out.querySelector('.cur')?.remove();out.insertAdjacentText('beforeend','\\n');
  for(const l of lines){out.insertAdjacentText('beforeend',l+'\\n');scroll();await new Promise(r=>setTimeout(r,delay))}
  window.prompt_()};
window.clearTerm=()=>{out.innerHTML='';window.prompt_()};
window.prompt_();
</script></body></html>"""


def terminal(page):
    """Open the terminal page (from about:blank so no site security policy blocks its script)."""
    page.goto("about:blank")
    page.set_content(TERMINAL)
    page.wait_for_timeout(300)


def term(command, delay_ms=25, keep_last=None, clear=False):
    """Action: type `command`, run it for real, and stream its real output."""
    def action(page):
        if clear:
            page.evaluate("clearTerm()")
        page.evaluate("t=>typeCmd(t)", command)
        page.wait_for_timeout(len(command) * 65 + 300)
        result = subprocess.run(command, shell=True, cwd=ROOT, capture_output=True, text=True, check=False,
                                stdin=subprocess.DEVNULL, timeout=600)
        lines = ANSI.sub("", result.stdout + result.stderr).rstrip("\n").splitlines()
        if keep_last:
            lines = lines[-keep_last:]
        page.evaluate("([l,d])=>feed(l,d)", [lines, delay_ms])
        page.wait_for_timeout(len(lines) * delay_ms + 400)
    return action


def editor(path, first, last, focus=None):
    """Setup/action: show real lines of a source file in an editor-style page."""
    def setup(page):
        lines = (ROOT / path).read_text(encoding="utf-8").splitlines()[first - 1:last]
        focus_lines = set(range(focus[0], focus[1] + 1)) if focus else set()
        rows = "".join(
            f"<div class='ln{' f' if first + i in focus_lines else ''}'><span class='n'>{first + i}</span>"
            f"<code class='language-python'>{html.escape(line) or ' '}</code></div>"
            for i, line in enumerate(lines))
        page.goto("about:blank")
        page.set_content(f"""<html><head>
<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css'>
<script src='https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js'></script>
<style>body{{margin:0;background:#1e2127;color:#abb2bf;font-family:-apple-system}}
.tabs{{height:38px;background:#16181d;display:flex;align-items:flex-end;padding-left:12px}}
.tab{{background:#1e2127;color:#e6e6e6;font-size:13px;padding:9px 18px;border-radius:6px 6px 0 0}}
.path{{color:#7f848e;font-size:12px;padding:8px 18px;border-bottom:1px solid #2c313a}}
.code{{padding:10px 0;font:16.5px/1.55 Menlo,monospace}} .ln{{display:flex;white-space:pre}}
.ln.f{{background:rgba(97,175,239,.10)}} .n{{width:58px;text-align:right;padding-right:20px;color:#495162;flex:none}}
code.hljs{{padding:0;background:transparent;font:inherit}}</style></head><body>
<div class='tabs'><div class='tab'>{html.escape(Path(path).name)}</div></div>
<div class='path'>cinegraph › {html.escape(path.replace('/', ' › '))}</div><div class='code'>{rows}</div>
<script>document.querySelectorAll('code').forEach(e=>hljs.highlightElement(e))</script></body></html>""")
        page.wait_for_timeout(600)
    return setup


def demo(tab=None):
    """Setup: the demo page, optionally on a given tab."""
    def setup(page):
        page.goto(f"{APP}/demo")
        page.wait_for_load_state("networkidle")
        if tab:
            page.click(f"nav button[data-tab={tab}]")
    return setup


def neo4j(page):
    """Log into Neo4j Browser (local dev password from .env) and clear the welcome frame."""
    password = re.search(r"^NEO4J_PASSWORD=(.*)$", (ROOT / ".env").read_text(), re.M).group(1).strip()
    page.goto("http://localhost:7474/browser/")
    page.wait_for_selector("input[type=password]", timeout=30000)
    page.fill("input[type=password]", password)
    page.keyboard.press("Enter")
    page.wait_for_timeout(4000)
    page.keyboard.press("Escape")
    cypher(":clear", human=False)(page)
    try:
        page.get_by_role("button", name="Collapse Sidebar").click(timeout=1500)
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def cypher(query, human=True, zoom=0):
    """Action: type a query in Neo4j Browser and run it; optionally zoom the graph view in."""
    def action(page):
        editor_box = page.get_by_role("textbox", name="Cypher Editor")
        if human:
            click(page, editor_box)
        else:
            editor_box.click()
        page.keyboard.press("Meta+A")
        page.keyboard.press("Backspace")
        page.keyboard.type(query, delay=28 if human else 0)
        page.keyboard.press("Escape")
        page.keyboard.press("Meta+Enter")
        page.wait_for_timeout(1800)
        for _ in range(zoom):
            try:
                button = page.get_by_role("button", name=re.compile("zoom in", re.I)).first
                click(page, button, pause=120)
                page.wait_for_timeout(250)
            except Exception:  # pylint: disable=broad-exception-caught
                break
    return action


def goto(url, wait_ms=2500):
    """Action/setup: open a URL and let it settle."""
    def action(page):
        page.goto(url, wait_until="domcontentloaded", timeout=90000)
        page.wait_for_timeout(wait_ms)
    return action


def scroll_to_text(text, offset=-90):
    """Action: smooth-scroll so the first element containing `text` is near the top."""
    def action(page):
        page.evaluate(
            """([t,o])=>{const el=[...document.querySelectorAll('h1,h2,h3,p,td,th,.comment-body p')]
                 .find(e=>e.textContent.includes(t)); if(el){window.scrollTo({top:el.getBoundingClientRect().top
                 +window.scrollY+o,behavior:'smooth'})}}""", [text, offset])
        page.wait_for_timeout(1200)
    return action


def swagger_call(page):
    """Open the recommendations operation in Swagger, fill it in and execute it."""
    op_id = "#operations-users-recommendations_api_users__user_id__recommendations_get"
    op = page.locator(op_id)
    click(page, op.locator(".opblock-summary"))
    page.wait_for_timeout(600)
    click(page, op.get_by_role("button", name="Try it out"))
    type_into(page, f"{op_id} input[placeholder='user_id']", "1")
    type_into(page, f"{op_id} input[placeholder='limit']", "3")
    click(page, op.get_by_role("button", name="Execute"))
    live = op.locator(".live-responses-table .response-col_description .highlight-code").first
    live.wait_for(state="visible", timeout=15000)
    page.evaluate("e=>window.scrollTo({top:e.getBoundingClientRect().top+window.scrollY-140,behavior:'smooth'})",
                  live.element_handle())
    page.wait_for_timeout(900)


# ---------------------------------------------------------------------------------------------------------------
# The walkthrough
# ---------------------------------------------------------------------------------------------------------------

SCENES = [
    ("the-app", demo("foryou"), [
        ("Hi, I'm Deha, and this is CineGraph, a movie recommender built on Neo4j with the MovieLens dataset. "
         "I'll show it as a user first, then go underneath.", None),
        ("Let's start with user number one.", lambda p: click(p, "#loadUser")),
        ("At the top is a profile computed from the graph. This person rated two hundred and thirty-two movies, "
         "mostly action and adventure.", lambda p: move_to(p, p.locator("#profile"))),
        ("Below are the recommendations. I find the thirty users whose taste overlaps most with this person, and "
         "collect the films they loved that this person hasn't seen. Terminator 2 comes first.",
         lambda p: move_to(p, p.locator("#recs .movie").first)),
        ("I wanted every recommendation to be explainable. So if I click on 'why'...",
         lambda p: click(p, "[data-why]")),
        ("...I get the evidence from the graph: similar users who loved it, its similarity to The Terminator, which "
         "this user rated five stars, and their taste for action. No language model writes this, so it can't invent "
         "a reason.", lambda p: move_to(p, p.locator(".why").first)),
        ("On the right, community detection suggests movies outside this person's usual taste.",
         lambda p: move_to(p, p.locator("#disc .movie").first)),
    ]),
    ("movies-and-paths", demo("foryou"), [
        ("Now a single movie. I'll search for The Matrix.",
         seq(lambda p: click(p, "nav button[data-tab=movies]"), lambda p: type_into(p, "#q", "The Matrix"),
             lambda p: click(p, "#search"))),
        ("On the left, similarity based on people: movies liked by the same users, like Fight Club and The Empire "
         "Strikes Back.", lambda p: move_to(p, p.locator("#simCollab .movie").first)),
        ("On the right, only shared genres and tags. Genres alone would never put Fight Club next to The Matrix. "
         "That's the value of behaviour in a graph.", lambda p: move_to(p, p.locator("#simContent .movie").first)),
        ("The taste path takes two movies, here Toy Story and The Godfather,",
         seq(lambda p: click(p, "nav button[data-tab=path]"), lambda p: click(p, "#findPath"))),
        ("and finds the strongest chain of similar movies between them. It goes through Star Wars, which makes "
         "sense.", lambda p: move_to(p, p.locator(".path .node").nth(1))),
    ]),
    ("the-assistant", demo("ask"), [
        ("The last tab is an assistant. I'll ask for the best sci-fi movies of the nineties with at least thirty "
         "ratings.", lambda p: click(p, "#askBtn")),
        ("It runs on a small local model through Ollama, so it's free and private. It turns my question into "
         "Cypher, Neo4j's query language.", None),
        ("Before the query runs, it goes through every check on the right: one read-only statement, only labels and "
         "properties from my schema, a row limit, and Neo4j confirming it doesn't write anything.",
         lambda p: move_to(p, p.locator(".checks").first)),
        ("The table is real data. The sentence above is written by the model, so every number in it must appear in "
         "the results.", lambda p: move_to(p, p.locator(".answer").first)),
        ("Now something the data can't answer: who directed Pulp Fiction?",
         seq(lambda p: type_into(p, "#question", "Who directed Pulp Fiction?"), lambda p: click(p, "#askBtn"))),
        ("There's no director data, so the model declines, which is what I want.", wait(500)),
        ("A request to delete all the movies is stopped before the model is even called.",
         seq(lambda p: type_into(p, "#question", "Delete all the movies from the database"),
             lambda p: click(p, "#askBtn"))),
        ("And this box shows what happens if the model itself writes a dangerous query, using a relationship that "
         "doesn't exist.", lambda p: move_to(p, p.locator("#badCypher"))),
        ("It fails three checks and never reaches the database. And every query runs read-only anyway, so Neo4j "
         "would refuse it.", lambda p: click(p, "#checkBtn")),
    ]),
    ("running-it", terminal, [
        ("That's the app. Now, how it's built. It all starts with one command, make docker-run.",
         term("make docker-run", 15)),
        ("It starts three containers: Neo4j with the Graph Data Science plugin, the FastAPI backend, and an Nginx "
         "reverse proxy in front.", None),
    ]),
    ("the-graph", neo4j, [
        ("This is the database. Here's the schema: users, movies, genres and tags.",
         cypher("CALL db.schema.visualization()")),
        ("A rating is a relationship, not a node. It only ever connects one user and one movie, so a node would add "
         "an extra step to every query.", None),
        ("Genres are nodes, so I can travel through them, but also a list on each movie, so filtering by genre "
         "needs no extra hop. A test keeps them in sync.", None),
        ("The similar relationship is computed by a graph algorithm and written back. Here's Toy Story and its "
         "most similar movies.",
         cypher("MATCH p = (m:Movie {title: 'Toy Story'})-[:SIMILAR]->(:Movie) RETURN p", zoom=2)),
        ("Toy Story 2, Star Wars, The Lion King. That comes only from who liked what.", None),
        ("For performance, here's the profile of the trending query.",
         cypher("PROFILE MATCH ()-[r:RATED]->(m:Movie) WHERE r.timestamp >= datetime('2018-06-24') "
                "RETURN m.title, r.rating")),
        ("It starts with an index seek on the rating timestamps. Without that index, it does about thirty-four "
         "times more work.", None),
    ]),
    ("the-ai-schema", goto(f"{REPO}#appendix--our-schema-vs-an-ai-generated-schema", 3000), [
        ("We also had to compare our schema with one generated by an AI. Mine came from the data. The AI's invented "
         "names, emails and ages, and stored every rating twice.",
         scroll_to_text("our schema vs an AI-generated schema", offset=-40)),
        ("The README explains each difference.", lambda p: wheel(p, 260)),
    ]),
    ("the-model", terminal, [
        ("There's also a machine learning part. The graph algorithms give each user and movie an embedding, and a "
         "model predicts whether a user will like a movie.", term("make train", 30)),
        ("I hide each user's latest likes and check if the model predicts them. It gets an AUC of 0.83, against 0.82 "
         "for just recommending popular movies.", None),
        ("A modest gain, and I want to be honest about that.", None),
    ]),
    ("the-code", editor("app/services/movie_service.py", 148, 176, focus=(160, 172)), [
        ("Here's the recommendation query. It's one pattern: from the user, to the movies they liked, to similar "
         "users, to what those users liked.", None),
        ("Not exists removes movies you've already seen, and everything is a parameter, which keeps it safe and "
         "fast.", None),
        ("And this is the LangGraph workflow. The model only proposes a query. Plain code validates and runs it, "
         "and errors go back to the model for up to three tries.", editor("app/agent/graph.py", 168, 194)),
    ]),
    ("the-api", goto(f"{APP}/docs", 1500), [
        ("Everything goes through a FastAPI service, documented in Swagger, with input validation and error codes.",
         lambda p: wheel(p, 380)),
        ("Here's the same recommendations call from the API.", swagger_call),
        ("Same result, Terminator 2 first.", None),
    ]),
    ("quality", terminal, [
        ("Pylint scores above the required 9.5.", term("make lint", 3, keep_last=14)),
        ("And the test suite: ninety-seven tests, all passing.", term("make test", 10, clear=True)),
        ("Each part was built on its own branch and merged through a pull request.",
         term("git log --oneline --graph -n 22", 30, clear=True)),
    ]),
    ("reviews", goto(f"{REPO}/pull/2", 3000), [
        ("Every pull request was reviewed. Here, the review found that the safety filter refused a normal question, "
         "just because it contained the word 'change'.", scroll_to_text("False positive", offset=-120)),
        ("It was fixed before merging, and GitHub Actions runs lint and tests on every pull request.",
         lambda p: wheel(p, 500)),
        ("The API image is published on Docker Hub, for Intel and ARM.",
         seq(terminal, term("docker buildx imagetools inspect dehatanaci06/cinegraph-api:1.0 | grep Platform", 80))),
    ]),
    ("wrap-up", demo("foryou"), [
        ("One limitation: new users with no ratings only get popular movies. And the model is small, which is why "
         "nothing it writes is trusted without checks.", lambda p: click(p, "#loadUser")),
        ("Everything is on GitHub. Thanks for watching.", None),
    ]),
]

# ---------------------------------------------------------------------------------------------------------------
# Audio, recording and assembly
# ---------------------------------------------------------------------------------------------------------------


def run(cmd):
    """Run a command, raising with its output on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(f"{cmd[0]} failed: {result.stderr[-1500:]}")
    return result.stdout


def spoken(text):
    """Apply pronunciation fixes."""
    for word, say_as in SPOKEN.items():
        text = re.sub(rf"(?<![\w.]){re.escape(word)}(?![\w])", say_as, text)
    return text


def synthesize_all():
    """Render every beat with Kokoro (cached by text); return {(scene, beat): (wav, seconds)}."""
    audio_dir = BUILD / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    jobs, paths = [], {}
    for name, _, beats in SCENES:
        for k, (text, _) in enumerate(beats):
            digest = hashlib.sha1(spoken(text).encode()).hexdigest()[:12]
            wav = audio_dir / f"{digest}.wav"
            if not wav.exists() and all(j["final"] != str(wav) for j in jobs):
                jobs.append({"text": spoken(text), "out": str(audio_dir / f"{digest}.raw.wav"), "final": str(wav)})
            paths[(name, k)] = wav
    if jobs:
        job_file = audio_dir / "jobs.json"
        job_file.write_text(json.dumps(jobs))
        print(f"Synthesizing {len(jobs)} beats with Kokoro...", flush=True)
        run([str(TTS_PYTHON), str(PRES / "tts.py"), str(job_file)])
        for job in jobs:
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", job["out"], "-af", "loudnorm=I=-16:TP=-1.5",
                 "-ar", str(RATE), "-ac", "1", job["final"]])
            Path(job["out"]).unlink()
    durations = {}
    for key, wav in paths.items():
        with wave.open(str(wav)) as w:
            durations[key] = (wav, w.getnframes() / w.getframerate())
    return durations


def build_audio(wavs, lengths, out):
    """Concatenate beat audio, padding each beat with silence to its on-screen length."""
    with wave.open(str(out), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(RATE)
        for wav_path, length in zip(wavs, lengths):
            with wave.open(str(wav_path)) as src:
                frames = src.readframes(src.getnframes())
            dst.writeframes(frames)
            pad = max(0, int(round(length * RATE)) - len(frames) // 2)
            dst.writeframes(b"\x00\x00" * pad)


def record_scene(browser, folder, setup, beats, audio):
    """Record one scene; return (video, trim offset, beat lengths)."""
    for old in folder.glob("*.webm"):
        old.unlink()
    context = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(folder),
                                  record_video_size={"width": W, "height": H}, color_scheme="dark")
    context.add_init_script(CURSOR_JS)
    t_create = time.monotonic()
    page = context.new_page()
    page.set_default_timeout(60000)
    setup(page)
    page.evaluate(CURSOR_JS)
    page.mouse.move(_MOUSE["x"], _MOUSE["y"])
    page.wait_for_timeout(600)
    t_start = time.monotonic()
    lengths = []
    for k, ((_, action), (_, seconds)) in enumerate(zip(beats, audio)):
        print(f"      beat {k + 1}/{len(beats)}", flush=True)
        began = time.monotonic()
        if action is not None:
            action(page)
            page.evaluate(CURSOR_JS)
        remaining = seconds + GAP - (time.monotonic() - began)
        if remaining > 0:
            page.wait_for_timeout(int(remaining * 1000))
        lengths.append(time.monotonic() - began)
    page.wait_for_timeout(400)
    video = Path(page.video.path())
    context.close()
    return video, t_start - t_create, lengths


def assemble_scene(folder, video, offset, lengths, wavs):
    """Trim the recording, attach the voice track, encode 1080p30 H.264."""
    voice = folder / "voice.wav"
    build_audio(wavs, lengths, voice)
    total = sum(lengths)
    out = folder / "scene.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{offset:.3f}", "-i", str(video), "-i", str(voice),
         "-filter_complex", "[0:v]scale=1920:1080:flags=lanczos,fps=30,format=yuv420p,"
         "tpad=stop_mode=clone:stop_duration=10[v]",
         "-map", "[v]", "-map", "1:a", "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
         "-c:a", "aac", "-b:a", "192k", "-ar", str(RATE), str(out)])
    return out, total


def write_script(manifest):
    """The narration with a timestamp for every sentence, for recording a voice-over."""
    lines = ["# CineGraph demo — narration\n",
             "Timestamps match `CineGraph_demo_silent.mp4`. Each line starts when its on-screen action starts; "
             "you don't have to match the words exactly, just keep roughly the same pace.\n"]
    clock = 0.0
    for i, (name, _, beats) in enumerate(SCENES, 1):
        entry = manifest.get(str(i), {})
        lines.append(f"## {i}. {name.replace('-', ' ').capitalize()}\n")
        for k, (text, _) in enumerate(beats):
            stamp = clock + sum(entry.get("beats", [])[:k])
            lines.append(f"**{int(stamp // 60)}:{int(stamp % 60):02d}** — {text}\n")
        clock += entry.get("seconds", 0)
    lines.append(f"_Total length: {int(clock // 60)}:{int(clock % 60):02d}_\n")
    (PRES / "narration.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    """Synthesize, record selected scenes (default all), then concatenate everything in order."""
    wanted = {int(a) for a in sys.argv[1:] if a.isdigit()} or set(range(1, len(SCENES) + 1))
    BUILD.mkdir(parents=True, exist_ok=True)
    audio = synthesize_all()
    total_speech = sum(seconds for _, seconds in audio.values())
    print(f"Speech alone: {total_speech / 60:.2f} min", flush=True)
    if "--audio-only" in sys.argv:
        return
    manifest_path = BUILD / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for i, (name, setup, beats) in enumerate(SCENES, 1):
            if i not in wanted:
                continue
            print(f"[{i}/{len(SCENES)}] {name}", flush=True)
            folder = BUILD / f"{i:02d}-{name}"
            folder.mkdir(parents=True, exist_ok=True)
            beat_audio = [audio[(name, k)] for k in range(len(beats))]
            video, offset, lengths = record_scene(browser, folder, setup, beats, beat_audio)
            out, total = assemble_scene(folder, video, offset, lengths, [wav for wav, _ in beat_audio])
            manifest[str(i)] = {"name": name, "file": str(out), "seconds": round(total, 2),
                                "beats": [round(x, 2) for x in lengths]}
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"      {total:.1f}s", flush=True)
        browser.close()
    write_script(manifest)
    parts = [manifest[str(i)]["file"] for i in range(1, len(SCENES) + 1)
             if str(i) in manifest and manifest[str(i)]["name"] == SCENES[i - 1][0]]
    (BUILD / "concat.txt").write_text("".join(f"file '{p}'\n" for p in parts))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(BUILD / "concat.txt"),
         "-c", "copy", "-movflags", "+faststart", str(OUT)])
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(OUT), "-an", "-c:v", "copy", "-movflags", "+faststart",
         str(OUT_SILENT)])
    total = sum(manifest[str(i)]["seconds"] for i in range(1, len(SCENES) + 1) if str(i) in manifest)
    print(f"Video: {OUT} ({total / 60:.2f} min, {len(parts)} scenes)")


if __name__ == "__main__":
    main()
