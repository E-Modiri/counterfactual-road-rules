"""Counterfactual: Road Rules.

Predict-first training for truck drivers. Learn the hours-of-service and roadside
inspection rules by predicting what happens when one condition changes.
Gemma 4 (run locally via Ollama) coaches the driver's reasoning, writes new
scenarios from the driver's own situation, and explains in the driver's language.
"""
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

import streamlit as st

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODEL_ID = os.getenv("GEMMA_MODEL", "gemma4:e2b")

LANGUAGES = {
    "English": "en-US",
    "Español": "es-US",
    "ਪੰਜਾਬੀ (Punjabi)": "pa-IN",
    "हिन्दी (Hindi)": "hi-IN",
    "Русский (Russian)": "ru-RU",
    "Português": "pt-BR",
}

# Verified rule sheet. Gemma is told to use ONLY these facts, so generated
# scenarios stay grounded instead of inventing regulations.
RULE_SHEET = """U.S. FMCSA hours-of-service, property-carrying CMV drivers:
- 11-hour limit: may drive at most 11 hours after 10 consecutive hours off duty.
- 14-hour window: may not drive beyond the 14th consecutive hour after coming on duty following 10 consecutive hours off duty. Off-duty time (lunch, waiting) does NOT extend the window, except qualifying sleeper-berth split periods.
- 30-minute break: required after 8 cumulative hours of driving without at least a 30-minute interruption. Any 30 consecutive minutes not driving counts: off duty, sleeper berth, or on-duty not driving (fueling, paperwork, loading).
- 60/70-hour limit: may not drive after 60 hours on duty in 7 consecutive days or 70 hours in 8 consecutive days. It is a rolling window, not a calendar week.
- 34-hour restart: 34 or more consecutive hours off duty resets the 60/70-hour clock to zero.
CVSA roadside inspection levels:
- Level I: full inspection of driver credentials/records and the vehicle, including under the vehicle (37 steps).
- Level II: walk-around; driver credentials plus all vehicle items that can be checked without going under the vehicle.
- Level III: driver-only; credentials, medical card, hours-of-service records and other paperwork.
- Level V: vehicle-only, without the driver present.
- Out-of-service violations found at any level can stop the truck or the driver."""

SCENARIOS: dict[str, dict[str, Any]] = {
    "30-minute break": {
        "topic": "Hours of service · 30-minute break",
        "claim": "My 30-minute break only counts if I'm off duty. Fueling doesn't count.",
        "context": "You've driven 7 hours 45 minutes. You pull into a truck stop and spend 30 straight minutes fueling and doing paperwork, logged as on-duty (not driving).",
        "question": "Have you satisfied the 30-minute break, so you can drive past the 8-hour mark?",
        "choices": [
            "No. Only off-duty or sleeper-berth time counts as a break.",
            "No. The break only counts if you take it after you hit 8 hours.",
            "Yes. Any 30 consecutive minutes of not driving counts, including on-duty work like fueling.",
            "Yes, but only if a second driver was at the wheel.",
        ],
        "correct": 2,
        "answer": "Yes. The break is required after 8 cumulative hours of driving, and any 30 consecutive minutes of non-driving time satisfies it: off duty, sleeper berth, or on-duty not driving, such as fueling or paperwork.",
        "assumption": "The hidden assumption is that a 'break' has to mean resting off duty. The rule only asks that you stop driving for 30 minutes in a row.",
        "transfer": "Now suppose you fuel for 20 minutes, drive 10 minutes to a parking spot, then do 10 minutes of paperwork. Is the break satisfied? Why or why not?",
        "hint": "The rule is about driving time, not about duty status.",
    },
    "14-hour window": {
        "topic": "Hours of service · 14-hour window",
        "claim": "I've only driven 5 hours today, so I still have 6 hours of driving left.",
        "context": "You came on duty at 6:00 AM after 10 hours off. The shipper held you for 9 hours. You've driven 5 hours total. It's now 8:00 PM.",
        "question": "Can you keep driving tonight?",
        "choices": [
            "Yes. You've only used 5 of your 11 driving hours.",
            "No. 14 hours have passed since you came on duty, so the driving window is closed.",
            "Yes, as long as you take a 30-minute break first.",
            "Yes, because waiting at a shipper doesn't count against any clock.",
        ],
        "correct": 1,
        "answer": "No. The 14-hour window runs on the clock from when you came on duty: 6:00 AM to 8:00 PM is 14 hours. Once it closes you may not drive, even with 6 hours left on the 11-hour driving limit. Detention time eats the window.",
        "assumption": "The hidden assumption is that the 11-hour driving limit is the only clock. Two clocks run at once, and whichever runs out first stops you.",
        "transfer": "What does it take to reset your 14-hour window so you can drive again? And how could you plan the day differently if you expect a long wait at the shipper?",
        "hint": "Count hours on the wall clock since you came on duty, not hours behind the wheel.",
    },
    "Long lunch": {
        "topic": "Hours of service · 14-hour window",
        "claim": "If I take a 2-hour lunch off duty, my 14-hour day stretches to 16 hours.",
        "context": "You came on duty at 5:00 AM. At noon you go off duty for a 2-hour lunch, then get back on the road at 2:00 PM. You're not using the sleeper berth.",
        "question": "What's the latest time you can drive today?",
        "choices": [
            "9:00 PM, because the 2 off-duty hours are added to the end of the day.",
            "7:00 PM, because off-duty time doesn't pause the 14-hour window.",
            "There's no limit as long as you stay under 11 hours of driving.",
            "5:00 PM, because a lunch break ends your driving day.",
        ],
        "correct": 1,
        "answer": "7:00 PM. The 14-hour window is consecutive: 5:00 AM plus 14 hours is 7:00 PM. Off-duty breaks like lunch do not pause or extend it. Only qualifying sleeper-berth split periods work differently.",
        "assumption": "The hidden assumption is that the 14-hour limit counts working hours. It counts wall-clock time from the moment you come on duty.",
        "transfer": "A dispatcher suggests a long off-duty break mid-day so you can 'drive later tonight.' What would you check before agreeing?",
        "hint": "Is it 14 hours of work, or 14 hours of clock time?",
    },
    "Break vs. 11-hour limit": {
        "topic": "Hours of service · 11-hour driving limit",
        "claim": "Taking my 30-minute break resets my driving hours.",
        "context": "After 10 hours off, you drove 8 hours, took a 30-minute break, then drove 3 more hours. You're 12 hours into your 14-hour window.",
        "question": "Can you drive one more hour to reach your customer?",
        "choices": [
            "No. You've driven 11 hours, the daily maximum, and the break doesn't reset it.",
            "Yes. The break reset your driving clock, so you have 8 hours available.",
            "Yes. You still have 2 hours left in your 14-hour window.",
            "Yes, if you log the extra hour as on-duty not driving.",
        ],
        "correct": 0,
        "answer": "No. You've driven 8 + 3 = 11 hours, the most you may drive after 10 consecutive hours off duty. The 30-minute break satisfies the break rule. It doesn't restore driving hours, and time left in the 14-hour window doesn't help once the 11 are gone.",
        "assumption": "The hidden assumption is that the rules share one clock. The break, the 11-hour limit and the 14-hour window are separate limits, and you must satisfy all three.",
        "transfer": "What actually gives you a fresh 11 hours of driving? And why would logging driving as 'on-duty not driving' be a serious problem?",
        "hint": "Add up every hour behind the wheel since your last 10 hours off.",
    },
    "Weekly 70-hour limit": {
        "topic": "Hours of service · 60/70-hour limit",
        "claim": "My 70 hours reset every Monday.",
        "context": "You run on the 70-hour/8-day schedule. Counting the last 8 days, you've been on duty 68 hours. Today is Monday.",
        "question": "Because it's a new week, do you have a full 70 hours available today?",
        "choices": [
            "Yes. The weekly limit resets at the start of each calendar week.",
            "Yes, as long as you took Sunday off.",
            "No. It's a rolling 8-day window. You only get hours back as old days roll off, or after a 34-hour restart.",
            "No. You can't do any work at all until next week.",
        ],
        "correct": 2,
        "answer": "No. The 70-hour limit looks back over any 8 consecutive days. It's rolling, not tied to the calendar. You get hours back as the oldest day drops off, or all at once by taking 34 or more consecutive hours off duty (a 34-hour restart).",
        "assumption": "The hidden assumption is that 'weekly' means the calendar week. The rule counts the last 8 days, whatever day it is.",
        "transfer": "You have 3 hours left on your 70 and a 6-hour run tomorrow. What are your two options to make it legal?",
        "hint": "Look back 8 days from today, not to the start of the week.",
    },
    "Level II inspection": {
        "topic": "Roadside inspection · CVSA levels",
        "claim": "If the officer doesn't go under my truck, they can't put it out of service for equipment problems.",
        "context": "You're pulled in for a Level II walk-around inspection. The officer checks your license and logs, then walks around the truck and finds a chafed air line and a broken turn signal, both visible without going under the truck.",
        "question": "Can a Level II inspection put your truck out of service for equipment?",
        "choices": [
            "No. Only a full Level I can find vehicle problems.",
            "No. A Level II only checks paperwork.",
            "It depends on whether you're carrying hazmat.",
            "Yes. Level II covers every vehicle item that can be checked without going under the truck, and out-of-service defects found there count.",
        ],
        "correct": 3,
        "answer": "Yes. A Level II walk-around covers driver credentials plus every vehicle item that can be inspected without going under the vehicle. If the officer finds an out-of-service defect, the truck can be stopped right there.",
        "assumption": "The hidden assumption is that only the most thorough inspection matters. Every level looks for problems in its own scope.",
        "transfer": "A Level III inspection is driver-only. What would the officer check, and what would you have ready before you roll in?",
        "hint": "Think about what an officer can see while walking around the truck.",
    },
}


# ---------------------------------------------------------------- Gemma 4 ----

@st.cache_data(ttl=15, show_spinner=False)
def gemma_status() -> tuple[bool, str]:
    """Check that Ollama is running and the Gemma model has been pulled."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as r:
            models = [m.get("name", "") for m in json.load(r).get("models", [])]
    except Exception:
        return False, "Ollama isn't running. Start it, then run `ollama pull " + MODEL_ID + "`."
    base = MODEL_ID.split(":")[0]
    if MODEL_ID in models or (":" not in MODEL_ID and any(m.startswith(base) for m in models)):
        return True, f"Gemma 4 ready · `{MODEL_ID}`"
    return False, f"Model not found. Run `ollama pull {MODEL_ID}`."


def ask_gemma(system: str, user: str, want_json: bool = False, timeout: int = 120) -> str:
    """Send one chat turn to Gemma 4 through Ollama's local API."""
    body: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,  # skip thinking mode: much faster on small models
        "options": {"temperature": 0.3},
    }
    if want_json:
        body["format"] = "json"
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["message"]["content"].strip()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Couldn't reach Gemma at {OLLAMA_URL}: {exc}") from exc


def parse_json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"```(?:json)?", "", text or "", flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def gemma_generate(situation: str, language: str) -> dict[str, Any]:
    system = (
        "You are Counterfactual, a training coach for professional truck drivers. "
        "Use ONLY the rules in this rule sheet. If the situation needs a rule that is "
        "not on the sheet, pick the closest rule from the sheet instead of inventing one.\n\n"
        + RULE_SHEET
    )
    user = f"""A driver described this situation: "{situation}"

Write one predict-first training challenge based on it. State a common mistaken belief as the claim, then describe a concrete situation (with times and hours) that tests it.
Write every text field in {language}. Return ONLY JSON with these keys:
{{"title": "short title", "claim": "the mistaken belief", "context": "the concrete situation",
 "question": "one question", "choices": ["four distinct answers"], "correct_index": 0,
 "answer": "why the correct answer follows, citing the rule", "assumption": "the hidden assumption",
 "transfer": "a follow-up question applying the rule to a new case", "hint": "a hint that doesn't give it away"}}
correct_index is an integer 0-3. Put the correct answer at a random position."""
    data = parse_json(ask_gemma(system, user, want_json=True))
    keys = ["title", "claim", "context", "question", "choices", "correct_index",
            "answer", "assumption", "transfer", "hint"]
    if not data or any(k not in data for k in keys):
        raise ValueError("Gemma's answer was missing fields. Try again, or rephrase the situation.")
    if not isinstance(data["choices"], list) or len(data["choices"]) != 4:
        raise ValueError("Gemma didn't return exactly four choices. Try again.")
    try:
        data["correct_index"] = int(data["correct_index"])
    except (TypeError, ValueError):
        raise ValueError("Gemma returned an invalid answer index. Try again.")
    if not 0 <= data["correct_index"] < 4:
        raise ValueError("Gemma returned an invalid answer index. Try again.")
    data["choices"] = [str(c) for c in data["choices"]]
    return data


def gemma_coach(s: dict[str, Any], picked: str, reasoning: str, language: str) -> str:
    system = (
        "You are a friendly, experienced trucking safety trainer. Be brief and concrete. "
        "Use only these rules:\n" + RULE_SHEET
    )
    user = f"""Scenario: {s['context']}
Question: {s['question']}
Correct answer: {s['choices'][s['correct']]}
Why: {s['answer']}
The driver picked: {picked}
The driver's reasoning: {reasoning or '(none given)'}

In {language}, give the driver 2-3 short sentences of feedback on their REASONING (not just whether the pick was right): what they got right, the exact misunderstanding if any, and one tip to remember the rule on the road. Plain text, no lists."""
    return ask_gemma(system, user)


def gemma_translate(s: dict[str, Any], language: str) -> str:
    system = "You translate trucking safety training into plain, simple language a driver can follow. Keep numbers and rule names exact."
    user = (f"Translate into {language}. Keep it short and clear.\n\n"
            f"Correct answer: {s['choices'][s['correct']]}\nWhy: {s['answer']}\n"
            f"Hidden assumption: {s['assumption']}")
    return ask_gemma(system, user)


# ------------------------------------------------------------- Audio --------

def speak(parts: list[str], lang: str, pause_ms: int = 0, label: str = "▶ Play") -> None:
    """Read text aloud with the browser's built-in speech (offline, no API).
    With pause_ms, waits between the question and the answer so the driver can think."""
    # Escape "</" so text (including Gemma output) can't close the <script> tag.
    payload = json.dumps(parts).replace("</", "<\\/")
    html = f"""
<style>html,body{{margin:0;overflow:hidden}}</style><button id="b" style="font-size:18px;padding:12px 18px;border-radius:12px;border:0;
  background:#1f3a8a;color:white;cursor:pointer;width:100%;font-weight:700;box-sizing:border-box">{label}</button>
<script>
const parts = {payload}, lang = "{lang}", pause = {pause_ms};
function say(t) {{ return new Promise(res => {{
  const u = new SpeechSynthesisUtterance(t); u.lang = lang; u.rate = 0.95;
  u.onend = res; u.onerror = res; speechSynthesis.speak(u); }}); }}
document.getElementById("b").onclick = async () => {{
  speechSynthesis.cancel();
  for (let i = 0; i < parts.length; i++) {{
    await say(parts[i]);
    if (pause && i === parts.length - 2) await new Promise(r => setTimeout(r, pause));
  }}
}};
</script>"""
    if hasattr(st, "iframe"):
        st.iframe(html, height=72)
    else:  # older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=72)


def basic_reasoning_check(reasoning: str) -> str:
    lower = reasoning.lower()
    if any(w in lower for w in ["because", "since", "clock", "window", "hours", "rule", "break", "limit"]):
        return "You gave a reason. Compare it with the rule above: did you name the clock or limit that decides it?"
    return "Try adding a reason: “I think this because…”. Naming the rule is what makes it stick."


# --------------------------------------------------------------- Page -------

st.set_page_config(page_title="Counterfactual: Road Rules", page_icon="🚛", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.4rem; max-width: 1200px; }
.hero { padding: 1.4rem 1.6rem; border: 1px solid #e6eaf1; border-radius: 22px;
        background: linear-gradient(120deg, #eef3ff 0%, #fff7ec 100%); }
.kicker { color:#1f3a8a; font-weight:800; text-transform:uppercase; letter-spacing:.11em; font-size:.76rem; }
.hero h1 { font-size:2.5rem; letter-spacing:-.03em; margin:.25rem 0 .4rem 0; color:#182238; }
.hero p { font-size:1.08rem; color:#4c5870; max-width:860px; margin-bottom:.4rem; }
.panel { border:1px solid #e6eaf1; border-radius:16px; padding:1rem 1.1rem; background:white; color:#172033; }
.panel h3, .panel b { color:#172033 !important; }
.small-muted { color:#68738a; font-size:.92rem; }
.tag { display:inline-block; background:#e3e9ff; color:#1f3a8a; border-radius:99px; padding:.25rem .7rem; font-size:.82rem; font-weight:700; margin-right:.3rem; }
.ai { display:inline-block; background:#fff1d6; color:#8a5300; border-radius:99px; padding:.2rem .6rem; font-size:.78rem; font-weight:700; }
/* Big touch targets for use in the cab */
div.stButton > button { min-height: 3.1rem; font-size: 1.05rem; font-weight: 700; border-radius: 12px; }
div[role="radiogroup"] label { font-size: 1.05rem; padding: .35rem 0; }
</style>""", unsafe_allow_html=True)

ATTEMPT_KEYS = ["submitted", "selected", "explanation", "transfer_answer", "show_hint",
                "coach", "translation"]


def reset_attempt() -> None:
    for k in ATTEMPT_KEYS:
        st.session_state.pop(k, None)


st.session_state.setdefault("scenario_name", next(iter(SCENARIOS)))
st.session_state.setdefault("submitted", False)
ready, status_msg = gemma_status()

st.markdown("""
<div class="hero">
  <div class="kicker">Life Behind the Wheel · Open-source AI with Gemma 4</div>
  <h1>🚛 Counterfactual: Road Rules</h1>
  <p>Hours-of-service violations put drivers out of service. Most come from rules that are easy to memorize and easy to misapply.
  Counterfactual changes one condition, asks you to predict what happens, then shows the rule you missed.</p>
  <span class="tag">Predict → Test → Explain → Transfer</span><span class="tag">Hands-free audio</span><span class="tag">Your language</span>
</div>""", unsafe_allow_html=True)
st.caption("🅿️ Use the screen only while parked. On the road, use hands-free mode: it reads the challenge aloud, pauses, then reads the answer, with nothing to touch.")

with st.sidebar:
    st.header("Challenges")
    names = list(SCENARIOS) + (["✨ My situation"] if "custom" in st.session_state else [])
    current = st.session_state.scenario_name if st.session_state.scenario_name in names else names[0]
    picked_name = st.radio("Challenge", names, index=names.index(current), label_visibility="collapsed")
    if picked_name != st.session_state.scenario_name:
        st.session_state.scenario_name = picked_name
        reset_attempt()
        st.rerun()

    st.divider()
    language = st.selectbox("Language", list(LANGUAGES), help="Gemma explains and coaches in this language. Audio uses your device's voice for it.")

    st.divider()
    st.subheader("✨ Train on my situation")
    if ready:
        st.success(status_msg)
    else:
        st.warning(status_msg)
    situation = st.text_area("What happened on your last run?",
                             placeholder="e.g. Got held at the receiver 5 hours, then dispatch wanted me to run another 400 miles.")
    if st.button("Build my challenge", type="primary", use_container_width=True, disabled=not ready):
        if not situation.strip():
            st.warning("Describe a situation first.")
        else:
            try:
                with st.spinner("Gemma 4 is building a challenge from the rule sheet…"):
                    g = gemma_generate(situation.strip(), language)
                st.session_state.custom = {
                    "topic": "Your situation · written by Gemma 4", "claim": g["claim"],
                    "context": g["context"], "question": g["question"], "choices": g["choices"],
                    "correct": g["correct_index"], "answer": g["answer"], "assumption": g["assumption"],
                    "transfer": g["transfer"], "hint": g["hint"], "title": g["title"], "ai": True,
                }
                st.session_state.scenario_name = "✨ My situation"
                reset_attempt()
                st.rerun()
            except Exception as exc:
                st.error(f"Gemma couldn't build it: {exc}")

if st.session_state.scenario_name == "✨ My situation" and "custom" in st.session_state:
    s = st.session_state.custom
    title = s["title"]
else:
    s = SCENARIOS[st.session_state.scenario_name]
    title = st.session_state.scenario_name
lang_code = LANGUAGES[language]

st.markdown(f"### {title}")
st.caption(s["topic"])
if s.get("ai"):
    st.markdown('<span class="ai">AI-generated by Gemma 4 from the rule sheet · check against FMCSA before relying on it</span>', unsafe_allow_html=True)

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.markdown("#### What a lot of drivers believe")
    st.markdown(f'<div class="panel"><h3 style="margin-top:0">“{s["claim"]}”</h3><p class="small-muted">Your job: test it, not repeat it.</p></div>', unsafe_allow_html=True)
    st.markdown("#### The situation")
    st.info(s["context"])
    st.markdown("#### Your call")
    st.write(s["question"])
    selected = st.radio("Pick the answer you think is right", s["choices"], index=None,
                        key="selected", disabled=st.session_state.submitted)
    st.text_area("Why? (optional, but Gemma coaches your reasoning)", key="explanation",
                 disabled=st.session_state.submitted, height=90,
                 placeholder="Which clock or rule decides this?")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Check my answer", type="primary", use_container_width=True,
                     disabled=st.session_state.submitted or selected is None):
            st.session_state.submitted = True
            st.rerun()
    with c2:
        if st.button("Hint", use_container_width=True):
            st.session_state.show_hint = True
    with c3:
        if st.button("Next challenge", use_container_width=True):
            keys = list(SCENARIOS)
            cur = st.session_state.scenario_name
            st.session_state.scenario_name = keys[(keys.index(cur) + 1) % len(keys)] if cur in keys else keys[0]
            reset_attempt()
            st.rerun()
    if st.session_state.get("show_hint"):
        st.warning(f"Hint: {s['hint']}")

with right:
    st.markdown("#### 🎧 Hands-free mode")
    st.markdown('<p class="small-muted">Reads the situation and question, pauses 8 seconds so you can think, then reads the answer. In English for built-in challenges; Gemma challenges play in the language they were written in.</p>', unsafe_allow_html=True)
    speak([f"{s['claim']}. {s['context']}", s["question"],
           f"The answer: {s['choices'][s['correct']]}. {s['answer']}"],
          "en-US" if not s.get("ai") else lang_code, pause_ms=8000, label="🎧 Play challenge")
    st.write("")
    st.markdown("#### Why predict first?")
    st.markdown('<div class="panel"><b>1 · Predict</b><p class="small-muted">Commit to an answer before you see the rule.</p><b>2 · Test</b><p class="small-muted">One changed condition shows whether your rule of thumb holds.</p><b>3 · Explain</b><p class="small-muted">Find the hidden assumption, not just the right option.</p><b>4 · Transfer</b><p class="small-muted">Apply it to the next situation you\'ll actually face.</p></div>', unsafe_allow_html=True)

if st.session_state.submitted:
    st.divider()
    right_pick = st.session_state.selected == s["choices"][s["correct"]]
    if right_pick:
        st.success("✅ Right call. Now check that your reason matches the rule.")
    else:
        st.error("❌ That's the trap. Here's the rule that decides it.")
    st.markdown(f"**Correct answer:** {s['choices'][s['correct']]}")
    st.markdown(f"**Why:** {s['answer']}")
    st.markdown(f"**Hidden assumption:** {s['assumption']}")

    reasoning = (st.session_state.get("explanation") or "").strip()
    st.markdown("#### 🧠 Coach")
    if ready:
        if "coach" not in st.session_state:
            try:
                with st.spinner("Gemma 4 is reading your reasoning…"):
                    st.session_state.coach = gemma_coach(s, st.session_state.selected, reasoning, language)
            except Exception as exc:
                st.session_state.coach = f"(Gemma unavailable: {exc}) " + basic_reasoning_check(reasoning)
        st.info(st.session_state.coach)
        st.caption("Feedback by Gemma 4, running locally. No driver data leaves this device.")
        if language != "English" and not s.get("ai"):
            if st.button(f"Explain the rule in {language}"):
                try:
                    with st.spinner("Translating…"):
                        st.session_state.translation = gemma_translate(s, language)
                except Exception as exc:
                    st.error(str(exc))
            if st.session_state.get("translation"):
                st.info(st.session_state.translation)
                speak([st.session_state.translation], lang_code, label=f"🔊 Read aloud in {language}")
    else:
        st.info(basic_reasoning_check(reasoning) if reasoning else "Add your reasoning next time. With Gemma 4 running, you get personal coaching on it.")

    st.markdown("#### Next situation")
    st.write(s["transfer"])
    st.text_area("Your answer", key="transfer_answer", placeholder="Name the rule, then predict what happens…")
    speak([f"The answer: {s['choices'][s['correct']]}. {s['answer']}"], "en-US" if not s.get("ai") else lang_code,
          label="🔊 Read the answer aloud")

st.divider()
st.caption("Built-in rules summarize FMCSA hours-of-service rules for property-carrying drivers and CVSA inspection levels. "
           "This is a training aid, not legal advice. Exceptions (short-haul, adverse conditions, sleeper-berth splits, state rules) can change the answer.")
