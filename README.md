# 🚛 Counterfactual: Road Rules

**Predict-first training for truck drivers, coached by Gemma 4 running locally.**

Built at RowdyLaunch (UTSA, Oct 31, 2026) for the **Life Behind the Wheel** track (Cruze Maps) and the **Best Open-Source AI Project** track (MLH · Gemma 4).

## The driver problem

New and experienced drivers get placed out of service and pick up CSA points for hours-of-service violations, and the rules are easy to memorize but easy to misapply. Driving past the 14-hour window after a long wait at a shipper is the classic case. Drivers memorize "11 hours of driving" and miss that two clocks run at once. Most training is a slideshow plus a quiz, and it doesn't stick.

**Who has it:** new CDL drivers in their first year, and fleets onboarding them.

## How it works

Each challenge runs one loop:

1. **Predict:** a common belief ("my 30-minute break has to be off duty") and a concrete situation. The driver commits to an answer and says why.
2. **Test:** the app shows the rule that decides it.
3. **Explain:** it names the hidden assumption behind the mistake.
4. **Transfer:** a new situation checks that the driver can apply the rule.

## What Gemma 4 does

Gemma 4 E2B runs **locally through Ollama**, so no driver data leaves the device and it works without cell signal.

- **Coaches the driver's reasoning.** Gemma reads why the driver picked an answer and points to the exact misunderstanding, not just right or wrong.
- **Builds a challenge from the driver's own situation.** "Got held at the receiver 5 hours, then dispatch wanted another 400 miles" becomes a new predict-first challenge.
- **Explains in the driver's language:** Spanish, Punjabi, Hindi, Russian or Portuguese.
- **Stays grounded.** Gemma gets a verified rule sheet (`RULE_SHEET` in `app.py`) and is told to use only those rules. Generated challenges are labeled as AI-generated.

## Safe from the cab

- **Hands-free mode** reads the situation and question aloud, pauses 8 seconds so the driver can think, then reads the answer. There's nothing to touch, so it works like a podcast while driving.
- **The screen is for when the truck is parked,** for example during the 30-minute break. It uses large buttons and large answer text.
- Audio uses the browser's built-in speech, so it works offline with no API key.

## Run it

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The six built-in challenges work with no model.

### Turn on Gemma 4

1. Install Ollama from https://ollama.com
2. Download the model (about 7 GB): `ollama pull gemma4:e2b`
3. Make sure Ollama is running, then start the app. The sidebar shows **Gemma 4 ready**.

Options: set `GEMMA_MODEL` to change the model (for example `gemma4:e4b`), or set `OLLAMA_URL` if Ollama runs on another machine.

## Built-in challenges

| Challenge | Rule it tests |
|---|---|
| 30-minute break | On-duty, not-driving time (fueling) counts toward the break |
| 14-hour window | Detention time uses up the 14-hour window |
| Long lunch | Off-duty breaks don't extend the 14-hour window |
| Break vs. 11-hour limit | The 30-minute break doesn't reset driving hours |
| Weekly 70-hour limit | The window is rolling 8 days, not the calendar week; 34-hour restart |
| Level II inspection | A walk-around can still find out-of-service defects |

## Sources

- FMCSA hours-of-service rules (property-carrying drivers): 11-hour limit, 14-hour window, 30-minute break after 8 hours of driving, 60/70-hour limit, 34-hour restart. https://www.fmcsa.dot.gov/regulations/hours-service/summary-hours-service-regulations
- CVSA North American Standard inspection levels. https://cvsa.org/inspections/all-inspection-levels/
- Gemma 4 by Google DeepMind (Apache 2.0), served with Ollama.

This is a training aid, not legal advice. Exceptions (short-haul, adverse driving conditions, sleeper-berth splits, state rules) can change an answer.

## What we built today

We built everything in `app.py` during the hacking window: the challenge engine, the driver scenarios, the Gemma 4 coaching, the challenge generator and translation, the grounding rule sheet and the hands-free audio mode. Libraries we used: Streamlit, Ollama and Gemma 4.

## AI tools used

- **Claude (Anthropic):** helped write and review code and the README.
- **Gemma 4:** runs inside the app.

## Team

**Team name:** _TEAM NAME_

| Name | GitHub | LinkedIn |
|---|---|---|
| _Full name_ | _@github_ | _linkedin.com/in/…_ |

## License

MIT. See `LICENSE`.
