"""
MacroNewsAgent.py  —  Groq Edition (100% Free)
===============================================
Uses Groq's free API (Llama 3 70B) instead of Claude.
Includes a keep-alive ping so Render's free tier never sleeps.

Stack cost breakdown:
  Groq API      -> FREE (14,400 req/day limit — you'll use ~20/month)
  FRED API      -> FREE
  Render hosting-> FREE (free tier, kept alive by self-ping)
  TradingView   -> FREE for chart + Pine Script
  Total         -> $0.00

requirements.txt:
  groq
  fredapi
  requests
  python-dotenv
  schedule
  flask

.env file:
  GROQ_API_KEY=your_groq_key        <- free at console.groq.com
  FRED_API_KEY=your_fred_key        <- free at fred.stlouisfed.org
  TV_ALERT_SECRET=any_secret_string <- make up any password
  RENDER_URL=https://your-app.onrender.com
"""

import os, json, time, logging, threading
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
import schedule
from fredapi import Fred
from groq import Groq
from flask import Flask, request, jsonify

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger("MacroAgent")

# ------------------------------------------------------------------ clients
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
TV_SECRET    = os.getenv("TV_ALERT_SECRET", "secret")
RENDER_URL   = os.getenv("RENDER_URL", "")

fred   = Fred(api_key=FRED_API_KEY)
client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = "llama-3.3-70b-versatile"   # best free model on Groq

app = Flask(__name__)

# ------------------------------------------------------------------ events
EVENTS = {
    "CPI": {
        "series_id"   : "CPIAUCSL",
        "name"        : "CPI YoY",
        "unit"        : "%",
        "transform"   : "pc1",
        "instruments" : ["DXY", "EURUSD", "XAUUSD", "US10Y"],
        "typical_move": 0.55,
        "sigma"       : 0.1,
    },
    "NFP": {
        "series_id"   : "PAYEMS",
        "name"        : "Nonfarm Payrolls",
        "unit"        : "K",
        "transform"   : "chg",
        "instruments" : ["DXY", "EURUSD", "US10Y"],
        "typical_move": 0.45,
        "sigma"       : 30,
    },
    "PCE": {
        "series_id"   : "PCEPI",
        "name"        : "PCE Deflator",
        "unit"        : "%",
        "transform"   : "pc1",
        "instruments" : ["DXY", "XAUUSD"],
        "typical_move": 0.35,
        "sigma"       : 0.1,
    },
    "PPI": {
        "series_id"   : "PPIACO",
        "name"        : "PPI",
        "unit"        : "%",
        "transform"   : "pc1",
        "instruments" : ["DXY"],
        "typical_move": 0.25,
        "sigma"       : 0.15,
    },
    "RETAIL": {
        "series_id"   : "RSAFS",
        "name"        : "Retail Sales MoM",
        "unit"        : "%",
        "transform"   : "pc1",
        "instruments" : ["DXY", "SPX"],
        "typical_move": 0.20,
        "sigma"       : 0.2,
    },
}

# Update these the day before each release.
# Source: investing.com/economic-calendar (free, no login needed)
FORECASTS = {
    "CPI"   : {"forecast": 3.2, "previous": 3.5},
    "NFP"   : {"forecast": 185, "previous": 232},
    "PCE"   : {"forecast": 2.6, "previous": 2.8},
    "PPI"   : {"forecast": 2.3, "previous": 2.7},
    "RETAIL": {"forecast": 0.4, "previous": 0.7},
}


# ------------------------------------------------------------------ step 1: fetch
def get_latest_actual(event_key: str):
    cfg = EVENTS[event_key]
    try:
        series = fred.get_series(
            cfg["series_id"],
            observation_start=(datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d"),
        )
        if cfg["transform"] == "pc1":
            pct = series.pct_change(periods=12) * 100
            return round(float(pct.dropna().iloc[-1]), 2)
        elif cfg["transform"] == "chg":
            return round(float(series.diff().dropna().iloc[-1]) / 1000, 1)
        return round(float(series.iloc[-1]), 2)
    except Exception as e:
        log.error(f"FRED fetch failed for {event_key}: {e}")
        return None


# ------------------------------------------------------------------ step 2: surprise
def compute_surprise(event_key: str, actual: float) -> dict:
    fc    = FORECASTS[event_key]
    sigma = EVENTS[event_key]["sigma"]
    diff  = round(actual - fc["forecast"], 2)
    shock = round(diff / sigma, 2)
    return {
        "actual"     : actual,
        "forecast"   : fc["forecast"],
        "previous"   : fc["previous"],
        "diff"       : diff,
        "shock_sigma": shock,
    }


# ------------------------------------------------------------------ step 3: groq AI
def get_ai_verdict(event_key: str, surprise: dict) -> dict:
    cfg = EVENTS[event_key]

    prompt = f"""You are a macro trading analyst. Analyze this economic data release.
Output ONLY a JSON object. No markdown. No explanation. Just raw JSON.

Event: {cfg['name']}
Actual: {surprise['actual']}{cfg['unit']}
Consensus Forecast: {surprise['forecast']}{cfg['unit']}
Previous: {surprise['previous']}{cfg['unit']}
Surprise: {surprise['diff']:+.2f} ({surprise['shock_sigma']:+.1f} standard deviations)
Instruments: {', '.join(cfg['instruments'])}
Historical avg move: +-{cfg['typical_move']}%

Rules:
- CPI/PPI/PCE ABOVE forecast = hot inflation = BULLISH_USD
- CPI/PPI/PCE BELOW forecast = soft inflation = BEARISH_USD
- NFP ABOVE forecast = strong jobs = BULLISH_USD
- NFP BELOW forecast = weak jobs = BEARISH_USD
- |shock| below 0.5 = NEUTRAL
- |shock| 0.5 to 1.5 = MILD
- |shock| above 1.5 = STRONG

Return exactly this JSON:
{{
  "bias": "BULLISH_USD" or "BEARISH_USD" or "NEUTRAL" or "HIGH_VOL",
  "strength": "STRONG" or "MILD" or "NEUTRAL",
  "confidence": <integer 30-85>,
  "context": "<2 sentences max — trading setup explanation>",
  "key_risk": "<1 sentence — what invalidates this bias>",
  "dxy_est": "<e.g. +0.4% or -0.3%>",
  "gold_est": "<e.g. +0.6% or -0.4%>"
}}"""

    try:
        response = client.chat.completions.create(
            model      = GROQ_MODEL,
            max_tokens = 400,
            temperature= 0.1,
            messages   = [{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Extract JSON object even if model adds surrounding text
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        return json.loads(raw)

    except json.JSONDecodeError as e:
        log.error(f"JSON parse failed: {e} | Raw output: {raw}")
        return _fallback_verdict()
    except Exception as e:
        log.error(f"Groq API error: {e}")
        return _fallback_verdict()


def _fallback_verdict() -> dict:
    return {
        "bias"      : "NEUTRAL",
        "strength"  : "NEUTRAL",
        "confidence": 35,
        "context"   : "AI unavailable. Verify manually before trading.",
        "key_risk"  : "System error — check Render logs.",
        "dxy_est"   : "0%",
        "gold_est"  : "0%",
    }


# ------------------------------------------------------------------ step 4: format
def format_result(event_key: str, surprise: dict, verdict: dict) -> dict:
    cfg   = EVENTS[event_key]
    arrow = "▲" if verdict["bias"] == "BULLISH_USD" else "▼" if verdict["bias"] == "BEARISH_USD" else "—"
    return {
        "event"      : event_key,
        "name"       : cfg["name"],
        "actual"     : surprise["actual"],
        "forecast"   : surprise["forecast"],
        "previous"   : surprise["previous"],
        "surprise"   : surprise["diff"],
        "shock_sigma": surprise["shock_sigma"],
        "bias"       : verdict["bias"],
        "strength"   : verdict["strength"],
        "confidence" : verdict["confidence"],
        "context"    : verdict["context"],
        "key_risk"   : verdict["key_risk"],
        "dxy_est"    : verdict["dxy_est"],
        "gold_est"   : verdict.get("gold_est", "n/a"),
        "timestamp"  : datetime.utcnow().isoformat() + "Z",
        # One-line summary — paste this into Pine Script label input
        "pine_label" : (
            f"{arrow} {event_key} {verdict['bias']} | "
            f"Conf:{verdict['confidence']}% | "
            f"Actual:{surprise['actual']} vs {surprise['forecast']} | "
            f"DXY:{verdict['dxy_est']}"
        ),
        # Full human-readable block for Render logs / Telegram
        "summary": (
            f"\n{'='*55}\n"
            f"  {arrow} {event_key} | {verdict['bias']} | {verdict['confidence']}% confidence\n"
            f"  Actual: {surprise['actual']}{cfg['unit']} | "
            f"Forecast: {surprise['forecast']}{cfg['unit']} | "
            f"Surprise: {surprise['diff']:+.2f} ({surprise['shock_sigma']:+.1f}sigma)\n"
            f"  DXY est: {verdict['dxy_est']} | Gold est: {verdict.get('gold_est','n/a')}\n"
            f"  {verdict['context']}\n"
            f"  Risk: {verdict['key_risk']}\n"
            f"{'='*55}"
        ),
    }


# ------------------------------------------------------------------ pipeline
def run_event_analysis(event_key: str):
    event_key = event_key.upper().strip()

    if event_key not in EVENTS:
        log.warning(f"Unknown event '{event_key}'. Valid: {list(EVENTS.keys())}")
        return None

    log.info(f"--- Running {event_key} analysis ---")

    actual = get_latest_actual(event_key)
    if actual is None:
        return None

    surprise = compute_surprise(event_key, actual)
    verdict  = get_ai_verdict(event_key, surprise)
    result   = format_result(event_key, surprise, verdict)

    print(result["summary"])
    print(f"\n  Pine label text:\n  {result['pine_label']}\n")

    return result


# ------------------------------------------------------------------ flask routes

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "agent" : "MacroNewsAgent — Groq Edition",
        "model" : GROQ_MODEL,
        "time"  : datetime.utcnow().isoformat() + "Z",
        "routes": ["/ping", "/status", "/webhook (POST)", "/manual/<EVENT>"],
    }), 200


@app.route("/ping", methods=["GET"])
def ping():
    """UptimeRobot / self-ping hits this every 14 min to prevent Render sleep."""
    return "pong", 200


@app.route("/status", methods=["GET"])
def status():
    """Check current forecasts loaded into the agent."""
    return jsonify({
        "model"    : GROQ_MODEL,
        "events"   : list(EVENTS.keys()),
        "forecasts": FORECASTS,
        "time"     : datetime.utcnow().isoformat() + "Z",
    }), 200


@app.route("/webhook", methods=["POST"])
def webhook():
    """
    TradingView alert sends POST here when it fires.
    Body: {"event": "CPI", "ticker": "EURUSD", "time": "..."}
    Returns full analysis JSON.
    """
    secret = request.headers.get("X-TV-Secret", "")
    if TV_SECRET and secret != TV_SECRET:
        return jsonify({"error": "unauthorized"}), 401

    data      = request.get_json(force=True, silent=True) or {}
    event_key = data.get("event", "CPI").upper()

    result = run_event_analysis(event_key)
    if result is None:
        return jsonify({"error": f"Analysis failed for {event_key}"}), 500

    return jsonify(result), 200


@app.route("/manual/<event_key>", methods=["GET"])
def manual_trigger(event_key):
    """
    Trigger from your browser — no curl needed.
    Just visit: https://your-app.onrender.com/manual/CPI
    """
    result = run_event_analysis(event_key.upper())
    if result is None:
        return jsonify({"error": "Analysis failed — check Render logs"}), 500
    return jsonify(result), 200


@app.route("/forecast/<event_key>", methods=["POST"])
def update_forecast(event_key):
    """
    Update the consensus forecast before an event.
    POST {"forecast": 3.1, "previous": 3.5}
    Use this the night before a release to set the correct street estimate.
    """
    event_key = event_key.upper()
    if event_key not in FORECASTS:
        return jsonify({"error": "Unknown event"}), 404

    data = request.get_json(force=True, silent=True) or {}
    if "forecast" in data:
        FORECASTS[event_key]["forecast"] = float(data["forecast"])
    if "previous" in data:
        FORECASTS[event_key]["previous"] = float(data["previous"])

    log.info(f"Forecast updated: {event_key} -> {FORECASTS[event_key]}")
    return jsonify({"event": event_key, "forecasts": FORECASTS[event_key]}), 200


# ------------------------------------------------------------------ keep-alive
def self_ping():
    """Ping /ping every 14 minutes so Render free tier stays awake."""
    if not RENDER_URL:
        return
    try:
        r = requests.get(f"{RENDER_URL}/ping", timeout=10)
        log.info(f"Keep-alive ping: {r.status_code}")
    except Exception as e:
        log.warning(f"Keep-alive ping failed: {e}")


# ------------------------------------------------------------------ scheduler
def run_scheduler():
    """
    Background thread: schedules auto-runs + keep-alive pings.

    Release times (all 08:30 ET = 13:30 UTC):
      CPI    -> usually 2nd or 3rd Wednesday of month
      NFP    -> 1st Friday of month
      PCE    -> last Friday of month
      PPI    -> ~2 weeks after CPI (usually Thursday)
      RETAIL -> same day as CPI or PPI

    Agent runs at 13:31 UTC (1 min after release) to ensure
    FRED has published the new data point.
    """
    schedule.every(14).minutes.do(self_ping)

    schedule.every().wednesday.at("13:31").do(lambda: run_event_analysis("CPI"))
    schedule.every().friday.at("13:31").do(lambda: run_event_analysis("NFP"))
    schedule.every().friday.at("13:31").do(lambda: run_event_analysis("PCE"))
    schedule.every().thursday.at("13:31").do(lambda: run_event_analysis("PPI"))

    log.info("Scheduler running. Monitoring macro events 24/7.")

    while True:
        schedule.run_pending()
        time.sleep(30)


# ------------------------------------------------------------------ entry point
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Quick terminal test: python MacroNewsAgent.py CPI
        result = run_event_analysis(sys.argv[1])
        if result:
            print("\nCopy this into your Pine Script label input:")
            print(result["pine_label"])
    else:
        # Production mode: scheduler thread + Flask server
        log.info(f"MacroNewsAgent starting — model: {GROQ_MODEL}")
        log.info(f"Render URL: {RENDER_URL or 'not set (self-ping disabled)'}")

        t = threading.Thread(target=run_scheduler, daemon=True)
        t.start()

        port = int(os.environ.get("PORT", 10000))
        log.info(f"Flask listening on port {port}")
        app.run(host="0.0.0.0", port=port, debug=False)
