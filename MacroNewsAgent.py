"""
MacroNewsAgent.py
=================
AI agent that:
1. Monitors economic calendar (Trading Economics / FRED)
2. Computes surprise score when data releases
3. Calls Claude API for direction bias + context
4. Sends webhook to TradingView to trigger Pine Script labels

Requirements:
    pip install fredapi requests anthropic python-dotenv schedule

.env file:
    FRED_API_KEY=your_fred_key
    ANTHROPIC_API_KEY=your_claude_key
    TV_WEBHOOK_URL=https://webhook.site/your-id   # or your server URL
    TV_ALERT_SECRET=your_tradingview_webhook_secret
"""

import os, json, time, schedule, logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
from fredapi import Fred
import anthropic

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("MacroAgent")

# ─── CONFIG ─────────────────────────────────────────────────────────────────
FRED_API_KEY    = os.getenv("FRED_API_KEY")
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY")
TV_WEBHOOK_URL  = os.getenv("TV_WEBHOOK_URL")          # your TradingView webhook endpoint
TV_SECRET       = os.getenv("TV_ALERT_SECRET", "")

fred   = Fred(api_key=FRED_API_KEY)
client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# ─── EVENT REGISTRY ─────────────────────────────────────────────────────────
# FRED series IDs + metadata for each tracked event
EVENTS = {
    "CPI": {
        "series_id"   : "CPIAUCSL",      # CPI All Urban Consumers
        "name"        : "CPI YoY",
        "unit"        : "%",
        "transform"   : "pc1",           # % change from year ago
        "instruments" : ["DXY", "EURUSD", "XAUUSD", "US10Y"],
        "typical_move": 0.55,            # % average USD move on ±0.1 surprise
    },
    "NFP": {
        "series_id"   : "PAYEMS",        # Nonfarm Payrolls
        "name"        : "Nonfarm Payrolls",
        "unit"        : "K",
        "transform"   : "chg",           # change
        "instruments" : ["DXY", "EURUSD", "US10Y"],
        "typical_move": 0.45,
    },
    "PCE": {
        "series_id"   : "PCEPI",
        "name"        : "PCE Deflator",
        "unit"        : "%",
        "transform"   : "pc1",
        "instruments" : ["DXY", "XAUUSD"],
        "typical_move": 0.35,
    },
    "PPI": {
        "series_id"   : "PPIACO",
        "name"        : "PPI",
        "unit"        : "%",
        "transform"   : "pc1",
        "instruments" : ["DXY"],
        "typical_move": 0.25,
    },
}

# ─── CONSENSUS FORECASTS ─────────────────────────────────────────────────────
# In production: fetch from Trading Economics API or Econoday
# Here we store manually or pull from a free source
FORECASTS = {
    "CPI" : {"forecast": 3.2, "previous": 3.5},
    "NFP" : {"forecast": 185, "previous": 232},
    "PCE" : {"forecast": 2.6, "previous": 2.8},
    "PPI" : {"forecast": 2.3, "previous": 2.7},
}


def get_latest_actual(event_key: str) -> float | None:
    """Pull the most recent data point from FRED."""
    cfg = EVENTS[event_key]
    try:
        series = fred.get_series(
            cfg["series_id"],
            observation_start=(datetime.today() - timedelta(days=60)).strftime("%Y-%m-%d"),
        )
        if cfg["transform"] == "pc1":
            pct = series.pct_change(periods=12) * 100   # YoY %
            return round(pct.dropna().iloc[-1], 2)
        elif cfg["transform"] == "chg":
            return round((series.diff().dropna().iloc[-1]) / 1000, 1)  # → thousands
        return round(series.iloc[-1], 2)
    except Exception as e:
        log.error(f"FRED fetch failed for {event_key}: {e}")
        return None


def compute_surprise(event_key: str, actual: float) -> dict:
    """Calculate surprise vs consensus and derive shock magnitude."""
    fc   = FORECASTS[event_key]
    diff = round(actual - fc["forecast"], 2)
    # Normalise: how many σ from consensus? (rough: 1σ ≈ 0.1 for CPI, 30K for NFP)
    sigma_map = {"CPI": 0.1, "NFP": 30, "PCE": 0.1, "PPI": 0.15}
    sigma = sigma_map.get(event_key, 0.1)
    shock = round(diff / sigma, 2)
    return {
        "actual"  : actual,
        "forecast": fc["forecast"],
        "previous": fc["previous"],
        "diff"    : diff,
        "shock_sigma": shock,     # >1 = hot, <-1 = soft
    }


def get_ai_verdict(event_key: str, surprise: dict) -> dict:
    """Call Claude to generate direction bias + 2-line context."""
    prompt = f"""
You are a macro trading agent. Analyze this economic data release and output a JSON decision.

Event: {EVENTS[event_key]['name']}
Actual: {surprise['actual']}{EVENTS[event_key]['unit']}
Consensus Forecast: {surprise['forecast']}{EVENTS[event_key]['unit']}
Previous: {surprise['previous']}{EVENTS[event_key]['unit']}
Surprise: {surprise['diff']:+.2f} ({surprise['shock_sigma']:+.1f}σ)
Instruments affected: {', '.join(EVENTS[event_key]['instruments'])}

Rules:
- If actual > forecast (hot print for inflation, strong for payrolls): bullish USD
- If actual < forecast (soft print): bearish USD
- Factor in magnitude: |shock| < 0.5σ → NEUTRAL, 0.5–1.5σ → mild, >1.5σ → strong
- For CPI specifically: hot = bearish risk assets, cold = bullish risk assets

Respond ONLY with valid JSON, no markdown, no preamble:
{{
  "bias": "BULLISH_USD" | "BEARISH_USD" | "NEUTRAL" | "HIGH_VOL",
  "strength": "STRONG" | "MILD" | "NEUTRAL",
  "confidence": <integer 30-90>,
  "context": "<2 sentence max trading context>",
  "key_risk": "<one sentence on what could invalidate this bias>",
  "dxy_est": "<estimated DXY % move, e.g. +0.5%>",
  "gold_est": "<estimated XAUUSD % move>"
}}
"""
    try:
        msg = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 400,
            messages   = [{"role": "user", "content": prompt}],
        )
        raw  = msg.content[0].text.strip()
        data = json.loads(raw)
        return data
    except Exception as e:
        log.error(f"Claude API error: {e}")
        return {
            "bias"      : "NEUTRAL",
            "strength"  : "NEUTRAL",
            "confidence": 40,
            "context"   : "AI analysis unavailable. Trade with caution.",
            "key_risk"  : "System error.",
            "dxy_est"   : "0%",
            "gold_est"  : "0%",
        }


def send_to_tradingview(event_key: str, surprise: dict, verdict: dict):
    """
    POST the signal to TradingView via webhook.

    TradingView setup:
    1. Create an alert on any chart → Webhook URL = your server
    2. On your server, receive POST → forward to TV via Pine's 'alert()' function
       OR use a service like webhook.site for testing.
    3. In Pine Script: use input.string() overrides to display the label.

    The payload below matches the Pine Script input structure.
    """
    payload = {
        "event"     : event_key,
        "name"      : EVENTS[event_key]["name"],
        "actual"    : surprise["actual"],
        "forecast"  : surprise["forecast"],
        "surprise"  : surprise["diff"],
        "shock"     : surprise["shock_sigma"],
        "bias"      : verdict["bias"],
        "strength"  : verdict["strength"],
        "confidence": verdict["confidence"],
        "context"   : verdict["context"],
        "key_risk"  : verdict["key_risk"],
        "dxy_est"   : verdict["dxy_est"],
        "gold_est"  : verdict.get("gold_est", "n/a"),
        "timestamp" : datetime.utcnow().isoformat() + "Z",
        # Pine Script will read these values via alert message parsing
        # Format: "EVENT=CPI|BIAS=BEARISH_USD|CONF=67|..."
        "tv_message": (
            f"MACRO_SIGNAL|EVENT={event_key}|"
            f"BIAS={verdict['bias']}|CONF={verdict['confidence']}|"
            f"ACTUAL={surprise['actual']}|FORECAST={surprise['forecast']}|"
            f"CONTEXT={verdict['context']}"
        ),
    }

    headers = {
        "Content-Type" : "application/json",
        "X-TV-Secret"  : TV_SECRET,
    }

    try:
        r = requests.post(TV_WEBHOOK_URL, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        log.info(f"✓ Webhook sent — {event_key} | {verdict['bias']} | {verdict['confidence']}% conf")
        log.info(f"  Context: {verdict['context']}")
    except requests.RequestException as e:
        log.error(f"Webhook failed: {e}")

    return payload


def run_event_analysis(event_key: str):
    """Full pipeline: fetch → surprise → AI → webhook."""
    log.info(f"─── Running analysis for {event_key} ───")

    actual = get_latest_actual(event_key)
    if actual is None:
        log.warning(f"No data for {event_key}, skipping.")
        return

    surprise = compute_surprise(event_key, actual)
    log.info(f"  Surprise: {surprise['diff']:+.2f} ({surprise['shock_sigma']:+.1f}σ)")

    verdict = get_ai_verdict(event_key, surprise)
    log.info(f"  Verdict: {verdict['bias']} | Conf: {verdict['confidence']}%")

    payload = send_to_tradingview(event_key, surprise, verdict)

    # Pretty print for terminal
    print("\n" + "="*60)
    print(f"  {event_key} ANALYSIS COMPLETE — {datetime.utcnow().strftime('%H:%M UTC')}")
    print("="*60)
    print(f"  Actual:     {surprise['actual']} (Forecast: {surprise['forecast']})")
    print(f"  Surprise:   {surprise['diff']:+.2f} | {surprise['shock_sigma']:+.1f}σ")
    print(f"  Bias:       {verdict['bias']} ({verdict['strength']}) — {verdict['confidence']}% conf")
    print(f"  DXY est:    {verdict['dxy_est']}")
    print(f"  Gold est:   {verdict.get('gold_est','n/a')}")
    print(f"  Context:    {verdict['context']}")
    print(f"  Key risk:   {verdict['key_risk']}")
    print("="*60 + "\n")

    return payload


# ─── SCHEDULER ───────────────────────────────────────────────────────────────
# Set exact release times (ET) here. The agent runs 1 min after release
# to ensure FRED has the data. Adjust dates before each release.

def schedule_events():
    """
    Example schedule — update dates before each release month.
    CPI: 2nd or 3rd week of month, 08:30 ET
    NFP: 1st Friday of month, 08:30 ET
    """
    # CPI — run at 08:31 ET on release day
    schedule.every().day.at("13:31").do(   # 08:31 ET = 13:31 UTC
        lambda: run_event_analysis("CPI")
    ).tag("CPI")

    # NFP — 1st Friday 08:31 ET
    schedule.every().friday.at("13:31").do(
        lambda: run_event_analysis("NFP")
    ).tag("NFP")

    # PCE — last Friday of month
    schedule.every().friday.at("13:31").do(
        lambda: run_event_analysis("PCE")
    ).tag("PCE")

    log.info("Scheduler configured. Waiting for events...")
    while True:
        schedule.run_pending()
        time.sleep(30)


# ─── CLI / MANUAL RUN ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Manual trigger: python MacroNewsAgent.py CPI
        event = sys.argv[1].upper()
        if event in EVENTS:
            run_event_analysis(event)
        else:
            print(f"Unknown event '{event}'. Valid: {list(EVENTS.keys())}")
    else:
        # Run scheduler
        print("Starting Macro News Agent scheduler...")
        print("Manual run: python MacroNewsAgent.py CPI|NFP|PCE|PPI\n")
        schedule_events()
