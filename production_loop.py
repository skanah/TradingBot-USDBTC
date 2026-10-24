"""
Production Online Loop for BTCUSD Scalping Bot
==============================================
Polls LIVE_DATA_URL every 5 minutes, ingests only the newest tail candle,
runs the latest candle through saved ML coefficients, checks GA thresholds,
and sends the resulting action to an exchange adapter.

Defaults are intentionally safe:
- DRY_RUN=1 unless EXCHANGE_API_URL is set or --adapter mt5/http is used.
- No trade is generated if models/ml_model.json is missing or disabled.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

LIVE_DATA_URL = os.getenv("LIVE_DATA_URL", "https://obilo.io/casdata/BTCUSDlive.txt")
HIST_DATA_URL = os.getenv("HIST_DATA_URL", "https://obilo.io/casdata/BTCUSD.txt")

MODEL_DIR = "models"
ML_MODEL_FILE = os.path.join(MODEL_DIR, "ml_model.json")
ROBUST_PARAMS_FILE = os.path.join(MODEL_DIR, "robust_params.json")
BEST_PARAMS_FILE = os.path.join(MODEL_DIR, "best_params.json")
SIGNAL_LOG_FILE = "production_signals.csv"

DEFAULT_POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))  # 5 minutes
DEFAULT_TAIL_CANDLES = int(os.getenv("LIVE_TAIL_CANDLES", "120"))
DEFAULT_MAX_DATA_AGE_SECONDS = int(os.getenv("LIVE_MAX_DATA_AGE_SECONDS", "900"))

DEFAULT_SYMBOL = os.getenv("SYMBOL", "BTCUSD")
DEFAULT_EXCHANGE_ADAPTER = os.getenv("EXCHANGE_ADAPTER", "dry_run")
DEFAULT_EXCHANGE_API_URL = os.getenv("EXCHANGE_API_URL", "")
DEFAULT_EXCHANGE_API_TOKEN = os.getenv("EXCHANGE_API_TOKEN", "")
DEFAULT_DRY_RUN = os.getenv("DRY_RUN", "1" if not DEFAULT_EXCHANGE_API_URL else "0") != "0"

DEFAULT_FEATURE_NAMES = [
    "momentum_short_pct",
    "momentum_med_pct",
    "range_pct",
    "close_position_pct",
    "trend_20_pct",
    "api_signal",
    "api_short",
    "api_med",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("production_loop.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GAParams:
    take_profit: float = 150.0
    stop_loss: float = 120.0
    max_hold: int = 600
    cooldown: int = 180
    momentum_short_period: int = 5
    momentum_med_period: int = 9
    momentum_short_threshold: float = 0.06
    momentum_med_threshold: float = 0.10
    use_api_filter: bool = True
    min_signal_age: int = 0
    fitness: float = 0.0
    wfo_passed: bool = False
    entry_threshold: float = 0.55  # Changed from 0.70 to 0.55 for more responsive trading
    api_signal_threshold: float = 5.0

    @classmethod
    def load(cls) -> "GAParams":
        candidates = [ROBUST_PARAMS_FILE, BEST_PARAMS_FILE]
        data: Dict[str, Any] = {}
        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if loaded.get("fitness", 0) >= data.get("fitness", -1):
                        data = loaded
                except Exception as exc:
                    logger.warning("Could not load GA params from %s: %s", path, exc)

        if not data:
            logger.warning("No GA params found. Using conservative defaults.")
            return cls()

        params = cls()
        params.take_profit = float(data.get("take_profit", params.take_profit))
        params.stop_loss = float(data.get("stop_loss", params.stop_loss))
        params.max_hold = int(data.get("max_hold", params.max_hold))
        params.cooldown = int(data.get("cooldown", params.cooldown))
        params.momentum_short_period = int(data.get("momentum_short_period", params.momentum_short_period))
        params.momentum_med_period = int(data.get("momentum_med_period", params.momentum_med_period))
        params.momentum_short_threshold = float(data.get("momentum_short_threshold", params.momentum_short_threshold))
        params.momentum_med_threshold = float(data.get("momentum_med_threshold", params.momentum_med_threshold))
        params.use_api_filter = bool(data.get("use_api_filter", params.use_api_filter))
        params.min_signal_age = int(data.get("min_signal_age", params.min_signal_age))
        params.fitness = float(data.get("fitness", params.fitness))
        params.wfo_passed = bool(data.get("wfo_passed", params.wfo_passed))
        params.entry_threshold = float(data.get("entry_threshold", params.entry_threshold))
        params.api_signal_threshold = float(data.get("api_signal_threshold", params.api_signal_threshold))

        # Keep live parameters inside the same safety bounds used by the MT5 bot.
        params.take_profit = max(80.0, min(params.take_profit, 250.0))
        params.stop_loss = max(100.0, min(params.stop_loss, 250.0))
        params.cooldown = max(60, params.cooldown)
        params.momentum_short_threshold = max(0.01, params.momentum_short_threshold)
        params.momentum_med_threshold = max(0.01, params.momentum_med_threshold)
        return params


@dataclass
class LiveCandle:
    timestamp: float
    row_index: int
    row: List[float]
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    has_timestamp: bool = False

    @property
    def signature(self) -> Tuple[Any, ...]:
        return tuple(round(float(x), 8) for x in self.row[: min(len(self.row), 12)])


@dataclass
class MLResult:
    available: bool
    enabled: bool
    direction: int
    score: float
    probability: Optional[float]
    threshold: float
    model_type: str
    reasons: List[str]


@dataclass
class SignalDecision:
    action: str
    timestamp: float
    symbol: str
    price: float
    ml_score: float
    ml_probability: Optional[float]
    ml_direction: int
    ga_direction: int
    momentum_short_pct: float
    momentum_med_pct: float
    api_signal: float
    take_profit: float
    stop_loss: float
    max_hold: int
    cooldown: int
    confidence: float
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_csv_rows(text: str) -> List[List[float]]:
    """Parse live/history endpoint text. Supports one CSV row or many CSV rows."""
    rows: List[List[float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            logger.debug("Skipping non-numeric CSV row: %s", raw_line[:120])
    return rows


def parse_live_payload(text: str) -> List[LiveCandle]:
    rows = parse_csv_rows(text)
    candles: List[LiveCandle] = []
    for idx, row in enumerate(rows):
        if len(row) < 5:
            continue
        raw_ts = safe_float(row[0], 0.0)
        has_timestamp = raw_ts > 0
        ts = raw_ts if has_timestamp else time.time() - (len(rows) - idx)
        candles.append(
            LiveCandle(
                timestamp=ts,
                row_index=idx,
                row=row,
                open=safe_float(row[1], safe_float(row[-1])),
                high=safe_float(row[2], safe_float(row[-1])),
                low=safe_float(row[3], safe_float(row[-1])),
                close=safe_float(row[4], safe_float(row[-1])),
                volume=safe_float(row[5], 0.0) if len(row) > 5 else 0.0,
                has_timestamp=has_timestamp,
            )
        )
    candles.sort(key=lambda c: (c.timestamp, c.row_index))
    return candles


def fetch_live_tail(url: str = LIVE_DATA_URL, tail_candles: int = DEFAULT_TAIL_CANDLES) -> List[LiveCandle]:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    candles = parse_live_payload(response.text)
    if not candles:
        return []
    return candles[-tail_candles:]


def fetch_history(url: str = HIST_DATA_URL) -> List[List[float]]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    rows = parse_csv_rows(response.text)
    rows.sort(key=lambda r: r[0] if r else 0.0)
    return rows


def candle_timestamp(candle: LiveCandle) -> float:
    return candle.timestamp


def candle_signature(candle: LiveCandle) -> Tuple[Any, ...]:
    return candle.signature


def extract_features(candles: List[LiveCandle], index: Optional[int] = None, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    if not candles:
        return {name: 0.0 for name in (feature_names or DEFAULT_FEATURE_NAMES)}

    if index is None:
        index = len(candles) - 1
    index = max(0, min(index, len(candles) - 1))
    candle = candles[index]
    closes = [c.close for c in candles[: index + 1]]

    short_period = 5
    med_period = 12
    if index >= 2:
        short_period = 5
    if index >= 12:
        med_period = 12

    def pct_change(period: int) -> float:
        if len(closes) <= period:
            return 0.0
        base = closes[-period - 1]
        return ((closes[-1] - base) / max(abs(base), 1.0)) * 100.0

    high = candle.high
    low = candle.low
    close = candle.close
    high_low = max(high - low, 1.0)

    api_short = safe_float(candle.row[9], 0.0) if len(candle.row) > 9 else 0.0
    api_med = safe_float(candle.row[10], 0.0) if len(candle.row) > 10 else 0.0
    api_signal = api_short + api_med

    features = {
        "momentum_short_pct": pct_change(short_period),
        "momentum_med_pct": pct_change(med_period),
        "range_pct": ((high - low) / max(abs(close), 1.0)) * 100.0,
        "close_position_pct": ((close - low) / high_low) * 100.0,
        "trend_20_pct": pct_change(20),
        "api_signal": api_signal,
        "api_short": api_short,
        "api_med": api_med,
    }

    if feature_names:
        return {name: float(features.get(name, 0.0)) for name in feature_names}
    return features


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class SavedMLModel:
    def __init__(self, model_file: str = ML_MODEL_FILE):
        self.model_file = model_file
        self.model_type = "linear"
        self.feature_names = DEFAULT_FEATURE_NAMES.copy()
        self.coefficients = np.zeros(len(self.feature_names), dtype=float)
        self.intercept = 0.0
        self.enabled = False
        self.probability = False
        self.probability_threshold = 0.55
        self.decision_threshold = 0.0
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.classes: Dict[str, int] = {"sell": 0, "buy": 1}
        self.loaded = False
        self.load_error: Optional[str] = None

    def load(self) -> bool:
        if not os.path.exists(self.model_file):
            self.load_error = f"ML model file not found: {self.model_file}"
            return False
        try:
            with open(self.model_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            self.load_error = f"Could not parse ML model JSON: {exc}"
            return False

        self.enabled = bool(data.get("enabled", True))
        self.model_type = str(data.get("model_type", "linear"))
        self.feature_names = list(data.get("feature_names") or data.get("features") or DEFAULT_FEATURE_NAMES)
        if not self.feature_names:
            self.feature_names = DEFAULT_FEATURE_NAMES.copy()

        coef_raw = (
            data.get("coefficients")
            or data.get("coef")
            or data.get("weights")
            or data.get("ml_coefficients")
            or []
        )
        if isinstance(coef_raw, dict):
            self.coefficients = np.array([float(coef_raw.get(name, 0.0)) for name in self.feature_names], dtype=float)
        else:
            coef_array = np.array([float(x) for x in coef_raw], dtype=float)
            if coef_array.size < len(self.feature_names):
                padded = np.zeros(len(self.feature_names), dtype=float)
                padded[: coef_array.size] = coef_array
                coef_array = padded
            self.coefficients = coef_array[: len(self.feature_names)]

        self.intercept = float(data.get("intercept", data.get("bias", 0.0)))
        self.probability = bool(data.get("probability", data.get("predict_proba", False)))
        self.probability_threshold = float(data.get("probability_threshold", 0.55))
        self.decision_threshold = float(data.get("decision_threshold", 0.0))
        scaler = data.get("standard_scaler") or data.get("scaler") or {}
        if scaler:
            self.mean = np.array([float(x) for x in scaler.get("mean", [])], dtype=float)
            self.scale = np.array([float(x) for x in scaler.get("scale", [])], dtype=float)
        classes = data.get("classes")
        if isinstance(classes, dict):
            self.classes = {str(k): int(v) for k, v in classes.items()}
        self.loaded = True
        logger.info(
            "Loaded ML model: enabled=%s type=%s features=%d probability=%s",
            self.enabled,
            self.model_type,
            len(self.feature_names),
            self.probability,
        )
        return True

    def predict(self, candles: List[LiveCandle]) -> MLResult:
        if not self.loaded:
            self.load()
        if not self.loaded:
            return MLResult(False, False, 0, 0.0, None, 0.0, self.model_type, [self.load_error or "ML model not loaded"])
        if not self.enabled:
            return MLResult(True, False, 0, 0.0, None, 0.0, self.model_type, ["ML model disabled"])
        if not candles:
            return MLResult(True, True, 0, 0.0, None, self.decision_threshold, self.model_type, ["No live candles"])

        features = extract_features(candles, len(candles) - 1, self.feature_names)
        x = np.array([features[name] for name in self.feature_names], dtype=float)
        if self.mean is not None and self.scale is not None:
            mean = self.mean[: x.size] if self.mean.size >= x.size else np.pad(self.mean, (0, max(0, x.size - self.mean.size)))
            scale = self.scale[: x.size] if self.scale.size >= x.size else np.pad(self.scale, (0, max(0, x.size - self.scale.size)))
            scale = np.where(scale == 0, 1.0, scale)
            x = (x - mean) / scale

        score = float(np.dot(x, self.coefficients) + self.intercept)
        probability = sigmoid(score) if self.probability else None

        if self.probability:
            threshold = self.probability_threshold
            if probability >= threshold:
                direction = 1
                reasons = [f"ML probability {probability:.3f} >= {threshold:.3f}"]
            elif probability <= (1.0 - threshold):
                direction = -1
                reasons = [f"ML probability {probability:.3f} <= {1.0 - threshold:.3f}"]
            else:
                direction = 0
                reasons = [f"ML probability {probability:.3f} inside neutral band"]
        else:
            threshold = self.decision_threshold
            if score >= threshold:
                direction = 1
                reasons = [f"ML score {score:.3f} >= {threshold:.3f}"]
            elif score <= -threshold:
                direction = -1
                reasons = [f"ML score {score:.3f} <= {-threshold:.3f}"]
            else:
                direction = 0
                reasons = [f"ML score {score:.3f} inside neutral band"]

        return MLResult(True, True, direction, score, probability, threshold, self.model_type, reasons)


class DryRunExchangeAdapter:
    def execute(self, payload: Dict[str, Any]) -> bool:
        logger.info("DRY RUN exchange payload: %s", json.dumps(payload, indent=2))
        return True


class HttpExchangeAdapter:
    def __init__(self, url: str, token: Optional[str] = None):
        self.url = url
        self.token = token

    def execute(self, payload: Dict[str, Any]) -> bool:
        if not self.url:
            logger.error("EXCHANGE_API_URL is not set; cannot transmit signal")
            return False
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response = requests.post(self.url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        logger.info("Exchange API accepted payload: %s", json.dumps(payload, indent=2))
        return True


class MT5ExchangeAdapter:
    """Optional adapter using the existing MT5 exchange implementation."""

    def __init__(self):
        self.connection = None

    def _connect(self) -> bool:
        if self.connection is not None:
            return True
        try:
            from scalping_bot_v3 import MT5Connection

            self.connection = MT5Connection()
            if not self.connection.connect():
                logger.error("MT5 connection failed")
                return False
            return True
        except Exception as exc:
            logger.error("Could not initialize MT5 exchange adapter: %s", exc)
            return False

    def execute(self, payload: Dict[str, Any]) -> bool:
        action = payload.get("action")
        if action not in {"buy", "sell"}:
            logger.info("MT5 adapter ignoring action=%s", action)
            return True
        if not self._connect():
            return False
        ok = self.connection.open_trade(action, tp=payload.get("take_profit"), sl=payload.get("stop_loss"))
        if self.connection:
            self.connection.disconnect()
            self.connection = None
        return ok


class ProductionLoop:
    def __init__(
        self,
        live_url: str = LIVE_DATA_URL,
        hist_url: str = HIST_DATA_URL,
        poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS,
        tail_candles: int = DEFAULT_TAIL_CANDLES,
        max_data_age_seconds: int = DEFAULT_MAX_DATA_AGE_SECONDS,
        model_file: str = ML_MODEL_FILE,
        exchange_adapter: str = DEFAULT_EXCHANGE_ADAPTER,
        exchange_url: str = DEFAULT_EXCHANGE_API_URL,
        exchange_token: Optional[str] = DEFAULT_EXCHANGE_API_TOKEN,
        dry_run: bool = DEFAULT_DRY_RUN,
        symbol: str = DEFAULT_SYMBOL,
        send_hold_signals: bool = False,
    ):
        self.live_url = live_url
        self.hist_url = hist_url
        self.poll_interval = poll_interval
        self.tail_candles = tail_candles
        self.max_data_age_seconds = max_data_age_seconds
        self.model = SavedMLModel(model_file)
        self.ga_params = GAParams.load()
        self.symbol = symbol
        self.send_hold_signals = send_hold_signals
        self.last_seen_ts = 0.0
        self.last_seen_signature: Optional[Tuple[Any, ...]] = None
        self.last_trade_time = 0.0
        self.running = True

        if dry_run or not exchange_url or exchange_adapter == "dry_run":
            self.exchange = DryRunExchangeAdapter()
            self.exchange_name = "dry_run"
        elif exchange_adapter == "http":
            self.exchange = HttpExchangeAdapter(exchange_url, exchange_token)
            self.exchange_name = "http"
        elif exchange_adapter == "mt5":
            self.exchange = MT5ExchangeAdapter()
            self.exchange_name = "mt5"
        else:
            raise ValueError(f"Unknown exchange adapter: {exchange_adapter}")

        self._init_signal_log()

    def _init_signal_log(self) -> None:
        if not os.path.exists(SIGNAL_LOG_FILE):
            with open(SIGNAL_LOG_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_utc",
                    "action",
                    "symbol",
                    "price",
                    "ml_score",
                    "ml_probability",
                    "ml_direction",
                    "ga_direction",
                    "momentum_short_pct",
                    "momentum_med_pct",
                    "api_signal",
                    "take_profit",
                    "stop_loss",
                    "cooldown",
                    "confidence",
                    "reasons",
                ])

    def append_signal_log(self, decision: SignalDecision) -> None:
        with open(SIGNAL_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                utc_now(),
                decision.action,
                decision.symbol,
                f"{decision.price:.8f}",
                f"{decision.ml_score:.6f}",
                "" if decision.ml_probability is None else f"{decision.ml_probability:.6f}",
                decision.ml_direction,
                decision.ga_direction,
                f"{decision.momentum_short_pct:.6f}",
                f"{decision.momentum_med_pct:.6f}",
                f"{decision.api_signal:.6f}",
                f"{decision.take_profit:.6f}",
                f"{decision.stop_loss:.6f}",
                decision.cooldown,
                f"{decision.confidence:.6f}",
                "; ".join(decision.reasons),
            ])

    def newest_tail(self) -> Optional[LiveCandle]:
        candles = fetch_live_tail(self.live_url, self.tail_candles)
        if not candles:
            return None
        latest = candles[-1]
        if latest.has_timestamp:
            age = time.time() - candle_timestamp(latest)
            if age > self.max_data_age_seconds:
                logger.warning("Latest live data is stale: age=%.0fs > max=%.0fs", age, self.max_data_age_seconds)
                return None

            # Timestamp-based de-duplication for normal multi-candle live feeds.
            if candle_timestamp(latest) <= self.last_seen_ts:
                return None

        # Fallback signature de-duplication for single-row indicator feeds without timestamps.
        sig = candle_signature(latest)
        if not latest.has_timestamp and sig == self.last_seen_signature:
            return None

        if latest.has_timestamp:
            self.last_seen_ts = candle_timestamp(latest)
        self.last_seen_signature = sig
        return latest

    def evaluate_latest(self, candles: List[LiveCandle]) -> SignalDecision:
        latest = candles[-1]
        features = extract_features(candles, len(candles) - 1)
        ml = self.model.predict(candles)
        reasons = list(ml.reasons)

        short_pct = features["momentum_short_pct"]
        med_pct = features["momentum_med_pct"]
        api_signal = features["api_signal"]

        ga_direction = 0
        if abs(short_pct) >= self.ga_params.momentum_short_threshold and abs(med_pct) >= self.ga_params.momentum_med_threshold:
            if short_pct > 0 and med_pct > 0:
                ga_direction = 1
            elif short_pct < 0 and med_pct < 0:
                ga_direction = -1

        if ga_direction == 0:
            reasons.append("GA momentum thresholds not satisfied")

        if self.ga_params.use_api_filter and api_signal != 0:
            api_direction = 1 if api_signal >= self.ga_params.api_signal_threshold else (-1 if api_signal <= -self.ga_params.api_signal_threshold else 0)
            if api_direction and ga_direction and api_direction != ga_direction:
                ga_direction = 0
                reasons.append(f"API signal opposes GA direction: api={api_signal:.1f}")

        # Add detailed logging for trade rejection reasons
        if not ml.enabled:
            action = "hold"
            reasons.append("ML model disabled")
        elif ml.direction == 0:
            action = "hold"
            reasons.append(f"ML score {ml.score:.3f} inside neutral band (threshold={ml.threshold:.3f})")
        elif ga_direction == 0:
            action = "hold"
            reasons.append("GA direction not confirmed")
        elif ml.direction != ga_direction:
            action = "hold"
            reasons.append(f"ML/GA disagreement: ML={ml.direction:+d}, GA={ga_direction:+d}")
        else:
            action = "buy" if ml.direction == 1 else "sell"
            reasons.append(f"Entry signal confirmed: ML={ml.direction:+d}, GA={ga_direction:+d}, score={ml.score:.3f}")

        confidence = float(abs(ml.score)) if ml.probability is None else float(max(ml.probability, 1.0 - ml.probability))
        decision = SignalDecision(
            action=action,
            timestamp=time.time(),
            symbol=self.symbol,
            price=latest.close,
            ml_score=ml.score,
            ml_probability=ml.probability,
            ml_direction=ml.direction,
            ga_direction=ga_direction,
            momentum_short_pct=short_pct,
            momentum_med_pct=med_pct,
            api_signal=api_signal,
            take_profit=self.ga_params.take_profit,
            stop_loss=self.ga_params.stop_loss,
            max_hold=self.ga_params.max_hold,
            cooldown=self.ga_params.cooldown,
            confidence=confidence,
            reasons=reasons,
        )
        return decision

    def execute_signal(self, decision: SignalDecision) -> bool:
        now = time.time()
        if decision.action in {"buy", "sell"} and now - self.last_trade_time < self.ga_params.cooldown:
            logger.info(
                "Cooldown active: %s signal suppressed (%.0fs remaining)",
                decision.action.upper(),
                self.ga_params.cooldown - (now - self.last_trade_time),
            )
            return False

        payload = decision.to_dict()
        payload.update(
            {
                "timestamp_utc": utc_now(),
                "exchange_adapter": self.exchange_name,
                "ga_params": {
                    "take_profit": self.ga_params.take_profit,
                    "stop_loss": self.ga_params.stop_loss,
                    "max_hold": self.ga_params.max_hold,
                    "cooldown": self.ga_params.cooldown,
                    "momentum_short_threshold": self.ga_params.momentum_short_threshold,
                    "momentum_med_threshold": self.ga_params.momentum_med_threshold,
                    "use_api_filter": self.ga_params.use_api_filter,
                    "fitness": self.ga_params.fitness,
                    "wfo_passed": self.ga_params.wfo_passed,
                },
            }
        )

        if decision.action == "hold" and not self.send_hold_signals:
            logger.info("HOLD signal not sent to exchange: %s", json.dumps(payload, indent=2))
            return True

        ok = self.exchange.execute(payload)
        if ok and decision.action in {"buy", "sell"}:
            self.last_trade_time = now
        return ok

    def poll_once(self) -> Optional[SignalDecision]:
        candles = fetch_live_tail(self.live_url, self.tail_candles)
        if not candles:
            logger.warning("No live candles fetched")
            return None
        latest = candles[-1]
        if latest.has_timestamp:
            age = time.time() - candle_timestamp(latest)
            if age > self.max_data_age_seconds:
                logger.warning("Latest live data is stale: age=%.0fs > max=%.0fs", age, self.max_data_age_seconds)
                return None

            sig = candle_signature(latest)
            if candle_timestamp(latest) <= self.last_seen_ts:
                logger.info("No new live candle since ts=%.0f", self.last_seen_ts)
                return None
        else:
            sig = candle_signature(latest)
            if sig == self.last_seen_signature:
                logger.info("No new live candle signature change")
                return None

        if latest.has_timestamp:
            self.last_seen_ts = candle_timestamp(latest)
        self.last_seen_signature = sig

        decision = self.evaluate_latest(candles)
        self.append_signal_log(decision)
        logger.info(
            "Decision: action=%s price=%.2f ML=%+.3f prob=%s GA=%+d S=%.3f%% M=%.3f%% API=%.1f reasons=%s",
            decision.action.upper(),
            decision.price,
            decision.ml_score,
            "" if decision.ml_probability is None else f"{decision.ml_probability:.3f}",
            decision.ga_direction,
            decision.momentum_short_pct,
            decision.momentum_med_pct,
            decision.api_signal,
            "; ".join(decision.reasons),
        )
        self.execute_signal(decision)
        return decision

    def run_forever(self) -> None:
        logger.info(
            "Production loop starting: live=%s poll=%ss tail=%d exchange=%s dry_run=%s",
            self.live_url,
            self.poll_interval,
            self.tail_candles,
            self.exchange_name,
            isinstance(self.exchange, DryRunExchangeAdapter),
        )
        while self.running:
            started = time.time()
            try:
                self.poll_once()
            except KeyboardInterrupt:
                logger.info("Interrupted; stopping production loop")
                self.running = False
                break
            except Exception as exc:
                logger.error("Production poll failed: %s", exc, exc_info=True)

            elapsed = time.time() - started
            sleep_for = max(1.0, self.poll_interval - elapsed)
            end_sleep = time.time() + sleep_for
            logger.info("Next poll in %.0fs", sleep_for)
            while self.running and time.time() < end_sleep:
                time.sleep(min(5.0, end_sleep - time.time()))


def train_linear_model_from_history(
    output_file: str = ML_MODEL_FILE,
    hist_url: str = HIST_DATA_URL,
    epochs: int = 1000,
    learning_rate: float = 0.1,
    l2: float = 0.001,
) -> bool:
    """Train a simple logistic-regression ML model from HIST_DATA_URL and save coefficients.

    Labels predict whether the next candle close is higher than the current close.
    This is intentionally lightweight and dependency-free beyond numpy.
    """
    candles_rows = fetch_history(hist_url)
    if len(candles_rows) < 100:
        logger.error("Not enough historical candles for ML training: %d", len(candles_rows))
        return False

    candles = []
    for row in candles_rows:
        if len(row) < 5:
            continue
        candles.append(
            LiveCandle(
                timestamp=safe_float(row[0], 0.0),
                row_index=0,
                row=row,
                open=safe_float(row[1], safe_float(row[-1])),
                high=safe_float(row[2], safe_float(row[-1])),
                low=safe_float(row[3], safe_float(row[-1])),
                close=safe_float(row[4], safe_float(row[-1])),
                volume=safe_float(row[5], 0.0) if len(row) > 5 else 0.0,
            )
        )
    candles.sort(key=lambda c: c.timestamp)

    feature_names = DEFAULT_FEATURE_NAMES.copy()
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for idx in range(20, len(candles) - 1):
        window = candles[max(0, idx - 30) : idx + 1]
        features = extract_features(window, len(window) - 1, feature_names)
        label = 1 if candles[idx + 1].close > candles[idx].close else 0
        xs.append(np.array([features[name] for name in feature_names], dtype=float))
        ys.append(label)

    if len(xs) < 50:
        logger.error("Not enough labeled samples for ML training: %d", len(xs))
        return False

    x = np.vstack(xs)
    y = np.array(ys, dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    x_norm = (x - mean) / scale

    weights = np.zeros(x_norm.shape[1], dtype=float)
    intercept = 0.0
    n = x_norm.shape[0]
    for _ in range(epochs):
        z = x_norm @ weights + intercept
        pred = sigmoid_array(z)
        error = pred - y
        grad_w = (x_norm.T @ error) / n + l2 * weights
        grad_b = error.mean()
        weights -= learning_rate * grad_w
        intercept -= learning_rate * grad_b

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    model = {
        "enabled": True,
        "model_type": "linear_probability",
        "probability": True,
        "feature_names": feature_names,
        "coefficients": [float(v) for v in weights],
        "intercept": float(intercept),
        "standard_scaler": {
            "mean": [float(v) for v in mean],
            "scale": [float(v) for v in scale],
        },
        "classes": {"sell": 0, "buy": 1},
        "probability_threshold": 0.55,
        "training": {
            "source": hist_url,
            "samples": int(n),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "positive_rate": float(y.mean()),
            "trained_at_utc": utc_now(),
        },
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    logger.info("Saved trained ML model to %s", output_file)
    return True


def sigmoid_array(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production online loop for BTCUSD scalping bot")
    parser.add_argument("--live-url", default=LIVE_DATA_URL)
    parser.add_argument("--hist-url", default=HIST_DATA_URL)
    parser.add_argument("--interval", type=int, default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--tail-candles", type=int, default=DEFAULT_TAIL_CANDLES)
    parser.add_argument("--max-data-age-seconds", type=int, default=DEFAULT_MAX_DATA_AGE_SECONDS)
    parser.add_argument("--model-file", default=ML_MODEL_FILE)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--adapter", choices=["dry_run", "http", "mt5"], default=DEFAULT_EXCHANGE_ADAPTER)
    parser.add_argument("--exchange-url", default=DEFAULT_EXCHANGE_API_URL)
    parser.add_argument("--exchange-token", default=DEFAULT_EXCHANGE_API_TOKEN)
    parser.add_argument("--dry-run", action="store_true", help="Log signals only; do not send to exchange")
    parser.add_argument("--send-hold-signals", action="store_true", help="Send HOLD payloads to the exchange adapter")
    parser.add_argument("--once", action="store_true", help="Run a single poll and exit")
    parser.add_argument(
        "--train-ml",
        action="store_true",
        help="Train/save models/ml_model.json from HIST_DATA_URL before running production loop",
    )
    parser.add_argument("--ml-epochs", type=int, default=1000)
    parser.add_argument("--ml-lr", type=float, default=0.1)
    parser.add_argument("--ml-l2", type=float, default=0.001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.train_ml:
        if not train_linear_model_from_history(
            output_file=args.model_file,
            hist_url=args.hist_url,
            epochs=args.ml_epochs,
            learning_rate=args.ml_lr,
            l2=args.ml_l2,
        ):
            return 1

    loop = ProductionLoop(
        live_url=args.live_url,
        hist_url=args.hist_url,
        poll_interval=args.interval,
        tail_candles=args.tail_candles,
        max_data_age_seconds=args.max_data_age_seconds,
        model_file=args.model_file,
        exchange_adapter=args.adapter,
        exchange_url=args.exchange_url,
        exchange_token=args.exchange_token,
        dry_run=args.dry_run,
        symbol=args.symbol,
        send_hold_signals=args.send_hold_signals,
    )

    def stop(_signum: int, _frame: Any) -> None:
        loop.running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    if args.once:
        loop.poll_once()
        return 0

    loop.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())