"""
Online Learning Engine for Adaptive Scalping Bot
=================================================
Tracks trade outcomes, identifies patterns, and adjusts behavior.
Supports automatic mode switching based on performance.
"""
import os, time, json, logging, random, csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("learning.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


TRADE_HISTORY_SIZE = 100
LEARNING_INTERVAL = 300
MODE_SWITCH_THRESHOLD = 5
CSV_LEARNING_FILE = "learning_log.csv"
CSV_PERFORMANCE_FILE = "performance_log.csv"


class TradeOutcome:
    def __init__(self, direction: str, entry_price: float, exit_price: float,
                 pnl: float, held_seconds: float, timestamp: float,
                 momentum_short: int, momentum_med: int, api_signal: float,
                 regime: str = "unknown"):
        self.direction = direction
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.pnl = pnl
        self.held_seconds = held_seconds
        self.timestamp = timestamp
        self.momentum_short = momentum_short
        self.momentum_med = momentum_med
        self.api_signal = api_signal
        self.regime = regime
        self.won = pnl > 0


@dataclass
class LearningStats:
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_hold_time: float
    best_direction: str
    best_momentum: str
    current_mode: str
    confidence: float


class OnlineLearningEngine:
    MODES = {
        'RECONNAISSANCE': {
            'description': 'Learning mode - more trades',
            'cooldown_mult': 0.5,
            'threshold_mult': 0.7,
            'max_hold_mult': 1.5,
            'sl_mult': 1.0,
        },
        'SCALPING': {
            'description': 'Normal trading mode',
            'cooldown_mult': 1.0,
            'threshold_mult': 1.0,
            'max_hold_mult': 1.0,
            'sl_mult': 1.0,
        },
        'AGGRESSIVE': {
            'description': 'Profit-taking mode',
            'cooldown_mult': 0.8,
            'threshold_mult': 0.8,
            'max_hold_mult': 0.7,
            'sl_mult': 1.0,
        },
        'DEFENSIVE': {
            'description': 'Protection mode after losses',
            'cooldown_mult': 2.0,
            'threshold_mult': 1.2,
            'max_hold_mult': 0.8,
            'sl_mult': 1.0,
        },
    }

    def __init__(self):
        self.trade_history = deque(maxlen=TRADE_HISTORY_SIZE)
        self.start_time = time.time()
        self.last_learning_time = 0
        self.current_mode = 'SCALPING'
        self.mode_consecutive_count = 0
        self.learning_enabled = True
        
        self._init_csv_files()
        
        logger.info(f"Learning engine initialized in {self.current_mode} mode")

    def _init_csv_files(self):
        if not os.path.exists(CSV_LEARNING_FILE):
            with open(CSV_LEARNING_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'direction', 'pnl', 'held', 'momentum_s', 
                              'momentum_m', 'api', 'regime', 'mode', 'params_changed'])
        
        if not os.path.exists(CSV_PERFORMANCE_FILE):
            with open(CSV_PERFORMANCE_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'balance', 'equity', 'daily_pnl',
                               'total_trades', 'win_rate', 'avg_pnl', 'current_mode'])

    def record_trade(self, direction: str, entry_price: float, exit_price: float,
                     pnl: float, held_seconds: float, momentum_short: int = 0,
                     momentum_med: int = 0, api_signal: float = 0,
                     regime: str = "unknown", current_params: Dict = None):
        outcome = TradeOutcome(
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            held_seconds=held_seconds,
            timestamp=time.time(),
            momentum_short=momentum_short,
            momentum_med=momentum_med,
            api_signal=api_signal,
            regime=regime
        )
        self.trade_history.append(outcome)
        
        with open(CSV_LEARNING_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.fromtimestamp(outcome.timestamp).isoformat(),
                outcome.direction,
                f"{outcome.pnl:.2f}",
                f"{outcome.held_seconds:.0f}",
                outcome.momentum_short,
                outcome.momentum_med,
                f"{outcome.api_signal:.1f}",
                outcome.regime,
                self.current_mode,
                'N/A'
            ])
        
        self._check_mode_switch()
        
        return outcome

    def _check_mode_switch(self):
        if len(self.trade_history) < 1:
            return
        
        recent_trades = list(self.trade_history)[-20:]
        recent_pnl = sum(t.pnl for t in recent_trades)
        recent_wins = sum(1 for t in recent_trades if t.won)
        recent_wr = recent_wins / len(recent_trades) if recent_trades else 0
        
        # Count consecutive losses
        consecutive_losses = 0
        for t in reversed(self.trade_history):
            if not t.won:
                consecutive_losses += 1
            else:
                break
        
        new_mode = self.current_mode
        should_switch = False
        
        # Switch to DEFENSIVE after 2+ consecutive losses to tighten up before hard stop
        if consecutive_losses >= 2 and self.current_mode != 'DEFENSIVE':
            new_mode = 'DEFENSIVE'
            should_switch = True
            logger.warning(f"MODE SWITCH: {self.current_mode} -> DEFENSIVE ({consecutive_losses} consecutive losses)")
        
        elif recent_pnl < -6 and self.current_mode != 'DEFENSIVE':
            new_mode = 'DEFENSIVE'
            should_switch = True
            logger.warning(f"MODE SWITCH: {self.current_mode} -> DEFENSIVE (recent loss: ${recent_pnl:.2f})")
        
        elif len(recent_trades) >= 3 and recent_wr >= 0.75 and self.current_mode == 'SCALPING':
            new_mode = 'AGGRESSIVE'
            should_switch = True
            logger.info(f"MODE SWITCH: {self.current_mode} -> AGGRESSIVE (win rate: {recent_wr:.0%} over {len(recent_trades)} trades)")
        
        elif recent_wr < 0.40 and self.current_mode == 'AGGRESSIVE':
            new_mode = 'SCALPING'
            should_switch = True
            logger.info(f"MODE SWITCH: {self.current_mode} -> SCALPING (win rate dropped: {recent_wr:.0%})")
        
        # Recovery from DEFENSIVE - require multiple wins before removing protection
        elif self.current_mode == 'DEFENSIVE':
            wins_in_last_5 = sum(1 for t in list(self.trade_history)[-5:] if t.won)
            if wins_in_last_5 >= 3:
                new_mode = 'SCALPING'
                should_switch = True
                logger.info(f"MODE SWITCH: DEFENSIVE -> SCALPING ({wins_in_last_5} wins in last 5, recovering)")
            # Also recover if enough time has passed without losses
            elif len(self.trade_history) >= 5 and recent_pnl >= -0.5:
                new_mode = 'SCALPING'
                should_switch = True
                logger.info(f"MODE SWITCH: DEFENSIVE -> SCALPING (stabilized: ${recent_pnl:.2f})")
        
        if should_switch:
            self._switch_mode(new_mode)

    def _switch_mode(self, new_mode: str):
        if new_mode in self.MODES:
            old_mode = self.current_mode
            self.current_mode = new_mode
            self.mode_consecutive_count += 1
            
            logger.info(f"MODE CHANGED: {old_mode} -> {new_mode}")
            logger.info(f"  {self.MODES[new_mode]['description']}")

    def get_adjusted_params(self, base_params: Dict) -> Dict:
        if not self.learning_enabled:
            return base_params
        
        mode_settings = self.MODES.get(self.current_mode, self.MODES['SCALPING'])
        
        adjusted = base_params.copy()
        adjusted['cooldown'] = int(base_params.get('cooldown', 180) * mode_settings['cooldown_mult'])
        adjusted['take_profit'] = base_params.get('take_profit', 15.0)
        adjusted['stop_loss'] = base_params.get('stop_loss', 10.0) * mode_settings['sl_mult']
        adjusted['max_hold'] = int(base_params.get('max_hold', 600) * mode_settings['max_hold_mult'])
        
        threshold_mult = mode_settings['threshold_mult']
        adjusted['momentum_short_threshold'] = base_params.get('momentum_short_threshold', 0.03) * threshold_mult
        adjusted['momentum_med_threshold'] = base_params.get('momentum_med_threshold', 0.05) * threshold_mult
        
        return adjusted

    def get_learning_stats(self) -> LearningStats:
        if not self.trade_history:
            return LearningStats(
                total_trades=0, win_rate=0, avg_win=0, avg_loss=0,
                profit_factor=0, avg_hold_time=0, best_direction="",
                best_momentum="", current_mode=self.current_mode, confidence=0
            )
        
        trades = list(self.trade_history)
        total = len(trades)
        wins = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]
        
        win_rate = len(wins) / total if total > 0 else 0
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 1
        
        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        avg_hold = sum(t.held_seconds for t in trades) / total if trades else 0
        
        dir_counts = {}
        for t in trades:
            key = f"{t.direction}_{t.momentum_short}"
            dir_counts[key] = dir_counts.get(key, 0) + (1 if t.won else -0.5)
        best_direction = max(dir_counts, key=dir_counts.get) if dir_counts else ""
        
        return LearningStats(
            total_trades=total,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_hold_time=avg_hold,
            best_direction=best_direction,
            best_momentum="",
            current_mode=self.current_mode,
            confidence=min(1.0, total / 50)
        )

    def should_quick_evolve(self) -> bool:
        if not self.learning_enabled:
            return False
        
        now = time.time()
        if now - self.last_learning_time < LEARNING_INTERVAL:
            return False
        
        if len(self.trade_history) < 10:
            return False
        
        self.last_learning_time = now
        return True

    def get_pattern_insights(self) -> Dict:
        if len(self.trade_history) < 5:
            return {}
        
        trades = list(self.trade_history)
        
        insights = {
            'best_hour': None,
            'best_direction': None,
            'avg_hold_winners': 0,
            'avg_hold_losers': 0,
            'optimal_hold_range': (0, 0),
            'confidence': min(1.0, len(trades) / 30)
        }
        
        dir_wins = {'buy': 0, 'sell': 0}
        dir_total = {'buy': 0, 'sell': 0}
        for t in trades:
            dir_total[t.direction] = dir_total.get(t.direction, 0) + 1
            if t.won:
                dir_wins[t.direction] = dir_wins.get(t.direction, 0) + 1
        
        best_dir = None
        best_rate = 0
        for d, total in dir_total.items():
            if total >= 3:
                rate = dir_wins.get(d, 0) / total
                if rate > best_rate:
                    best_rate = rate
                    best_dir = d
        insights['best_direction'] = best_dir
        
        winners = [t for t in trades if t.won]
        losers = [t for t in trades if not t.won]
        if winners:
            insights['avg_hold_winners'] = sum(t.held_seconds for t in winners) / len(winners)
        if losers:
            insights['avg_hold_losers'] = sum(t.held_seconds for t in losers) / len(losers)
        
        if winners and len(winners) >= 3:
            hold_times = sorted([t.held_seconds for t in winners])
            insights['optimal_hold_range'] = (hold_times[len(hold_times)//4], hold_times[len(hold_times)*3//4])
        
        return insights

    def log_performance(self, balance: float, equity: float, daily_pnl: float,
                      total_trades: int, win_rate: float, avg_pnl: float):
        with open(CSV_PERFORMANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                f"{balance:.2f}",
                f"{equity:.2f}",
                f"{daily_pnl:.2f}",
                total_trades,
                f"{win_rate:.2%}",
                f"{avg_pnl:.3f}",
                self.current_mode
            ])

    def get_summary(self) -> str:
        stats = self.get_learning_stats()
        insights = self.get_pattern_insights()
        
        summary = f"""
=== LEARNING ENGINE SUMMARY ============================
  Mode: {stats.current_mode}
  Trades: {stats.total_trades}
  Win Rate: {stats.win_rate:.1%} ({len([t for t in self.trade_history if t.won])}/{stats.total_trades})
  Profit Factor: {stats.profit_factor:.2f}
  Avg Win: ${stats.avg_win:.2f}  |  Avg Loss: ${stats.avg_loss:.2f}
  Avg Hold: {stats.avg_hold_time:.0f}s
  Confidence: {stats.confidence:.0%}
=== PATTERN INSIGHTS ===================================
  Best Direction: {insights.get('best_direction', 'N/A')}
  Avg Hold (W): {insights.get('avg_hold_winners', 0):.0f}s | (L): {insights.get('avg_hold_losers', 0):.0f}s
======================================================
"""
        return summary


def test_learning():
    engine = OnlineLearningEngine()
    
    print("Testing Online Learning Engine...")
    print(f"Initial mode: {engine.current_mode}")
    
    engine.record_trade('buy', 77000, 77020, 2.0, 60, 1, 1, 2, 'trending')
    engine.record_trade('buy', 77100, 77090, -1.0, 90, -1, 0, -1, 'ranging')
    engine.record_trade('sell', 77200, 77180, 2.0, 45, -1, -1, -2, 'trending')
    engine.record_trade('buy', 77300, 77320, 2.0, 55, 1, 1, 1, 'trending')
    engine.record_trade('buy', 77400, 77420, 2.0, 70, 1, 1, 2, 'trending')
    
    print(engine.get_summary())
    
    adjusted = engine.get_adjusted_params({
        'cooldown': 180, 'take_profit': 20, 'stop_loss': 5,
        'max_hold': 120, 'momentum_short_threshold': 0.03, 'momentum_med_threshold': 0.05
    })
    print(f"\nAdjusted params for {engine.current_mode} mode:")
    for k, v in adjusted.items():
        print(f"  {k}: {v}")
    
    print("\nMode after 5 trades:", engine.current_mode)
    
    for i in range(10):
        pnl = 2.0 if random.random() > 0.3 else -1.0
        engine.record_trade('buy', 77000+i*10, 77000+i*10+pnl, pnl, 60)
    
    print(f"Mode after 15 trades: {engine.current_mode}")
    print(engine.get_summary())


if __name__ == "__main__":
    test_learning()