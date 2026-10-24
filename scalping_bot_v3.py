"""
BTC Scalping Bot - MT5 LIVE TRADING VERSION
Connects to MetaTrader 5 on same Windows PC
Uses real account balance, opens real 0.01 lot trades
Uses MT5 prices for momentum, dual confirmation entries
"""
import os, sys, time, random, logging, requests, datetime, json, numpy as np, threading
from typing import List, Optional
import MetaTrader5 as mt5

try:
    from model_versioning import ModelVersioningSystem, MAX_FITNESS_CAP
    VERSIONING_AVAILABLE = True
except ImportError:
    VERSIONING_AVAILABLE = False
    MAX_FITNESS_CAP = 3000.0  # Fallback

try:
    from robust_ga_framework import RobustGA, fetch_historical_data as ga_fetch_hist
    ROBUST_GA_AVAILABLE = True
except ImportError:
    ROBUST_GA_AVAILABLE = False

try:
    from trading_gui import TradingGUI
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    # trading_gui.py not available

try:
    from online_learning import OnlineLearningEngine
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False

# ========== CONFIGURATION ==========
LIVE_DATA_URL = "https://obilo.io/casdata/BTCUSDlive.txt"
HIST_DATA_URL = "https://obilo.io/casdata/BTCUSD.txt"

# MT5 Settings - CHANGE THESE TO MATCH YOUR BROKER
MT5_SYMBOL = "BTCUSD"        # Symbol name in MT5 (check exact name in your Market Watch)
MT5_LOT_SIZE = 0.01           # Trade size
MT5_MAGIC_NUMBER = 123456     # Unique ID for this bot's trades
MT5_DEVIATION = 20            # Slippage tolerance in points

# Trading parameters - CORRECTED for actual BTCUSD 0.01 lot economics
# Verified from live logs: $1 price move = $0.01 profit on 0.01 lot
PIP_VALUE = 0.01              # $ per 1.0 price movement for 0.01 lots
SPREAD = 2.0                  # Expected spread for backtest only
# From live logs: TP of 23 points yielded net ~$0.06 => commission ~$0.17
COMMISSION = 0.17             # Per-trade commission (round trip)
MIN_PROFIT = 0.35              # Lowered: allow earlier momentum exits for small scalps
TAKE_PROFIT = 120.0            # Reduced to match realistic scalp size for BTCUSD (~$1.03 net after commission)
STOP_LOSS = 120.0              # Matched to TP for better risk/reward symmetry
COOLDOWN_MIN = 120             # Minimum seconds between trades
MAX_DAILY_LOSS = -5.0          # Tightened: ~3.7% of $134 account; preserves capital
MAX_CONSECUTIVE_LOSSES = 3     # Tighter: stop after 3 losses to prevent digging deeper
MIN_ACCOUNT_BALANCE = 50.0     # Absolute floor - halt trading if balance drops this low
MAX_SPREAD = 100.0             # Skip entries if broker spread exceeds $100 (digits=0 BTCUSD)
LEGACY_FITNESS_THRESHOLD = 150.0  # Old simple-GA fitness scale is incomparable with RobustGA

# GA settings
POP_SIZE = 50
GENERATIONS = 10
MUTATION_RATE = 0.3
ELITE = 5
# Increase evolution interval and data requirements:
EVOLVE_INTERVAL = 1800  # 30 minutes instead of 5
MIN_CANDLES_FOR_EVOLUTION = 1000  # Require more data

# More conservative but reasonable for live trading (scaled to real BTCUSD pip value)
INITIAL_TP = 150.0  # 150 price units for decent profit (~$1.33 net after fees)
INITIAL_SL = 150.0  # Match MIN_SL_DISTANCE for BTCUSD volatility

# After opening a position, track profit and protect capital
# All values in DOLLARS (account currency) for 0.01 lot BTCUSD
# CRITICAL FIX: breakeven must cover COMMISSION (~$0.17 round trip) plus swap plus spread.
# Floating PnL of $0.30 at SL=entry means actual net = $0.30 - $0.17(commission) = $0.13.
# Raised to $0.60 so even after commission+swap, the trade locks in ~$0.40 net profit.
BREAKEVEN_PROFIT = 0.60   # Move SL to entry after $0.60 floating profit (covers ~$0.17 commission + ~$0.03 swap)
# Trailing stop parameters are now loaded from GA (trail_activation_profit, trail_distance)
# These are defaults if GA params are not available
DEFAULT_TRAIL_ACTIVATION = 0.50   # Activate software trailing stop at $0.50 profit
DEFAULT_TRAIL_DISTANCE = 0.30       # Trail by $0.30 (dollars for 0.01 lot)

USE_ROBUST_GA = True  # Use robust GA framework for re-evolution

LOG_FILE = "trading_bot_mt5.log"
MODEL_DIR = "models"
PARAMS_FILE = os.path.join(MODEL_DIR, "best_params.json")
ROBUST_PARAMS_FILE = os.path.join(MODEL_DIR, "robust_params.json")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def ts():
    return datetime.datetime.now().strftime("%A %I:%M:%S %p")

# ========== DATA FETCHING (for indicators and backtest) ==========
def fetch_live():
    """Fetch indicator data from obilo.io"""
    try:
        r = requests.get(LIVE_DATA_URL, timeout=10)
        parts = r.text.strip().split(',')
        if len(parts) < 11:
            return None
        return [float(x) for x in parts]
    except:
        return None

def fetch_hist():
    """Fetch historical candles from obilo.io for backtesting"""
    try:
        r = requests.get(HIST_DATA_URL, timeout=15)
        rows = []
        for line in r.text.strip().split('\n'):
            if not line.strip():
                continue
            p = line.strip().split(',')
            if len(p) >= 10:
                try:
                    rows.append([float(x) for x in p])
                except ValueError:
                    continue
        rows.sort(key=lambda x: x[0])
        return rows
    except:
        return []

# Add to the fetch functions in scalping_bot_v2.py:

def fetch_live_with_retry(max_retries=3):
    """Fetch live data with retry and backup sources"""
    urls = [
        "https://obilo.io/casdata/BTCUSDlive.txt",
        # Add backup URLs if available
    ]
    
    for url in urls:
        for attempt in range(max_retries):
            try:
                r = requests.get(url, timeout=10)
                parts = r.text.strip().split(',')
                if len(parts) >= 11:
                    return [float(x) for x in parts]
            except:
                time.sleep(2)
    return None
# ========== MT5 CONNECTION ==========
class MT5Connection:
    """Handles all MetaTrader 5 interactions"""
    
    def __init__(self):
        self.connected = False
        self.account_info = None
        self.starting_balance = 0.0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self._last_warn_time = 0
        self._cooloff_until = 0
        # Risk management: maximum fraction of free margin to risk per trade
        self.risk_per_trade = 0.02  # 2% of free margin per trade
        # Maximum lot size cap (absolute ceiling for safety)
        self.max_lot_size = 1.0
        # Minimum lot size floor (broker minimum for BTCUSD)
        self.min_lot_size = 0.01
        
    def connect(self) -> bool:
        """Initialize MT5 connection with retry logic"""
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting MT5 connection (attempt {attempt + 1}/{max_retries})...")
                
                # Check if already connected
                if self.connected:
                    logger.info("Already connected to MT5")
                    return True
                
                # Initialize MT5 connection
                if not mt5.initialize():
                    error = mt5.last_error()
                    logger.error(f"MT5 initialize failed (attempt {attempt + 1}): {error}")
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return False
                
                logger.info(f"MT5 Version: {mt5.version()}")
                
                # Enable symbol BEFORE getting info (critical for accurate data)
                if not mt5.symbol_select(MT5_SYMBOL, True):
                    logger.warning(f"Could not select symbol {MT5_SYMBOL}")
                
                # Give MT5 a moment to populate symbol data
                time.sleep(0.5)
                
                # Check supported filling modes and symbol properties
                symbol_info = mt5.symbol_info(MT5_SYMBOL)
                if symbol_info:
                    logger.info(f"Symbol {MT5_SYMBOL} filling modes: {symbol_info.filling_mode}")
                    # Access symbol properties safely
                    try:
                        logger.info(f"Symbol {MT5_SYMBOL} trade_stops_level: {symbol_info.trade_stops_level}")
                    except:
                        logger.info(f"Symbol {MT5_SYMBOL} trade_stops_level: not available")
                    try:
                        logger.info(f"Symbol {MT5_SYMBOL} spread: {symbol_info.spread}")
                    except:
                        pass
                    logger.info(f"Symbol {MT5_SYMBOL} digits: {symbol_info.digits}")
                    logger.info(f"Symbol {MT5_SYMBOL} point: {symbol_info.point}")
                    
                    # CRITICAL WARNINGS for abnormal BTCUSD symbol config
                    if symbol_info.digits == 0:
                        logger.warning(f"Symbol {MT5_SYMBOL} has digits=0 (integer pricing). "
                                       f"SL/TP will be rounded to whole numbers.")
                    if symbol_info.spread == 0:
                        logger.warning(f"Symbol {MT5_SYMBOL} reports spread=0. "
                                       f"This may indicate zero-spread account or stale data.")
                    if symbol_info.point == 1.0:
                        logger.info(f"Symbol {MT5_SYMBOL} point=1.0 (minimum price step = $1.0)")
                else:
                    logger.warning(f"Could not get symbol info for {MT5_SYMBOL}")
                
                # Get account info
                self.account_info = mt5.account_info()
                if self.account_info is None:
                    error = mt5.last_error()
                    logger.error(f"Failed to get account info (attempt {attempt + 1}): {error}")
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        mt5.shutdown()
                        time.sleep(retry_delay)
                        continue
                    else:
                        return False
                
                self.starting_balance = self.account_info.balance
                
                logger.info(f"Connected to MT5 account #{self.account_info.login}")
                logger.info(f"Balance: ${self.account_info.balance:.2f}")
                logger.info(f"Equity: ${self.account_info.equity:.2f}")
                logger.info(f"Symbol: {MT5_SYMBOL}")
                
                # Enable symbol if needed
                if not mt5.symbol_select(MT5_SYMBOL, True):
                    logger.warning(f"Could not select symbol {MT5_SYMBOL}")
                
                self.connected = True
                return True
                
            except Exception as e:
                logger.error(f"Exception during MT5 connection (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    if self.connected:
                        mt5.shutdown()
                        self.connected = False
                    time.sleep(retry_delay)
                else:
                    return False
        
        return False
    
    def disconnect(self):
        """Shutdown MT5 connection"""
        if self.connected:
            mt5.shutdown()
            logger.info("MT5 disconnected")
    
    def get_balance(self) -> float:
        """Get current account balance"""
        if not self.connected:
            return 0.0
        info = mt5.account_info()
        if info:
            return info.balance
        return 0.0
    
    def get_equity(self) -> float:
        """Get current account equity"""
        if not self.connected:
            return 0.0
        info = mt5.account_info()
        if info:
            return info.equity
        return 0.0
    
    def _reconnect(self) -> bool:
        """Try to reconnect to MT5 if connection was lost"""
        logger.warning("Attempting MT5 reconnection...")
        mt5.shutdown()
        time.sleep(1)
        if mt5.initialize():
            # Re-select symbol
            mt5.symbol_select(MT5_SYMBOL, True)
            self.account_info = mt5.account_info()
            if self.account_info:
                logger.info(f"Reconnected to MT5 account #{self.account_info.login}")
                self.connected = True
                return True
        logger.error("MT5 reconnection failed")
        self.connected = False
        return False
    
    def get_tick(self):
        """Get current bid/ask from MT5 with auto-reconnect"""
        tick = mt5.symbol_info_tick(MT5_SYMBOL)
        if tick is None:
            # Try to reconnect
            error = mt5.last_error()
            if error[0] != 0:  # Non-zero error code means connection issue
                logger.warning(f"MT5 tick failed with error: {error}, attempting reconnect...")
                if self._reconnect():
                    tick = mt5.symbol_info_tick(MT5_SYMBOL)
        else:
            # Log tick feed quality (spread check)
            # CRITICAL FIX: For integer-priced symbols (digits=0, point=1.0),
            # spread is in whole dollars. Convert to dollars for comparison.
            symbol_info = mt5.symbol_info(MT5_SYMBOL)
            point = getattr(symbol_info, 'point', 0.01) if symbol_info else 0.01
            spread_dollars = (tick.ask - tick.bid) * point
            if spread_dollars <= 0.0:
                # Broker reports zero spread for BTCUSD — this is normal for some brokers.
                # Prices are clearly updating (see tick.bid values), so this is NOT bad data.
                # Do NOT return None; just log at DEBUG level once per 60s to avoid log spam.
                if not hasattr(self, '_last_zero_spread_log') or time.time() - self._last_zero_spread_log > 60:
                    self._last_zero_spread_log = time.time()
                    logger.info(f"Broker reports zero spread for BTCUSD (Bid={tick.bid:.2f} Ask={tick.ask:.2f}) (rate-limited 60s)")
            elif spread_dollars < 0.5:
                # Tight but non-zero spread is normal for liquid BTCUSD
                logger.debug(f"Tight spread: Bid={tick.bid:.2f} Ask={tick.ask:.2f} Spread=${spread_dollars:.2f}")
        return tick

    def _get_position_ticket_from_order(self, order_ticket: int) -> Optional[int]:
        """
        Convert an order ticket to a position ticket by querying MT5 positions.
        MT5's order_send returns an order ticket (result.order), but deal history
        references position_id. This helper finds the position that was opened by
        the given order ticket.
        """
        try:
            # Get all positions and find the one opened by this order
            # (positions have no 'order' field, so we match by magic + symbol + recent open)
            # Alternative: search deal history for the entry deal with this order ticket
            from_time = int(time.time() - 86400)
            to_time = int(time.time()) + 3600
            deals = mt5.history_deals_get(from_time, to_time)
            if deals:
                for deal in deals:
                    if (deal.order == order_ticket and 
                        deal.symbol == MT5_SYMBOL and 
                        deal.magic == MT5_MAGIC_NUMBER and 
                        deal.entry == mt5.DEAL_ENTRY_IN):
                        return deal.position_id
            return None
        except Exception as e:
            logger.error(f"Error converting order ticket to position: {e}")
            return None

    
    def get_open_positions(self):
        """Get all open positions for this symbol"""
        positions = mt5.positions_get(symbol=MT5_SYMBOL)
        if positions is None:
            error = mt5.last_error()
            if error[0] != 0:  # Connection error, not just "no positions"
                logger.warning(f"MT5 positions_get failed: {error}, attempting reconnect...")
                if self._reconnect():
                    positions = mt5.positions_get(symbol=MT5_SYMBOL)
            if positions is None:
                logger.debug(f"No positions returned for {MT5_SYMBOL} (error: {mt5.last_error()})")
                return []
        # Log all positions found (for debugging)
        if positions:
            for p in positions:
                logger.debug(f"Position found: ticket={p.ticket}, magic={p.magic}, "
                           f"type={'BUY' if p.type == mt5.POSITION_TYPE_BUY else 'SELL'}, "
                           f"volume={p.volume}, profit={p.profit:.2f}")
        bot_positions = [p for p in positions if p.magic == MT5_MAGIC_NUMBER]
        if positions and not bot_positions:
            logger.warning(f"Found {len(positions)} position(s) for {MT5_SYMBOL} but none match "
                          f"magic={MT5_MAGIC_NUMBER}. Position magics: {[p.magic for p in positions]}")
        return bot_positions
    
    def has_open_position(self) -> bool:
        """Check if we have an open position"""
        return len(self.get_open_positions()) > 0
    
    def get_position_details(self):
        """Get details of current position"""
        positions = self.get_open_positions()
        if not positions:
            return None
        p = positions[0]
        return {
            'ticket': p.ticket,
            'direction': 'buy' if p.type == mt5.POSITION_TYPE_BUY else 'sell',
            'entry_price': p.price_open,
            'current_price': p.price_current,
            'profit': p.profit,
            'volume': p.volume,
        }
    
    def calculate_lot_size(self) -> float:
        """
        Calculate position size based on free margin.
        
        Formula: lot_size = (free_margin * risk_per_trade) / (stop_loss_pips * pip_value_per_lot)
        
        For BTCUSD with digits=0 and point=1.0:
        - 1.0 price move = $0.01 per 0.01 lot (PIP_VALUE)
        - pip_value_per_lot = PIP_VALUE / 0.01 = $1.00 per standard lot per $1 move
        
        Caps: min_lot_size <= result <= max_lot_size
        Falls back to MT5_LOT_SIZE if account info unavailable.
        
        FIX: Reduced risk_per_trade from 0.02 (2%) to 0.01 (1%) per trade for safer
        position sizing on small accounts (~$114). Two or three full stop losses in a row
        at 2% risk would noticeably dent the account; 1% is more sustainable.
        """
        # Check if we can get margin info
        info = mt5.account_info()
        if info is None:
            logger.warning(f"Cannot get account info for lot sizing, using default {MT5_LOT_SIZE}")
            return MT5_LOT_SIZE
        
        margin_free = info.margin_free
        if margin_free <= 0:
            logger.warning(f"No free margin (${margin_free:.2f}), using minimum lot {self.min_lot_size}")
            return self.min_lot_size
        
        # Risk per trade in dollars: 1% of free margin (safer for small accounts)
        effective_risk = max(0.005, min(0.02, self.risk_per_trade))
        risk_dollars = margin_free * effective_risk
        
        # For BTCUSD at 0.01 lot, $1 move = $0.01 PnL
        # Per standard lot (1.0): $1 move = $1.00 PnL
        # We use a conservative SL assumption of 150 price units for sizing
        assumed_sl_pips = 150.0
        pip_value_per_std_lot = 1.0  # $1 per $1 move per standard lot
        
        # lot = risk_dollars / (assumed_sl_pips * pip_value_per_std_lot)
        raw_lot = risk_dollars / (assumed_sl_pips * pip_value_per_std_lot)
        
        # Round to nearest 0.01 lot step (MT5 standard)
        lot = round(raw_lot * 100) / 100.0
        
        # Apply caps
        lot = max(self.min_lot_size, min(lot, self.max_lot_size))
        
        logger.info(f"Lot sizing: balance=${info.balance:.2f} free_margin=${margin_free:.2f} "
                    f"risk_pct={effective_risk:.1%} risk_dollars=${risk_dollars:.2f} "
                    f"assumed_sl={assumed_sl_pips} raw_lot={raw_lot:.4f} final_lot={lot:.2f}")
        
        return lot
    
    def _get_filling_mode(self):
        """Detect the correct filling mode for the symbol"""
        symbol_info = mt5.symbol_info(MT5_SYMBOL)
        if symbol_info is None:
            logger.warning(f"Cannot get symbol info for {MT5_SYMBOL}, defaulting to FOK")
            return mt5.ORDER_FILLING_FOK
        
        filling_mode = symbol_info.filling_mode
        logger.debug(f"Symbol {MT5_SYMBOL} filling_mode flags: {filling_mode}")
        
        # filling_mode is a bitmask:
        # 1 = ORDER_FILLING_FOK supported
        # 2 = ORDER_FILLING_IOC supported  
        # Bit 2 (4) = ORDER_FILLING_RETURN supported (hedge accounts)
        if filling_mode & 1:
            return mt5.ORDER_FILLING_FOK
        elif filling_mode & 2:
            return mt5.ORDER_FILLING_IOC
        else:
            # Default to RETURN for hedge accounts (most crypto brokers)
            return mt5.ORDER_FILLING_RETURN
    
    def open_trade(self, direction: str, tp: float = None, sl: float = None) -> bool:
        """Open a new trade with SL/TP. Uses provided tp/sl or falls back to constants."""
        if self.has_open_position():
            logger.warning("Already have an open position")
            return False
        
        tick = self.get_tick()
        if tick is None:
            logger.error("Cannot get price from MT5")
            return False
        
        use_tp = tp if tp is not None else TAKE_PROFIT
        use_sl = sl if sl is not None else STOP_LOSS
        
        # Get symbol digits for proper price formatting
        symbol_info = mt5.symbol_info(MT5_SYMBOL)
        digits = symbol_info.digits if symbol_info else 0
        
        # ENSURE MINIMUM STOP DISTANCE for BTCUSD volatility
        # With digits=0/point=1.0, broker stop level is in whole dollars
        broker_min_stop = getattr(symbol_info, 'trade_stops_level', 0) if symbol_info else 0
        broker_point = getattr(symbol_info, 'point', 1.0) if symbol_info else 1.0
        MIN_SL_DISTANCE = 120.0  # Hard floor: $120 minimum stop distance for BTCUSD scalping
        if broker_min_stop > 0:
            MIN_SL_DISTANCE = max(MIN_SL_DISTANCE, broker_min_stop * broker_point + 20.0)
        use_sl = max(use_sl, MIN_SL_DISTANCE)
        logger.info(f"Stop distance enforced: SL={use_sl:.2f} (broker_min={broker_min_stop}, point={broker_point})")
        
        if direction == 'buy':
            price = round(tick.ask, digits)
            order_type = mt5.ORDER_TYPE_BUY
            sl_price = round(price - use_sl, digits)
            tp_price = round(price + use_tp, digits)
        else:
            price = round(tick.bid, digits)
            order_type = mt5.ORDER_TYPE_SELL
            sl_price = round(price + use_sl, digits)
            tp_price = round(price - use_tp, digits)
        
        # Calculate dynamic lot size from free margin
        lot_size = self.calculate_lot_size()
        
        filling_type = self._get_filling_mode()
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": MT5_SYMBOL,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": MT5_DEVIATION,
            "magic": MT5_MAGIC_NUMBER,
            "comment": "ScalpingBot",
            "type_filling": filling_type,
        }
        
        logger.info(f"Sending {direction.upper()} order: price={price:.2f} SL={sl_price:.2f} TP={tp_price:.2f} "
                    f"filling={filling_type} volume={lot_size:.2f}")
        
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Order send returned None! MT5 error: {mt5.last_error()}")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment} (code: {result.retcode})")
            logger.error(f"Request was: {request}")
            # Handle network errors with retry
            if result.retcode in [10031, 10019, 10021]:
                logger.warning(f"Network error (code: {result.retcode}), retrying in 2 seconds...")
                time.sleep(2)
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f">>> Retry succeeded! OPENED {direction.upper()} {lot_size:.2f} lot @ {price:.2f} SL={sl_price:.2f} TP={tp_price:.2f} [Ticket: {result.order}]")
                    return True
                else:
                    logger.error(f"Retry failed: {result.comment if result else 'None'}")
                    return False
            # Try with alternative filling mode if first attempt failed
            elif result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
                alt_filling = mt5.ORDER_FILLING_IOC if filling_type == mt5.ORDER_FILLING_FOK else mt5.ORDER_FILLING_FOK
                logger.warning(f"Retrying with alternative filling mode: {alt_filling}")
                request["type_filling"] = alt_filling
                result = mt5.order_send(request)
                if result is None:
                    logger.error(f"Retry order send returned None! MT5 error: {mt5.last_error()}")
                    return False
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f">>> Alternative filling succeeded! OPENED {direction.upper()} {lot_size:.2f} lot @ {price:.2f} SL={sl_price:.2f} TP={tp_price:.2f} [Ticket: {result.order}]")
                    return True
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Retry also failed: {result.comment} (code: {result.retcode})")
                    # Try RETURN mode as last resort
                    request["type_filling"] = mt5.ORDER_FILLING_RETURN
                    logger.warning(f"Last resort retry with RETURN filling mode")
                    result = mt5.order_send(request)
                    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                        logger.error(f"All filling modes failed. Last error: {mt5.last_error()}")
                        return False
                    else:
                        logger.info(f">>> RETURN filling succeeded! OPENED {direction.upper()} {lot_size:.2f} lot @ {price:.2f} SL={sl_price:.2f} TP={tp_price:.2f} [Ticket: {result.order}]")
                        return True
            elif result.retcode == 10016:  # Invalid stops - retry with wider SL
                logger.warning(f"Invalid stops detected (entry={price}, SL={sl_price}), retrying with wider SL...")
                sl_buffer = use_sl * 0.5
                if direction == 'buy':
                    new_sl = round(price - use_sl - sl_buffer, digits)
                    new_tp = round(price + use_tp + sl_buffer * 0.5, digits)
                else:
                    new_sl = round(price + use_sl + sl_buffer, digits)
                    new_tp = round(price - use_tp - sl_buffer * 0.5, digits)
                request["sl"] = new_sl
                request["tp"] = new_tp
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f">>> Retry succeeded! OPENED {direction.upper()} {lot_size:.2f} lot @ {price:.2f} SL={new_sl:.2f} TP={new_tp:.2f} [Ticket: {result.order}]")
                    return True
                else:
                    err = result.comment if result else 'None'
                    logger.error(f"Wider stops retry failed: {err}")
                    return False
            else:
                # CRITICAL FIX: Unhandled errors (e.g. 10016 Invalid Stops) must NOT fall through to success
                logger.error(f"Unhandled order error {result.retcode}, aborting trade")
                return False
        
        logger.info(f">>> OPENED {direction.upper()} {lot_size:.2f} lot @ {price:.2f} SL={sl_price:.2f} TP={tp_price:.2f} [Ticket: {result.order}]")
        
        # CRITICAL FIX: Verify position actually exists in MT5 after order reports success.
        # Some brokers accept orders but immediately close them (invalid stops, margin issues).
        # We retry position detection for up to 2 seconds to catch MT5 propagation delays.
        time.sleep(0.3)
        for verify_attempt in range(6):
            if self.has_open_position():
                logger.debug(f">>> Position verified in MT5 after {verify_attempt + 1} attempt(s)")
                return True
            if verify_attempt < 5:
                logger.debug(f"Position not yet visible in MT5 (attempt {verify_attempt + 1}/6), retrying...")
                time.sleep(0.3)
        
        logger.error(">>> CRITICAL: Order sent successfully but position NOT found in MT5 after 2 seconds!")
        logger.error("Possible causes: broker rejected SL/TP, margin stop-out, or magic number mismatch.")
        return False
    
    def _get_deal_details(self, position_ticket: int) -> dict:
        """
        Query MT5 deal history to get actual profit, commission, and swap for a closed trade.
        
        MT5 stores deals when positions are closed. We search for the closing deal
        associated with this position ticket and extract the real PnL including costs.
        
        Returns dict with 'profit', 'commission', 'swap', and 'net' keys.
        Returns zeros if deal not found (fallback to position.profit).
        """
        try:
            # Get all deals for this position within the last 24 hours
            from_time = int(time.time() - 86400)  # 24 hours ago
            to_time = int(time.time()) + 3600  # 1 hour in future (clock skew)
            deals = mt5.history_deals_get(from_time, to_time)
            
            if deals is None or len(deals) == 0:
                logger.warning(f"No deals found in history for position {position_ticket}")
                return {'profit': 0.0, 'commission': 0.0, 'swap': 0.0, 'net': 0.0}
            
            # Find the closing deal for this position (entry=position ticket)
            closing_deal = None
            for deal in deals:
                if deal.position_id == position_ticket and deal.entry == mt5.DEAL_ENTRY_OUT:
                    closing_deal = deal
                    break
            
            if closing_deal is None:
                # Fallback: search for the most recent OUT deal for this symbol with our magic
                for deal in reversed(deals):
                    if (deal.symbol == MT5_SYMBOL and deal.magic == MT5_MAGIC_NUMBER 
                        and deal.entry == mt5.DEAL_ENTRY_OUT):
                        closing_deal = deal
                        break
            
            if closing_deal:
                profit = getattr(closing_deal, 'profit', 0.0)
                commission = getattr(closing_deal, 'commission', 0.0)
                swap = getattr(closing_deal, 'swap', 0.0)
                net = profit + commission + swap
                return {'profit': profit, 'commission': commission, 'swap': swap, 'net': net}
            else:
                logger.warning(f"No closing deal found for position {position_ticket} in history")
                return {'profit': 0.0, 'commission': 0.0, 'swap': 0.0, 'net': 0.0}
                
        except Exception as e:
            logger.error(f"Error querying deal history: {e}")
            return {'profit': 0.0, 'commission': 0.0, 'swap': 0.0, 'net': 0.0}
    
    def close_position(self):
        """Close the current open position. Returns profit amount on success, None on failure"""
        positions = self.get_open_positions()
        if not positions:
            logger.debug("No position to close (already closed by MT5)")
            return None
        
        position = positions[0]
        position_ticket = position.ticket
        tick = self.get_tick()
        if tick is None:
            return None
        
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        
        filling_type = self._get_filling_mode()
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": MT5_SYMBOL,
            "volume": position.volume,
            "type": order_type,
            "position": position_ticket,
            "price": price,
            "deviation": MT5_DEVIATION,
            "magic": MT5_MAGIC_NUMBER,
            "comment": "Close",
            "type_filling": filling_type,
        }
        
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Close order send returned None! MT5 error: {mt5.last_error()}")
            return None
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            # CRITICAL FIX: Wait briefly then query deal history for accurate PnL.
            # position.profit is FLOATING PnL and does NOT include commission/swap correctly.
            # Only history_deals_get() gives the true closed trade result.
            time.sleep(0.3)  # Allow MT5 to finalize the deal record
            deal_info = self._get_deal_details(position_ticket)
            
            # If deal history has valid data, use it. Otherwise fall back to position.profit.
            if deal_info['net'] != 0.0:
                actual_pnl = deal_info['net']
                logger.info(f"<<< CLOSED | DealProfit=${deal_info['profit']:.2f} "
                           f"Commission=${deal_info['commission']:.2f} "
                           f"Swap=${deal_info['swap']:.2f} "
                           f"NET=${actual_pnl:.2f} "
                           f"[Floating PnL was ${position.profit:.2f}]")
                return actual_pnl
            else:
                # Fallback: use floating profit, but log the discrepancy
                logger.warning(f"<<< CLOSED | No deal record found — using floating PnL=${position.profit:.2f} "
                              f"(balance-derived will differ by commission+swap)")
                return position.profit
        
        # Position already closed (code 10036) - not an error, just a race condition
        if result.retcode == 10036:
            logger.info(f"Close: Position already closed by MT5 (code: {result.retcode})")
            return None
        
        logger.error(f"Close failed: {result.comment} (code: {result.retcode})")
        
        # Try alternative filling modes
        if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
            for alt_filling in [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]:
                if alt_filling == filling_type:
                    continue
                logger.warning(f"Retrying close with filling mode: {alt_filling}")
                request["type_filling"] = alt_filling
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"<<< CLOSED (alt fill) | PnL: ${position.profit:.2f}")
                    return position.profit
            logger.error(f"All filling modes failed for close")
            return None
        
        if result.retcode == 10029:
            logger.warning(f"Close failed: position too close to market (code 10029). Retrying with 2x deviation...")
            request["deviation"] = MT5_DEVIATION * 2
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"<<< CLOSED (retry) | PnL: ${position.profit:.2f}")
                return position.profit
            
            # If retry failed, broker may have closed position via SL/TP in the meantime.
            # Check if position still exists; if not, return last known profit.
            time.sleep(0.2)
            if not self.has_open_position():
                logger.info(f"<<< CLOSED by broker (SL/TP) | PnL (est): ${position.profit:.2f}")
                return position.profit
            logger.error(f"Retry close also failed: {result.comment if result else 'None'}")
            return None
        
        return None
    
    def modify_sl(self, ticket: int, new_sl: float) -> bool:
        """Modify stop loss on an open position via TRADE_ACTION_SLTP"""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"Cannot modify SL: position {ticket} not found")
            return False
        
        position = positions[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": MT5_SYMBOL,
            "position": ticket,
            "sl": new_sl,
            "tp": position.tp,
        }
        
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Modify SL failed: {mt5.last_error()}")
            return False
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Modified SL to {new_sl:.2f} for ticket {ticket}")
            return True
        else:
            logger.warning(f"Modify SL failed: {result.comment} (code: {result.retcode})")
            return False
    
    def update_daily_pnl(self):
        """Track daily P&L"""
        current_balance = self.get_balance()
        self.daily_pnl = current_balance - self.starting_balance
    
    def should_stop_trading(self) -> bool:
        """Check if we should stop trading due to losses. Includes auto-recovery cool-off."""
        self.update_daily_pnl()
        now = time.time()

        if self.daily_pnl <= MAX_DAILY_LOSS:
            if now - self._last_warn_time > 60:
                self._last_warn_time = now
                logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return True

        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            # First time hitting the limit: start a cool-off period
            if self._cooloff_until == 0:
                cooloff = 900  # 15 minutes base cool-off
                self._cooloff_until = now + cooloff
                logger.warning(
                    f"Max consecutive losses reached: {self.consecutive_losses}. "
                    f"Trading paused for {cooloff // 60} minutes (cool-off)."
                )
                logger.warning(f"Daily PnL: ${self.daily_pnl:.2f}")

            # Cool-off expired: reduce streak and allow trading again
            if now >= self._cooloff_until:
                old = self.consecutive_losses
                self.consecutive_losses = max(0, self.consecutive_losses - 2)
                self._cooloff_until = 0
                logger.info(
                    f"Cool-off complete. Consecutive losses {old} -> {self.consecutive_losses}. "
                    f"Resuming trading with elevated caution."
                )
                return False

            # Still in cool-off: rate-limit warnings
            if now - self._last_warn_time > 60:
                self._last_warn_time = now
                remaining = int(self._cooloff_until - now)
                logger.warning(
                    f"Max consecutive losses: {self.consecutive_losses}. "
                    f"Cool-off remaining: {remaining}s"
                )
            return True

        # Below threshold: reset cool-off tracker
        self._cooloff_until = 0
        return False

# ========== STRATEGY PARAMETERS (for GA optimization) ==========
class StrategyParams:
    def __init__(self):
        self.take_profit = 10.0   # Price units for TP (gives ~$1.00 gross)
        self.stop_loss = 80.0    # Price units for SL (BTCUSD needs wide stops)
        self.max_hold = 600       # Max hold seconds
        self.cooldown = 60       # Seconds between trades
        self.fitness = 0.0
        self.momentum_short_period = 4
        self.momentum_med_period = 9
        self.momentum_short_threshold = 0.001  # 0.1% for sensitive entry
        self.momentum_med_threshold = 0.002    # 0.2% for medium momentum
        self.use_api_filter = True
        self.min_signal_age = 0
        self.metrics = None  # Will be set by GA framework
        # ---- New: lookback window sizes (in ticks) ----
        self.lookback_short = 8    # Default short lookback (tick count for fast momentum)
        self.lookback_med = 24     # Default medium lookback (3x short for smoother signal)
        self.max_momentum_age = 30 # Seconds: momentum must have flipped within this window to allow entry
        # ---- Trailing stop parameters (in dollars for 0.01 lot BTCUSD) ----
        self.trail_activation_profit = 0.50  # Activate trailing stop at this profit
        self.trail_distance = 0.30           # Trail this distance behind peak profit
    
    def to_dict(self):
        return {
            'take_profit': self.take_profit,
            'stop_loss': self.stop_loss,
            'max_hold': self.max_hold,
            'cooldown': self.cooldown,
            'fitness': self.fitness,
            'momentum_short_period': self.momentum_short_period,
            'momentum_med_period': self.momentum_med_period,
            'momentum_short_threshold': self.momentum_short_threshold,
            'momentum_med_threshold': self.momentum_med_threshold,
            'lookback_short': self.lookback_short,
            'lookback_med': self.lookback_med,
            'max_momentum_age': self.max_momentum_age,
            'trail_activation_profit': self.trail_activation_profit,
            'trail_distance': self.trail_distance,
        }
    
    def mutate(self):
        # CORRECTED: BTCUSD at 0.01 lot needs 100s of points to overcome fees/volatility
        self.take_profit = max(100.0, min(500.0, self.take_profit * random.uniform(0.7, 1.3)))
        self.stop_loss = max(80.0, min(400.0, self.stop_loss * random.uniform(0.7, 1.3)))
        self.max_hold = max(120, min(900, self.max_hold + random.randint(-120, 120)))
        self.cooldown = max(60, min(600, self.cooldown + random.randint(-60, 60)))

    def crossover(self, other):
        child = StrategyParams()
        child.take_profit = random.choice([self.take_profit, other.take_profit])
        child.stop_loss = random.choice([self.stop_loss, other.stop_loss])
        child.max_hold = random.choice([self.max_hold, other.max_hold])
        child.cooldown = random.choice([self.cooldown, other.cooldown])
        return child

# ========== BACKTEST (uses historical data for GA) ==========
def backtest(params: StrategyParams, candles: List[List[float]]) -> float:
    balance = 200.0
    trade = None
    last_trade_time = -9999
    price_history = []
    
    for i in range(20, len(candles) - 1):
        c = candles[i]
        next_c = candles[i + 1]
        mid = c[4]
        
        price_history.append(mid)
        if len(price_history) > 30:
            price_history.pop(0)
        
        mom_short = 0
        if len(price_history) >= 4:
            change = (price_history[-1] - price_history[-4]) / max(price_history[-4], 1) * 100
            if change > params.momentum_short_threshold * 100:
                mom_short = 1
            elif change < -params.momentum_short_threshold * 100:
                mom_short = -1

        mom_med = 0
        if len(price_history) >= 9:
            change = (price_history[-1] - price_history[-9]) / max(price_history[-9], 1) * 100
            if change > params.momentum_med_threshold * 100:
                mom_med = 1
            elif change < -params.momentum_med_threshold * 100:
                mom_med = -1
        
        if trade is None:
            if c[0] - last_trade_time < params.cooldown:
                continue
            
            direction = None
            if mom_short == 1 and mom_med == 1:
                direction = 'buy'
            elif mom_short == -1 and mom_med == -1:
                direction = 'sell'
            
            if direction is None:
                continue
            
            trade = {
                'dir': direction, 
                'entry': mid, 
                'time': c[0],
                'trailing_active': False,
                'trail_floor': 0.0
            }
            last_trade_time = c[0]
        else:
            entry = trade['entry']
            direction = trade['dir']
            high, low = next_c[2], next_c[3]
            tp = params.take_profit
            sl = params.stop_loss
            held = next_c[0] - trade['time']
            
            exit_price = None
            pnl = 0.0
            
            # Calculate current profit in dollars
            if direction == 'buy':
                profit = (mid - entry) * PIP_VALUE
            else:
                profit = (entry - mid) * PIP_VALUE
            
            # TRAILING STOP LOGIC
            trailing_active = trade.get('trailing_active', False)
            trail_floor = trade.get('trail_floor', 0.0)
            
            trail_activation = getattr(params, 'trail_activation_profit', DEFAULT_TRAIL_ACTIVATION)
            trail_dist = getattr(params, 'trail_distance', DEFAULT_TRAIL_DISTANCE)
            
            if profit >= trail_activation:
                trailing_active = True
                if trail_floor == 0.0:
                    trail_floor = profit - trail_dist
                else:
                    candidate_floor = profit - trail_dist
                    if candidate_floor > trail_floor:
                        trail_floor = candidate_floor
            
            if trailing_active and profit <= trail_floor:
                if direction == 'buy':
                    exit_price = entry + trail_floor / PIP_VALUE
                else:
                    exit_price = entry - trail_floor / PIP_VALUE
            
            if exit_price is None:
                if direction == 'buy':
                    if high >= entry + tp:
                        exit_price = entry + tp
                    elif low <= entry - sl:
                        exit_price = entry - sl
                else:
                    if low <= entry - tp:
                        exit_price = entry - tp
                    elif high >= entry + sl:
                        exit_price = entry + sl
            
            if exit_price is None and held > 30:
                if direction == 'buy' and mom_short == -1:
                    exit_price = next_c[4]
                elif direction == 'sell' and mom_short == 1:
                    exit_price = next_c[4]
            
            if exit_price is not None:
                if direction == 'buy':
                    profit = (exit_price - entry - SPREAD) * PIP_VALUE - COMMISSION
                else:
                    profit = (entry - exit_price - SPREAD) * PIP_VALUE - COMMISSION
                balance += profit
                trade = None
            elif held > params.max_hold:
                exit_close = next_c[4]
                if direction == 'buy':
                    profit = (exit_close - entry - SPREAD) * PIP_VALUE - COMMISSION
                else:
                    profit = (entry - exit_close - SPREAD) * PIP_VALUE - COMMISSION
                balance += profit
                trade = None
    
    if trade:
        last_close = candles[-1][4]
        entry = trade['entry']
        direction = trade['dir']
        if direction == 'buy':
            profit = (last_close - entry - SPREAD) * PIP_VALUE - COMMISSION
        else:
            profit = (entry - last_close - SPREAD) * PIP_VALUE - COMMISSION
        balance += profit
    
    return balance - 200.0

# ========== GENETIC ALGORITHM ==========
def evolve_params(candles):
    pop = [StrategyParams() for _ in range(POP_SIZE)]
    for p in pop:
        # CORRECTED: Ranges now match actual BTCUSD economics (100s of points, not 10s)
        p.take_profit = random.uniform(100.0, 400.0)
        p.stop_loss = random.uniform(100.0, 300.0)
        p.max_hold = random.randint(120, 900)
        p.cooldown = random.randint(60, 600)
    
    for gen in range(GENERATIONS):
        for p in pop:
            p.fitness = backtest(p, candles)
        pop.sort(key=lambda x: x.fitness, reverse=True)
        best = pop[0]
        logger.info(f"  Gen {gen+1}: Best=${best.fitness:.2f} TP={best.take_profit:.1f} "
                    f"SL={best.stop_loss:.1f} Hold={best.max_hold}s Cool={best.cooldown}s")
        
        new_pop = pop[:ELITE]
        while len(new_pop) < POP_SIZE:
            p1, p2 = random.sample(pop[:15], 2)
            child = p1.crossover(p2)
            if random.random() < MUTATION_RATE:
                child.mutate()
            new_pop.append(child)
        pop = new_pop
    
    for p in pop:
        p.fitness = backtest(p, candles)
    pop.sort(key=lambda x: x.fitness, reverse=True)
    return pop[0]

def save_params(p):
    os.makedirs(MODEL_DIR, exist_ok=True)
    d = {'take_profit': p.take_profit, 'stop_loss': p.stop_loss,
         'max_hold': p.max_hold, 'cooldown': p.cooldown, 'fitness': p.fitness,
         'momentum_short_threshold': getattr(p, 'momentum_short_threshold', 0.003),
         'momentum_med_threshold': getattr(p, 'momentum_med_threshold', 0.005),
         'momentum_short_period': getattr(p, 'momentum_short_period', 4),
         'momentum_med_period': getattr(p, 'momentum_med_period', 9),
         'lookback_short': getattr(p, 'lookback_short', 5),
         'lookback_med': getattr(p, 'lookback_med', 12),
         'max_momentum_age': getattr(p, 'max_momentum_age', 10),
         'trail_activation_profit': getattr(p, 'trail_activation_profit', DEFAULT_TRAIL_ACTIVATION),
         'trail_distance': getattr(p, 'trail_distance', DEFAULT_TRAIL_DISTANCE)}
    with open(PARAMS_FILE, 'w') as f:
        json.dump(d, f, indent=2)
    logger.info(f"Saved: TP={d['take_profit']:.1f} SL={d['stop_loss']:.1f} "
                f"Hold={d['max_hold']}s Cool={d['cooldown']}s Fit=${d['fitness']:.2f} "
                f"TrailAct={d['trail_activation_profit']:.2f} TrailDist={d['trail_distance']:.2f}")

def load_params():
    if VERSIONING_AVAILABLE:
        vs = ModelVersioningSystem()
        loaded = vs.load_best()
        if loaded:
            p = StrategyParams()
            p.take_profit = loaded.get('take_profit', 15.0)
            p.stop_loss = loaded.get('stop_loss', 150.0)
            p.max_hold = loaded.get('max_hold', 600)
            p.cooldown = loaded.get('cooldown', 180)
            p.fitness = loaded.get('fitness', 0)
            if p.fitness > LEGACY_FITNESS_THRESHOLD:
                logger.warning(f"Loaded legacy/simple-GA model fitness={p.fitness:.1f}. Resetting to 0.0 for RobustGA comparability.")
                p.fitness = 0.0
            stats = vs.get_stats()
            logger.info(f"Loaded best model: {stats['current_best']} "
                       f"(Fitness={stats['best_fitness']:.1f}, "
                       f"{stats['total_improvements']} improvements)")
            return p
        else:
            logger.info("No models in versioning system")
    
    if os.path.exists(ROBUST_PARAMS_FILE):
        with open(ROBUST_PARAMS_FILE) as f:
            d = json.load(f)
        # FIX: Reject robust params with negative fitness — proven losing strategy
        fitness = d.get('fitness', 0)
        if fitness < 0:
            logger.warning(f"REJECTING robust params (fitness={fitness:.2f} < 0) — proven losing strategy")
            logger.info("Deleting bad robust params; bot will evolve fresh params.")
            try:
                os.remove(ROBUST_PARAMS_FILE)
            except:
                pass
            return None
        p = StrategyParams()
        p.take_profit = d.get('take_profit', 15.0)
        p.stop_loss = d.get('stop_loss', 150.0)
        p.max_hold = d.get('max_hold', 600)
        p.cooldown = d.get('cooldown', 180)
        p.fitness = fitness
        p.momentum_short_threshold = d.get('momentum_short_threshold', 0.003)
        p.momentum_med_threshold = d.get('momentum_med_threshold', 0.005)
        p.momentum_short_period = d.get('momentum_short_period', 4)
        p.momentum_med_period = d.get('momentum_med_period', 9)
        p.use_api_filter = d.get('use_api_filter', True)
        p.min_signal_age = d.get('min_signal_age', 0)
        # Load new lookback params if present
        p.lookback_short = d.get('lookback_short', 5)
        p.lookback_med = d.get('lookback_med', 12)
        p.max_momentum_age = d.get('max_momentum_age', 10)
        # Load trailing stop params if present (GA-optimized)
        p.trail_activation_profit = d.get('trail_activation_profit', DEFAULT_TRAIL_ACTIVATION)
        p.trail_distance = d.get('trail_distance', DEFAULT_TRAIL_DISTANCE)
        if p.fitness > LEGACY_FITNESS_THRESHOLD:
            logger.warning(f"Loaded legacy/simple-GA params fitness={p.fitness:.1f}. Resetting to 0.0.")
            p.fitness = 0.0

        # ADAPTIVE MOMENTUM THRESHOLD ADJUSTMENT
        # If thresholds are too high for current market conditions, reduce them dynamically
        original_short_thresh = p.momentum_short_threshold
        original_med_thresh = p.momentum_med_threshold

        # For scalping mode, use more sensitive thresholds to catch smaller moves
        if p.cooldown > 300:  # This indicates scalping mode
            p.momentum_short_threshold = max(0.001, original_short_thresh * 0.3)  # Reduce by 70%
            p.momentum_med_threshold = max(0.001, original_med_thresh * 0.3)    # Reduce by 70%

        logger.info(f"Loaded ROBUST params: TP={p.take_profit:.1f} SL={p.stop_loss:.1f} "
                   f"Fitness={p.fitness:.1f} S-Thresh={original_short_thresh:.4f}->{p.momentum_short_threshold:.4f} "
                   f"M-Thresh={original_med_thresh:.4f}->{p.momentum_med_threshold:.4f} "
                   f"TrailAct={p.trail_activation_profit:.2f} TrailDist={p.trail_distance:.2f}")
        return p
    
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            d = json.load(f)
        # CRITICAL FIX: Reject legacy params with negative fitness (proven losers)
        # These were likely generated by the old simple GA with bad backtest data.
        fitness = d.get('fitness', 0)
        if fitness < 0:
            logger.warning(f"REJECTING legacy params (fitness={fitness:.2f} < 0) - proven losing strategy")
            logger.info("Deleting bad legacy params and will use default/robust params instead")
            try:
                os.remove(PARAMS_FILE)
            except:
                pass
            return None
        p = StrategyParams()
        p.take_profit = d['take_profit']
        p.stop_loss = d['stop_loss']
        p.max_hold = d.get('max_hold', 600)
        p.cooldown = d.get('cooldown', 180)
        p.fitness = fitness
        p.momentum_short_threshold = d.get('momentum_short_threshold', 0.003)
        p.momentum_med_threshold = d.get('momentum_med_threshold', 0.005)
        # Load trailing stop params if present (GA-optimized)
        p.trail_activation_profit = d.get('trail_activation_profit', DEFAULT_TRAIL_ACTIVATION)
        p.trail_distance = d.get('trail_distance', DEFAULT_TRAIL_DISTANCE)
        if p.fitness > LEGACY_FITNESS_THRESHOLD:
            logger.warning(f"Loaded legacy/simple-GA params fitness={p.fitness:.1f}. Resetting to 0.0.")
            p.fitness = 0.0
        logger.warning("Using legacy params (run GA framework for robust params)")
        return p
    return None

# ========== LIVE BOT WITH MT5 ==========
class LiveBot:
    def __init__(self):
        # Connect to MT5
        self.mt5 = MT5Connection()
        if not self.mt5.connect():
            logger.error("Cannot connect to MT5. Make sure MT5 is running.")
            logger.error("1. Open MT5")
            logger.error("2. Go to Tools > Options > Expert Advisors")
            logger.error("3. Enable 'Allow Automated Trading'")
            sys.exit(1)
        
        # Clean overfit params
        if os.path.exists(PARAMS_FILE):
            try:
                with open(PARAMS_FILE) as f:
                    d = json.load(f)
                if d.get('fitness', 0) > MAX_FITNESS_CAP:
                    os.remove(PARAMS_FILE)
            except:
                pass
        if os.path.exists(ROBUST_PARAMS_FILE):
            try:
                with open(ROBUST_PARAMS_FILE) as f:
                    d = json.load(f)
                if d.get('fitness', 0) > MAX_FITNESS_CAP:
                    os.remove(ROBUST_PARAMS_FILE)
                    logger.warning(f"Removed overfit robust params (fitness={d.get('fitness', 0):.1f} > MAX_FITNESS_CAP)")
            except:
                pass

            # Force reset of overfit models
        if VERSIONING_AVAILABLE:
            vs = ModelVersioningSystem()
            tracker = vs._load_fitness_tracker()
            if tracker.best_fitness >= MAX_FITNESS_CAP:
                logger.warning(f"Resetting overfit model (fitness={tracker.best_fitness})")
                vs._reset_overfit_data(tracker.best_fitness)
        
        self.params = load_params()
        
        if self.params and self.params.fitness > LEGACY_FITNESS_THRESHOLD:
            logger.warning(f"Startup: resetting legacy fitness artifact {self.params.fitness:.1f} -> 0.0")
            self.params.fitness = 0.0
        if not self.params:
            logger.info("Training initial agent...")
            candles = fetch_hist()
            if not candles:
                logger.error("No historical data!")
                self.mt5.disconnect()
                sys.exit(1)
            self.params = evolve_params(candles)
            self.params.take_profit = max(100.0, min(self.params.take_profit, 500.0))
            self.params.stop_loss = max(120.0, min(self.params.stop_loss, 400.0))
            self.params.cooldown = max(self.params.cooldown, COOLDOWN_MIN)
            save_params(self.params)
        
        # Safety caps - enforce our trading rules (corrected for BTCUSD $0.01/point reality)
        # CAP TP at 250: trades avg ~180s hold, rarely reach 500+ points before momentum reverses
        self.params.take_profit = max(80.0, min(self.params.take_profit, 250.0))
        self.params.stop_loss = max(100.0, min(self.params.stop_loss, 250.0))
        self.params.cooldown = max(self.params.cooldown, COOLDOWN_MIN)
        # Live-appropriate momentum thresholds: 0.03-0.15% range for BTC at ~$60k
        self.params.momentum_short_threshold = max(0.03, min(self.params.momentum_short_threshold, 0.15))
        self.params.momentum_med_threshold = max(0.04, min(self.params.momentum_med_threshold, 0.20))
        
        self.total_trades = 0
        self.wins = 0
        self.total_pnl = 0.0
        self.last_trade_direction = 'buy'  # FIX: Track direction for MT5-closed trades
        self.last_position_profit = 0.0  # FIX: Track last known profit for MT5-closed detection
        self.last_evolve = time.time()
        self.model_count = 1
        # ========== MODEL LOADING DIAGNOSTICS ==========
        if self.params:
            logger.info(f"=== MODEL LOADING DIAGNOSTICS ===")
            logger.info(f"Model object: StrategyParams (fitness={self.params.fitness:.1f})")
            logger.info(f"Model version: v{self.model_count}")
            logger.info(f"Model fitness: {self.params.fitness:.1f}")
            logger.info(f"Threshold BUY: {self.params.momentum_short_threshold:.4f}/{self.params.momentum_med_threshold:.4f}")
            logger.info(f"Threshold SELL: {self.params.momentum_short_threshold:.4f}/{self.params.momentum_med_threshold:.4f}")
            logger.info(f"TP={self.params.take_profit:.1f} SL={self.params.stop_loss:.1f} Hold={self.params.max_hold}s Cool={self.params.cooldown}s")
            # Check if we have a proper version ID from the versioning system
            if VERSIONING_AVAILABLE:
                vs_check = ModelVersioningSystem()
                stats_check = vs_check.get_stats()
                logger.info(f"Versioning system current_best: {stats_check['current_best']}")
                logger.info(f"Versioning system best_fitness: {stats_check['best_fitness']:.1f}")
                logger.info(f"Versioning system improvements: {stats_check['total_improvements']}")
                if stats_check['current_best'] is None and self.params.fitness > 0:
                    logger.warning("Versioning system has NO current_best despite valid params — creating synthetic version entry")
                    # Create a synthetic version entry so _status() shows a meaningful name
                    from datetime import datetime
                    syn_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    tracker_check = vs_check._load_fitness_tracker()
                    tracker_check.best_fitness = self.params.fitness
                    tracker_check.best_version_id = f"v1_{syn_ts}"
                    tracker_check.total_improvements = 1
                    tracker_check.last_update = datetime.now().isoformat()
                    vs_check._save_fitness_tracker(tracker_check)
                    logger.info(f"Created synthetic version entry: {tracker_check.best_version_id} (fitness={self.params.fitness:.1f})")
        # ========== END DIAGNOSTICS ==========
        
        self.last_signal = ""
        self.last_api_signal = 0
        self.last_api_time = 0
        self.price_history = []
        self.last_trade_time = 0
        self.position_open_time = 0
        self.had_open_position = False
        self.balance_before_trade = 0.0  # Track balance before opening a trade for accurate PnL
        self.position_gone_count = 0  # Consecutive ticks position not found (prevents false MT5-close detections)
        self.position_open_verified = False  # Position has been confirmed visible in MT5 after opening
        self.position_peak_profit = 0.0  # Peak profit for trailing stop calculation
        self.trailing_stop_active = False  # Once true, stays true until position closes
        self.position_trail_floor = 0.0    # Locked floor profit; only ratchets up
        self.position_breakeven_applied = False  # Whether SL has been moved to breakeven
        
        self.current_mode = "SCALPING"
        self.learning_active = False
        self.momentum_short = 0
        self.momentum_med = 0
        self.entry_score = 0.0
        
        # Momentum freshness tracking (when momentum last flipped)
        self.momentum_short_time = 0.0
        self.momentum_med_time = 0.0
        
        self.learning = None
        if LEARNING_AVAILABLE:
            self.learning = OnlineLearningEngine()
            logger.info(f"Learning engine started: {self.learning.current_mode} mode")
            # Apply learning engine adjustments to params
            adjusted = self.learning.get_adjusted_params({
                'cooldown': self.params.cooldown,
                'take_profit': self.params.take_profit,
                'stop_loss': self.params.stop_loss,
                'max_hold': self.params.max_hold,
                'momentum_short_threshold': self.params.momentum_short_threshold,
                'momentum_med_threshold': self.params.momentum_med_threshold,
            })
            self.params.cooldown = adjusted['cooldown']
            self.params.stop_loss = adjusted['stop_loss']
            self.params.max_hold = adjusted['max_hold']
            self.params.momentum_short_threshold = adjusted['momentum_short_threshold']
            self.params.momentum_med_threshold = adjusted['momentum_med_threshold']
            # Apply safety caps AFTER adjustments - tighter for realistic scalping R:R
            self.params.take_profit = max(80.0, min(self.params.take_profit, 250.0))
            self.params.stop_loss = max(100.0, min(self.params.stop_loss, 250.0))
            self.params.cooldown = max(self.params.cooldown, COOLDOWN_MIN)
            logger.info(f"Learning mode '{self.learning.current_mode}' adjusted: "
                       f"cool={self.params.cooldown}s SL={self.params.stop_loss:.1f} "
                       f"hold={self.params.max_hold}s "
                       f"thresh={self.params.momentum_short_threshold:.4f}/{self.params.momentum_med_threshold:.4f}")
        
        # CRITICAL FIX: Store base params to prevent compounding when mode
        # adjustments are applied repeatedly. Without this, each call to
        # get_adjusted_params() multiplies already-adjusted values, causing
        # SL to spiral: 120 -> 180 -> 270 -> 405 -> 607.5 ...
        self.base_params = {
            'cooldown': self.params.cooldown,
            'take_profit': self.params.take_profit,
            'stop_loss': self.params.stop_loss,
            'max_hold': self.params.max_hold,
            'momentum_short_threshold': self.params.momentum_short_threshold,
            'momentum_med_threshold': self.params.momentum_med_threshold,
        }
        
        # Start GUI if available
        self.gui = None
        if GUI_AVAILABLE:
            try:
                self.gui = TradingGUI(self)
                self.gui.start()
                logger.info("Trading GUI started successfully")
            except Exception as e:
                logger.warning(f"Could not start GUI: {e}")
        
        self.green = '#00cc66'
        self.red = '#ff4444'
        self.yellow = '#ffcc00'
        self.white = '#ffffff'
        self.gray = '#888888'
        
        # DO NOT close existing open positions - keep running trades alive
        positions = self.mt5.get_open_positions()
        if positions:
            logger.info(f"Found {len(positions)} existing open position(s) - will continue managing them")
            # Re-attach to existing position instead of closing it
            self.position_open_time = time.time()  # Reset hold timer (approximate)
        
        logger.info(f"Bot ready. TP={self.params.take_profit:.1f} SL={self.params.stop_loss:.1f} "
                     f"Hold={self.params.max_hold}s Cool={self.params.cooldown}s")
        
        if self.learning:
            logger.info(f"Learning: {self.learning.current_mode} mode, "
                     f"cooldown={self.params.cooldown}s, thresholds={self.params.momentum_short_threshold:.3f}/{self.params.momentum_med_threshold:.3f}")
    
    def run(self):
        logger.info("=" * 60)
        logger.info(f"[{ts()}] LIVE MT5 SCALPING BOT STARTED")
        logger.info(f"Account: ${self.mt5.get_balance():.2f}")
        logger.info("=" * 60)
        last_status = time.time()
        live = None
        
        while True:
            try:
                if self.mt5.should_stop_trading():
                    if not hasattr(self, '_last_run_stop_log') or time.time() - self._last_run_stop_log > 60:
                        self._last_run_stop_log = time.time()
                        logger.error("LOSS LIMIT REACHED - Bot will not open new trades")
                        logger.error(f"Daily PnL: ${self.mt5.daily_pnl:.2f}")
                        logger.error(f"Consecutive losses: {self.mt5.consecutive_losses}")
                    if self.gui:
                        self.gui.set_mode("DEFENSIVE")
                
                live = fetch_live_with_retry()
                now = time.time()
                
                if now - self.last_evolve > EVOLVE_INTERVAL:
                    self._evolve()
                    self.last_evolve = now
                
                tick = self.mt5.get_tick()
                if self.gui:
                    if tick:
                        self.gui.bid_label.config(text=f"Bid: ${tick.bid:.2f}")
                        self.gui.ask_label.config(text=f"Ask: ${tick.ask:.2f}")
                        self.gui.spread_label.config(text=f"Spread: {tick.ask - tick.bid:.1f}")
                    else:
                        self.gui.bid_label.config(text="Bid: ---")
                        self.gui.ask_label.config(text="Ask: ---")
                        self.gui.spread_label.config(text="Spread: ---")
                    
                    self.gui.api_label.config(text=f"API: {int(self.last_api_signal)}")
                    mom_color = {1: self.green, -1: self.red, 0: self.gray}.get(self.momentum_short, self.gray)
                    self.gui.smom_label.config(text=f"S-Mom: {self.momentum_short:+d}", fg=mom_color)
                    mom_color = {1: self.green, -1: self.red, 0: self.gray}.get(self.momentum_med, self.gray)
                    self.gui.mmom_label.config(text=f"M-Mom: {self.momentum_med:+d}", fg=mom_color)
                    score_color = self.green if self.entry_score > 0.5 else (self.red if self.entry_score < -0.5 else self.yellow)
                    self.gui.score_label.config(text=f"Score: {self.entry_score:.2f}", fg=score_color)
                    
                    if self.mt5.has_open_position():
                        pos = self.mt5.get_position_details()
                        if pos:
                            dir_color = self.green if pos['direction'] == 'buy' else self.red
                            self.gui.pos_dir_label.config(text=f"Direction: {pos['direction'].upper()}", fg=dir_color)
                            self.gui.pos_entry_label.config(text=f"Entry: ${pos['entry_price']:.2f}")
                            if pos['profit'] >= 0:
                                self.gui.pos_pnl_label.config(text=f"PnL: +${pos['profit']:.2f}", fg=self.green)
                            else:
                                self.gui.pos_pnl_label.config(text=f"PnL: ${pos['profit']:.2f}", fg=self.red)
                            held = now - self.position_open_time if self.position_open_time > 0 else 0
                            self.gui.pos_held_label.config(text=f"Held: {int(held)}s / {self.params.max_hold}s")
                            self.gui.pos_tp_label.config(text=f"TP: {self.params.take_profit:.1f}")
                            self.gui.pos_sl_label.config(text=f"SL: {self.params.stop_loss:.1f}")
                    else:
                        self.gui.pos_dir_label.config(text="Direction: ---", fg=self.white)
                        self.gui.pos_entry_label.config(text="Entry: ---")
                        self.gui.pos_pnl_label.config(text="PnL: ---", fg=self.white)
                        self.gui.pos_held_label.config(text="Held: No position")
                        self.gui.pos_tp_label.config(text="TP: ---")
                        self.gui.pos_sl_label.config(text="SL: ---")
                    
                    self.gui.balance_label.config(text=f"${self.mt5.get_balance():.2f}")
                    self.gui.equity_label.config(text=f"${self.mt5.get_equity():.2f}")
                    if self.mt5.daily_pnl >= 0:
                        self.gui.daily_label.config(text=f"+${self.mt5.daily_pnl:.2f}", fg=self.green)
                    else:
                        self.gui.daily_label.config(text=f"${self.mt5.daily_pnl:.2f}", fg=self.red)
                    self.gui.trades_label.config(text=str(self.total_trades))
                    if self.total_trades > 0:
                        wr = (self.wins / self.total_trades) * 100
                        avg = self.total_pnl / self.total_trades
                        self.gui.winrate_label.config(text=f"{wr:.0f}%")
                        self.gui.avg_label.config(text=f"${avg:.3f}")
                
                if live:
                    self._step(live)
                
                if now - last_status > 30:
                    self._status(live)
                    last_status = now
                
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("\nShutting down...")
                self._shutdown()
                break
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                time.sleep(10)

# In the _step method, replace the loss limit check and add recovery:

    def _step(self, live):
        now = time.time()
        
        # Get prices from MT5 directly (real broker prices)
        tick = self.mt5.get_tick()
        if tick is None:
            return
        
        bid = tick.bid
        ask = tick.ask
        mid = (bid + ask) / 2
        
        # Get indicators from obilo.io
        ind1, ind2 = live[9], live[10]
        api_signal = ind1 + ind2
        
        if api_signal != self.last_api_signal:
            self.last_api_signal = api_signal
            self.last_api_time = now
            logger.info(f"[{ts()}] Indicator update: API={api_signal:.0f}")
        
        # Track MT5 price history for momentum
        self.price_history.append(mid)
        if len(self.price_history) > 120:
            self.price_history.pop(0)
        
        # Short momentum using MT5 prices - absolute dollar move instead of %
        # to avoid noise at $60k prices. Minimum floor prevents micro-whipsaws.
        # FIX: Use parameterised lookbacks for faster scalping signals.
        # FIX 2026-06-29: Lowered dollar thresholds from 35/55 to 20/30 to increase
        # momentum detection sensitivity. At $60k BTC, $20 change over 8 ticks (8s)
        # and $30 over 24 ticks (24s) is reasonable for scalping.
        MOMENTUM_SHORT_LOOKBACK = self.params.lookback_short
        MOMENTUM_MED_LOOKBACK = self.params.lookback_med
        
        prev_momentum_short = self.momentum_short
        prev_momentum_med = self.momentum_med
        
        momentum_short = 0
        if len(self.price_history) >= MOMENTUM_SHORT_LOOKBACK:
            change = self.price_history[-1] - self.price_history[-MOMENTUM_SHORT_LOOKBACK]
            dollar_thresh = max(15.0, self.params.momentum_short_threshold * 400.0)
            if change > dollar_thresh:
                momentum_short = 1
            elif change < -dollar_thresh:
                momentum_short = -1
        
        # Medium momentum using MT5 prices
        momentum_med = 0
        if len(self.price_history) >= MOMENTUM_MED_LOOKBACK:
            change = self.price_history[-1] - self.price_history[-MOMENTUM_MED_LOOKBACK]
            dollar_thresh = max(20.0, self.params.momentum_med_threshold * 400.0)
            if change > dollar_thresh:
                momentum_med = 1
            elif change < -dollar_thresh:
                momentum_med = -1
        
        # Track when momentum last flipped so we can reject stale signals.
        if momentum_short != prev_momentum_short and momentum_short != 0:
            self.momentum_short_time = now
            logger.debug(f"S-Mom flipped to {momentum_short:+d} at t={now:.1f}")
        if momentum_med != prev_momentum_med and momentum_med != 0:
            self.momentum_med_time = now
            logger.debug(f"M-Mom flipped to {momentum_med:+d} at t={now:.1f}")
        
        self.momentum_short = momentum_short
        self.momentum_med = momentum_med
        
        # FIXED 2026-06-29: REMOVED API directional bias from entry score calculation.
        # The old formula (api_dir*0.10 + momentum_short*0.50 + momentum_med*0.40) created
        # a permanent BUY bias because API=3 always gives api_dir=+1.0. This made SELL entries
        # require much stronger momentum to overcome the +0.10 baseline, causing missed
        # sell opportunities during downtrends. API is now used ONLY as a hard veto below.
        # Both BUY and SELL signals are now purely momentum-driven, giving each equal opportunity.
        entry_score = (momentum_short * 0.55 + momentum_med * 0.45)
        self.entry_score = entry_score
        
        # Update GUI market data
        if self.gui:
            self.gui.update_market(bid, ask, api_signal, momentum_short, momentum_med, entry_score)
            if self.mt5.has_open_position():
                pos = self.mt5.get_position_details()
                if pos:
                    held = now - self.position_open_time if self.position_open_time > 0 else 0
                    self.gui.update_position(pos['direction'], pos['entry_price'], 
                                           (bid + ask) / 2, pos['profit'], held,
                                           self.params.max_hold, self.params.take_profit,
                                           self.params.stop_loss)
        
        signal_age = now - self.last_api_time if self.last_api_time > 0 else 999
        self.last_signal = (f"API={api_signal:.0f} S-Mom={momentum_short:+d} M-Mom={momentum_med:+d} "
                           f"Bid={bid:.2f} Ask={ask:.2f}")
        
        # ---- DETECT MT5-INITIATED CLOSE (SL/TP hit by broker) ----
        has_position = self.mt5.has_open_position()
        if has_position:
            self.position_gone_count = 0
            self.position_open_verified = True
        
        if self.had_open_position and not has_position:
            # CRITICAL FIX: Require position to have been verified AND gone for 2+ consecutive ticks
            # before declaring it closed. This prevents false detections from MT5 API propagation delays.
            if not self.position_open_verified:
                # Position never confirmed visible after open - likely failed immediately
                logger.warning(">>> Position disappeared before verification - not recording as closed trade")
                self.had_open_position = False
                self.position_open_verified = False
                self.position_gone_count = 0
                return
            
            self.position_gone_count += 1
            if self.position_gone_count < 2:
                logger.debug(f"Position not found (gone_count={self.position_gone_count}), waiting for confirmation...")
                return
            
            # Require minimum 3 seconds since open before accepting a close
            # (prevents false "closed in 2 seconds" detections)
            time_since_open = now - self.position_open_time if self.position_open_time > 0 else 999
            if time_since_open < 3:
                logger.debug(f"Position gone after only {time_since_open:.1f}s, ignoring (likely MT5 delay)")
                return
            
            logger.info(f"<<< Position closed by MT5 (SL/TP hit) - recording trade outcome")
            
            # CRITICAL FIX: Try to get the actual deal details from MT5 history.
            # Balance-derived PnL is correct but doesn't tell us WHY. The deal history
            # gives exact profit, commission, and swap breakdown.
            last_ticket = getattr(self, '_last_position_ticket', 0)
            deal_info = self.mt5._get_deal_details(last_ticket)
            if deal_info['net'] != 0.0:
                actual_pnl = deal_info['net']
                logger.info(f"<<< MT5 CLOSED | DealProfit=${deal_info['profit']:.2f} "
                           f"Commission=${deal_info['commission']:.2f} "
                           f"Swap=${deal_info['swap']:.2f} "
                           f"NET=${actual_pnl:.2f}")
                estimated_profit = actual_pnl
            else:
                # Fallback: use balance-derived PnL
                current_balance = self.mt5.get_balance()
                if self.balance_before_trade > 0:
                    estimated_profit = current_balance - self.balance_before_trade
                else:
                    estimated_profit = current_balance - (self.mt5.starting_balance + self.total_pnl)
                
                logger.warning(f"<<< MT5 CLOSED (no deal record) | Balance-derived PnL=${estimated_profit:.2f}")
            
            self._update_stats(estimated_profit)
            self.had_open_position = False
            self.position_open_verified = False
            self.position_gone_count = 0
            self.balance_before_trade = 0.0
            self.last_position_profit = 0.0
            self.position_peak_profit = 0.0
            self.trailing_stop_active = False
            self.position_trail_floor = 0.0
            self.position_breakeven_applied = False
            # CRITICAL FIX: Reset cooldown clock so bot doesn't re-enter immediately
            self.last_trade_time = time.time()
            
            # Auto-recover from loss streak if we just had a win
            if estimated_profit > 0 and self.mt5.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                logger.info("WIN detected after loss streak! Resuming trading...")
                self.mt5.consecutive_losses = 0
            return
        
        # ---- MANAGE OPEN POSITION ----
        if has_position:
            pos = self.mt5.get_position_details()
            if pos is None:
                return
            
            self.had_open_position = True
            direction = pos['direction']
            entry = pos['entry_price']
            profit = pos['profit']
            self.last_position_profit = profit  # FIX: Save last known profit for MT5-closed detection
            held = now - self.position_open_time if self.position_open_time > 0 else 999
            
            tp = self.params.take_profit
            sl = self.params.stop_loss
            
            # FAST-EXIT: If position goes immediately against us with momentum
            # already flipping, cut within seconds before small loss becomes big.
            # This stops the "Trade 2" failure: entered at local peak, instantly
            # reversed -$0.30+ with momentum neutralising.
            FAST_EXIT_LOSS = -0.35     # Dollar loss threshold (0.01 lot)
            FAST_EXIT_MIN_AGE = 8      # Seconds: let spread/commission settle
            if profit <= FAST_EXIT_LOSS and held <= FAST_EXIT_MIN_AGE:
                if (direction == 'buy' and momentum_short <= 0) or \
                   (direction == 'sell' and momentum_short >= 0):
                    self._safe_close("fast stop (adverse pop)", expected_pnl=profit)
                    return
            elif profit <= FAST_EXIT_LOSS and held <= 20:
                # Slightly longer leash: both momentum against
                if (direction == 'buy' and momentum_short == -1 and momentum_med == -1) or \
                   (direction == 'sell' and momentum_short == 1 and momentum_med == 1):
                    self._safe_close("fast stop (dual momentum against)", expected_pnl=profit)
                    return
            
            # BREAKEVEN: once profit hits BREAKEVEN_PROFIT, move SL to entry to lock
            # in at least a minimal winner even if everything reverses.
            # MINIMUM 45s hold prevents micro-whipsaws on BTC volatility.
            if profit >= BREAKEVEN_PROFIT and not self.position_breakeven_applied and held > 45:
                # Move SL to entry ± tiny buffer (0 for digits=0)
                if direction == 'buy':
                    new_sl = round(entry - 1.0, 0)
                else:
                    new_sl = round(entry + 1.0, 0)
                pos_ticket = pos.get('ticket')
                if pos_ticket:
                    if self.mt5.modify_sl(pos_ticket, new_sl):
                        self.position_breakeven_applied = True
                        logger.info(f"BREAKEVEN applied: SL moved to {new_sl:.2f}")

            # Check TP using MT5 prices - close when profit target hit
            if direction == 'buy' and bid >= entry + tp:
                self._safe_close("TAKE PROFIT")
                return
            if direction == 'sell' and ask <= entry - tp:
                self._safe_close("TAKE PROFIT")
                return
            
            # Check SL using MT5 prices - cut losses when SL hit
            if direction == 'buy' and bid <= entry - sl:
                self._safe_close("STOP LOSS")
                return
            if direction == 'sell' and ask >= entry + sl:
                self._safe_close("STOP LOSS")
                return
            
            # HARDENED Momentum reversal exit - require higher profit AND both momentum against
            # to avoid noise whipsawing winners out before they reach TP.
            if profit >= 1.00 and held > 90:
                if direction == 'buy' and momentum_short == -1 and momentum_med <= 0:
                    self._safe_close("momentum reversal", expected_pnl=profit)
                    return
                if direction == 'sell' and momentum_short == 1 and momentum_med >= 0:
                    self._safe_close("momentum reversal", expected_pnl=profit)
                    return
            
            # HARDENED CUT LOSERS EARLY: deeper drawdown, longer hold time, full momentum confirmation.
            # This prevents -$0.60 noise cuts that would have recovered into winners.
            if profit <= -1.00 and held > 90:
                if direction == 'buy' and momentum_short == -1 and momentum_med == -1:
                    self._safe_close("early stop (momentum against)", expected_pnl=profit)
                    return
                if direction == 'sell' and momentum_short == 1 and momentum_med == 1:
                    self._safe_close("early stop (momentum against)", expected_pnl=profit)
                    return
            
            # TRAILING STOP: once we hit trail_activation_profit, we NEVER give back more than trail_distance.
            # CRITICAL FIX: must track a "trailing_stop_active" flag so we don't bail out of
            # the check when profit temporarily dips below trail_activation_profit (this bug caused
            # the live trade to reverse from +$0.61 all the way to -$0.03 on max hold).
            # Use GA-optimized trailing parameters if available, otherwise use defaults.
            trail_activation = getattr(self.params, 'trail_activation_profit', DEFAULT_TRAIL_ACTIVATION)
            trail_dist = getattr(self.params, 'trail_distance', DEFAULT_TRAIL_DISTANCE)
            
            if profit > self.position_peak_profit:
                self.position_peak_profit = profit
            if profit >= trail_activation:
                self.trailing_stop_active = True
                # Lock trail_floor the FIRST time we cross threshold, then only ratchet up
                if self.position_trail_floor == 0.0:
                    self.position_trail_floor = profit - trail_dist
                else:
                    # Ratchet: only raise floor, never lower it
                    candidate_floor = profit - trail_dist
                    if candidate_floor > self.position_trail_floor:
                        self.position_trail_floor = candidate_floor
            if getattr(self, 'trailing_stop_active', False):
                if profit <= self.position_trail_floor:
                    self._safe_close("trailing stop", expected_pnl=profit)
                    self.position_peak_profit = 0.0
                    self.trailing_stop_active = False
                    self.position_trail_floor = 0.0
                    return
            
            # DYNAMIC MAX HOLD: only extend for winners; never retroactively shrink
            effective_max_hold = self.params.max_hold
            if profit > 0.15:  # Slightly profitable and momentum still aligned?
                if (direction == 'buy' and momentum_short >= 0) or (direction == 'sell' and momentum_short <= 0):
                    effective_max_hold = int(self.params.max_hold * 1.5)  # Give winners more room
                else:
                    effective_max_hold = int(self.params.max_hold * 1.2)
            # REMOVED: Retroactive reduction caused trades open for 300s to die instantly
            # when a brief -$0.31 dip dropped effective_max_hold from 400 -> 240.
            
            # Max hold time exit - close when time is up
            if held > effective_max_hold:
                self._safe_close("max hold time")
                self.position_peak_profit = 0.0
            return
        
        # ---- CHECK ENTRY ----
        effective_cooldown = max(self.params.cooldown, COOLDOWN_MIN)
        # Progressive cooldown after consecutive losses to stop revenge trading
        if self.mt5.consecutive_losses > 0:
            penalty = min(self.mt5.consecutive_losses * 60, 300)  # +1min per loss, max +5min
            effective_cooldown += penalty
            logger.debug(f"Cooldown extended by {penalty}s due to {self.mt5.consecutive_losses} consecutive losses")
        if now - self.last_trade_time < effective_cooldown:
            return
        

        if self.mt5.should_stop_trading():
            # Only log once per 30 seconds
            if not hasattr(self, '_last_stop_log') or time.time() - self._last_stop_log > 30:
                self._last_stop_log = time.time()
                logger.error("LOSS LIMIT REACHED - Bot will not open new trades")
                logger.error(f"Daily PnL: ${self.mt5.daily_pnl:.2f}")
                logger.error(f"Consecutive losses: {self.mt5.consecutive_losses}")
            
            if self.gui:
                self.gui.set_mode("DEFENSIVE")
            return
        
        # DYNAMIC ENTRY THRESHOLDS based on mode and performance
        # AGGRESSIVE mode handling: require more trades (>=10) before lowering threshold
        # to prevent premature risk escalation after a lucky streak. Also double the 
        # lot size penalty: in AGGRESSIVE mode we don't double lots, we tighten threshold
        # less aggressively.
        entry_threshold = 0.50  # More sensitive threshold for current market conditions
        insights = self.learning.get_pattern_insights() if self.learning else {}
        current_mode = self.learning.current_mode if self.learning else 'SCALPING'
        best_dir = insights.get('best_direction', '').replace('_', ' ').split()[0] if insights.get('best_direction') else None

        if current_mode == 'DEFENSIVE':
            entry_threshold = 0.65  # Slightly harder threshold in DEFENSIVE
        elif current_mode == 'AGGRESSIVE':
            # Only lower threshold if we have statistically meaningful sample
            if self.total_trades >= 10:
                entry_threshold = 0.40  # More sensitive when winning
                logger.debug(f"AGGRESSIVE mode active: threshold=0.40 ({self.total_trades} trades evaluated)")
            else:
                # Keep SCALPING threshold until 10+ trades are collected
                entry_threshold = 0.50
                logger.debug(f"AGGRESSIVE mode deferred: only {self.total_trades}/10 trades collected, threshold=0.50")

        # Progressive difficulty: each consecutive loss makes entries harder.
        # Raised penalty from 0.04 to 0.06 per loss for more effective loss prevention
        loss_penalty = self.mt5.consecutive_losses * 0.06  # Stronger penalty
        entry_threshold += loss_penalty
        if self.mt5.consecutive_losses > 0:
            logger.debug(f"Entry threshold elevated by {loss_penalty:.2f} after {self.mt5.consecutive_losses} loss(es): {entry_threshold:.2f}")

        direction = None

        # RANGE/CHOP FILTER: skip entries if price has been oscillating without a clear trend.
        # FIX 2026-06-29: Raised from $40 to $80 to reduce false positives.
        # At $59k BTC, $40 range over 4 minutes is normal micro-oscillation even during
        # trending markets. $80 over 240 ticks better captures true non-trending conditions.
        CHOP_WINDOW = 240  # 4-minute window
        if not self.mt5.has_open_position() and len(self.price_history) >= CHOP_WINDOW:
            recent = self.price_history[-CHOP_WINDOW:]
            price_range = max(recent) - min(recent)
            if price_range < 80.0:  # Less aggressive chop filter
                if not hasattr(self, '_last_chop_log') or now - getattr(self, '_last_chop_log', 0) > 30:
                    self._last_chop_log = now
                    logger.info(f"CHOP FILTER: No entry — price range too tight (${price_range:.0f} in 4m, threshold=$80)")
                # direction remains None; falls through to return below

        # CRITICAL FIX: Require both momentum indicators to concur for entry.
        # The old code allowed API-biased entries with only one momentum signal,
        # leading to repeated counter-trend entries that drained the account.
        if entry_score > entry_threshold and momentum_short == 1 and momentum_med == 1:
            direction = 'buy'
        elif entry_score < -entry_threshold and momentum_short == -1 and momentum_med == -1:
            direction = 'sell'
        
        if direction is None:
            # ====== DIAGNOSTIC: Log exactly why no entry was taken ======
            diagnostic = {
                'score': f'{entry_score:.3f}',
                'buy_threshold': f'{entry_threshold:.3f}',
                'sell_threshold': f'{-entry_threshold:.3f}',
                'cooldown': f'{max(0, effective_cooldown - (now - self.last_trade_time)):.0f}s remaining' if self.last_trade_time > 0 else 'none',
                'mom_short': momentum_short,
                'mom_med': momentum_med,
                'api_signal': f'{api_signal:.0f}',
            }
            # Determine the specific reason(s)
            reasons = []
            if momentum_short != 1 or momentum_med != 1:
                if entry_score > entry_threshold:
                    reasons.append(f"score={entry_score:.3f}>=threshold but momentum not both +1 (S={momentum_short}, M={momentum_med})")
            if momentum_short != -1 or momentum_med != -1:
                if entry_score < -entry_threshold:
                    reasons.append(f"score={entry_score:.3f}<=threshold but momentum not both -1 (S={momentum_short}, M={momentum_med})")
            if abs(entry_score) <= entry_threshold:
                reasons.append(f"|score|={abs(entry_score):.3f} <= entry_threshold={entry_threshold:.3f}")
            
            diagnostic['reason'] = ' | '.join(reasons) if reasons else 'chop_filter_active'
            logger.info(f"No trade: score={diagnostic['score']} buy={diagnostic['buy_threshold']} sell={diagnostic['sell_threshold']} cooldown={diagnostic['cooldown']} S-mom={diagnostic['mom_short']:+d} M-mom={diagnostic['mom_med']:+d} api={diagnostic['api_signal']} — {diagnostic['reason']}")
            return
        
        # MOMENTUM FRESHNESS FILTER: only reject entries when momentum is NEUTRAL (0)
        # AND stale. If momentum is actively +1 or -1 (directional), it doesn't matter
        # when it last flipped — the trend is still intact. The old filter incorrectly
        # blocked valid trend continuations where M-Mom was -1 for 52+ seconds because
        # the last *flip* was old, even though momentum was still actively bearish.
        max_age = self.params.max_momentum_age
        
        # Only check freshness for neutrally-signed momentum (0).
        # +1/-1 means the trend direction is active regardless of age.
        s_active = self.momentum_short != 0  # Short momentum is +1 or -1 (still active)
        m_active = self.momentum_med != 0    # Medium momentum is +1 or -1 (still active)
        
        s_age = now - self.momentum_short_time if self.momentum_short_time > 0 else 999
        m_age = now - self.momentum_med_time if self.momentum_med_time > 0 else 999
        
        # Only reject if BOTH are stale AND both are neutral (safest filter)
        stale_s = not s_active and s_age > max_age
        stale_m = not m_active and m_age > max_age
        
        if stale_s or stale_m:
            logger.info(f"FRESHNESS FILTER: Rejecting {direction.upper()} — stale neutral momentum "
                       f"(S-active={s_active} age={s_age:.0f}s, M-active={m_active} age={m_age:.0f}s, max={max_age}s)")
            return
        
        # DIRECTIONAL FILTER: only block a direction when we have statistically significant
        # evidence it performs poorly (>=5 trades, WR < 35%) while the other direction is better.
        if self.learning and self.total_trades >= 5:
            buy_trades = [t for t in self.learning.trade_history if t.direction == 'buy']
            sell_trades = [t for t in self.learning.trade_history if t.direction == 'sell']
            buy_wr = sum(1 for t in buy_trades if t.won) / len(buy_trades) if len(buy_trades) >= 3 else 0.5
            sell_wr = sum(1 for t in sell_trades if t.won) / len(sell_trades) if len(sell_trades) >= 3 else 0.5
            
            if direction == 'buy' and len(buy_trades) >= 3 and buy_wr < 0.35 and sell_wr > buy_wr + 0.15:
                logger.info(f"DIRECTIONAL FILTER: Blocking BUY (WR={buy_wr:.0%} vs SELL WR={sell_wr:.0%})")
                return
            if direction == 'sell' and len(sell_trades) >= 3 and sell_wr < 0.35 and buy_wr > sell_wr + 0.15:
                logger.info(f"DIRECTIONAL FILTER: Blocking SELL (WR={sell_wr:.0%} vs BUY WR={buy_wr:.0%})")
                return
        
        # RELAXED API filter: only veto if API STRONGLY opposes AND signal is fresh.
        # The entry_score already accounts for API at 10% weight; a hard veto here
        # caused the bot to entirely skip SELL trades for 30+ minutes while BTC
        # dropped $300 with bearish momentum, because a stale API=3 signal
        # overrode clear dual-momentum bearish signals.
        api_fresh = (now - self.last_api_time) < 600 if self.last_api_time > 0 else False
        if api_fresh:
            if direction == 'buy' and api_signal <= -5:
                logger.info(f"Skipping BUY - API strongly opposes direction (API={api_signal:.0f})")
                return
            if direction == 'sell' and api_signal >= 5:
                logger.info(f"Skipping SELL - API strongly opposes direction (API={api_signal:.0f})")
                return
        
        # PRE-TRADE RISK GUARDS
        # Guard 1: minimum balance floor
        if self.mt5.get_balance() < MIN_ACCOUNT_BALANCE:
            if not hasattr(self, '_last_balance_warn') or time.time() - self._last_balance_warn > 60:
                self._last_balance_warn = time.time()
                logger.warning(f"Balance ${self.mt5.get_balance():.2f} below floor ${MIN_ACCOUNT_BALANCE} - no new trades")
            return
        
        # Guard 2: spread check (broker may show 0 initially; only reject when definitely excessive)
        tick_check = self.mt5.get_tick()
        if tick_check and tick_check.ask and tick_check.bid:
            spread = tick_check.ask - tick_check.bid
            if spread > MAX_SPREAD:
                logger.warning(f"Spread ${spread:.2f} exceeds max ${MAX_SPREAD} - skipping entry")
                return
        
        # Open trade on MT5 with optimized TP/SL
        # Capture balance BEFORE order send so commission deduction doesn't
        # distort the PnL calculation for MT5-closed trades.
        pre_trade_balance = self.mt5.get_balance()
        if self.mt5.open_trade(direction, tp=self.params.take_profit, sl=self.params.stop_loss):
            self.last_trade_time = now
            self.position_open_time = now
            self.had_open_position = True
            self.last_trade_direction = direction  # FIX: Save direction for stats when MT5 closes
            self.balance_before_trade = pre_trade_balance
            self.position_peak_profit = 0.0
            # CRITICAL FIX: Save the position ticket so MT5-closed detection can query deal history.
            pos = self.mt5.get_position_details()
            if pos:
                self._last_position_ticket = pos['ticket']
            logger.info(f">>> {direction.upper()} signal (score={entry_score:.2f}, {self.last_signal})")
        else:
            self.balance_before_trade = 0.0

    def _safe_close(self, reason: str, expected_pnl: float = None):
        """Close position, handling already-closed race condition gracefully.
        Defined as a proper method (not a closure re-created on every tick)
        to avoid repeated function definition overhead."""
        if not self.mt5.has_open_position():
            logger.debug(f"Position already closed by MT5 before {reason} close")
            return None
        close_result = self.mt5.close_position()
        if close_result is not None and close_result is not False:
            self._update_stats(close_result)
            self.had_open_position = False
            self.balance_before_trade = 0.0
            self.last_position_profit = 0.0
            self.position_peak_profit = 0.0
            self.trailing_stop_active = False
            self.position_trail_floor = 0.0
            self.position_breakeven_applied = False
            logger.info(f"<<< CLOSED on {reason} | PnL: ${close_result:.2f}")
        elif close_result is None:
            # Position was likely closed by MT5 between check and close
            logger.debug(f"Close returned None for {reason} - MT5 likely already closed it")
        return close_result

    def _update_stats(self, profit):
        self.total_trades += 1
        if profit > 0:
            self.wins += 1
            self.mt5.consecutive_losses = 0
        else:
            self.mt5.consecutive_losses += 1
        
        # After any trade, enforce cooldown before next entry (prevents overtrading)
        self.last_trade_time = time.time()
        
        self.total_pnl += profit
        self.mt5.update_daily_pnl()
        
        if self.learning:
            pos = self.mt5.get_position_details()
            direction = self.last_trade_direction  # FIX: Use saved direction when MT5 already closed position
            entry_price = 0
            if pos:
                direction = pos['direction']
                entry_price = pos['entry_price']
            tick = self.mt5.get_tick()
            exit_price = (tick.bid + tick.ask) / 2 if tick else entry_price
            self.learning.record_trade(
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=profit,
                held_seconds=time.time() - self.position_open_time if self.position_open_time > 0 else 0,
                momentum_short=self.momentum_short,
                momentum_med=self.momentum_med,
                api_signal=self.last_api_signal
            )
            
            # CRITICAL FIX: Pass base_params to get_adjusted_params() instead of
            # already-adjusted self.params. This prevents compounding mode
            # multipliers on every trade (e.g., AGGRESSIVE SL 1.5x applied
            # repeatedly: 120 -> 180 -> 270 -> 405 -> 607.5).
            adjusted = self.learning.get_adjusted_params(self.base_params)
            old_sl = self.params.stop_loss
            self.params.cooldown = max(adjusted['cooldown'], COOLDOWN_MIN)
            self.params.stop_loss = max(adjusted['stop_loss'], 120.0)
            self.params.max_hold = adjusted['max_hold']
            self.params.momentum_short_threshold = adjusted['momentum_short_threshold']
            self.params.momentum_med_threshold = adjusted['momentum_med_threshold']
            if self.params.stop_loss != old_sl:
                logger.info(f"Learning adjusted SL: {old_sl:.1f} -> {self.params.stop_loss:.1f} "
                           f"(mode={self.learning.current_mode})")
            
            if self.gui:
                self.gui.set_mode(self.learning.current_mode)
        
        if self.gui:
            held = time.time() - self.position_open_time if self.position_open_time > 0 else 0
            pos = self.mt5.get_position_details()
            direction = self.last_trade_direction  # FIX: Use saved direction when MT5 already closed position
            if pos:
                direction = pos['direction']
            self.gui.record_trade(direction, profit, held)
            
            wr = (self.wins / max(1, self.total_trades)) * 100
            avg = self.total_pnl / max(1, self.total_trades)
            self.gui.update_performance(
                self.mt5.get_balance(),
                self.mt5.get_equity(),
                self.mt5.daily_pnl,
                self.total_trades,
                self.wins,
                avg,
                wr
            )

    def _evolve(self):
        logger.info(f"\n{'='*40}\n[{ts()}] EVOLVING PARAMETERS")
        
        if USE_ROBUST_GA and ROBUST_GA_AVAILABLE:
            try:
                logger.info("Using ROBUST GA Framework (WFO + Multi-Period Validation)")
                candles = ga_fetch_hist()
                if not candles:
                    return
                
                ga = RobustGA(candles, n_folds=3)
                new_params = ga.walk_forward_optimize()
                validation = ga.multi_period_validate(new_params, n_periods=3)
                
                logger.info(f"Validation: Passed={validation['passed']}, "
                         f"Avg Profit=${validation['avg_profit']:.2f}")
                
                # Determine if we should save/update
                wfo_passed = getattr(new_params, 'wfo_passed', False)
                current_is_legacy = self.params.fitness > LEGACY_FITNESS_THRESHOLD
                unvalidated = self.params.fitness <= 0 or current_is_legacy
                should_save = False
                
                if unvalidated and wfo_passed:
                    logger.info(f"Replacing stale/legacy params with WFO-validated RobustGA params (fitness={new_params.fitness:.1f})")
                    should_save = True
                elif validation['passed'] and new_params.fitness > self.params.fitness * 0.8:
                    should_save = True
                    logger.info(f"ROBUST IMPROVED! Fitness={new_params.fitness:.1f}")
                elif new_params.fitness > self.params.fitness:
                    should_save = True
                    logger.info(f"ROBUST IMPROVED! Fitness={new_params.fitness:.1f}")
                else:
                    logger.info(f"No improvement (Fitness={new_params.fitness:.1f} <= {self.params.fitness:.1f})")
                
                if should_save:
                    # Check for overfit before updating live params
                    if new_params.fitness > MAX_FITNESS_CAP:
                        logger.warning(f"Rejecting overfit model update (fitness={new_params.fitness:.1f} > {MAX_FITNESS_CAP})")
                        logger.warning("Live params unchanged - model would be overfit")
                        logger.info(f"DEBUG: MAX_FITNESS_CAP = {MAX_FITNESS_CAP}, new_params.fitness = {new_params.fitness}")
                    else:
                        # Apply safety caps - tighter TP/SL for realistic scalping R:R
                        new_params.take_profit = max(80.0, min(new_params.take_profit, 250.0))
                        new_params.stop_loss = max(100.0, min(new_params.stop_loss, 250.0))
                        new_params.cooldown = max(new_params.cooldown, COOLDOWN_MIN)
                        
                        self.params.take_profit = new_params.take_profit
                        self.params.stop_loss = new_params.stop_loss
                        self.params.max_hold = new_params.max_hold
                        self.params.cooldown = new_params.cooldown
                        self.params.fitness = new_params.fitness
                        self.model_count += 1
                        # CRITICAL FIX: Update base_params so mode adjustments apply
                        # to the newly evolved values, not old compounded ones.
                        self.base_params = {
                            'cooldown': self.params.cooldown,
                            'take_profit': self.params.take_profit,
                            'stop_loss': self.params.stop_loss,
                            'max_hold': self.params.max_hold,
                            'momentum_short_threshold': self.params.momentum_short_threshold,
                            'momentum_med_threshold': self.params.momentum_med_threshold,
                        }
                        logger.info(f"Updated live params from evolved model (fitness={new_params.fitness:.1f})")
                        logger.info(f"New TP={new_params.take_profit:.1f} SL={new_params.stop_loss:.1f} "
                                  f"Hold={new_params.max_hold}s Cool={new_params.cooldown}s")
                    
                    # Save via versioning system
                    if VERSIONING_AVAILABLE:
                        vs = ModelVersioningSystem()
                        params_dict = {
                            'take_profit': new_params.take_profit,
                            'stop_loss': new_params.stop_loss,
                            'max_hold': new_params.max_hold,
                            'cooldown': new_params.cooldown,
                            'momentum_short_period': new_params.momentum_short_period,
                            'momentum_med_period': new_params.momentum_med_period,
                            'momentum_short_threshold': new_params.momentum_short_threshold,
                            'momentum_med_threshold': new_params.momentum_med_threshold,
                            'fitness': min(new_params.fitness, MAX_FITNESS_CAP),
                            'use_api_filter': getattr(new_params, 'use_api_filter', True),
                            'min_signal_age': getattr(new_params, 'min_signal_age', 0),
                            'lookback_short': getattr(new_params, 'lookback_short', 5),
                            'lookback_med': getattr(new_params, 'lookback_med', 12),
                            'max_momentum_age': getattr(new_params, 'max_momentum_age', 10),
                            'trail_activation_profit': getattr(new_params, 'trail_activation_profit', DEFAULT_TRAIL_ACTIVATION),
                            'trail_distance': getattr(new_params, 'trail_distance', DEFAULT_TRAIL_DISTANCE)
                        }
                        metrics_dict = {
                            'total_profit': new_params.metrics.total_profit,
                            'profit_factor': new_params.metrics.profit_factor,
                            'win_rate': new_params.metrics.win_rate,
                            'max_drawdown': new_params.metrics.max_drawdown,
                            'regime_scores': new_params.metrics.regime_scores
                        }
                        logger.info(f"Fitness before save: {new_params.fitness:.1f} (cap={MAX_FITNESS_CAP}) "
                                   f"PF={new_params.metrics.profit_factor:.2f} WR={new_params.metrics.win_rate:.1%} "
                                   f"Trades={new_params.metrics.total_trades}")
                        saved, version_id = vs.save_model(params_dict, metrics_dict, validation['passed'])
                        if saved:
                            logger.info(f"Model saved as {version_id} via versioning system")
                            stats = vs.get_stats()
                            logger.info(f"Total improvements: {stats['total_improvements']}, "
                                      f"Archived models: {stats['archived_count']}")
                    else:
                        os.makedirs(MODEL_DIR, exist_ok=True)
                        with open(ROBUST_PARAMS_FILE, 'w') as f:
                            json.dump(new_params.to_dict(), f, indent=2)
                        logger.info(f"Saved to {ROBUST_PARAMS_FILE}")
                
                # Update GUI with new model info
                if self.gui:
                    try:
                        vs = ModelVersioningSystem()
                        stats = vs.get_stats()
                        self.gui.update_model(
                            stats.get('current_best', 'v1'),
                            stats.get('best_fitness', 0),
                            datetime.datetime.now().strftime("%H:%M"),
                            stats.get('total_improvements', 0)
                        )
                    except:
                        pass
                return
            except Exception as e:
                logger.error(f"Robust GA failed: {e}, falling back to simple GA")
                import traceback
                logger.error(traceback.format_exc())
        
        # Fallback to simple GA
        candles = fetch_hist()
        if not candles:
            return
        new_params = evolve_params(candles)
        # FIX: Align fallback GA safety caps with BTCUSD volatility requirements
        new_params.take_profit = max(80.0, min(new_params.take_profit, 250.0))
        new_params.stop_loss = max(100.0, min(new_params.stop_loss, 250.0))
        new_params.cooldown = max(new_params.cooldown, COOLDOWN_MIN)
        if new_params.fitness > self.params.fitness:
            logger.info(f"IMPROVED! ${self.params.fitness:.2f} -> ${new_params.fitness:.2f}")
            self.params = new_params
            self.model_count += 1
            # CRITICAL FIX: Keep base_params in sync with new evolved params
            self.base_params = {
                'cooldown': self.params.cooldown,
                'take_profit': self.params.take_profit,
                'stop_loss': self.params.stop_loss,
                'max_hold': self.params.max_hold,
                'momentum_short_threshold': self.params.momentum_short_threshold,
                'momentum_med_threshold': self.params.momentum_med_threshold,
            }
            save_params(self.params)
        else:
            logger.info(f"No improvement ({new_params.fitness:.2f})")

    def _status(self, live):
        balance = self.mt5.get_balance()
        equity = self.mt5.get_equity()
        wr = (self.wins / max(1, self.total_trades)) * 100
        avg = self.total_pnl / max(1, self.total_trades)
        
        trade_info = "No trade"
        if self.mt5.has_open_position():
            pos = self.mt5.get_position_details()
            if pos:
                trade_info = f"{pos['direction'].upper()} PnL=${pos['profit']:.2f}"
        
        model_info = f"v{self.model_count}"
        if VERSIONING_AVAILABLE:
            vs = ModelVersioningSystem()
            stats = vs.get_stats()
            if stats['current_best']:
                model_info = stats['current_best']
        
        mode_info = ""
        if self.learning:
            mode_info = f"[{self.learning.current_mode}] "
            insights = self.learning.get_pattern_insights()
            if insights.get('best_direction'):
                mode_info += f"BestDir={insights['best_direction']} "
        
        logger.info(f"--- [{ts()}] --- {mode_info}\n{self.last_signal}\n"
                    f"Bal=${balance:.2f} Eq=${equity:.2f} | {trade_info}\n"
                    f"Model={model_info} Trades={self.total_trades} Win={wr:.0f}% Avg=${avg:.3f} "
                    f"Daily=${self.mt5.daily_pnl:.2f} LossStreak={self.mt5.consecutive_losses}")
        
        if self.learning and self.total_trades % 10 == 0 and self.total_trades > 0:
            logger.info(self.learning.get_summary())

    def _shutdown(self):
        if self.gui:
            try:
                self.gui.destroy()
            except:
                pass
        # Do NOT close open positions on shutdown - let SL/TP handle them or manage manually
        if self.mt5.has_open_position():
            pos = self.mt5.get_position_details()
            if pos:
                logger.warning(f"Open position remains: {pos['direction'].upper()} @ {pos['entry_price']:.2f} "
                              f"PnL=${pos['profit']:.2f} - Position will be managed by MT5 SL/TP")
        final_balance = self.mt5.get_balance()
        logger.info(f"Final balance: ${final_balance:.2f}")
        self.mt5.disconnect()


# ========== MAIN ==========
if __name__ == "__main__":
    print("\n" + "="*60)
    print("BTC Scalping Bot - MT5 Live Trading")
    print("="*60)
    print("\nRequirements:")
    print("1. MetaTrader 5 must be OPEN and LOGGED IN")
    print("2. Go to Tools > Options > Expert Advisors")
    print("3. Enable 'Allow Automated Trading'")
    print("4. Make sure BTCUSD is visible in Market Watch")
    print("\nPress Ctrl+C to stop the bot\n")
    print("="*60 + "\n")
    
    # Disabled: deleting params on every startup prevents learning persistence.
    # Uncomment below only if you explicitly want a fresh start every time.
    # if os.path.exists(PARAMS_FILE):
    #     os.remove(PARAMS_FILE)
    
    bot = LiveBot()
    bot.run()