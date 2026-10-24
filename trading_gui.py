"""
BTC Scalping Bot - Trading GUI (Standalone)
=====================================
Tkinter-based visual interface for real-time trading metrics.
Can connect directly to MT5 for data.
"""
import tkinter as tk
from tkinter import ttk
import time
import threading
from datetime import datetime

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

MT5_SYMBOL = "BTCUSD"

class TradingGUI:
    """Lightweight Tkinter GUI showing real-time trading metrics."""
    
    def __init__(self, bot_instance=None):
        self.bot = bot_instance
        
        self.root = tk.Tk()
        self.root.title("BTC Scalping Bot v3.0")
        self.root.geometry("480x520")
        self.root.resizable(False, False)
        self.root.configure(bg='#1a1a2e')
        
        # Connect to MT5 directly if no bot instance
        self.mt5 = None
        self.mt5_connected = False
        if not self.bot and MT5_AVAILABLE:
            self._connect_mt5()
        
        self._setup_styles()
        self._create_widgets()
        
        self.update_thread = None
        self.running = True
        self._last_trades = []
        
    def _connect_mt5(self):
        """Connect directly to MT5 with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            if mt5.initialize():
                self.mt5_connected = True
                self.mt5 = mt5
                print("GUI connected to MT5 successfully")
                # Verify connection by getting account info
                info = mt5.account_info()
                if info:
                    print(f"MT5 Account: #{info.login} Balance: ${info.balance:.2f}")
                else:
                    print("MT5 connected but could not get account info")
                return
            else:
                error = mt5.last_error()
                print(f"GUI MT5 connection attempt {attempt+1}/{max_retries} failed: {error}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    # Make sure MT5 is shut down before retrying
                    mt5.shutdown()
        
        self.mt5_connected = False
        self.mt5 = None
        print(f"GUI MT5: Could not connect after {max_retries} attempts.")
        print("Make sure MT5 is running and logged in.")
        
    def _setup_styles(self):
        self.bg_color = '#1a1a2e'
        self.panel_color = '#16213e'
        self.green = '#00cc66'
        self.red = '#ff4444'
        self.yellow = '#ffcc00'
        self.blue = '#4da6ff'
        self.white = '#ffffff'
        self.gray = '#888888'
        self.bold_font = ('Consolas', 11, 'bold')
        self.normal_font = ('Consolas', 10)
        self.small_font = ('Consolas', 9)
        
    def _create_widgets(self):
        self.root.columnconfigure(0, weight=1)
        
        header = tk.Frame(self.root, bg=self.blue, pady=8)
        header.pack(fill='x')
        
        title_label = tk.Label(header, text="BTC SCALPING BOT", font=('Consolas', 14, 'bold'),
               fg=self.white, bg=self.blue)
        title_label.pack(side='left', padx=15)
        
        self.mode_label = tk.Label(header, text="[MODE: INIT]",
                                font=('Consolas', 10, 'bold'),
                                fg=self.yellow, bg=self.blue)
        self.mode_label.pack(side='right', padx=15)
        
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        top_row = tk.Frame(main_frame, bg=self.bg_color)
        top_row.pack(fill='x')
        
        market_frame = self._create_market_panel(top_row)
        market_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        pos_perf_row = tk.Frame(top_row, bg=self.bg_color)
        pos_perf_row.pack(side='left', fill='both', expand=True, padx=(5, 0))
        
        position_frame = self._create_position_panel(pos_perf_row)
        position_frame.pack(fill='both', expand=True, pady=(0, 5))
        
        performance_frame = self._create_performance_panel(pos_perf_row)
        performance_frame.pack(fill='both', expand=True, pady=(5, 0))
        
        model_trades_row = tk.Frame(main_frame, bg=self.bg_color)
        model_trades_row.pack(fill='x', pady=(10, 0))
        
        model_frame = self._create_model_panel(model_trades_row)
        model_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        trades_frame = self._create_trades_panel(model_trades_row)
        trades_frame.pack(side='left', fill='x', expand=True, padx=(5, 0))
        
    def _create_panel(self, parent, title=""):
        frame = tk.Frame(parent, bg=self.panel_color, padx=10, pady=8)
        if title:
            tk.Label(frame, text=title, font=('Consolas', 9, 'bold'),
                   fg=self.gray, bg=self.panel_color).pack(anchor='w')
            tk.Frame(frame, bg=self.gray, height=1).pack(fill='x', pady=(2, 6))
        return frame
    
    def _create_market_panel(self, parent):
        outer = tk.Frame(parent, bg=self.panel_color)
        
        title = tk.Label(outer, text="MARKET", font=('Consolas', 9, 'bold'),
                        fg=self.gray, bg=self.panel_color)
        title.pack(anchor='w')
        tk.Frame(outer, bg=self.gray, height=1).pack(fill='x', pady=(2, 6))
        
        inner = tk.Frame(outer, bg=self.panel_color)
        inner.pack(fill='both', expand=True)
        
        self.bid_label = tk.Label(inner, text="Bid: ---", font=self.bold_font,
                                fg=self.white, bg=self.panel_color, anchor='w')
        self.bid_label.pack(fill='x')
        
        self.ask_label = tk.Label(inner, text="Ask: ---", font=self.bold_font,
                                fg=self.white, bg=self.panel_color, anchor='w')
        self.ask_label.pack(fill='x')
        
        self.spread_label = tk.Label(inner, text="Spread: ---", font=self.small_font,
                                   fg=self.gray, bg=self.panel_color, anchor='w')
        self.spread_label.pack(fill='x')
        
        tk.Frame(inner, bg=self.bg_color, height=6).pack()
        
        self.api_label = tk.Label(inner, text="API: ---", font=self.normal_font,
                                 fg=self.blue, bg=self.panel_color, anchor='w')
        self.api_label.pack(fill='x')
        
        self.smom_label = tk.Label(inner, text="S-Mom: ---", font=self.normal_font,
                                 fg=self.white, bg=self.panel_color, anchor='w')
        self.smom_label.pack(fill='x')
        
        self.mmom_label = tk.Label(inner, text="M-Mom: ---", font=self.normal_font,
                                 fg=self.white, bg=self.panel_color, anchor='w')
        self.mmom_label.pack(fill='x')
        
        self.score_label = tk.Label(inner, text="Score: ---", font=self.normal_font,
                                  fg=self.yellow, bg=self.panel_color, anchor='w')
        self.score_label.pack(fill='x')
        
        outer.pack(fill='both', expand=True)
        return outer
    
    def _create_position_panel(self, parent):
        outer = tk.Frame(parent, bg=self.panel_color)
        
        title = tk.Label(outer, text="POSITION", font=('Consolas', 9, 'bold'),
                        fg=self.gray, bg=self.panel_color)
        title.pack(anchor='w')
        tk.Frame(outer, bg=self.gray, height=1).pack(fill='x', pady=(2, 6))
        
        inner = tk.Frame(outer, bg=self.panel_color)
        inner.pack(fill='both', expand=True)
        
        self.pos_dir_label = tk.Label(inner, text="Direction: ---", font=self.bold_font,
                                     fg=self.white, bg=self.panel_color, anchor='w')
        self.pos_dir_label.pack(fill='x')
        
        self.pos_entry_label = tk.Label(inner, text="Entry: ---", font=self.normal_font,
                                      fg=self.gray, bg=self.panel_color, anchor='w')
        self.pos_entry_label.pack(fill='x')
        
        self.pos_pnl_label = tk.Label(inner, text="PnL: $0.00", font=self.bold_font,
                                     fg=self.white, bg=self.panel_color, anchor='w')
        self.pos_pnl_label.pack(fill='x')
        
        self.pos_held_label = tk.Label(inner, text="Held: 0s / 0s", font=self.small_font,
                                       fg=self.gray, bg=self.panel_color, anchor='w')
        self.pos_held_label.pack(fill='x')
        
        self.pos_tp_label = tk.Label(inner, text="TP: ---", font=self.small_font,
                                     fg=self.green, bg=self.panel_color, anchor='w')
        self.pos_tp_label.pack(fill='x')
        
        self.pos_sl_label = tk.Label(inner, text="SL: ---", font=self.small_font,
                                     fg=self.red, bg=self.panel_color, anchor='w')
        self.pos_sl_label.pack(fill='x')
        
        outer.pack(fill='both', expand=True)
        return outer
    
    def _create_performance_panel(self, parent):
        outer = tk.Frame(parent, bg=self.panel_color)
        
        title = tk.Label(outer, text="PERFORMANCE", font=('Consolas', 9, 'bold'),
                        fg=self.gray, bg=self.panel_color)
        title.pack(anchor='w')
        tk.Frame(outer, bg=self.gray, height=1).pack(fill='x', pady=(2, 6))
        
        inner = tk.Frame(outer, bg=self.panel_color)
        inner.pack(fill='both', expand=True)
        
        row1 = tk.Frame(inner, bg=self.panel_color)
        row1.pack(fill='x')
        tk.Label(row1, text="Balance:", font=self.small_font, fg=self.gray, bg=self.panel_color).pack(side='left')
        self.balance_label = tk.Label(row1, text="$0.00", font=self.bold_font,
                                       fg=self.green, bg=self.panel_color, anchor='e')
        self.balance_label.pack(side='right')
        
        row2 = tk.Frame(inner, bg=self.panel_color)
        row2.pack(fill='x')
        tk.Label(row2, text="Equity:", font=self.small_font, fg=self.gray, bg=self.panel_color).pack(side='left')
        self.equity_label = tk.Label(row2, text="$0.00", font=self.normal_font,
                                    fg=self.white, bg=self.panel_color, anchor='e')
        self.equity_label.pack(side='right')
        
        row3 = tk.Frame(inner, bg=self.panel_color)
        row3.pack(fill='x')
        tk.Label(row3, text="Daily PnL:", font=self.small_font, fg=self.gray, bg=self.panel_color).pack(side='left')
        self.daily_label = tk.Label(row3, text="$0.00", font=self.normal_font,
                                    fg=self.green, bg=self.panel_color, anchor='e')
        self.daily_label.pack(side='right')
        
        row4 = tk.Frame(inner, bg=self.panel_color)
        row4.pack(fill='x')
        tk.Label(row4, text="Trades:", font=self.small_font, fg=self.gray, bg=self.panel_color).pack(side='left')
        self.trades_label = tk.Label(row4, text="0", font=self.normal_font,
                                     fg=self.white, bg=self.panel_color, anchor='e')
        self.trades_label.pack(side='right')
        
        row5 = tk.Frame(inner, bg=self.panel_color)
        row5.pack(fill='x')
        tk.Label(row5, text="Win Rate:", font=self.small_font, fg=self.gray, bg=self.panel_color).pack(side='left')
        self.winrate_label = tk.Label(row5, text="0%", font=self.bold_font,
                                      fg=self.yellow, bg=self.panel_color, anchor='e')
        self.winrate_label.pack(side='right')
        
        row6 = tk.Frame(inner, bg=self.panel_color)
        row6.pack(fill='x')
        tk.Label(row6, text="Avg:", font=self.small_font, fg=self.gray, bg=self.panel_color).pack(side='left')
        self.avg_label = tk.Label(row6, text="$0.00", font=self.normal_font,
                                  fg=self.white, bg=self.panel_color, anchor='e')
        self.avg_label.pack(side='right')
        
        outer.pack(fill='both', expand=True)
        return outer
    
    def _create_model_panel(self, parent):
        outer = tk.Frame(parent, bg=self.panel_color)
        
        title = tk.Label(outer, text="MODEL", font=('Consolas', 9, 'bold'),
                        fg=self.gray, bg=self.panel_color)
        title.pack(anchor='w')
        tk.Frame(outer, bg=self.gray, height=1).pack(fill='x', pady=(2, 6))
        
        inner = tk.Frame(outer, bg=self.panel_color)
        inner.pack(fill='both', expand=True)
        
        row1 = tk.Frame(inner, bg=self.panel_color)
        row1.pack(fill='x')
        tk.Label(row1, text="Version:", font=self.small_font, fg=self.gray, bg=self.panel_color).pack(side='left')
        self.model_version_label = tk.Label(row1, text="v1", font=self.bold_font,
                                            fg=self.blue, bg=self.panel_color, anchor='e')
        self.model_version_label.pack(side='right')
        
        row2 = tk.Frame(inner, bg=self.panel_color)
        row2.pack(fill='x')
        tk.Label(row2, text="Fitness:", font=self.small_font, fg=self.gray, bg=self.panel_color).pack(side='left')
        self.fitness_label = tk.Label(row2, text="0.0", font=self.normal_font,
                                      fg=self.white, bg=self.panel_color, anchor='e')
        self.fitness_label.pack(side='right')
        
        row3 = tk.Frame(inner, bg=self.panel_color)
        row3.pack(fill='x')
        tk.Label(row3, text="Last Update:", font=self.small_font, fg=self.gray, bg=self.panel_color).pack(side='left')
        self.update_time_label = tk.Label(row3, text="--:--", font=self.small_font,
                                          fg=self.gray, bg=self.panel_color, anchor='e')
        self.update_time_label.pack(side='right')
        
        row4 = tk.Frame(inner, bg=self.panel_color)
        row4.pack(fill='x')
        tk.Label(row4, text="Improvements:", font=self.small_font, fg=self.gray, bg=self.panel_color).pack(side='left')
        self.improvements_label = tk.Label(row4, text="0", font=self.small_font,
                                          fg=self.yellow, bg=self.panel_color, anchor='e')
        self.improvements_label.pack(side='right')
        
        outer.pack(fill='x', pady=(10, 0))
        return outer
    
    def _create_trades_panel(self, parent):
        outer = tk.Frame(parent, bg=self.panel_color)
        
        title = tk.Label(outer, text="RECENT TRADES", font=('Consolas', 9, 'bold'),
                        fg=self.gray, bg=self.panel_color)
        title.pack(anchor='w')
        tk.Frame(outer, bg=self.gray, height=1).pack(fill='x', pady=(2, 6))
        
        self.trades_listbox = tk.Listbox(outer, font=self.small_font,
                                        bg=self.bg_color, fg=self.white,
                                        selectbackground=self.panel_color,
                                        selectforeground=self.white,
                                        borderwidth=0, highlightthickness=0,
                                        height=4)
        self.trades_listbox.pack(fill='x', pady=(5, 0))
        
        outer.pack(fill='x', pady=(10, 0))
        return outer
    
    def start(self):
        if self.update_thread is None or not self.update_thread.is_alive():
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
    
    def _update_loop(self):
        while self.running:
            try:
                self.root.after(0, self._refresh_display)
            except:
                pass
            time.sleep(0.5)
    
    def _refresh_display(self):
        try:
            if self.bot:
                self._update_from_bot()
            elif self.mt5_connected:
                self._update_from_mt5()
            else:
                # Show disconnected status
                self.mode_label.config(text="[MT5: DISCONNECTED]", fg=self.red)
            self.root.update_idletasks()
        except:
            pass
    
    def _update_from_mt5(self):
        """Update from direct MT5 connection"""
        if not self.mt5_connected or self.mt5 is None:
            # Try to reconnect
            if MT5_AVAILABLE:
                self._connect_mt5()
            return
        
        try:
            tick = self.mt5.symbol_info_tick(MT5_SYMBOL)
            if tick:
                self.bid_label.config(text=f"Bid: ${tick.bid:.2f}")
                self.ask_label.config(text=f"Ask: ${tick.ask:.2f}")
                self.spread_label.config(text=f"Spread: {tick.ask - tick.bid:.1f}")
            else:
                self.bid_label.config(text="Bid: ---")
                self.ask_label.config(text="Ask: ---")
                self.spread_label.config(text="Spread: ---")
            
            # Account info
            info = self.mt5.account_info()
            if info:
                self.balance_label.config(text=f"${info.balance:.2f}")
                self.equity_label.config(text=f"${info.equity:.2f}")
            
            # Positions - get ALL positions for the symbol (not filtered by magic)
            positions = self.mt5.positions_get(symbol=MT5_SYMBOL)
            if positions and len(positions) > 0:
                p = positions[0]
                dir_color = self.green if p.type == self.mt5.POSITION_TYPE_BUY else self.red
                direction = "BUY" if p.type == self.mt5.POSITION_TYPE_BUY else "SELL"
                self.pos_dir_label.config(text=f"Direction: {direction}", fg=dir_color)
                self.pos_entry_label.config(text=f"Entry: ${p.price_open:.2f}")
                if p.profit >= 0:
                    self.pos_pnl_label.config(text=f"PnL: +${p.profit:.2f}", fg=self.green)
                else:
                    self.pos_pnl_label.config(text=f"PnL: ${p.profit:.2f}", fg=self.red)
                # Show all positions count if multiple
                if len(positions) > 1:
                    self.pos_held_label.config(text=f"Positions: {len(positions)}")
            else:
                self.pos_dir_label.config(text="Direction: ---", fg=self.white)
                self.pos_entry_label.config(text="Entry: ---")
                self.pos_pnl_label.config(text="PnL: ---", fg=self.white)
                self.pos_held_label.config(text="Held: No position")
        except Exception as e:
            # MT5 connection may have been lost
            print(f"GUI MT5 update error: {e}")
            self.mt5_connected = False
            self.mt5 = None
    
    def _update_from_bot(self):
        try:
            tick = self.bot.mt5.get_tick()
            if tick:
                self.bid_label.config(text=f"Bid: ${tick.bid:.2f}")
                self.ask_label.config(text=f"Ask: ${tick.ask:.2f}")
                spread = tick.ask - tick.bid
                self.spread_label.config(text=f"Spread: {spread:.1f}")
                
                self.api_label.config(text=f"API: {int(self.bot.last_api_signal)}")
                
                mom_s = self.bot.momentum_short
                mom_m = self.bot.momentum_med
                mom_color = {1: self.green, -1: self.red, 0: self.gray}.get(mom_s, self.gray)
                self.smom_label.config(text=f"S-Mom: {mom_s:+d}", fg=mom_color)
                
                mom_color = {1: self.green, -1: self.red, 0: self.gray}.get(mom_m, self.gray)
                self.mmom_label.config(text=f"M-Mom: {mom_m:+d}", fg=mom_color)
                
                score = self.bot.entry_score
                score_color = self.green if score > 0.5 else (self.red if score < -0.5 else self.yellow)
                self.score_label.config(text=f"Score: {score:.2f}", fg=score_color)
            
            balance = self.bot.mt5.get_balance()
            equity = self.bot.mt5.get_equity()
            daily_pnl = self.bot.mt5.daily_pnl
            
            self.balance_label.config(text=f"${balance:.2f}")
            self.equity_label.config(text=f"${equity:.2f}")
            
            if daily_pnl >= 0:
                self.daily_label.config(text=f"+${daily_pnl:.2f}", fg=self.green)
            else:
                self.daily_label.config(text=f"${daily_pnl:.2f}", fg=self.red)
            
            self.trades_label.config(text=str(self.bot.total_trades))
            
            if self.bot.total_trades > 0:
                wr = (self.bot.wins / self.bot.total_trades) * 100
                avg = self.bot.total_pnl / self.bot.total_trades
                self.winrate_label.config(text=f"{wr:.0f}%")
                self.avg_label.config(text=f"${avg:.3f}")
            else:
                self.winrate_label.config(text="0%")
                self.avg_label.config(text="$0.00")
            
            if VERSIONING_AVAILABLE:
                vs = ModelVersioningSystem()
                stats = vs.get_stats()
                if stats['current_best']:
                    self.model_version_label.config(text=stats['current_best'])
                self.fitness_label.config(text=f"{stats['best_fitness']:.1f}")
                self.improvements_label.config(text=str(stats['total_improvements']))
            
            if self.bot.mt5.has_open_position():
                pos = self.bot.mt5.get_position_details()
                if pos:
                    dir_color = self.green if pos['direction'] == 'buy' else self.red
                    self.pos_dir_label.config(text=f"Direction: {pos['direction'].upper()}",
                                           fg=dir_color)
                    self.pos_entry_label.config(text=f"Entry: ${pos['entry_price']:.2f}")
                    
                    if pos['profit'] >= 0:
                        self.pos_pnl_label.config(text=f"PnL: +${pos['profit']:.2f}", fg=self.green)
                    else:
                        self.pos_pnl_label.config(text=f"PnL: ${pos['profit']:.2f}", fg=self.red)
                    
                    held = time.time() - self.bot.position_open_time if self.bot.position_open_time > 0 else 0
                    max_hold = self.bot.params.max_hold
                    self.pos_held_label.config(text=f"Held: {int(held)}s / {max_hold}s")
                    
                    tp = self.bot.params.take_profit
                    sl = self.bot.params.stop_loss
                    self.pos_tp_label.config(text=f"TP: {tp:.1f}")
                    self.pos_sl_label.config(text=f"SL: {sl:.1f}")
            else:
                self.pos_dir_label.config(text="Direction: ---", fg=self.white)
                self.pos_entry_label.config(text="Entry: ---")
                self.pos_pnl_label.config(text="PnL: ---", fg=self.white)
                self.pos_held_label.config(text="Held: No position")
                self.pos_tp_label.config(text="TP: ---")
                self.pos_sl_label.config(text="SL: ---")
                
        except Exception as e:
            pass
    
    def update_market(self, bid, ask, api_signal, s_mom, m_mom, score=0):
        try:
            self.bid_label.config(text=f"Bid: ${bid:.2f}")
            self.ask_label.config(text=f"Ask: ${ask:.2f}")
            
            spread = ask - bid
            self.spread_label.config(text=f"Spread: {spread:.1f}")
            
            self.api_label.config(text=f"API: {int(api_signal)}")
            
            mom_color = {1: self.green, -1: self.red, 0: self.gray}.get(s_mom, self.gray)
            self.smom_label.config(text=f"S-Mom: {s_mom:+d}", fg=mom_color)
            
            mom_color = {1: self.green, -1: self.red, 0: self.gray}.get(m_mom, self.gray)
            self.mmom_label.config(text=f"M-Mom: {m_mom:+d}", fg=mom_color)
            
            score_color = self.green if score > 0.5 else (self.red if score < -0.5 else self.yellow)
            self.score_label.config(text=f"Score: {score:.2f}", fg=score_color)
            
        except:
            pass
    
    def update_position(self, direction, entry, current, pnl, held, max_hold, tp, sl):
        try:
            dir_color = self.green if direction == 'buy' else self.red
            self.pos_dir_label.config(text=f"Direction: {direction.upper()}", fg=dir_color)
            self.pos_entry_label.config(text=f"Entry: ${entry:.2f}")
            
            if pnl >= 0:
                self.pos_pnl_label.config(text=f"PnL: +${pnl:.2f}", fg=self.green)
            else:
                self.pos_pnl_label.config(text=f"PnL: ${pnl:.2f}", fg=self.red)
            
            self.pos_held_label.config(text=f"Held: {int(held)}s / {max_hold}s")
            self.pos_tp_label.config(text=f"TP: {tp:.1f}")
            self.pos_sl_label.config(text=f"SL: {sl:.1f}")
            
        except:
            pass
    
    def update_performance(self, balance, equity, daily, trades, wins, avg, wr):
        try:
            self.balance_label.config(text=f"${balance:.2f}")
            self.equity_label.config(text=f"${equity:.2f}")
            
            if daily >= 0:
                self.daily_label.config(text=f"+${daily:.2f}", fg=self.green)
            else:
                self.daily_label.config(text=f"${daily:.2f}", fg=self.red)
            
            self.trades_label.config(text=str(trades))
            self.winrate_label.config(text=f"{wr:.0f}%")
            self.avg_label.config(text=f"${avg:.3f}")
            
        except:
            pass
    
    def update_model(self, version, fitness, update_time, improvements):
        try:
            self.model_version_label.config(text=version)
            self.fitness_label.config(text=f"{fitness:.1f}")
            self.update_time_label.config(text=update_time)
            self.improvements_label.config(text=str(improvements))
            
        except:
            pass
    
    def record_trade(self, direction, pnl, held):
        try:
            if pnl >= 0:
                line = f"BUY  " if direction == 'buy' else "SELL "
                line += f"@ ${pnl:.2f} ({int(held)}s)"
                color = self.green
            else:
                line = f"BUY  " if direction == 'buy' else "SELL "
                line += f"@ ${pnl:.2f} ({int(held)}s)"
                color = self.red
            
            self.trades_listbox.insert(0, line)
            
            if self.trades_listbox.size() > 10:
                self.trades_listbox.delete(10)
            
            self._last_trades.insert(0, {'dir': direction, 'pnl': pnl, 'held': held, 'time': time.time()})
            if len(self._last_trades) > 20:
                self._last_trades.pop()
                
        except:
            pass
    
    def set_mode(self, mode):
        mode_text = {
            'RECONNAISSANCE': '[MODE: RECON]',
            'SCALPING': '[MODE: SCALPING]',
            'AGGRESSIVE': '[MODE: AGGRESSIVE]',
            'DEFENSIVE': '[MODE: DEFENSIVE]'
        }.get(mode, f'[MODE: {mode}]')
        
        mode_color = {
            'RECONNAISSANCE': self.yellow,
            'SCALPING': self.green,
            'AGGRESSIVE': self.red,
            'DEFENSIVE': self.blue
        }.get(mode, self.yellow)
        
        self.mode_label.config(text=mode_text, fg=mode_color)
    
    def set_learning(self, active):
        if active:
            self.mode_label.config(text=self.mode_label.cget('text') + ' [LEARNING]')
        else:
            text = self.mode_label.cget('text').replace(' [LEARNING]', '')
            self.mode_label.config(text=text)
    
    def shutdown(self):
        """Gracefully shut down the GUI and clean up resources."""
        self.running = False
        # Wait for update thread to finish
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
        # Disconnect from MT5
        if self.mt5_connected and self.mt5 is not None:
            try:
                self.mt5.shutdown()
                print("GUI: MT5 connection closed.")
            except:
                pass
            self.mt5_connected = False
            self.mt5 = None
        # Destroy the window
        try:
            self.root.destroy()
        except:
            pass

    def destroy(self):
        self.running = False
        try:
            self.root.destroy()
        except:
            pass


def start_gui(bot_instance=None):
    gui = TradingGUI(bot_instance)
    gui.root.protocol("WM_DELETE_WINDOW", gui.shutdown)
    gui.start()
    return gui


if __name__ == "__main__":
    print("Starting standalone GUI test...")
    gui = TradingGUI()
    
    # Handle window close button (X) gracefully
    gui.root.protocol("WM_DELETE_WINDOW", gui.shutdown)
    
    if gui.mt5_connected:
        print("MT5 connected - starting live data updates")
        gui.start()
    else:
        print("MT5 not connected - showing demo data")
        gui.mode_label.config(text="[MODE: DEMO]", fg=gui.yellow)
        gui.update_market(77423.50, 77424.50, -2, 1, 0, 0.72)
        gui.update_position('buy', 77420.00, 77422.00, 2.10, 45, 120, 22.0, 4.5)
        gui.update_performance(203.45, 205.55, 3.45, 47, 32, 0.52, 68)
        gui.update_model('v3_143205', 1850.2, '14:30', 3)
        gui.record_trade('buy', 2.10, 45)
        gui.record_trade('sell', -0.50, 89)
        gui.record_trade('buy', 1.20, 67)
        gui.set_mode('SCALPING')
    
    try:
        gui.root.mainloop()
    except KeyboardInterrupt:
        print("\nShutting down gracefully (Ctrl+C)...")
        gui.shutdown()
