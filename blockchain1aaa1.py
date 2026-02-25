from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Optional
import time
import traceback
import requests
import panel as pn
import json
import os

pn.extension()

# ----------------------------
# Helpers
# ----------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def fmt_ts(dt: Optional[datetime]) -> str:
    return dt.isoformat() if dt else "n/a"

def safe_add_next_tick(fn) -> None:
    """
    UI updates must happen on the Bokeh/Panel document thread.
    """
    try:
        doc = pn.state.curdoc
        if doc is not None:
            doc.add_next_tick_callback(fn)
            return
    except Exception:
        pass
    try:
        fn()
    except Exception:
        pass

def log_line(log_pane: pn.pane.Markdown, line: str) -> None:
    def _do():
        log_pane.object = (log_pane.object or "") + line + "\n"
    safe_add_next_tick(_do)

def set_status(pane: pn.pane.Markdown, md: str) -> None:
    def _do():
        pane.object = md
    safe_add_next_tick(_do)

# ----------------------------
# Config + State
# ----------------------------
@dataclass(frozen=True)
class LivePriceConfig:
    poll_seconds: int = 5
    timeout_seconds: int = 10
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15"
    )
    coinbase_url: str = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
    binance_url: str = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    coingecko_url: str = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"

@dataclass(frozen=True)
class AtrConfig:
    base_url: str = "https://data-api.binance.vision"
    symbol: str = "BTCUSDT"
    interval: str = "1d"
    limit: int = 60
    period: int = 14

@dataclass
class DcaParams:
    buy_usd: float
    drop_pct: float

@dataclass
class TradingRules:
    budget_usd: float
    dca: DcaParams

@dataclass
class Portfolio:
    cash_usd: float
    btc_dca: float
    ref_price: Optional[float]
    buys: int

@dataclass
class AtrTrade:
    open: bool
    entry: Optional[float]
    stop: Optional[float]
    k: float
    usd: float
    btc: float
    realized_pnl: float

@dataclass
class State:
    rules: TradingRules
    portfolio: Portfolio
    atr_trade: AtrTrade
    last_price: Optional[float]
    last_source: Optional[str]
    last_ts: Optional[datetime]
    lock: Lock
    pause_dca: Event
    stop: Event

def build_state() -> State:
    return State(
        rules=TradingRules(budget_usd=0.0, dca=DcaParams(buy_usd=0.0, drop_pct=0.0)),
        portfolio=Portfolio(cash_usd=0.0, btc_dca=0.0, ref_price=None, buys=0),
        atr_trade=AtrTrade(open=False, entry=None, stop=None, k=1.5, usd=0.0, btc=0.0, realized_pnl=0.0),
        last_price=None,
        last_source=None,
        last_ts=None,
        lock=Lock(),
        pause_dca=Event(),
        stop=Event(),
    )

# ----------------------------
# Market data
# ----------------------------
def _get_json(session: requests.Session, url: str, cfg: LivePriceConfig) -> dict:
    headers = {"User-Agent": cfg.user_agent, "Accept": "application/json"}
    r = session.get(url, headers=headers, timeout=cfg.timeout_seconds)
    r.raise_for_status()
    return r.json()

def fetch_coinbase(cfg: LivePriceConfig, session: requests.Session) -> float:
    data = _get_json(session, cfg.coinbase_url, cfg)
    return float(data["data"]["amount"])

def fetch_binance(cfg: LivePriceConfig, session: requests.Session) -> float:
    data = _get_json(session, cfg.binance_url, cfg)
    return float(data["price"])

def fetch_coingecko(cfg: LivePriceConfig, session: requests.Session) -> float:
    data = _get_json(session, cfg.coingecko_url, cfg)
    return float(data["bitcoin"]["usd"])

def fetch_live_price(cfg: LivePriceConfig, session: requests.Session) -> tuple[float, str]:
    try:
        return fetch_coinbase(cfg, session), "coinbase"
    except Exception:
        pass
    try:
        return fetch_binance(cfg, session), "binance"
    except Exception:
        pass
    return fetch_coingecko(cfg, session), "coingecko"

def fetch_binance_hlc(atr_cfg: AtrConfig) -> list[tuple[float, float, float]]:
    url = f"{atr_cfg.base_url}/api/v3/klines"
    params = {"symbol": atr_cfg.symbol, "interval": atr_cfg.interval, "limit": atr_cfg.limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    out: list[tuple[float, float, float]] = []
    for row in r.json():
        high = float(row[2])
        low = float(row[3])
        close = float(row[4])
        out.append((high, low, close))
    return out

def atr_from_hlc(hlc: list[tuple[float, float, float]], period: int) -> float:
    if len(hlc) < period + 1:
        raise RuntimeError("Not enough candles for ATR")
    prev_close = hlc[0][2]
    trs: list[float] = []
    for (h, l, c) in hlc[1:]:
        trs.append(max(h - l, abs(h - prev_close), abs(l - prev_close)))
        prev_close = c
    window = trs[-period:]
    return sum(window) / float(period)

# ----------------------------
# Trading logic
# ----------------------------
def maybe_dca(state: State, price: float) -> Optional[str]:
    p = state.portfolio
    d = state.rules.dca
    if p.ref_price is None:
        p.ref_price = price
        return None
    ref = p.ref_price
    drop = ((ref - price) / ref) * 100.0
    if drop < d.drop_pct:
        return None
    buy_usd = min(d.buy_usd, p.cash_usd)
    if buy_usd <= 0:
        p.ref_price = price
        return "DCA trigger hit but cash is 0"
    buy_btc = buy_usd / price
    p.cash_usd -= buy_usd
    p.btc_dca += buy_btc
    p.buys += 1
    p.ref_price = price
    return f"DCA BUY drop={drop:.2f}% usd={buy_usd:.2f} btc={buy_btc:.8f} price={price:.2f} cash={p.cash_usd:.2f}"

def atr_open_trade(state: State, atr_cfg: AtrConfig, usd: float, k: float) -> str:
    t = state.atr_trade
    if t.open:
        return "ATR trade already open"
    price = state.last_price
    if price is None:
        return "No live price yet"
    hlc = fetch_binance_hlc(atr_cfg)
    atr_val = atr_from_hlc(hlc, atr_cfg.period)
    stop = price - (k * atr_val)
    btc = usd / price
    t.open = True
    t.entry = price
    t.stop = stop
    t.k = k
    t.usd = usd
    t.btc = btc
    return f"ATR OPEN entry={price:.2f} ATR={atr_val:.2f} k={k:.2f} stop={stop:.2f} usd={usd:.2f} btc={btc:.8f}"

def atr_close_trade(state: State, reason: str) -> str:
    t = state.atr_trade
    if not t.open:
        return "No ATR trade open"
    price = state.last_price
    if price is None:
        return "No live price yet"
    entry = float(t.entry or 0.0)
    pnl = (price - entry) * float(t.btc)
    t.realized_pnl += pnl
    t.open = False
    t.entry = None
    t.stop = None
    t.usd = 0.0
    t.btc = 0.0
    return f"ATR CLOSE reason={reason} exit={price:.2f} pnl={pnl:.2f} realized={t.realized_pnl:.2f}"

# ----------------------------
# LLM Tool Functions
# ----------------------------

def get_portfolio_value(state: State) -> dict:
    """
    Calculate current portfolio value and breakdown.
    Returns: dict with cash, btc_holdings, current_value, total_value
    """
    with state.lock:
        cash = state.portfolio.cash_usd
        btc = state.portfolio.btc_dca
        price = state.last_price or 0.0
        btc_value = btc * price
        total = cash + btc_value
        
    return {
        "cash_usd": round(cash, 2),
        "btc_holdings": round(btc, 8),
        "btc_value_usd": round(btc_value, 2),
        "total_value_usd": round(total, 2),
        "current_btc_price": round(price, 2)
    }

def get_dca_performance(state: State) -> dict:
    """
    Calculate DCA strategy performance metrics.
    Returns: dict with total_invested, current_value, profit_loss, roi_percentage
    """
    with state.lock:
        budget = state.rules.budget_usd
        cash = state.portfolio.cash_usd
        btc = state.portfolio.btc_dca
        price = state.last_price or 0.0
        buys = state.portfolio.buys
        
    invested = budget - cash
    current_value = btc * price
    pnl = current_value - invested
    roi = (pnl / invested * 100) if invested > 0 else 0.0
    
    return {
        "total_invested_usd": round(invested, 2),
        "current_value_usd": round(current_value, 2),
        "profit_loss_usd": round(pnl, 2),
        "roi_percentage": round(roi, 2),
        "number_of_buys": buys
    }

def get_atr_performance(state: State) -> dict:
    """
    Calculate ATR trading strategy performance.
    Returns: dict with trade_status, unrealized_pnl, realized_pnl, total_pnl
    """
    with state.lock:
        is_open = state.atr_trade.open
        entry = state.atr_trade.entry
        btc = state.atr_trade.btc
        realized = state.atr_trade.realized_pnl
        price = state.last_price or 0.0
        
    unrealized = 0.0
    if is_open and entry:
        unrealized = (price - entry) * btc
        
    total_pnl = realized + unrealized
    
    return {
        "trade_open": is_open,
        "unrealized_pnl_usd": round(unrealized, 2),
        "realized_pnl_usd": round(realized, 2),
        "total_pnl_usd": round(total_pnl, 2),
        "current_btc_price": round(price, 2)
    }

def compare_strategies(state: State) -> dict:
    """
    Compare DCA vs ATR strategy performance.
    Returns: dict comparing both strategies with recommendation
    """
    dca = get_dca_performance(state)
    atr = get_atr_performance(state)
    
    dca_pnl = dca["profit_loss_usd"]
    atr_pnl = atr["total_pnl_usd"]
    
    if dca_pnl > atr_pnl:
        better_strategy = "DCA"
        difference = dca_pnl - atr_pnl
    else:
        better_strategy = "ATR"
        difference = atr_pnl - dca_pnl
    
    return {
        "dca_profit_loss": dca_pnl,
        "atr_profit_loss": atr_pnl,
        "better_strategy": better_strategy,
        "profit_difference_usd": round(difference, 2),
        "dca_roi_percentage": dca["roi_percentage"]
    }

def simulate_dca_change(state: State, new_drop_pct: float, new_buy_usd: float) -> dict:
    """
    Simulate what would happen with different DCA parameters.
    Returns: dict with simulation results
    """
    with state.lock:
        current_drop = state.rules.dca.drop_pct
        current_buy = state.rules.dca.buy_usd
        
    return {
        "current_drop_pct": current_drop,
        "current_buy_usd": current_buy,
        "proposed_drop_pct": new_drop_pct,
        "proposed_buy_usd": new_buy_usd,
        "recommendation": f"Lowering drop % from {current_drop}% to {new_drop_pct}% will trigger more frequent buys. Increasing buy amount from ${current_buy} to ${new_buy_usd} will deploy capital faster but with fewer total buys."
    }

# ----------------------------
# Telegram Notifications
# ----------------------------

class TelegramNotifier:
    """Send trading notifications via Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
    def send_message(self, message: str) -> bool:
        """Send a message to Telegram (non-blocking)"""
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(self.base_url, json=payload, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram send error: {e}")
            return False
    
    def send_async(self, message: str):
        """Send message in background thread to avoid blocking"""
        Thread(target=self.send_message, args=(message,), daemon=True).start()

def load_telegram_config() -> tuple[Optional[str], Optional[str]]:
    """Load Telegram bot token and chat ID from telegram_config.txt"""
    config_file = "telegram_config.txt"
    if not os.path.exists(config_file):
        return None, None
    
    bot_token = None
    chat_id = None
    
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("BOT_TOKEN="):
                    bot_token = line.split("=", 1)[1].strip()
                elif line.startswith("CHAT_ID="):
                    chat_id = line.split("=", 1)[1].strip()
        
        if bot_token and chat_id:
            return bot_token, chat_id
    except Exception as e:
        print(f"Error loading Telegram config: {e}")
    
    return None, None

# ----------------------------
# LLM Integration - Using Google Gemini
# ----------------------------

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class TradingLLM:
    """
    Google Gemini LLM integration for trading strategy suggestions.
    Uses tool-calling to analyze portfolio and provide recommendations.
    """
    
    def __init__(self, api_key: str):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Try multiple model names for compatibility
        model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro', 'gemini-pro']
        self.model = None
        
        for model_name in model_names:
            try:
                self.model = genai.GenerativeModel(model_name)
                # Test the model with minimal query
                self.model.generate_content("test", generation_config={"max_output_tokens": 1})
                break
            except Exception:
                self.model = None
                continue
        
        # Dynamic fallback: find any supported model
        if self.model is None:
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        self.model = genai.GenerativeModel(m.name)
                        break
            except Exception as e:
                raise RuntimeError(f"Failed to initialize any Gemini model: {e}")
        
        if not self.model:
            raise RuntimeError("No compatible Gemini models found for this API key.")
    
    def query_gemini(self, prompt: str) -> str:
        """Query Google Gemini API"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error querying Gemini: {str(e)}"
    
    def answer_query(self, user_query: str, state: State) -> str:
        """
        Process user query by calling appropriate tool functions and generating response.
        """
        user_query_lower = user_query.lower()
        
        # Determine which tools to call based on query
        context_data = {}
        
        if "portfolio" in user_query_lower or "value" in user_query_lower or "total" in user_query_lower:
            context_data["portfolio"] = get_portfolio_value(state)
            
        if "dca" in user_query_lower:
            context_data["dca_performance"] = get_dca_performance(state)
            
        if "atr" in user_query_lower:
            context_data["atr_performance"] = get_atr_performance(state)
            
        if "compare" in user_query_lower or "better" in user_query_lower or "which" in user_query_lower or "profitable" in user_query_lower:
            context_data["comparison"] = compare_strategies(state)
            
        if "change" in user_query_lower or "simulate" in user_query_lower:
            context_data["simulation"] = simulate_dca_change(state, 2.5, 600.0)
        
        # If no specific tool needed, get all data
        if not context_data:
            context_data["portfolio"] = get_portfolio_value(state)
            context_data["dca_performance"] = get_dca_performance(state)
            context_data["atr_performance"] = get_atr_performance(state)
            context_data["comparison"] = compare_strategies(state)
        
        # Build prompt with context
        prompt = self._build_prompt(user_query, context_data)
        
        # Get LLM response
        llm_response = self.query_gemini(prompt)
        
        # Format final response with data
        final_response = self._format_response(llm_response, context_data)
        
        return final_response
    
    def _build_prompt(self, query: str, data: dict) -> str:
        """Build prompt for LLM with trading data context"""
        prompt = f"""You are an expert Bitcoin trading strategy advisor. 

User Question: {query}

Trading Data:
{json.dumps(data, indent=2)}

Instructions:
1. Analyze the trading data carefully
2. Focus on profit/loss metrics
3. Recommend which strategy (DCA or ATR) is performing better
4. Keep your response under 100 words
5. Be specific and data-driven
6. Use simple language

Provide your trading recommendation:"""
        return prompt
    
    def _format_response(self, llm_text: str, data: dict) -> str:
        """Format LLM response with actual data"""
        response = "### 🤖 AI Trading Advisor Response\n\n"
        response += f"**Strategy Recommendation:**\n{llm_text}\n\n"
        response += "---\n**Key Metrics:**\n"
        
        if "comparison" in data:
            comp = data["comparison"]
            response += f"- 📊 DCA Profit/Loss: **${comp['dca_profit_loss']:.2f}**\n"
            response += f"- 📈 ATR Profit/Loss: **${comp['atr_profit_loss']:.2f}**\n"
            response += f"- ✅ **Best Strategy: {comp['better_strategy']}** (${comp['profit_difference_usd']:.2f} higher)\n"
        
        if "portfolio" in data:
            port = data["portfolio"]
            response += f"- 💰 Total Portfolio Value: **${port['total_value_usd']:.2f}**\n"
            response += f"- ₿ BTC Price: **${port['current_btc_price']:.2f}**\n"
            
        if "dca_performance" in data:
            dca = data["dca_performance"]
            response += f"- 📉 DCA ROI: **{dca['roi_percentage']:.2f}%**\n"
            response += f"- 🔢 Number of DCA Buys: **{dca['number_of_buys']}**\n"
            
        return response

# ----------------------------
# Load API Token
# ----------------------------
def load_api_token() -> Optional[str]:
    """Load API token from hf_token.txt file (now works for Gemini too)"""
    token_file = "hf_token.txt"
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            token = f.read().strip()
            if token:
                return token
    return None

# ----------------------------
# App UI
# ----------------------------
price_cfg = LivePriceConfig()
atr_cfg = AtrConfig()
state = build_state()

# Initialize Telegram
telegram_bot_token, telegram_chat_id = load_telegram_config()
telegram_notifier = None

if telegram_bot_token and telegram_chat_id:
    telegram_notifier = TelegramNotifier(telegram_bot_token, telegram_chat_id)

# Initialize LLM
api_token = load_api_token()
trading_llm = None
llm_status_msg = ""

if api_token and GEMINI_AVAILABLE:
    try:
        trading_llm = TradingLLM(api_token)
        llm_status_msg = "### 🤖 AI Trading Advisor\n✅ **AI Powered by Google Gemini** - Ask me about trading strategies!"
    except Exception as e:
        llm_status_msg = f"### 🤖 AI Trading Advisor\n⚠️ **AI Setup Error**: {str(e)}"
elif not GEMINI_AVAILABLE:
    llm_status_msg = "### 🤖 AI Trading Advisor\n⚠️ **Missing Library** - Run: `pip install google-generativeai --break-system-packages`"
else:
    llm_status_msg = "### 🤖 AI Trading Advisor\n⚠️ **AI Disabled** - Add your Google Gemini API key to 'hf_token.txt'"

# UI Panes
status = pn.pane.Markdown("## LIVE BTC-USD\nStarting...", sizing_mode="stretch_width")
portfolio = pn.pane.Markdown("### Portfolio\nNot initialized", sizing_mode="stretch_width")
atr_box = pn.pane.Markdown("### ATR Trade\nNone", sizing_mode="stretch_width")
log_pane = pn.pane.Markdown("", sizing_mode="stretch_width")

# LLM Chat Panel
llm_chat_output = pn.pane.Markdown(llm_status_msg, sizing_mode="stretch_width", height=400, styles={'overflow-y': 'scroll', 'background': '#f0f8ff', 'padding': '15px', 'border-radius': '5px'})
llm_input = pn.widgets.TextInput(name="Ask AI Advisor", placeholder="e.g., Which strategy is more profitable?", sizing_mode="stretch_width")
llm_submit = pn.widgets.Button(name="🚀 Get AI Suggestion", button_type="success")

# Trading Controls
budget = pn.widgets.FloatInput(name="Budget USD", value=10000.0, step=100.0)
dca_buy = pn.widgets.FloatInput(name="DCA Buy USD", value=500.0, step=50.0)
dca_drop = pn.widgets.FloatInput(name="DCA Drop %", value=3.0, step=0.1)
apply_btn = pn.widgets.Button(name="Apply DCA + Reset", button_type="primary")
pause_btn = pn.widgets.Button(name="Pause DCA", button_type="warning")
resume_btn = pn.widgets.Button(name="Resume DCA", button_type="success")

atr_usd = pn.widgets.FloatInput(name="ATR Trade USD", value=500.0, step=50.0)
atr_k = pn.widgets.FloatInput(name="ATR k", value=1.5, step=0.1)
atr_open_btn = pn.widgets.Button(name="Open ATR Trade", button_type="primary")
atr_close_btn = pn.widgets.Button(name="Manual Close ATR", button_type="danger")
stop_btn = pn.widgets.Button(name="Stop App", button_type="danger")

def refresh_panels() -> None:
    with state.lock:
        price = state.last_price
        src = state.last_source
        ts = state.last_ts
        cash = state.portfolio.cash_usd
        btc = state.portfolio.btc_dca
        buys = state.portfolio.buys
        ref = state.portfolio.ref_price
        atr_open = state.atr_trade.open
        atr_entry = state.atr_trade.entry
        atr_stop = state.atr_trade.stop
        atr_real = state.atr_trade.realized_pnl

    if price is None:
        set_status(status, "## LIVE BTC-USD\nWaiting for first tick...")
    else:
        set_status(
            status,
            f"## LIVE BTC-USD\n**${price:,.2f}**  \nSource: {src}  \nUpdated: {fmt_ts(ts)}",
        )

    value = (cash + (btc * price)) if price is not None else cash
    portfolio_md = (
        "### Portfolio\n"
        f"- Cash USD: **${cash:,.2f}**\n"
        f"- BTC (DCA): **{btc:.8f}**\n"
        f"- Value USD: **${value:,.2f}**\n"
        f"- DCA Buys: **{buys}**\n"
        f"- DCA Ref Price: **${ref:.2f}**" if ref else "- DCA Ref Price: **n/a**\n"
    )
    set_status(portfolio, portfolio_md)

    atr_md = (
        "### ATR Trade\n"
        f"- Open: **{'Yes ✅' if atr_open else 'No ❌'}**\n"
        f"- Entry: **${atr_entry:.2f}**\n" if atr_entry else "- Entry: **n/a**\n"
        f"- Stop: **${atr_stop:.2f}**\n" if atr_stop else "- Stop: **n/a**\n"
        f"- Realized PnL: **${atr_real:,.2f}**\n"
    )
    set_status(atr_box, atr_md)

def on_apply(_):
    if budget.value <= 0 or dca_buy.value <= 0 or dca_drop.value <= 0:
        log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] ❌ Invalid inputs, must be > 0")
        return
    with state.lock:
        state.rules.budget_usd = float(budget.value)
        state.rules.dca.buy_usd = float(dca_buy.value)
        state.rules.dca.drop_pct = float(dca_drop.value)
        state.portfolio.cash_usd = float(budget.value)
        state.portfolio.btc_dca = 0.0
        state.portfolio.buys = 0
        state.portfolio.ref_price = state.last_price
    log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] ✅ Applied DCA + reset portfolio")
    
    # Send Telegram notification
    if telegram_notifier:
        telegram_msg = f"⚙️ *DCA Settings Updated*\n\nBudget: ${budget.value:.2f}\nBuy Amount: ${dca_buy.value:.2f}\nDrop %: {dca_drop.value}%"
        telegram_notifier.send_async(telegram_msg)
    
    refresh_panels()

def on_pause(_):
    state.pause_dca.set()
    log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] ⏸️ DCA paused")

def on_resume(_):
    state.pause_dca.clear()
    log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] ▶️ DCA resumed")

def on_atr_open(_):
    if atr_usd.value <= 0 or atr_k.value <= 0:
        log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] ❌ ATR inputs must be > 0")
        return
    try:
        with state.lock:
            msg = atr_open_trade(state, atr_cfg, float(atr_usd.value), float(atr_k.value))
        log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] {msg}")
        
        # Send Telegram notification for ATR open
        if telegram_notifier and "ATR OPEN" in msg:
            telegram_msg = f"🔵 *ATR Trade Opened*\n\n{msg}"
            telegram_notifier.send_async(telegram_msg)
            
    except Exception as exc:
        log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] ❌ ATR OPEN ERROR: {exc}")
    refresh_panels()

def on_atr_close(_):
    with state.lock:
        msg = atr_close_trade(state, "MANUAL")
    log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] {msg}")
    
    # Send Telegram notification for ATR close
    if telegram_notifier and "ATR CLOSE" in msg:
        telegram_msg = f"🔴 *ATR Trade Closed*\n\n{msg}"
        telegram_notifier.send_async(telegram_msg)
    
    refresh_panels()

def on_stop(_):
    state.stop.set()
    log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] 🛑 Stop signal sent")

def on_llm_submit(_):
    global trading_llm
    
    if trading_llm is None:
        set_status(llm_chat_output, "### 🤖 AI Trading Advisor\n⚠️ **AI Disabled**\n\nTo enable:\n1. Get free API key from https://aistudio.google.com/app/apikey\n2. Save it in 'hf_token.txt'\n3. Restart the app")
        return
    
    query = llm_input.value
    if not query or query.strip() == "":
        set_status(llm_chat_output, "### 🤖 AI Trading Advisor\n⚠️ Please enter a question!")
        return
    
    set_status(llm_chat_output, "### 🤖 AI Trading Advisor\n🤔 *Analyzing your portfolio with AI...*")
    log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] 🤖 AI Query: {query}")
    
    try:
        response = trading_llm.answer_query(query, state)
        set_status(llm_chat_output, response)
        log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] ✅ AI Response generated")
    except Exception as exc:
        error_msg = f"### 🤖 AI Trading Advisor\n❌ **Error:** {str(exc)}\n\nPlease check your API key and internet connection."
        set_status(llm_chat_output, error_msg)
        log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] ❌ AI Error: {exc}")
    
    llm_input.value = ""

# Button callbacks
apply_btn.on_click(on_apply)
pause_btn.on_click(on_pause)
resume_btn.on_click(on_resume)
atr_open_btn.on_click(on_atr_open)
atr_close_btn.on_click(on_atr_close)
stop_btn.on_click(on_stop)
llm_submit.on_click(on_llm_submit)

def price_thread():
    sess = requests.Session()
    while not state.stop.is_set():
        try:
            price, src = fetch_live_price(price_cfg, sess)
            with state.lock:
                state.last_price = price
                state.last_source = src
                state.last_ts = utc_now()
            refresh_panels()
        except Exception as exc:
            log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] ⚠️ PRICE ERROR: {exc}")
        time.sleep(price_cfg.poll_seconds)

def dca_thread():
    while not state.stop.is_set():
        if state.pause_dca.is_set():
            time.sleep(1)
            continue
        msg = None
        with state.lock:
            if state.last_price is not None:
                msg = maybe_dca(state, float(state.last_price))
        if msg:
            log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] {msg}")
            
            # Send Telegram notification for DCA buy
            if telegram_notifier and "DCA BUY" in msg:
                telegram_msg = f"🟢 *DCA Trade Executed*\n\n{msg}"
                telegram_notifier.send_async(telegram_msg)
            
            refresh_panels()
        time.sleep(1)

def atr_stop_thread():
    while not state.stop.is_set():
        should_close = False
        with state.lock:
            if state.last_price is not None and state.atr_trade.open and state.atr_trade.stop is not None:
                if float(state.last_price) <= float(state.atr_trade.stop):
                    should_close = True
        if should_close:
            with state.lock:
                msg = atr_close_trade(state, "STOP_LOSS_HIT")
            log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] {msg}")
            
            # Send Telegram notification for stop loss
            if telegram_notifier:
                telegram_msg = f"⚠️ *ATR Stop Loss Hit*\n\n{msg}"
                telegram_notifier.send_async(telegram_msg)
            
            refresh_panels()
        time.sleep(1)

def start_threads():
    Thread(target=price_thread, daemon=True).start()
    Thread(target=dca_thread, daemon=True).start()
    Thread(target=atr_stop_thread, daemon=True).start()
    log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] ✅ Threads started")
    
    if trading_llm:
        log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] 🤖 AI Trading Advisor: ENABLED (Google Gemini)")
    else:
        log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] ⚠️ AI Trading Advisor: DISABLED")
    
    if telegram_notifier:
        log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] 📱 Telegram Notifications: ENABLED")
        telegram_notifier.send_async("🚀 *Bitcoin Trading Bot Started*\n\nMonitoring for trades...")
    else:
        log_line(log_pane, f"[{utc_now().strftime('%H:%M:%S')}] 📱 Telegram Notifications: DISABLED")
    
    refresh_panels()

pn.state.onload(start_threads)

# Layout
layout = pn.Column(
    status,
    pn.Row(
        pn.Column(
            "### DCA Controls",
            budget,
            dca_buy,
            dca_drop,
            apply_btn,
            pn.Row(pause_btn, resume_btn)
        ),
        pn.Column(
            "### ATR Controls",
            atr_usd,
            atr_k,
            pn.Row(atr_open_btn, atr_close_btn),
            stop_btn
        ),
    ),
    pn.Row(
        pn.Column(portfolio),
        pn.Column(atr_box)
    ),
    pn.pane.Markdown("---"),
    pn.pane.Markdown("## 🤖 AI Trading Strategy Advisor (Powered by Google Gemini)"),
    llm_chat_output,
    pn.Row(llm_input, llm_submit),
    pn.pane.Markdown("---"),
    pn.pane.Markdown("### 📋 Activity Logs"),
    log_pane,
    sizing_mode="stretch_width",
)

layout.servable()