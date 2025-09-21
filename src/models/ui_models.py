import logging
from datetime import datetime
import pytz
from src.config import *
from typing import List, Dict
from src.models.bastion import SHA1664, EphemeralBastion
from collections import defaultdict  # Add this import
import asyncio
import json
import random

logger = logging.getLogger(__name__)

class CandlestickChart:
    def __init__(self, data: List[dict]):
        self.data = data

    def render(self):
        """Render an ASCII candlestick chart."""
        try:
            print("\n=== Candlestick Chart (ASCII) ===")
            for d in self.data[:3]:
                print(f"{d['date']}: O:{d['open']:.2f} H:{d['high']:.2f} L:{d['low']:.2f} C:{d['close']:.2f} V:{d['vol']}")
        except Exception as e:
            logger.error(f"Chart render error: {e}")

class OrderBookUI:
    def __init__(self):
        self.bids = []
        self.asks = []

    def add_bid(self, price: float, volume: float):
        """Add a bid to the order book."""
        try:
            self.bids.append({'price': price, 'volume': volume})
        except Exception as e:
            logger.error(f"Add bid error: {e}")

    def add_ask(self, price: float, volume: float):
        """Add an ask to the order book."""
        try:
            self.asks.append({'price': price, 'volume': volume})
        except Exception as e:
            logger.error(f"Add ask error: {e}")

    def render(self, current_price: float):
        """Render the order book UI."""
        try:
            print("\n=== Order Book ===")
            print(f"Current Price: ${current_price:.2f}")
            print("Bids:")
            for bid in self.bids[:3]:
                print(f"${bid['price']:.2f} | Vol: {bid['volume']:.2f}")
            print("Asks:")
            for ask in self.asks[:3]:
                print(f"${ask['price']:.2f} | Vol: {ask['volume']:.2f}")
        except Exception as e:
            logger.error(f"Order book render error: {e}")

class PortfolioUI:
    def __init__(self):
        self.trades = []

    def add_trade(self, ticker: str, price: float, amount: float, type_: str, tx_hash: str, current_price: float = 0.0) -> dict:
        """Add a trade to the portfolio."""
        try:
            pl = (current_price - price) * amount if type_ == "Buy" and current_price else 0.0
            trade = {
                'time': datetime.now(pytz.timezone('Australia/Sydney')).strftime("%Y-%m-%d %H:%M:%S"),
                'ticker': ticker, 'price': price, 'amount': amount, 'type': type_,
                'tx_hash': f"https://solscan.io/tx/{tx_hash}", 'pl': pl
            }
            self.trades.append(trade)
            return trade
        except Exception as e:
            logger.error(f"Add trade error: {e}")
            return None

    def render(self):
        """Render the portfolio UI."""
        try:
            total_pl = sum(t['pl'] for t in self.trades)
            print("\n=== Portfolio ===")
            print(f"{'Time':<19} | {'Ticker':<6} | {'Type':<4} | {'Price':>8} | {'Amount':>8} | {'P/L':>8} | {'Tx Hash':<8}")
            print("-" * 64)
            for t in self.trades[:3]:
                print(f"{t['time']:<19} | {t['ticker']:<6} | {t['type']:<4} | {t['price']:>8.2f} | {t['amount']:>8.2f} | {t['pl']:>8.2f} | {t['tx_hash'][-8:]}")
            print(f"Total P/L: ${total_pl:.2f}")
        except Exception as e:
            logger.error(f"Portfolio render error: {e}")

class PillboxUI:
    def __init__(self, stocks: List[str]):
        self.stocks = stocks
        self.pinned = []
        self.arbitrage_mode = False

    def select_stock(self, ticker: str):
        """Select a stock for trading."""
        try:
            if ticker in self.stocks and ticker not in self.pinned:
                self.pinned.append(ticker)
                print(f"Pinned {ticker} for trading")
        except Exception as e:
            logger.error(f"Select stock error: {e}")

    def toggle_arbitrage_mode(self):
        """Toggle arbitrage mode for stock selection."""
        try:
            self.arbitrage_mode = not self.arbitrage_mode
            print(f"Switched to {'Arbitrage' if self.arbitrage_mode else 'Normal'} mode")
        except Exception as e:
            logger.error(f"Toggle arbitrage mode error: {e}")

    def render(self, prices: dict[str, float]):
        """Render the pillbox asset selector UI."""
        try:
            print("\n=== Pillbox Asset Selector ===")
            sorted_stocks = sorted(self.stocks, key=lambda x: prices.get(x, 0), reverse=self.arbitrage_mode)
            for i, stock in enumerate(sorted_stocks[:3], 1):
                status = "Pinned" if stock in self.pinned else ""
                print(f"{i}. {stock} (${prices.get(stock, 0):.2f}) {status}")
        except Exception as e:
            logger.error(f"Pillbox render error: {e}")

class QuoteBoxUI:
    def __init__(self):
        self.quote = None

    def set_quote(self, ticker: str, price: float, amount: float):
        """Set a quote for a swap."""
        try:
            self.quote = {
                'ticker': ticker, 'price': price, 'amount': amount,
                'fees': FEE_RATE, 'slippage': 0.005, 'path': [ticker, 'USDT']
            }
        except Exception as e:
            logger.error(f"Set quote error: {e}")

    def render(self):
        """Render the quote box UI."""
        try:
            print("\n=== Quote Box ===")
            if self.quote:
                total_cost = self.quote['amount'] * self.quote['price'] * (1 + self.quote['fees'] + self.quote['slippage'])
                print(f"Swap: {self.quote['amount']:.2f} {self.quote['ticker']} @ ${self.quote['price']:.2f}")
                print(f"Path: {' -> '.join(self.quote['path'])}")
                print(f"Fees: {self.quote['fees']*100:.2f}% | Slippage: {self.quote['slippage']*100:.2f}%")
                print(f"Total Cost: ${total_cost:.2f}")
            else:
                print("No active quote")
        except Exception as e:
            logger.error(f"Quote box render error: {e}")

class TimeSelectorUI:
    def __init__(self):
        self.timeframe = '1m'
        self.current_time = datetime(2025, 9, 18, 0, 5, tzinfo=pytz.timezone('Australia/Sydney'))  # 12:05 AM AEST

    def set_timeframe(self, timeframe: str):
        """Set the timeframe for the UI."""
        try:
            if timeframe in ['1m', '5m', '15m', '1H', '4H', 'D', '1W', '1M']:
                self.timeframe = timeframe
                print(f"Timeframe set to {timeframe}")
        except Exception as e:
            logger.error(f"Set timeframe error: {e}")

    def render(self):
        """Render the time selector UI."""
        try:
            aest_time = self.current_time.strftime("%H:%M:%S")
            print("\n=== Time Selector ===")
            print(f"Current Time: {aest_time} AEST")
            print(f"Selected Timeframe: {self.timeframe}")
        except Exception as e:
            logger.error(f"Time selector render error: {e}")

class ChannelSelectorUI:
    def __init__(self, sha1664: SHA1664, bastion: EphemeralBastion):
        self.sha1664 = sha1664
        self.bastion = bastion
        self.current_channel = CHANNELS[0]
        self.channel_data = defaultdict(list)

    def select_channel(self, channel: str):
        """Select a channel for data processing."""
        try:
            if channel in CHANNELS:
                self.current_channel = channel
                self.bastion.set_ternary_state(1 if channel == "Bastions" else 0)
                logger.info(f"Switched to channel: {channel}")
                print(f"Switched to channel: {channel}")
            else:
                logger.warning(f"Invalid channel: {channel}")
                print(f"Invalid channel. Available: {', '.join(CHANNELS)}")
        except Exception as e:
            logger.error(f"Select channel error: {e}")

    def add_data(self, data: dict):
        """Add data to the current channel."""
        try:
            self.channel_data[self.current_channel].append(data)
            if self.current_channel == "Gossip":
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.sha1664.receive_gossip(data, f"node_{random.randint(1, MAX_NODES)}"))
                else:
                    asyncio.run(self.sha1664.receive_gossip(data, f"node_{random.randint(1, MAX_NODES)}"))
                logger.info(f"Added data to {self.current_channel} channel")
        except Exception as e:
            logger.error(f"Add channel data error: {e}")

    def render(self):
        """Render the channel selector UI."""
        try:
            print(f"\n=== Channel Selector ===")
            print(f"Current Channel: {self.current_channel}")
            print(f"Data Count: {len(self.channel_data[self.current_channel])}")
            if self.channel_data[self.current_channel]:
                sample_data = self.channel_data[self.current_channel][-1]
                print(f"Sample Data: {json.dumps(sample_data)[:50]}...")
        except Exception as e:
            logger.error(f"Channel selector render error: {e}")

class DashboardUI:
    def __init__(self, chart, order_book, portfolio, pillbox, quote, time_selector, channel_selector):
        self.chart = chart
        self.order_book = order_book
        self.portfolio = portfolio
        self.pillbox = pillbox
        self.quote = quote
        self.time_selector = time_selector
        self.channel_selector = channel_selector

    def render(self):
        """Render the dashboard UI with all components."""
        try:
            print("\n=== Dashboard ===")
            self.chart.render()
            self.order_book.render(200.0)
            self.portfolio.render()
            self.pillbox.render({"BTC": 200.0, "ETH": 3000.0, "SOL": 150.0})
            self.quote.render()
            self.time_selector.render()
            self.channel_selector.render()
        except Exception as e:
            logger.error(f"Dashboard render error: {e}")
