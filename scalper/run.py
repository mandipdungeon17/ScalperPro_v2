#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
SCALPER PRO - Quick Start Runner
═══════════════════════════════════════════════════════════════════════

Quick start commands:

  1. BACKTEST (with sample data, no API needed):
     python run.py backtest --sample

  2. BACKTEST (with real Dhan data):
     python run.py backtest

  3. PAPER TRADING (sends Telegram alerts, no real trades):
     python run.py paper

  4. LIVE TRADING (real money — use only after paper validation!):
     python run.py live

═══════════════════════════════════════════════════════════════════════
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # dotenv not installed, use system env vars


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ███████╗ ██████╗ █████╗ ██╗     ██████╗ ███████╗██████╗       ║
║   ██╔════╝██╔════╝██╔══██╗██║     ██╔══██╗██╔════╝██╔══██╗      ║
║   ███████╗██║     ███████║██║     ██████╔╝█████╗  ██████╔╝      ║
║   ╚════██║██║     ██╔══██║██║     ██╔═══╝ ██╔══╝  ██╔══██╗      ║
║   ███████║╚██████╗██║  ██║███████╗██║     ███████╗██║  ██║      ║
║   ╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝      ║
║                      P R O                                       ║
║                                                                  ║
║   Indian Index Options Scalping & Swing System                   ║
║   NIFTY • BANKNIFTY • FINNIFTY • MIDCPNIFTY • SENSEX            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def main():
    print_banner()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run.py backtest [--sample] [--days N]")
        print("  python run.py paper")
        print("  python run.py live")
        print()
        print("Options:")
        print("  --sample         Use generated sample data (no API key needed)")
        print("  --days N         Backtest period in days (default: 180)")
        print("  --indices X Y    Indices to trade (default: all 5)")
        print("  --scan N         Scan interval in seconds (default: 30)")
        return

    mode = sys.argv[1].lower()
    indices = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"]
    days = 180
    use_sample = False
    scan_interval = 30

    # Parse args
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--sample":
            use_sample = True
        elif sys.argv[i] == "--days" and i + 1 < len(sys.argv):
            days = int(sys.argv[i + 1])
            i += 1
        elif sys.argv[i] == "--indices":
            indices = []
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith("--"):
                indices.append(sys.argv[i].upper())
                i += 1
            continue
        elif sys.argv[i] == "--scan" and i + 1 < len(sys.argv):
            scan_interval = int(sys.argv[i + 1])
            i += 1
        i += 1

    from scalper.main import ScalperPro

    if mode == "backtest":
        print(f"📊 Running backtest: {days} days on {', '.join(indices)}")
        print(f"   Data source: {'Sample (generated)' if use_sample else 'Dhan API'}")
        print()
        system = ScalperPro(mode="BACKTEST", indices=indices)
        system.run_backtest(days=days, use_sample=use_sample)

    elif mode == "paper":
        print(f"📝 Starting PAPER trading on {', '.join(indices)}")
        print(f"   Scan interval: {scan_interval}s")
        print(f"   Telegram alerts: ENABLED")
        print(f"   Real orders: DISABLED")
        print()
        print("   Press Ctrl+C to stop")
        print()
        system = ScalperPro(mode="PAPER", indices=indices)
        system.run_live(scan_interval=scan_interval)

    elif mode == "live":
        print("⚠️  WARNING: LIVE TRADING MODE")
        print("   This will place REAL ORDERS with REAL MONEY!")
        print()
        confirm = input("   Type 'YES I UNDERSTAND' to proceed: ")
        if confirm != "YES I UNDERSTAND":
            print("   Aborted.")
            return
        print()
        print(f"💰 Starting LIVE trading on {', '.join(indices)}")
        system = ScalperPro(mode="LIVE", indices=indices)
        system.run_live(scan_interval=scan_interval)

    else:
        print(f"Unknown mode: {mode}")
        print("Use: backtest, paper, or live")


if __name__ == "__main__":
    main()
