"""Credential verification helper — called by START_PAPER.bat."""
import pathlib, sys

try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).parent / ".env")
except ImportError:
    pass

from scalper.config.settings import DHAN_CLIENT_ID, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

errors = []
if "YOUR_" in str(DHAN_CLIENT_ID):     errors.append("DHAN_CLIENT_ID not set in scalper/.env")
if "YOUR_" in str(TELEGRAM_BOT_TOKEN): errors.append("TELEGRAM_BOT_TOKEN not set in scalper/.env")
if "YOUR_" in str(TELEGRAM_CHAT_ID):   errors.append("TELEGRAM_CHAT_ID not set in scalper/.env")

if errors:
    for e in errors:
        print("  MISSING:", e)
    sys.exit(1)

print("  Credentials OK.")
