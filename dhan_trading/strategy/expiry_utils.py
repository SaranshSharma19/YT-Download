import datetime
from datetime import date, timedelta
from typing import Optional

# Holidays are dynamically extensible. Here are known major upcoming holidays 2025/2026.
NSE_HOLIDAYS = {
    date(2025, 1, 26), date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31),
    date(2025, 4, 10), date(2025, 4, 14), date(2025, 4, 18), date(2025, 5, 1),
    date(2025, 8, 15), date(2025, 10, 2), date(2025, 10, 21), date(2025, 11, 5),
    date(2025, 12, 25), date(2026, 1, 26), date(2026, 3, 10)
}

# New SEBI Compliant Single-Weekly Expiry Rules
EXPIRY_CONFIG = {
    'NIFTY': {'type': 'weekly', 'weekday': 1},   # Tuesday
    'BANKNIFTY': {'type': 'monthly', 'weekday': 1}, # Last Tuesday
    'FINNIFTY': {'type': 'monthly', 'weekday': 1},  # Last Tuesday
}

def _is_trading_day(d: date) -> bool:
    if d.weekday() >= 5: return False
    if d in NSE_HOLIDAYS: return False
    return True

def _previous_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not _is_trading_day(d):
        d -= timedelta(days=1)
    return d

def _get_next_weekly_expiry(today: date, weekday: int) -> date:
    days_ahead = weekday - today.weekday()
    if days_ahead < 0: days_ahead += 7
    elif days_ahead == 0 and not _is_trading_day(today):
        return _previous_trading_day(today)
    
    expiry = today + timedelta(days=days_ahead)
    return expiry if _is_trading_day(expiry) else _previous_trading_day(expiry)

def _get_last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    while last_day.weekday() != weekday:
        last_day -= timedelta(days=1)
    return last_day

def _get_next_monthly_expiry(today: date, weekday: int) -> date:
    this_month = _get_last_weekday_of_month(today.year, today.month, weekday)
    if this_month >= today:
        expiry = this_month
    else:
        expiry = _get_last_weekday_of_month(today.year + 1, 1, weekday) if today.month == 12 else _get_last_weekday_of_month(today.year, today.month + 1, weekday)
    return expiry if _is_trading_day(expiry) else _previous_trading_day(expiry)

def get_next_expiry(index_name: str, ref_date: Optional[date] = None) -> date:
    today = ref_date or date.today()
    config = EXPIRY_CONFIG.get(index_name, {'type': 'weekly', 'weekday': 1})
    if config['type'] == 'weekly':
        return _get_next_weekly_expiry(today, config['weekday'])
    return _get_next_monthly_expiry(today, config['weekday'])

def is_expiry_day(index_name: str, ref_date: Optional[date] = None) -> bool:
    today = ref_date or date.today()
    return get_next_expiry(index_name, today) == today

def is_stt_trap_time(index_name: str, current_time: datetime.datetime) -> bool:
    """Check if we are in the last 10 mins of expiry day (avoid ITM short option settlement)"""
    if not is_expiry_day(index_name, current_time.date()): return False
    # If time >= 15:20
    if current_time.hour == 15 and current_time.minute >= 20: return True
    return False
