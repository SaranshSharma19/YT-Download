"""
Expiry Day Awareness Utilities for Indian Index Options Trading.

Contract expiry rules:
- NIFTY:     Weekly options expire on TUESDAY
- BANKNIFTY: Monthly options expire on LAST TUESDAY of the month
- FINNIFTY:  Monthly options expire on LAST TUESDAY of the month

If expiry falls on a holiday, it shifts to the PREVIOUS trading day.
"""

from datetime import datetime, timedelta, date
from typing import Optional
import logging

logger = logging.getLogger('ExpiryUtils')

# Known NSE holidays for 2025
NSE_HOLIDAYS_2025 = {
    date(2025, 1, 26),   # Republic Day
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-ul-Fitr (Ramadan Id)
    date(2025, 4, 10),   # Shri Ram Navami
    date(2025, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 10, 2),   # Mahatma Gandhi Jayanti
    date(2025, 10, 21),  # Diwali-Laxmi Pujan
    date(2025, 11, 5),   # Gurunanak Jayanti
    date(2025, 12, 25),  # Christmas
}

# Known NSE holidays for 2026
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 26),   # Maha Shivaratri (Tentative)
    date(2026, 3, 10),   # Holi
    date(2026, 3, 30),   # Id-Ul-Fitr (Tentative)
    date(2026, 3, 31),   # Id-Ul-Fitr (Tentative)
    date(2026, 4, 2),    # Ram Navami
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 25),   # Buddha Purnima (Tentative)
    date(2026, 6, 5),    # Bakri Id (Tentative)
    date(2026, 7, 6),    # Muharram (Tentative)
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 19),   # Janmashtami
    date(2026, 9, 4),    # Milad-Un-Nabi (Tentative)
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 10, 21),  # Dussehra
    date(2026, 11, 9),   # Diwali (Laxmi Puja)
    date(2026, 11, 10),  # Diwali (Balipratipada)
    date(2026, 11, 30),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
}

# Combine holidays
NSE_HOLIDAYS = NSE_HOLIDAYS_2025.union(NSE_HOLIDAYS_2026)
HOLIDAYS_COVERED = {2025, 2026}

# Expiry day for each index
EXPIRY_CONFIG = {
    'NIFTY': {
        'type': 'weekly',
        'weekday': 1,  # Tuesday (0=Mon, 1=Tue, ..., 6=Sun)
    },
    'BANKNIFTY': {
        'type': 'monthly',
        'weekday': 1,  # Last Tuesday of month
    },
    'FINNIFTY': {
        'type': 'monthly',
        'weekday': 1,  # Last Tuesday of month
    },
}


def _is_trading_day(d: date) -> bool:
    """Check if a date is a valid trading day (not weekend, not holiday)."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    # Check if we have coverage for the year
    if d.year not in HOLIDAYS_COVERED:
        logger.warning(f"Holiday coverage missing for year {d.year}. Please update expiry_utils.py.")
        
    if d in NSE_HOLIDAYS:
        return False
    return True


def _previous_trading_day(d: date) -> date:
    """Get the previous valid trading day (roll backward past weekends/holidays)."""
    d = d - timedelta(days=1)
    while not _is_trading_day(d):
        d = d - timedelta(days=1)
    return d


def _get_next_weekly_expiry(today: date, weekday: int = 1) -> date:
    """Get the next weekly expiry (e.g., next Tuesday)."""
    days_ahead = weekday - today.weekday()
    if days_ahead < 0:
        days_ahead += 7
    elif days_ahead == 0:
        # Today is expiry day — return today (caller decides if it's expired)
        expiry = today
        if not _is_trading_day(expiry):
            return _previous_trading_day(expiry)
        return expiry
    
    expiry = today + timedelta(days=days_ahead)
    if not _is_trading_day(expiry):
        return _previous_trading_day(expiry)
    return expiry


def _get_last_weekday_of_month(year: int, month: int, weekday: int = 1) -> date:
    """Get the last occurrence of a specific weekday in a given month."""
    # Start from the last day of the month
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    
    last_day = next_month - timedelta(days=1)
    
    # Walk backward to find the last occurrence of the weekday
    while last_day.weekday() != weekday:
        last_day -= timedelta(days=1)
    
    return last_day


def _get_next_monthly_expiry(today: date, weekday: int = 1) -> date:
    """Get the next monthly expiry (last Tuesday of current or next month)."""
    # Check this month first
    this_month_expiry = _get_last_weekday_of_month(today.year, today.month, weekday)
    
    if this_month_expiry >= today:
        expiry = this_month_expiry
    else:
        # Move to next month
        if today.month == 12:
            expiry = _get_last_weekday_of_month(today.year + 1, 1, weekday)
        else:
            expiry = _get_last_weekday_of_month(today.year, today.month + 1, weekday)
    
    # Handle holidays
    if not _is_trading_day(expiry):
        return _previous_trading_day(expiry)
    return expiry


def get_next_expiry(index_name: str, ref_date: Optional[date] = None) -> date:
    """
    Get the next expiry date for an index.
    
    Args:
        index_name: 'NIFTY', 'BANKNIFTY', or 'FINNIFTY'
        ref_date: Reference date (defaults to today)
    
    Returns:
        Next expiry date
    """
    today = ref_date or date.today()
    config = EXPIRY_CONFIG.get(index_name)
    
    if not config:
        logger.warning(f"Unknown index {index_name}. Defaulting to weekly Tuesday expiry.")
        return _get_next_weekly_expiry(today, weekday=1)
    
    if config['type'] == 'weekly':
        return _get_next_weekly_expiry(today, config['weekday'])
    else:  # monthly
        return _get_next_monthly_expiry(today, config['weekday'])


def get_days_to_expiry(index_name: str, ref_date: Optional[date] = None) -> int:
    """
    A-3: Calculate number of calendar days between today and the next expiry.
    """
    today = ref_date or date.today()
    expiry = get_next_expiry(index_name, today)
    return (expiry - today).days


def is_expiry_day(index_name: str, ref_date: Optional[date] = None) -> bool:
    """
    Check if today (or ref_date) is the expiry day for an index.
    
    Args:
        index_name: 'NIFTY', 'BANKNIFTY', or 'FINNIFTY'
        ref_date: Reference date (defaults to today)
    
    Returns:
        True if it's expiry day
    """
    today = ref_date or date.today()
    expiry = get_next_expiry(index_name, today)
    return expiry == today


def is_expiry_week(index_name: str, ref_date: Optional[date] = None) -> bool:
    """
    Check if the current week contains the expiry day for an index.
    
    Args:
        index_name: 'NIFTY', 'BANKNIFTY', or 'FINNIFTY'
        ref_date: Reference date (defaults to today)
    
    Returns:
        True if expiry is within this week (Mon-Fri)
    """
    today = ref_date or date.today()
    expiry = get_next_expiry(index_name, today)
    
    # Calculate start of week (Monday)
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=4)  # Friday
    
    return week_start <= expiry <= week_end


def get_expiry_adjustment(index_name: str, ref_date: Optional[date] = None) -> dict:
    """
    Get trading parameter adjustments for expiry proximity.
    
    Returns dict with:
        - is_expiry_day: bool
        - is_expiry_week: bool
        - position_size_multiplier: float (1.0 = no change, 0.5 = halved)
        - sl_tightening: float (multiplier, e.g., 0.7 = 30% tighter)
        - tp_tightening: float
        - validator_threshold_boost: int (added to min_score)
        - should_skip: bool (if True, don't trade at all)
    """
    today = ref_date or date.today()
    _is_expiry_day = is_expiry_day(index_name, today)
    _is_expiry_week = is_expiry_week(index_name, today)
    
    if _is_expiry_day:
        return {
            'is_expiry_day': True,
            'is_expiry_week': True,
            'position_size_multiplier': 0.5,
            'sl_tightening': 0.7,    # 30% tighter SL
            'tp_tightening': 0.6,    # 40% tighter TP (quick scalps only)
            'validator_threshold_boost': 15,  # Much stricter validation
            'should_skip': False,     # Set to True for Option A (skip entirely)
        }
    elif _is_expiry_week:
        return {
            'is_expiry_day': False,
            'is_expiry_week': True,
            'position_size_multiplier': 0.75,
            'sl_tightening': 0.85,
            'tp_tightening': 0.85,
            'validator_threshold_boost': 5,
            'should_skip': False,
        }
    else:
        return {
            'is_expiry_day': False,
            'is_expiry_week': False,
            'position_size_multiplier': 1.0,
            'sl_tightening': 1.0,
            'tp_tightening': 1.0,
            'validator_threshold_boost': 0,
            'should_skip': False,
        }
