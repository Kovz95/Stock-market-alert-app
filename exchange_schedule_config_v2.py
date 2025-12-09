#!/usr/bin/env python3

"""

Exchange Closing Times Configuration (Version 2)

Based on actual exchange names from the database, not countries.

Times are provided for both EST (winter) and EDT (summer) - already converted from local exchange times.



39 exchanges from main_database_with_etfs.json

Times include 40-minute delay after market close for data availability.



IMPORTANT: Date handling for international markets

- Asian/Pacific markets that close after midnight ET are processed on the PREVIOUS US day

  (e.g., Tokyo Monday close at 1:20 AM ET is processed as Sunday night in US)

- This ensures we capture the correct trading day for each market

"""



from datetime import datetime, timedelta, date

import pandas as pd
import pytz



def is_dst_active():

    """Check if US Eastern Daylight Time is currently active"""

    eastern = pytz.timezone('America/New_York')

    now = datetime.now(eastern)

    return bool(now.dst())



def is_in_dst_transition_period():

    """

    Check if we're in the DST transition period where US and European DST don't align.

    Returns True during:

    - 2nd Sunday in March to last Sunday in March (US in EDT, Europe still in standard time)

    - Last Sunday in October to 1st Sunday in November (Europe back to standard time, US still in EDT)

    """

    eastern = pytz.timezone('America/New_York')

    london = pytz.timezone('Europe/London')

    berlin = pytz.timezone('Europe/Berlin')

    now_et = datetime.now(eastern)

    now_uk = datetime.now(london)

    now_de = datetime.now(berlin)

    

    # Check if US is in DST

    us_dst = bool(now_et.dst())

    uk_dst = bool(now_uk.dst())

    de_dst = bool(now_de.dst())

    

    # Transition periods occur when US is in EDT but Europe is not in summer time

    # UK and Germany switch to/from DST on the same dates (last Sunday of March/October)

    # This happens in March (US switches first) and October/November (Europe switches first)

    return us_dst and not (uk_dst or de_dst)



def get_market_days_for_exchange(exchange_config):

    """

    Get the US days of week when we should check an exchange's alerts.

    

    For exchanges with day_offset = -1 (Asia/Pacific):

    - Their Monday-Friday maps to US Monday-Friday early morning for daily checks

    - Their Friday (local) maps to US Friday early morning for weekly checks

    

    For exchanges with day_offset = 0 (Americas/Europe):

    - Their Monday-Friday maps to US Monday-Friday for daily checks

    - Their Friday maps to US Friday for weekly checks

    

    Returns:

        dict: {'daily': 'mon-fri', 'weekly': 'fri'}

    """

    day_offset = exchange_config.get('day_offset', 0)

    

    if day_offset == -1:

        # Asia/Pacific markets - check Monday-Friday early morning US time

        # Both daily and weekly on Friday (Friday in Asia)

        return {

            'daily': 'mon-fri',  # Monday-Friday early AM in US = Monday-Friday close in Asia

            'weekly': 'fri'      # Friday early AM in US = Friday close in Asia

        }

    else:

        # Americas/Europe markets - check Monday through Friday US time

        return {

            'daily': 'mon-fri',  # Monday-Friday in US = Monday-Friday locally

            'weekly': 'fri'      # Friday in US = Friday locally

        }


def _is_date_in_period(current_date: date, period: dict) -> bool:
    """Return True if current_date falls within the provided month/day period."""
    start = date(current_date.year, period['start_month'], period['start_day'])
    end = date(current_date.year, period['end_month'], period['end_day'])

    if start <= end:
        return start <= current_date <= end


    # Handle wrap-around periods that cross the year boundary
    return current_date >= start or current_date <= end


EXCHANGE_TIMEZONES = {
    "LONDON": "Europe/London",
    "XETRA": "Europe/Berlin",
    "EURONEXT PARIS": "Europe/Paris",
    "EURONEXT AMSTERDAM": "Europe/Amsterdam",
    "EURONEXT BRUSSELS": "Europe/Brussels",
    "EURONEXT LISBON": "Europe/Lisbon",
    "EURONEXT DUBLIN": "Europe/Dublin",
    "MILAN": "Europe/Rome",
    "SPAIN": "Europe/Madrid",
    "SIX SWISS": "Europe/Zurich",
    "SIX": "Europe/Zurich",
    "OMX NORDIC STOCKHOLM": "Europe/Stockholm",
    "OMX NORDIC COPENHAGEN": "Europe/Copenhagen",
    "OMX NORDIC HELSINKI": "Europe/Helsinki",
    "OMX NORDIC ICELAND": "Atlantic/Reykjavik",
    "OSLO": "Europe/Oslo",
    "WARSAW": "Europe/Warsaw",
    "VIENNA": "Europe/Vienna",
    "ATHENS": "Europe/Athens",
    "PRAGUE": "Europe/Prague",
    "BUDAPEST": "Europe/Budapest",
}

def get_exchange_close_time(exchange_config, use_edt=None, current_date=None, in_transition=None, exchange_name=None):
    """
    Determine the appropriate close time (hour, minute) for an exchange, accounting for DST overrides.
    """
    if use_edt is None:
        use_edt = is_dst_active()

    if not use_edt:
        return exchange_config['est_close_hour'], exchange_config['est_close_minute']

    if current_date is None:
        est_tz = pytz.timezone('America/New_York')
        current_date = datetime.now(est_tz).date()

    if in_transition is None:
        in_transition = is_in_dst_transition_period()

    hour = exchange_config['edt_close_hour']
    minute = exchange_config['edt_close_minute']

    tz_name = exchange_config.get('timezone') or EXCHANGE_TIMEZONES.get(exchange_name)
    is_local_dst = None
    if tz_name:
        try:
            local_tz = pytz.timezone(tz_name)
            is_local_dst = bool(datetime.now(local_tz).dst())
        except Exception:
            is_local_dst = None

    override = exchange_config.get('dst_transition_override')
    if not override:
        return hour, minute

    condition = override.get('condition')
    apply_override = False

    if condition == 'both_dst':
        if is_local_dst is None:
            apply_override = bool(use_edt)
        else:
            apply_override = bool(use_edt and is_local_dst)
    elif condition == 'misaligned_dst':
        if is_local_dst is None:
            apply_override = bool(in_transition)
        else:
            apply_override = bool(use_edt) != bool(is_local_dst)
    else:
        periods = override.get('periods')
        if periods:
            for period in periods:
                if _is_date_in_period(current_date, period):
                    apply_override = True
                    break
        elif is_local_dst is not None:
            apply_override = bool(use_edt) != bool(is_local_dst)
        else:
            apply_override = in_transition

    if apply_override:
        hour = override.get('edt_close_hour', hour)
        minute = override.get('edt_close_minute', minute)

    return hour, minute



# Exchange closing times with both EST and EDT times + 20-min delay for data availability

EXCHANGE_SCHEDULES = {

    # US Markets (4:20 PM ET - same for both EST and EDT)

    "NASDAQ": {

        "est_close_hour": 16,

        "est_close_minute": 40,

        "edt_close_hour": 16,

        "edt_close_minute": 40,

        "name": "NASDAQ Stock Market",

        "notes": "4:20 PM ET (4:00 PM close + 40 min delay)"

    },

    "NYSE": {

        "est_close_hour": 16,

        "est_close_minute": 40,

        "edt_close_hour": 16,

        "edt_close_minute": 40,

        "name": "New York Stock Exchange",

        "notes": "4:20 PM ET (4:00 PM close + 40 min delay)"

    },

    "NYSE AMERICAN": {

        "est_close_hour": 16,

        "est_close_minute": 40,

        "edt_close_hour": 16,

        "edt_close_minute": 40,

        "name": "NYSE American",

        "notes": "4:20 PM ET (4:00 PM close + 40 min delay)"

    },

    "NYSE ARCA": {

        "est_close_hour": 16,

        "est_close_minute": 40,

        "edt_close_hour": 16,

        "edt_close_minute": 40,

        "name": "NYSE Arca",

        "notes": "4:20 PM ET (4:00 PM close + 40 min delay)"

    },

    "CBOE BZX": {

        "est_close_hour": 16,

        "est_close_minute": 40,

        "edt_close_hour": 16,

        "edt_close_minute": 40,

        "name": "Cboe BZX Exchange",

        "notes": "4:20 PM ET (4:00 PM close + 40 min delay)"

    },

    

    # Canadian Markets (4:20 PM ET - same for both EST and EDT)

    "TORONTO": {

        "est_close_hour": 16,

        "est_close_minute": 40,

        "edt_close_hour": 16,

        "edt_close_minute": 40,

        "name": "Toronto Stock Exchange",

        "notes": "4:20 PM ET (4:00 PM close + 40 min delay)"

    },

    

    # Latin American Markets

    "SAO PAULO": {

        "est_close_hour": 15,  # 4:55 PM BRT = 2:55 PM EST + 40 min delay = 3:15 PM EST (winter)

        "est_close_minute": 35,

        "edt_close_hour": 16,  # 4:55 PM BRT = 3:55 PM EDT + 40 min delay = 4:15 PM EDT (summer)

        "edt_close_minute": 35,

        "name": "B3 Sao Paulo Stock Exchange",

        "notes": "3:15 PM EST / 4:15 PM EDT"

    },

    "MEXICO": {

        "est_close_hour": 16,  # Same as NYSE

        "est_close_minute": 40,

        "edt_close_hour": 16,  # Same as NYSE

        "edt_close_minute": 40,

        "name": "Mexican Stock Exchange",

        "notes": "4:20 PM ET (same as NYSE)"

    },

    "BUENOS AIRES": {

        "est_close_hour": 15,  # 5:00 PM ART = 3:00 PM EST (winter)

        "est_close_minute": 40,

        "edt_close_hour": 16,  # 5:00 PM ART = 4:00 PM EDT (summer)

        "edt_close_minute": 40,

        "name": "Buenos Aires Stock Exchange",

        "notes": "3:20 PM EST / 4:20 PM EDT"

    },

    "SANTIAGO": {

        "est_close_hour": 15,  # 4:00 PM CLT = 3:00 PM EST (winter)

        "est_close_minute": 40,

        "edt_close_hour": 16,  # 4:00 PM CLT = 4:00 PM EDT (summer)

        "edt_close_minute": 40,

        "name": "Santiago Stock Exchange",

        "notes": "3:20 PM EST / 4:20 PM EDT"

    },

    

    # European Markets - UK/Ireland

    "LONDON": {

        "est_close_hour": 12,  # 4:30 PM GMT = 11:30 AM EST (winter)

        "est_close_minute": 10,

        "edt_close_hour": 12,  # 4:30 PM BST = 11:30 AM EDT (summer when UK also in DST)

        "edt_close_minute": 10,

        "name": "London Stock Exchange",

        "notes": "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",

        # Special handling needed:

        # - 2nd Sunday March to last Sunday March: US is EDT, UK still GMT = 12:50 PM EDT

        # - Last Sunday October to 1st Sunday November: UK back to GMT, US still EDT = 12:50 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 13,  # During transition periods

            "edt_close_minute": 10

        }

    },



    "EURONEXT DUBLIN": {

        "est_close_hour": 12,

        "est_close_minute": 10,

        "edt_close_hour": 12,  # 4:30 PM IST = 11:30 AM EDT (summer when Ireland also in DST)

        "edt_close_minute": 10,

        "name": "Euronext Dublin",

        "notes": "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",

        # Special handling needed (Ireland follows UK DST schedule):

        # - 2nd Sunday March to last Sunday March: US is EDT, Ireland still GMT = 12:50 PM EDT

        # - Last Sunday October to 1st Sunday November: Ireland back to GMT, US still EDT = 12:50 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 13,  # During transition periods

            "edt_close_minute": 10

        }

    },

    

    # European Markets - Central Europe

    "XETRA": {

        "est_close_hour": 12,  # 5:30 PM CET = 11:30 AM EST (winter)

        "est_close_minute": 10,

        "edt_close_hour": 12,  # 5:30 PM CEST = 11:30 AM EDT (summer when Germany also in DST)

        "edt_close_minute": 10,

        "name": "Frankfurt Stock Exchange (Xetra)",

        "notes": "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",

        # Special handling needed (Germany DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Germany still CET = 12:50 PM EDT

        # - Last Sunday October to 1st Sunday November: Germany back to CET, US still EDT = 12:50 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 13,  # During transition periods

            "edt_close_minute": 10

        }

    },

    "EURONEXT PARIS": {

        "est_close_hour": 12,

        "est_close_minute": 10,

        "edt_close_hour": 12,  # 5:30 PM CEST = 11:30 AM EDT (summer when France also in DST)

        "edt_close_minute": 10,

        "name": "Euronext Paris",

        "notes": "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",

        # Special handling needed (France DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, France still CET = 12:50 PM EDT

        # - Last Sunday October to 1st Sunday November: France back to CET, US still EDT = 12:50 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 13,  # During transition periods

            "edt_close_minute": 10

        }

    },

    "EURONEXT AMSTERDAM": {

        "est_close_hour": 12,

        "est_close_minute": 10,

        "edt_close_hour": 12,  # 5:30 PM CEST = 11:30 AM EDT (summer when Netherlands also in DST)

        "edt_close_minute": 10,

        "name": "Euronext Amsterdam",

        "notes": "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",

        # Special handling needed (Netherlands DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Netherlands still CET = 12:50 PM EDT

        # - Last Sunday October to 1st Sunday November: Netherlands back to CET, US still EDT = 12:50 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 13,  # During transition periods

            "edt_close_minute": 10

        }

    },

    "EURONEXT BRUSSELS": {

        "est_close_hour": 12,

        "est_close_minute": 10,

        "edt_close_hour": 12,  # 5:30 PM CEST = 11:30 AM EDT (summer when Belgium also in DST)

        "edt_close_minute": 10,

        "name": "Euronext Brussels",

        "notes": "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",

        # Special handling needed (Belgium DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Belgium still CET = 12:50 PM EDT

        # - Last Sunday October to 1st Sunday November: Belgium back to CET, US still EDT = 12:50 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 13,  # During transition periods

            "edt_close_minute": 10

        }

    },

    "MILAN": {

        "est_close_hour": 12,

        "est_close_minute": 10,

        "edt_close_hour": 12,  # 5:30 PM CEST = 11:30 AM EDT (summer when Italy also in DST)

        "edt_close_minute": 10,

        "name": "Borsa Italiana Milan",

        "notes": "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",

        # Special handling needed (Italy DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Italy still CET = 12:50 PM EDT

        # - Last Sunday October to 1st Sunday November: Italy back to CET, US still EDT = 12:50 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 13,  # During transition periods

            "edt_close_minute": 10

        }

    },

    "SPAIN": {

        "est_close_hour": 12,

        "est_close_minute": 10,

        "edt_close_hour": 12,  # 5:30 PM CEST = 11:30 AM EDT (summer when Spain also in DST)

        "edt_close_minute": 10,

        "name": "Madrid Stock Exchange",

        "notes": "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",

        # Special handling needed (Spain DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Spain still CET = 12:50 PM EDT

        # - Last Sunday October to 1st Sunday November: Spain back to CET, US still EDT = 12:50 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 13,  # During transition periods

            "edt_close_minute": 10

        }

    },

    "SIX SWISS": {

        "est_close_hour": 12,

        "est_close_minute": 10,

        "edt_close_hour": 12,  # 5:30 PM CEST = 11:30 AM EDT (summer when Switzerland also in DST)

        "edt_close_minute": 10,

        "name": "SIX Swiss Exchange",

        "notes": "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",

        # Special handling needed (Switzerland DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Switzerland still CET = 12:50 PM EDT

        # - Last Sunday October to 1st Sunday November: Switzerland back to CET, US still EDT = 12:50 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 13,  # During transition periods

            "edt_close_minute": 10

        }

    },

    "VIENNA": {

        "est_close_hour": 12,

        "est_close_minute": 10,

        "edt_close_hour": 12,  # 5:30 PM CEST = 11:30 AM EDT (summer when Austria also in DST)

        "edt_close_minute": 10,

        "name": "Vienna Stock Exchange",

        "notes": "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",

        # Special handling needed (Austria DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Austria still CET = 12:50 PM EDT

        # - Last Sunday October to 1st Sunday November: Austria back to CET, US still EDT = 12:50 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 13,  # During transition periods

            "edt_close_minute": 10

        }

    },

    

    # European Markets - Portugal

    "EURONEXT LISBON": {

        "est_close_hour": 12,  # 4:30 PM WET = 11:30 AM EST

        "est_close_minute": 10,

        "edt_close_hour": 12,  # 4:30 PM WEST = 11:30 AM EDT (summer when Portugal also in DST)

        "edt_close_minute": 10,

        "name": "Euronext Lisbon",

        "notes": "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",

        # Special handling needed (Portugal DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Portugal still WET = 12:50 PM EDT

        # - Last Sunday October to 1st Sunday November: Portugal back to WET, US still EDT = 12:50 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 13,  # During transition periods

            "edt_close_minute": 10

        }

    },

    

    # Nordic Markets

    "OMX NORDIC STOCKHOLM": {

        "est_close_hour": 12,  # 5:30 PM CET = 11:30 AM EST

        "est_close_minute": 10,

        "edt_close_hour": 12,  # 5:30 PM CEST = 11:30 AM EDT

        "edt_close_minute": 10,

        "name": "Nasdaq Stockholm",

        "notes": "11:30 AM EST / 11:30 AM EDT (12:30 PM EDT during DST transition periods)",

        # Special handling needed (Sweden DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Sweden still CET = 12:30 PM EDT

        # - Last Sunday October to 1st Sunday November: Sweden back to CET, US still EDT = 12:30 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 13,  # During transition periods

            "edt_close_minute": 10

        }

    },

    "OMX NORDIC COPENHAGEN": {

        "est_close_hour": 11,  # 5:00 PM CET = 11:00 AM EST

        "est_close_minute": 40,

        "edt_close_hour": 11,  # 5:00 PM CEST = 11:00 AM EDT

        "edt_close_minute": 40,

        "name": "Nasdaq Copenhagen",

        "notes": "11:00 AM EST / 11:00 AM EDT (12:00 PM EDT during DST transition periods)",

        # Special handling needed (Denmark DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Denmark still CET = 12:00 PM EDT

        # - Last Sunday October to 1st Sunday November: Denmark back to CET, US still EDT = 12:00 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 12,  # During transition periods

            "edt_close_minute": 40

        }

    },

    "OMX NORDIC HELSINKI": {

        "est_close_hour": 12,  # 6:30 PM EET = 11:30 AM EST

        "est_close_minute": 10,

        "edt_close_hour": 12,  # 6:30 PM EEST = 11:30 AM EDT

        "edt_close_minute": 10,

        "name": "Nasdaq Helsinki",

        "notes": "11:30 AM EST / 11:30 AM EDT (12:30 PM EDT during DST transition periods)",

        # Special handling needed (Finland DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Finland still EET = 12:30 PM EDT

        # - Last Sunday October to 1st Sunday November: Finland back to EET, US still EDT = 12:30 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 13,  # During transition periods

            "edt_close_minute": 10

        }

    },

    "OMX NORDIC ICELAND": {

        "est_close_hour": 11,  # 3:30 PM GMT = 10:30 AM EST

        "est_close_minute": 10,

        "edt_close_hour": 12,  # 3:30 PM GMT = 11:30 AM EDT (Iceland doesn't use DST)

        "edt_close_minute": 10,

        "name": "Nasdaq Iceland",

        "notes": "10:50 AM EST / 11:50 AM EDT"

    },

    "OSLO": {

        "est_close_hour": 11,  # 4:25 PM CET = 10:25 AM EST

        "est_close_minute": 5,

        "edt_close_hour": 11,  # 4:25 PM CEST = 10:25 AM EDT

        "edt_close_minute": 5,

        "name": "Oslo Stock Exchange",

        "notes": "10:25 AM EST / 10:25 AM EDT (11:25 AM EDT during DST transition periods)",

        # Special handling needed (Norway DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Norway still CET = 11:25 AM EDT

        # - Last Sunday October to 1st Sunday November: Norway back to CET, US still EDT = 11:25 AM EDT

        "dst_transition_override": {

            "edt_close_hour": 12,  # During transition periods

            "edt_close_minute": 5

        }

    },

    

    # Eastern European Markets

    "WARSAW": {

        "est_close_hour": 11,  # 5:00 PM CET = 11:00 AM EST

        "est_close_minute": 40,

        "edt_close_hour": 11,  # 5:00 PM CEST = 11:00 AM EDT

        "edt_close_minute": 40,

        "name": "Warsaw Stock Exchange",

        "notes": "11:00 AM EST / 11:00 AM EDT (12:00 PM EDT during DST transition periods)",

        # Special handling needed (Poland DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Poland still CET = 12:00 PM EDT

        # - Last Sunday October to 1st Sunday November: Poland back to CET, US still EDT = 12:00 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 12,  # During transition periods

            "edt_close_minute": 40

        }

    },

    "PRAGUE": {

        "est_close_hour": 10,  # 4:00 PM CET = 10:00 AM EST

        "est_close_minute": 55,

        "edt_close_hour": 10,  # 4:00 PM CEST = 10:00 AM EDT

        "edt_close_minute": 55,

        "name": "Prague Stock Exchange",

        "notes": "10:15 AM EST / 10:15 AM EDT (11:15 AM EDT during DST transition periods)",

        # Special handling needed (Czech Republic DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Czech Republic still CET = 11:15 AM EDT

        # - Last Sunday October to 1st Sunday November: Czech Republic back to CET, US still EDT = 11:15 AM EDT

        "dst_transition_override": {

            "edt_close_hour": 11,  # During transition periods

            "edt_close_minute": 55

        }

    },

    "BUDAPEST": {

        "est_close_hour": 11,  # 5:00 PM CET = 11:00 AM EST

        "est_close_minute": 55,

        "edt_close_hour": 11,  # 5:00 PM CEST = 11:00 AM EDT

        "edt_close_minute": 55,

        "name": "Budapest Stock Exchange",

        "notes": "11:05 AM EST / 11:05 AM EDT (12:05 PM EDT during DST transition periods)",

        # Special handling needed (Hungary DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Hungary still CET = 12:05 PM EDT

        # - Last Sunday October to 1st Sunday November: Hungary back to CET, US still EDT = 12:05 PM EDT

        "dst_transition_override": {

            "edt_close_hour": 12,

            "edt_close_minute": 55

        }

    },

    "ATHENS": {

        "est_close_hour": 10,  # 5:20 PM EET = 10:20 AM EST

        "est_close_minute": 50,

        "edt_close_hour": 10,  # 5:20 PM EEST = 10:20 AM EDT

        "edt_close_minute": 50,

        "name": "Athens Stock Exchange",

        "notes": "10:10 AM EST / 10:10 AM EDT (11:10 AM EDT during DST transition periods)",

        # Special handling needed (Greece DST differs from US):

        # - 2nd Sunday March to last Sunday March: US is EDT, Greece still EET = 11:10 AM EDT

        # - Last Sunday October to 1st Sunday November: Greece back to EET, US still EDT = 11:10 AM EDT

        "dst_transition_override": {

            "edt_close_hour": 11,  # During transition periods

            "edt_close_minute": 50

        }

    },

    

    # Middle East & Africa

    "ISTANBUL": {

        "est_close_hour": 11,  # 6:00 PM TRT = 10:00 AM EST

        "est_close_minute": 0,

        "edt_close_hour": 12,  # 6:00 PM TRT = 11:00 AM EDT (Turkey doesn't observe DST)

        "edt_close_minute": 0,

        "name": "Borsa Istanbul",

        "notes": "10:20 AM EST / 11:20 AM EDT (Turkey doesn't observe DST)"

    },

    "JSE": {

        "est_close_hour": 11,  # 5:00 PM SAST = 10:00 AM EST

        "est_close_minute": 0,

        "edt_close_hour": 12,  # 5:00 PM SAST = 11:00 AM EDT (SA doesn't use DST)

        "edt_close_minute": 0,

        "name": "Johannesburg Stock Exchange",

        "notes": "10:20 AM EST / 11:20 AM EDT (South Africa doesn't observe DST)"

    },

    

    # Asia-Pacific Markets (all close overnight in US time)

    "TOKYO": {

        "est_close_hour": 2,  # 3:00 PM JST = 1:00 AM EST

        "est_close_minute": 15,

        "edt_close_hour": 3,  # 3:00 PM JST = 2:00 AM EDT (Japan doesn't use DST)

        "edt_close_minute": 15,

        "name": "Tokyo Stock Exchange",

        "notes": "1:20 AM EST / 2:20 AM EDT",

        "day_offset": -1  # Monday in Tokyo closes Sunday night/Monday morning in US

    },

    "HONG KONG": {

        "est_close_hour": 3,  # 4:00 PM HKT = 3:00 AM EST

        "est_close_minute": 50,

        "edt_close_hour": 4,  # 4:00 PM HKT = 4:00 AM EDT (HK doesn't use DST)

        "edt_close_minute": 50,

        "name": "Hong Kong Stock Exchange",

        "notes": "3:20 AM EST / 4:20 AM EDT",

        "day_offset": -1  # Monday in HK closes Sunday night/Monday morning in US

    },

    "SINGAPORE": {

        "est_close_hour": 4,  # 5:00 PM SGT = 4:00 AM EST

        "est_close_minute": 50,

        "edt_close_hour": 5,  # 5:00 PM SGT = 5:00 AM EDT (Singapore doesn't use DST)

        "edt_close_minute": 50,

        "name": "Singapore Exchange",

        "notes": "4:20 AM EST / 5:20 AM EDT",

        "day_offset": -1  # Monday in Singapore closes Sunday night/Monday morning in US

    },

    "ASX": {

        "est_close_hour": 1,  # 4:00 PM AEDT = 12:00 AM EST (next day)

        "est_close_minute": 0,

        "edt_close_hour": 3,  # 4:00 PM AEST = 2:00 AM EDT (Australia DST opposite of US)

        "edt_close_minute": 0,

        "name": "Australian Securities Exchange",

        "notes": "12:40 AM EST / 2:40 AM EDT (1:40 AM EDT when both regions observe DST)",

        "timezone": "Australia/Sydney",

        "dst_transition_override": {

            "condition": "both_dst",

            "edt_close_hour": 2,  # When US and Australia are both observing DST

            "edt_close_minute": 0

        },

        "day_offset": -1  # Monday in Australia closes Sunday night/Monday morning in US

    },

    "TAIWAN": {

        "est_close_hour": 1,  # 1:30 PM CST = 12:30 AM EST

        "est_close_minute": 10,

        "edt_close_hour": 2,  # 1:30 PM CST = 1:30 AM EDT (Taiwan doesn't use DST)

        "edt_close_minute": 10,

        "name": "Taiwan Stock Exchange",

        "notes": "12:50 AM EST / 1:50 AM EDT",

        "day_offset": -1  # Monday in Taiwan closes Sunday night/Monday morning in US

    },

    "NSE INDIA": {

        "est_close_hour": 5,  # 3:30 PM IST = 5:00 AM EST

        "est_close_minute": 40,

        "edt_close_hour": 6,  # 3:30 PM IST = 6:00 AM EDT (India doesn't use DST)

        "edt_close_minute": 40,

        "name": "National Stock Exchange of India",

        "notes": "5:20 AM EST / 6:20 AM EDT",

        "day_offset": -1  # Monday in India closes Sunday night/Monday morning in US

    },

    "INDONESIA": {

        "est_close_hour": 4,  # 4:00 PM WIB = 4:00 AM EST

        "est_close_minute": 40,

        "edt_close_hour": 5,  # 4:00 PM WIB = 5:00 AM EDT (Indonesia doesn't use DST)

        "edt_close_minute": 40,

        "name": "Indonesia Stock Exchange",

        "notes": "4:20 AM EST / 5:20 AM EDT",

        "day_offset": -1  # Monday in Indonesia closes Sunday night/Monday morning in US

    },

    "THAILAND": {

        "est_close_hour": 5,  # 4:30 PM ICT = 4:30 AM EST

        "est_close_minute": 20,

        "edt_close_hour": 6,  # 4:30 PM ICT = 5:30 AM EDT (Thailand doesn't use DST)

        "edt_close_minute": 20,

        "name": "Stock Exchange of Thailand",

        "notes": "4:50 AM EST / 5:50 AM EDT",

        "day_offset": -1  # Monday in Thailand closes Sunday night/Monday morning in US

    },

    "MALAYSIA": {

        "est_close_hour": 4,  # 5:00 PM MYT = 4:00 AM EST

        "est_close_minute": 40,

        "edt_close_hour": 5,  # 5:00 PM MYT = 5:00 AM EDT (Malaysia doesn't use DST)

        "edt_close_minute": 40,

        "name": "Bursa Malaysia",

        "notes": "4:20 AM EST / 5:20 AM EDT",

        "day_offset": -1  # Monday in Malaysia closes Sunday night/Monday morning in US

    },

    
    # ===== NEWLY ADDED EXCHANGES (2025-09-05) =====
    
    "BSE INDIA": {
        "est_close_hour": 5,  # 3:30 PM IST = 5:00 AM EST (same as NSE)
        "est_close_minute": 40,
        "edt_close_hour": 6,  # 3:30 PM IST = 6:00 AM EDT (India doesn't use DST)
        "edt_close_minute": 40,
        "name": "Bombay Stock Exchange",
        "notes": "5:40 AM EST / 6:40 AM EDT - Same hours as NSE",
        "day_offset": -1  # Monday in India closes Sunday night/Monday morning in US
    },
    
    "BUCHAREST SPOT": {
        "est_close_hour": 11,  # 5:20 PM EET = 10:20 AM EST + 40 min delay
        "est_close_minute": 0,
        "edt_close_hour": 11,  # 5:20 PM EEST = 10:20 AM EDT + 40 min delay
        "edt_close_minute": 0,
        "name": "Bucharest Stock Exchange",
        "notes": "11:00 AM EST / 11:00 AM EDT",
        "day_offset": 0
    },
    
    "COLOMBIA": {
        "est_close_hour": 16,  # 3:00 PM COT = 3:00 PM EST + 40 min delay (Colombia is EST-0)
        "est_close_minute": 40,
        "edt_close_hour": 15,  # 3:00 PM COT = 2:00 PM EDT + 40 min delay (Colombia doesn't use DST)
        "edt_close_minute": 40,
        "name": "Colombia Stock Exchange (BVC)",
        "notes": "4:40 PM EST / 3:40 PM EDT",
        "day_offset": 0
    }

}



def get_exchanges_by_closing_time(reference_time=None):
    """Group exchanges by their next scheduled daily run (expressed in ET)."""
    from calendar_adapter import get_next_daily_run_time, get_session_bounds

    reference = (
        pd.Timestamp(reference_time, tz="UTC") if reference_time else pd.Timestamp.now(tz="UTC")
    )
    reference_et = reference.tz_convert(pytz.timezone("America/New_York"))

    groups = {}
    for exchange, config in EXCHANGE_SCHEDULES.items():
        try:
            run_dt = get_next_daily_run_time(exchange, reference)
            run_ts = pd.Timestamp(run_dt)
            # Try to use the current/most recent session close when we're between close and the post-close run window.
            open_now, close_now = get_session_bounds(exchange, reference, next_if_closed=False)
            buffer_close = close_now + pd.Timedelta(minutes=40)
            if reference <= buffer_close:
                run_ts = buffer_close
            _, close_time = get_session_bounds(exchange, reference, next_if_closed=True)
        except Exception:
            continue

        run_et = run_ts.tz_convert(pytz.timezone("America/New_York"))
        key = run_et.strftime("%H:%M")
        day_offset = (run_et.date() - reference_et.date()).days

        groups.setdefault(key, []).append(
            {
                "exchange": exchange,
                "name": config.get("name", exchange),
                "hour": run_et.hour,
                "minute": run_et.minute,
                "day_offset": day_offset,
                "close_utc": close_time,
                "run_utc": run_ts,
            }
        )

    return groups



# Weekly schedule configuration

# All exchanges close on Friday at the same daily time for weekly alerts

WEEKLY_SCHEDULE = {

    # All exchanges use Friday as the weekly close with same times as daily

}



def get_weekly_schedule(exchange, use_edt=None):

    """Get weekly schedule for a specific exchange"""

    # Auto-detect DST if not specified

    if use_edt is None:

        use_edt = is_dst_active()

    

    # All exchanges use Friday with same closing time as daily

    if exchange in EXCHANGE_SCHEDULES:

        daily_config = EXCHANGE_SCHEDULES[exchange]

        hour, minute = get_exchange_close_time(
            daily_config,
            use_edt=use_edt,
            exchange_name=exchange,
        )
        tz_label = "EDT" if use_edt else "EST"

        return {

            "day_of_week": 4,  # Friday (0=Monday, 4=Friday)

            "close_hour": hour,

            "close_minute": minute,

            "notes": f"Weekly close on Friday at {hour}:{minute:02d} {tz_label}"

        }

    

    return None



if __name__ == "__main__":

    print("Exchange Schedule Configuration V2 (EST/EDT)")

    print("=" * 60)

    print(f"Total exchanges configured: {len(EXCHANGE_SCHEDULES)}")

    

    # Check current DST status

    is_edt = is_dst_active()

    current_tz = "EDT" if is_edt else "EST"

    print(f"\nCurrent timezone: {current_tz}")

    print(f"DST Active: {is_edt}")

    

    print(f"\nExchanges grouped by {current_tz} closing time (with 20-min delay):")

    

    groups = get_exchanges_by_closing_time()

    for time_key in sorted(groups.keys()):

        exchanges = groups[time_key]

        print(f"\n{time_key} {current_tz}: {len(exchanges)} exchange(s)")

        for ex in exchanges[:5]:  # Show first 5

            print(f"  - {ex['exchange']}: {ex['name']}")

        if len(exchanges) > 5:

            print(f"  ... and {len(exchanges) - 5} more")


