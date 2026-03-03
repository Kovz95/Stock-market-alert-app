# Add Alert Page Parity Plan: Web UI ↔ Streamlit (Add_Alert.py)

This plan aligns `apps/ui/web/app/alerts/add/` with `pages/Add_Alert.py` for timeframes, exchanges, countries, and optional feature parity.

---

## 1. Timeframe: 1h, 1d, 1wk

**Streamlit:** `["1h", "1d", "1wk"]`, default index 1 → **1d**.

**Web (current):** `["1D", "1W", "1M"]`, default `"1D"`.

**Backend:** Scheduler and alert services already use `1h`, `1d`, `1wk` (e.g. `apps/scheduler/internal/handler/common.go` maps daily→1d, weekly→1wk, hourly→1h; alert evaluation accepts these).

### Tasks

| # | Task | File(s) |
|---|------|--------|
| 1.1 | Change timeframe options from `["1D", "1W", "1M"]` to `["1h", "1d", "1wk"]`. | `_components/AlertBasicFields.tsx` |
| 1.2 | Use display labels: e.g. "1h (Hourly)", "1d (Daily)", "1wk (Weekly)" in the select dropdown. | `_components/AlertBasicFields.tsx` |
| 1.3 | Set default form state `timeframe` to `"1d"` (already correct in `AddAlertForm` defaultFormState if values are updated). | `_components/AddAlertForm.tsx`, `_components/types.ts` (if default is defined there) |
| 1.4 | Ensure create-alert payload sends lowercase `1h` / `1d` / `1wk` (no change needed if form state uses these). | N/A (verify after 1.1) |

---

## 2. Exchanges: Full list (match Streamlit / reference data)

**Streamlit:** Exchanges come from **market data** (DB): `market_data['Exchange'].dropna().unique()`, optionally filtered by country. So the list is dynamic and includes values like NYSE, NASDAQ, TORONTO, LONDON, XETRA, etc.

**Web (current):** Hardcoded `["NYSE", "NASDAQ", "US"]`.

**Reference (Python):**  
- `src/utils/reference_data/exchange_country_mapping.py` → `EXCHANGE_COUNTRY_MAP` keys (e.g. TOKYO, TAIWAN, NYSE, NASDAQ, LONDON, XETRA, TORONTO, ASX, …).  
- `src/utils/reference_data/exchange_mappings.py` → `EXCHANGE_CODE_TO_NAME` (e.g. NYSE, NASDAQ, LSE, TSE, FRA, …) and `EXCHANGE_COUNTRY_MAP` (same style as exchange_country_mapping).

### Tasks

| # | Task | File(s) |
|---|------|--------|
| 2.1 | Add a shared list of exchange codes used by the app (either static or from API). Recommended: **static list** derived from Python `EXCHANGE_COUNTRY_MAP` keys so UI and backend stay in sync. | New: `_components/constants.ts` or `lib/reference-data.ts`; or extend existing constants. |
| 2.2 | Populate the list with all exchanges from `EXCHANGE_COUNTRY_MAP` (e.g. NYSE, NASDAQ, NYSE AMERICAN, NYSE ARCA, CBOE BZX, TORONTO, LONDON, XETRA, FRANKFURT, EURONEXT PARIS, EURONEXT AMSTERDAM, MILAN, MADRID, ASX, TOKYO, TAIWAN, HONG KONG, SINGAPORE, NSE INDIA, BSE INDIA, etc.). | Same file as 2.1. |
| 2.3 | Replace hardcoded `EXCHANGES = ["NYSE", "NASDAQ", "US"]` in `AlertBasicFields.tsx` with the new list (import from constants/reference). | `_components/AlertBasicFields.tsx` |
| 2.4 | Keep "US" in the list if it appears in your DB/market data; otherwise use only the exchange codes from reference data. | Same as 2.2. |
| 2.5 | (Optional) Add an API that returns exchanges (and optionally countries) from the backend so the web app can show the same dynamic list as Streamlit. | Backend + `lib/` or `actions/` + hook. |

---

## 3. Countries: Full list (match Streamlit / reference data)

**Streamlit:** Countries from **market data**: `market_data['Country'].dropna().unique()`.

**Web (current):** Hardcoded `["US", "CA"]`.

**Reference (Python):** `src/utils/reference_data/country_mapping.py` → `COUNTRY_CODE_TO_NAME` (US, CA, UK, JP, DE, FR, AU, CH, NL, IT, ES, SE, NO, DK, FI, BE, IE, PT, AT, PL, GR, HU, CZ, TR, MX, HK, SG, TW, MY, KR, IN, CN, TH, ID, PH, VN).

### Tasks

| # | Task | File(s) |
|---|------|--------|
| 3.1 | Add a shared list of country codes (and optional display names). Recommended: mirror `COUNTRY_CODE_TO_NAME` keys. | Same constants/reference file as exchanges (e.g. `_components/constants.ts` or `lib/reference-data.ts`). |
| 3.2 | Replace hardcoded `COUNTRIES = ["US", "CA"]` in `AlertBasicFields.tsx` with the full list. | `_components/AlertBasicFields.tsx` |
| 3.3 | (Optional) Show "Country (Code)" or "Code – Country" in the dropdown using the display-name mapping. | `_components/AlertBasicFields.tsx` |

---

## 4. Default form state

| # | Task | File(s) |
|---|------|--------|
| 4.1 | Set default `timeframe` to `"1d"`. | `_components/AddAlertForm.tsx` (defaultFormState) |
| 4.2 | Set default `exchange` to a sensible value (e.g. `"NYSE"` or first in the new list). | Same. |
| 4.3 | Set default `country` to `"US"` (or first in list). | Same. |

---

## 5. Backend / API checks

| # | Task | Notes |
|---|------|--------|
| 5.1 | Confirm create-alert API accepts `timeframe` values `1h`, `1d`, `1wk` (no case sensitivity issues). | Already used in scheduler/alert services. |
| 5.2 | Confirm DB and alert evaluation support all new exchange/country values (no allowlist). | If backend validates against a fixed list, extend that list to match. |

---

## 6. Optional parity (Streamlit features not yet on web)

These are in Add_Alert.py but not required for the “timeframe + exchanges + countries” parity. Can be scheduled later.

| Feature | Streamlit behavior | Web follow-up |
|--------|--------------------|----------------|
| **Multi-timeframe comparison** | Checkbox “Enable Multi-Timeframe Comparison”; primary/comparison timeframe (1d, 1wk). | Add optional section: enable multi-TF, primary TF, comparison TF; pass to API if backend supports it. |
| **Mixed timeframe conditions** | Checkbox “Enable Mixed Timeframe Conditions”; docs for e.g. `rsi_weekly`, `Close_weekly`. | Add UI/docs for weekly/daily mixed conditions if backend supports. |
| **Asset type** | Sidebar: All, Stocks, ETFs, Futures. | Optional: asset type filter or selector when picking tickers. |
| **Ratio: per-leg exchange** | For ratio alerts, “Select First Exchange” / “Select Second Exchange” then pick symbol per exchange. | Optional: when isRatio, allow exchange (and country) per leg (ticker1/ticker2). |
| **Futures adjustment** | Select “none”, “panama”, “ratio” for futures. | Optional: show adjustment method when asset type is Futures (or when exchange suggests futures). |
| **Dynamic exchanges/countries** | Loaded from market_data (DB). | Optional: replace static lists with API that returns exchanges/countries from DB. |

---

## 7. Implementation order (recommended)

1. **Timeframe (Section 1)** – small, clear change; unblocks correct scheduler/alert behavior.
2. **Constants / reference data (Sections 2.1–2.2, 3.1)** – one shared file with exchanges and countries.
3. **AlertBasicFields (Sections 2.3, 3.2, 1.1–1.3)** – wire timeframes (1h, 1d, 1wk), exchanges, and countries into the form.
4. **Defaults (Section 4)** – ensure `timeframe: "1d"`, sensible exchange/country defaults.
5. **Verification (Section 5)** – create an alert with 1h/1d/1wk and various exchanges/countries; confirm backend and list views accept them.
6. **Optional (Section 6)** – multi-timeframe, mixed conditions, asset type, ratio exchanges, futures adjustment, dynamic lists – as needed.

---

## 8. Quick reference: values to use

**Timeframes (select value → label):**

- `1h` → "1h (Hourly)"
- `1d` → "1d (Daily)"  
- `1wk` → "1wk (Weekly)"

**Exchanges (examples from Python EXCHANGE_COUNTRY_MAP):**

NYSE, NASDAQ, NYSE AMERICAN, NYSE ARCA, CBOE BZX, TORONTO, LONDON, XETRA, FRANKFURT, EURONEXT AMSTERDAM, EURONEXT BRUSSELS, EURONEXT DUBLIN, EURONEXT LISBON, EURONEXT PARIS, MILAN, MADRID, SPAIN, VIENNA, WARSAW, ATHENS, PRAGUE, BUDAPEST, OSLO, SIX SWISS, TOKYO, TAIWAN, HONG KONG, SINGAPORE, MALAYSIA, INDONESIA, THAILAND, ASX, OMX NORDIC ICELAND, OMX NORDIC STOCKHOLM, OMX NORDIC HELSINKI, OMX NORDIC COPENHAGEN, BSE INDIA, NSE INDIA, SANTIAGO, BUENOS AIRES, MEXICO, COLOMBIA, SAO PAULO, ISTANBUL, JSE.

(Include “US” if your data uses it as an exchange.)

**Countries (from COUNTRY_CODE_TO_NAME):**

US, CA, UK, JP, DE, FR, AU, CH, NL, IT, ES, SE, NO, DK, FI, BE, IE, PT, AT, PL, GR, HU, CZ, TR, MX, HK, SG, TW, MY, KR, IN, CN, TH, ID, PH, VN.

---

*Document generated to align web Add Alert page with Streamlit Add_Alert.py (timeframes 1d/1wk/1h, exchanges, countries).*
