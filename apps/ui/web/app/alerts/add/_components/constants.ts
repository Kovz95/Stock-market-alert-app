/**
 * Reference data for Add Alert form: timeframes, exchanges, countries.
 * Aligned with Streamlit Add_Alert.py and Python src/utils/reference_data.
 */

/** Timeframe value + label for select. Backend expects 1h, 1d, 1wk. */
export const TIMEFRAMES = [
  { value: "1h", label: "1h (Hourly)" },
  { value: "1d", label: "1d (Daily)" },
  { value: "1wk", label: "1wk (Weekly)" },
] as const;

export const DEFAULT_TIMEFRAME = "1d";

/** Exchange codes. Matches Python EXCHANGE_COUNTRY_MAP + US when used as exchange in data. */
export const EXCHANGES = [
  "All",
  "NYSE",
  "NASDAQ",
  "US",
  "NYSE AMERICAN",
  "NYSE ARCA",
  "CBOE BZX",
  "TORONTO",
  "LONDON",
  "XETRA",
  "FRANKFURT",
  "EURONEXT AMSTERDAM",
  "EURONEXT BRUSSELS",
  "EURONEXT DUBLIN",
  "EURONEXT LISBON",
  "EURONEXT PARIS",
  "MILAN",
  "MADRID",
  "SPAIN",
  "VIENNA",
  "WARSAW",
  "ATHENS",
  "PRAGUE",
  "BUDAPEST",
  "OSLO",
  "SIX SWISS",
  "TOKYO",
  "TAIWAN",
  "HONG KONG",
  "SINGAPORE",
  "MALAYSIA",
  "INDONESIA",
  "THAILAND",
  "ASX",
  "OMX NORDIC ICELAND",
  "OMX NORDIC STOCKHOLM",
  "OMX NORDIC HELSINKI",
  "OMX NORDIC COPENHAGEN",
  "BSE INDIA",
  "NSE INDIA",
  "SANTIAGO",
  "BUENOS AIRES",
  "MEXICO",
  "COLOMBIA",
  "SAO PAULO",
  "ISTANBUL",
  "JSE",
] as const;

export const DEFAULT_EXCHANGE = "All";

/** Country code → display name. Matches Python COUNTRY_CODE_TO_NAME. */
export const COUNTRY_CODE_TO_NAME: Record<string, string> = {
  All: "All Countries",
  US: "United States",
  CA: "Canada",
  UK: "United Kingdom",
  JP: "Japan",
  DE: "Germany",
  FR: "France",
  AU: "Australia",
  CH: "Switzerland",
  NL: "Netherlands",
  IT: "Italy",
  ES: "Spain",
  SE: "Sweden",
  NO: "Norway",
  DK: "Denmark",
  FI: "Finland",
  BE: "Belgium",
  IE: "Ireland",
  PT: "Portugal",
  AT: "Austria",
  PL: "Poland",
  GR: "Greece",
  HU: "Hungary",
  CZ: "Czech Republic",
  TR: "Turkey",
  MX: "Mexico",
  HK: "Hong Kong",
  SG: "Singapore",
  TW: "Taiwan",
  MY: "Malaysia",
  KR: "South Korea",
  IN: "India",
  CN: "China",
  TH: "Thailand",
  ID: "Indonesia",
  PH: "Philippines",
  VN: "Vietnam",
};

/** Country codes for select - manually ordered with All first, then sorted. */
export const COUNTRIES = ["All", ...Object.keys(COUNTRY_CODE_TO_NAME).filter(c => c !== "All").sort()] as readonly string[];

export const DEFAULT_COUNTRY = "All";
