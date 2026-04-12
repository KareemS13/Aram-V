import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

CPI_RAW_PATH = os.path.join(DATA_RAW_DIR, "armstat_cpi_coicop.xlsx")
CPI_WEIGHTS_PATH = os.path.join(DATA_RAW_DIR, "armstat_cpi_weights.xlsx")
WB_PINKSHEET_PATH = os.path.join(DATA_RAW_DIR, "CMO-Historical-Data-Monthly.xlsx")

# ---------------------------------------------------------------------------
# FRED commodity series IDs
# ---------------------------------------------------------------------------
FRED_COMMODITY_SERIES = {
    "oil_wti":       "POILWTIUSDM",
    "wheat":         "PWHEAMTUSDM",
    "energy_idx":    "PNRGINDEXM",    # Energy commodity index (replaces aluminum)
    "commodity_idx": "PALLFNFINDEXM",
    "nat_gas":       "PNGASEUUSDM",   # European natural gas price (USD/MMBtu)
}

# RUB cross-rate series for gap-filling post-2022 sanctions
FRED_RUB_USD_SERIES = "CCUSMA02RUM618N"

# ---------------------------------------------------------------------------
# CBA exchange rate SOAP endpoint
# ---------------------------------------------------------------------------
CBA_WSDL_URL = "https://api.cba.am/exchangerates.asmx?wsdl"
CBA_ISO_CODES = "USD,EUR,RUB"

# IMF IFS fallback (AMD/USD end-of-period)
IMF_IFS_URL = (
    "http://dataservices.imf.org/REST/SDMX_JSON.svc/"
    "CompactData/IFS/M.AM.ENDA_XDC_USD_RATE"
)

# ---------------------------------------------------------------------------
# COICOP column mapping
# The Excel from ArmStatBank uses the full text names; we map them to short codes.
# Key: substring that appears in the ArmStat column header (case-insensitive)
# Value: short column name used throughout the pipeline
# ---------------------------------------------------------------------------
COICOP_MAP = {
    "00":  "cpi_headline",
    "01":  "cp01_food",
    "02":  "cp02_alc",
    "03":  "cp03_clothing",
    "04":  "cp04_housing",
    "05":  "cp05_furnishings",
    "06":  "cp06_health",
    "07":  "cp07_transport",
    "08":  "cp08_comms",
    "09":  "cp09_recreation",
    "10":  "cp10_education",
    "11":  "cp11_restaurants",
    "12":  "cp12_misc",
}

COICOP_LABELS = {
    "cpi_headline":    "Headline CPI",
    "cp01_food":       "Food & non-alcoholic beverages",
    "cp02_alc":        "Alcoholic beverages & tobacco",
    "cp03_clothing":   "Clothing & footwear",
    "cp04_housing":    "Housing, water, electricity, gas",
    "cp05_furnishings":"Furnishings & household equipment",
    "cp06_health":     "Health",
    "cp07_transport":  "Transport",
    "cp08_comms":      "Communication",
    "cp09_recreation": "Recreation & culture",
    "cp10_education":  "Education",
    "cp11_restaurants":"Restaurants & hotels",
    "cp12_misc":       "Miscellaneous",
}

# ---------------------------------------------------------------------------
# Structural break periods (inclusive on both ends)
# ---------------------------------------------------------------------------
STRUCTURAL_BREAKS = {
    "covid_2020":      ("2020-03-01", "2021-03-01"),
    "ukraine_2022":    ("2022-02-01", "2022-12-01"),
    "rub_crisis_2014": ("2014-11-01", "2015-06-01"),
}

# ---------------------------------------------------------------------------
# Feature engineering: lag structure
# ---------------------------------------------------------------------------
LAG_CONFIG = {
    "cpi_headline_mom": [1, 2, 3, 6, 12],
    "cp01_food":        [1, 2, 3],
    "cp04_housing":     [1, 2],
    "cp07_transport":   [1, 2],
    "fx_usd_amd":       [0, 1, 2, 3],
    "fx_eur_amd":       [1, 2],
    "fx_rub_amd":       [1, 2, 3],
    "oil_wti":          [1, 2, 3],
    "wheat":            [2, 3, 4],
    "energy_idx":       [1, 2],
    "nat_gas":          [1, 2, 3],    # Natural gas: 1-3 month pass-through to heating/transport
    "cba_rate":         [1, 2],       # CBA policy rate: 1-2 month transmission lag
    "ppi_mom":          [1, 2, 3],    # PPI MoM%: producer→consumer price pass-through
    # UN Comtrade trade features (MoM% changes)
    "imports_total_mom":     [1, 2],  # Total import growth → overall import inflation
    "imports_energy_mom":    [1, 2],  # Energy imports → fuel/heating pass-through
    "imports_cereals_mom":   [1, 2],  # Cereal imports → food price pressure
    "exports_total_mom":     [1, 2],  # Export growth → domestic supply pressure
}

# Columns to apply log-differencing (non-stationary price levels)
LOG_DIFF_COLS = ["fx_usd_amd", "fx_eur_amd", "fx_rub_amd",
                 "oil_wti", "wheat", "energy_idx", "commodity_idx", "nat_gas"]
# Note: cba_rate is kept as level (already a % rate, not a price index)

# ---------------------------------------------------------------------------
# SARIMAX exogenous columns (after feature engineering)
# ---------------------------------------------------------------------------
SARIMA_EXOG_COLS = [
    "fx_usd_amd_ld_lag1",
    "oil_wti_ld_lag2",
    "wheat_ld_lag3",
    "covid_2020",
    "ukraine_2022",
]

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
DATA_START = "2017-01-01"       # ArmStatBank table goes back to 2017
EVAL_START = "2023-01-01"       # Walk-forward CV begins here
FORECAST_HORIZON = 12           # Months ahead

GBM_DEFAULT_LAGS = [1, 2, 3, 6, 12]

# SHAP feature grouping for driver attribution charts
SHAP_GROUPS = {
    "Food prices":       ["cp01_food"],
    "Energy":            ["oil_wti", "nat_gas", "cp04_housing", "cp07_transport"],
    "Exchange rate":     ["fx_usd_amd", "fx_eur_amd", "fx_rub_amd"],
    "Global commodities":["wheat", "energy_idx", "commodity_idx"],
    "Monetary policy":   ["cba_rate"],
    "Producer prices":   ["ppi_mom"],
    "Trade flows":       ["imports_total_mom", "imports_energy_mom",
                          "imports_cereals_mom", "exports_total_mom"],
    "Own momentum":      ["cpi_headline_mom"],
    "Seasonality":       ["month_sin", "month_cos", "is_q1", "is_summer", "year_trend"],
    "Structural shocks": list(STRUCTURAL_BREAKS.keys()) + ["rub_ukraine"],
}

SHAP_GROUP_COLORS = {
    "Food prices":        "#4CAF50",
    "Energy":             "#FF9800",
    "Exchange rate":      "#2196F3",
    "Global commodities": "#795548",
    "Monetary policy":    "#00BCD4",
    "Producer prices":    "#FF5722",
    "Trade flows":        "#607D8B",
    "Own momentum":       "#9E9E9E",
    "Seasonality":        "#9C27B0",
    "Structural shocks":  "#F44336",
}


@dataclass
class Config:
    cpi_raw_path: str = CPI_RAW_PATH
    cpi_weights_path: str = CPI_WEIGHTS_PATH
    wb_pinksheet_path: str = WB_PINKSHEET_PATH
    fred_api_key: str = field(default_factory=lambda: os.environ.get("FRED_API_KEY", ""))
    data_start: str = DATA_START
    eval_start: str = EVAL_START
    forecast_horizon: int = FORECAST_HORIZON
    sarima_weight: float | None = None   # None = use inverse-MAE from CV
    gbm_weight: float | None = None

    @classmethod
    def from_args(cls, args) -> "Config":
        return cls(
            fred_api_key=args.fred_key or os.environ.get("FRED_API_KEY", ""),
            eval_start=args.eval_start,
            forecast_horizon=args.horizon,
        )
