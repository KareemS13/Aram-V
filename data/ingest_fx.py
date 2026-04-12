"""
Exchange rate ingestion for AMD/USD, AMD/EUR, AMD/RUB.

Primary source: Central Bank of Armenia (CBA) SOAP web service.
  WSDL: https://api.cba.am/exchangerates.asmx?wsdl
  Method: ExchangeRatesByDateRangeByISO
  Returns: daily rates → resampled to monthly mean.

Fallback: IMF IFS SDMX-JSON REST API (AMD/USD only).

RUB gap handling (March 2022 ~2 weeks):
  Fill using cross-rate: AMD/RUB = (AMD/USD) / (RUB/USD from FRED).
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from config import CBA_WSDL_URL, CBA_ISO_CODES, IMF_IFS_URL, FRED_RUB_USD_SERIES


class CBAExchangeRateLoader:
    """
    Fetch daily AMD exchange rates from CBA SOAP API using raw XML requests.
    Avoids zeep response-parsing issues with CBA's non-standard SOAP envelope.
    """

    SOAP_ENDPOINT = "https://api.cba.am/exchangerates.asmx"
    SOAP_ACTION   = "http://www.cba.am/ExchangeRatesByDateRangeByISO"

    def _soap_request(self, iso_codes: str, date_from: str, date_to: str) -> str:
        """
        Build and send a raw SOAP request to the CBA API.
        Returns the raw XML response text.
        """
        body = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xmlns:xsd="http://www.w3.org/2001/XMLSchema"
               xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <ExchangeRatesByDateRangeByISO xmlns="http://www.cba.am/">
      <ISOCodes>{iso_codes}</ISOCodes>
      <DateFrom>{date_from}T00:00:00</DateFrom>
      <DateTo>{date_to}T00:00:00</DateTo>
    </ExchangeRatesByDateRangeByISO>
  </soap:Body>
</soap:Envelope>"""

        headers = {
            "Content-Type": "text/xml; charset=utf-8",
            "SOAPAction":   self.SOAP_ACTION,
        }
        resp = requests.post(self.SOAP_ENDPOINT, data=body.encode("utf-8"),
                             headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.text

    def _parse_response(self, xml_text: str) -> list[dict]:
        """
        Parse CBA SOAP XML response into a list of {date, iso, rate} dicts.

        CBA response structure:
          .../diffgram/DocumentElement/ExchangeRatesByRange
            Rate       → AMD per 1 foreign unit
            ISO        → currency code (USD, EUR, RUB)
            RateDate   → date (ISO 8601 with timezone)
        """
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)

        records = []
        # Strip namespaces from all tags for robust matching
        for elem in root.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag == "ExchangeRatesByRange":
                iso_el   = elem.find("ISO")
                rate_el  = elem.find("Rate")
                date_el  = elem.find("RateDate")
                if iso_el is not None and rate_el is not None and date_el is not None:
                    try:
                        records.append({
                            "date": pd.to_datetime(date_el.text.split("T")[0]),
                            "iso":  iso_el.text.upper().strip(),
                            "rate": float(rate_el.text),
                        })
                    except Exception:
                        continue
        return records

    def fetch_daily(
        self,
        date_from: str = "2017-01-01",
        date_to: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch daily AMD exchange rates for USD, EUR, RUB.

        Returns pd.DataFrame with DatetimeIndex (daily) and columns:
            fx_usd_amd, fx_eur_amd, fx_rub_amd  (AMD per 1 foreign unit)
        """
        if date_to is None:
            date_to = datetime.today().strftime("%Y-%m-%d")

        # CBA API limits date range — chunk into yearly batches if range > 1 year
        dt_from = datetime.strptime(date_from, "%Y-%m-%d")
        dt_to   = datetime.strptime(date_to,   "%Y-%m-%d")
        all_records = []

        current = dt_from
        while current <= dt_to:
            chunk_end = min(current.replace(year=current.year + 1) - timedelta(days=1), dt_to)
            try:
                xml_text = self._soap_request(
                    CBA_ISO_CODES,
                    current.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                )
                chunk_records = self._parse_response(xml_text)
                all_records.extend(chunk_records)
            except Exception as e:
                raise RuntimeError(f"CBA API failed for {current.strftime('%Y-%m-%d')}: {e}")
            current = chunk_end + timedelta(days=1)

        if not all_records:
            raise ValueError("CBA API returned no records.")

        df = pd.DataFrame(all_records)
        df = df.pivot_table(index="date", columns="iso", values="rate", aggfunc="mean")
        df.columns.name = None

        rename_map = {"USD": "fx_usd_amd", "EUR": "fx_eur_amd", "RUB": "fx_rub_amd"}
        df = df.rename(columns=rename_map)
        df = df.reindex(columns=["fx_usd_amd", "fx_eur_amd", "fx_rub_amd"])
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def to_monthly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Resample daily rates to monthly mean (month-start frequency)."""
        return daily_df.resample("MS").mean()

    def fetch_monthly(
        self,
        date_from: str = "2017-01-01",
        date_to: str | None = None,
    ) -> pd.DataFrame:
        """Convenience: fetch daily + resample to monthly."""
        daily = self.fetch_daily(date_from, date_to)
        monthly = self.to_monthly(daily)
        return monthly


class IMFFallbackLoader:
    """Fetch AMD/USD from IMF IFS SDMX-JSON API (no key required)."""

    def fetch_monthly(
        self,
        date_from: str = "2017-01-01",
        date_to: str | None = None,
    ) -> pd.Series:
        """
        Returns monthly AMD/USD rate as pd.Series (AMD per 1 USD).
        Index is DatetimeIndex at month-start frequency.
        """
        url = IMF_IFS_URL
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"IMF IFS API failed: {e}")

        try:
            obs = (
                data["CompactData"]["DataSet"]["Series"]["Obs"]
            )
            records = []
            for o in obs:
                date_str = o["@TIME_PERIOD"]   # e.g. "2023-01"
                val = o.get("@OBS_VALUE")
                if val and val not in ("", "NA"):
                    records.append({
                        "date": pd.to_datetime(date_str + "-01"),
                        "fx_usd_amd": float(val),
                    })
        except (KeyError, TypeError) as e:
            raise ValueError(f"Could not parse IMF IFS response: {e}")

        series = (
            pd.DataFrame(records)
            .set_index("date")["fx_usd_amd"]
            .sort_index()
        )
        series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp("MS")

        if date_from:
            series = series[series.index >= pd.Timestamp(date_from)]
        if date_to:
            series = series[series.index <= pd.Timestamp(date_to)]

        return series


def _fill_rub_gap(monthly_df: pd.DataFrame, fred_api_key: str = "") -> pd.DataFrame:
    """
    Fill missing RUB/AMD values using cross-rate:
        AMD/RUB = (AMD/USD) / (RUB/USD)

    RUB/USD from FRED series CCUSMA02RUM618N (if API key provided),
    otherwise linearly interpolates the gap.
    """
    rub = monthly_df["fx_rub_amd"].copy()
    missing = rub[rub.isna()]

    if missing.empty:
        return monthly_df

    if fred_api_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=fred_api_key)
            rub_usd = fred.get_series(
                FRED_RUB_USD_SERIES,
                observation_start=str(missing.index.min().date()),
                observation_end=str(missing.index.max().date()),
            )
            rub_usd = rub_usd.resample("MS").mean()

            for date in missing.index:
                if date in rub_usd.index and not pd.isna(rub_usd[date]):
                    usd_amd = monthly_df.loc[date, "fx_usd_amd"]
                    if not pd.isna(usd_amd) and rub_usd[date] != 0:
                        monthly_df.loc[date, "fx_rub_amd"] = usd_amd / rub_usd[date]
            return monthly_df
        except Exception:
            pass  # Fall through to linear interpolation

    # Fallback: linear interpolation for short gaps (≤4 months)
    gap_len = len(missing)
    if gap_len <= 4:
        monthly_df["fx_rub_amd"] = monthly_df["fx_rub_amd"].interpolate(method="linear")
        warnings.warn(
            f"RUB/AMD: filled {gap_len} missing months by linear interpolation."
        )
    else:
        warnings.warn(
            f"RUB/AMD: {gap_len} missing months could not be filled. "
            "Provide FRED_API_KEY for cross-rate fill."
        )

    return monthly_df


def _load_fx_from_fred(
    fred_api_key: str,
    date_from: str,
    date_to: str | None,
) -> pd.DataFrame | None:
    """
    Load AMD exchange rates from FRED as a fallback.

    FRED series:
      DEXARMEN  - not available for AMD directly, but we can use:
      CCUSMA02ARM618N - Armenian Dram to USD (monthly, national currency per USD)
    """
    if not fred_api_key:
        return None
    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_api_key)

        # Armenian Dram per USD (AMD per 1 USD)
        # FRED series: Armenian Dram to US Dollar spot exchange rate
        usd_series = None
        for series_id in ["CCUSMA02ARM618N", "DEXARMEN"]:
            try:
                s = fred.get_series(series_id, observation_start=date_from,
                                    observation_end=date_to)
                if len(s) > 0:
                    usd_series = s.resample("MS").mean()
                    usd_series.name = "fx_usd_amd"
                    print(f"FX: loaded AMD/USD from FRED ({series_id}).")
                    break
            except Exception:
                continue

        if usd_series is None:
            return None

        df = pd.DataFrame({"fx_usd_amd": usd_series})
        df["fx_eur_amd"] = np.nan
        df["fx_rub_amd"] = np.nan
        df.index = pd.DatetimeIndex(df.index)
        return df

    except Exception as e:
        warnings.warn(f"FRED FX fetch failed: {e}")
        return None


def load_fx(
    date_from: str = "2017-01-01",
    date_to: str | None = None,
    fred_api_key: str = "",
) -> pd.DataFrame:
    """
    Load monthly AMD exchange rates (USD, EUR, RUB).

    Tries CBA SOAP API first; falls back to IMF for USD if CBA fails.

    Returns
    -------
    pd.DataFrame with DatetimeIndex (MS) and columns:
        fx_usd_amd, fx_eur_amd, fx_rub_amd
    """
    try:
        loader = CBAExchangeRateLoader()
        df = loader.fetch_monthly(date_from, date_to)
        print("FX: loaded from CBA SOAP API.")
    except Exception as cba_err:
        warnings.warn(f"CBA API failed ({cba_err}). Trying FRED for exchange rates.")
        df = _load_fx_from_fred(fred_api_key, date_from, date_to)
        if df is None:
            warnings.warn("FRED FX failed too. Trying IMF IFS for USD only.")
            try:
                imf = IMFFallbackLoader()
                usd = imf.fetch_monthly(date_from, date_to)
                df = pd.DataFrame({"fx_usd_amd": usd})
                df["fx_eur_amd"] = np.nan
                df["fx_rub_amd"] = np.nan
                print("FX: loaded USD from IMF IFS.")
            except Exception as imf_err:
                raise RuntimeError(
                    f"All FX sources failed. CBA: {cba_err}. IMF: {imf_err}. "
                    "Check internet connection."
                )

    # Fill RUB gap around March 2022
    df = _fill_rub_gap(df, fred_api_key=fred_api_key)

    # Forward-fill up to 3 days worth (weekends / public holidays)
    df = df.ffill(limit=1)

    return df
