
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import re
import numpy as np

# NOTE: Keep dependencies minimal for Streamlit Cloud.

def _to_utc(series: pd.Series, source_tz: str) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=False)
    if getattr(s.dt, "tz", None) is not None:
        return s.dt.tz_convert("UTC")
    return s.dt.tz_localize(source_tz).dt.tz_convert("UTC")

def _clean_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def _safe_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

def bucket_3h(ts_report: pd.Series) -> pd.Series:
    return ts_report.dt.floor("3H")

def plan_category(plan: pd.Series) -> pd.Series:
    s = plan.astype(str).str.lower()
    return np.where(s.str.contains("futures", na=False), "Futures", "CFD")

def is_automation(internal_status: pd.Series) -> pd.Series:
    return internal_status.astype(str).str.lower().str.contains("automation", na=False)


def _extract_rise_email_from_desc(desc: pd.Series) -> pd.Series:
    """Extract email from Rise Description between 'paid to' and 'rise id'.

    We rely on the current Rise report format used in the app.
    """
    d = desc.astype(str)
    # Example pattern: "Paid to someone@example.com ... Rise ID: 123"
    return d.str.extract(r"paid\s*to\s*([\w.+'-]+@[\w.-]+)\s+rise\s*id", flags=re.IGNORECASE)[0]


def find_rise_wallet_payout_not_in_backend(
    backend_df: pd.DataFrame,
    rise_df: pd.DataFrame,
    *,
    backend_ts_col: str,
    backend_tz: str,
    backend_email_col: str,
    backend_amount_col: str,
    rise_ts_col: str,
    rise_tz: str,
    rise_amount_col: str,
    rise_desc_col: str,
    report_start: pd.Timestamp,
    report_end: pd.Timestamp,
    report_tz: str,
    tolerance_minutes: int = 15,
    amount_tolerance_usd: float = 0.10,
) -> pd.DataFrame:
    """Find Rise wallet payout-like rows that do not exist in backend.

    Payout-like = Rise rows where we can extract an email from the Description.
    Matching rule = same email + abs(amount diff) <= amount_tolerance_usd
    AND abs(time diff) <= tolerance_minutes, one-to-one (no reuse of backend rows).

    Returned dataframe is in report_tz and filtered to the selected report window.
    """
    if backend_df is None or rise_df is None or len(backend_df) == 0 or len(rise_df) == 0:
        return pd.DataFrame()

    # --- Backend prep (in report_tz window)
    b = backend_df.copy()
    b_ts = _to_utc(b[backend_ts_col], backend_tz).dt.tz_convert(report_tz)
    b_email = b[backend_email_col].astype(str).str.strip().str.lower()
    b_amt = pd.to_numeric(b[backend_amount_col], errors="coerce")

    b_win = b[(b_ts >= report_start) & (b_ts < report_end)].copy()
    if len(b_win) == 0:
        return rise_df.iloc[0:0].copy()
    b_win["_ts_report"] = b_ts.loc[b_win.index]
    b_win["_email"] = b_email.loc[b_win.index]
    b_win["_amt"] = b_amt.loc[b_win.index]
    b_win = b_win.dropna(subset=["_ts_report", "_email", "_amt"])

    # --- Rise prep (in report_tz window)
    r = rise_df.copy()
    r_ts = _to_utc(r[rise_ts_col], rise_tz).dt.tz_convert(report_tz)
    r_amt = pd.to_numeric(r[rise_amount_col], errors="coerce")
    r_email = _extract_rise_email_from_desc(r[rise_desc_col]).astype(str).str.strip().str.lower()

    r_rep = r[(r_ts >= report_start) & (r_ts < report_end)].copy()
    if len(r_rep) == 0:
        return r.iloc[0:0].copy()
    r_rep["_ts_report"] = r_ts.loc[r_rep.index]
    r_rep["_email"] = r_email.loc[r_rep.index]
    r_rep["_amt"] = r_amt.loc[r_rep.index]
    r_rep = r_rep.dropna(subset=["_ts_report", "_email", "_amt"])
    r_rep = r_rep[r_rep["_email"].str.contains("@", na=False)]
    if len(r_rep) == 0:
        return r.iloc[0:0].copy()

    # Work in cents to avoid float issues
    tol_cents = int(round(amount_tolerance_usd * 100))
    b_win["_cents"] = (b_win["_amt"].round(2) * 100).round().astype("Int64")
    r_rep["_cents"] = (r_rep["_amt"].round(2) * 100).round().astype("Int64")
    b_win = b_win.dropna(subset=["_cents"]).copy()
    r_rep = r_rep.dropna(subset=["_cents"]).copy()

    # Index backend by email for quick candidate lookups
    b_by_email = {}
    for idx, row in b_win.iterrows():
        b_by_email.setdefault(row["_email"], []).append((idx, int(row["_cents"]), row["_ts_report"]))
    for k in b_by_email:
        b_by_email[k].sort(key=lambda x: x[2])

    used_backend = set()
    missing_rows = []

    r_sorted = r_rep.sort_values("_ts_report")
    for r_idx, rrow in r_sorted.iterrows():
        email = rrow["_email"]
        if email not in b_by_email:
            missing_rows.append(r_idx)
            continue

        target_cents = int(rrow["_cents"])
        target_ts = rrow["_ts_report"]
        best = None
        best_key = None

        for b_idx, b_cents, b_tsrep in b_by_email[email]:
            if b_idx in used_backend:
                continue
            if abs(b_cents - target_cents) > tol_cents:
                continue
            dt_min = abs((b_tsrep - target_ts).total_seconds()) / 60.0
            if dt_min > float(tolerance_minutes):
                continue
            score = (abs(b_cents - target_cents), dt_min)
            if best is None or score < best:
                best = score
                best_key = b_idx

        if best_key is None:
            missing_rows.append(r_idx)
        else:
            used_backend.add(best_key)

    out = r_rep.loc[missing_rows].copy()
    if len(out) == 0:
        return out

    # Add friendly columns for display
    out["Report Time"] = out["_ts_report"].dt.strftime("%Y-%m-%d %I:%M %p")
    out["Rise Email"] = out["_email"]
    out["Wallet Amount"] = out["_amt"].round(2)
    return out

@dataclass
class ReconResult:
    matched: pd.DataFrame
    late_sync: pd.DataFrame
    missing_true: pd.DataFrame
    summary_3h: pd.DataFrame

def _build_summary(matched: pd.DataFrame, late_sync: pd.DataFrame, missing_true: pd.DataFrame, report_start: pd.Timestamp, report_end: pd.Timestamp, report_tz: str) -> pd.DataFrame:
    if matched.empty:
        summary = pd.DataFrame(columns=["bucket_3h","matched_count","backend_total","wallet_total","diff_total","abs_diff_total"])
    else:
        summary = (
            matched.groupby("bucket_3h")
            .agg(
                matched_count=("amount_backend","size"),
                backend_total=("amount_backend","sum"),
                wallet_total=("amount_wallet","sum"),
                diff_total=("amount_diff","sum"),
                abs_diff_total=("amount_diff", lambda s: float(np.nansum(np.abs(s)))),
            )
            .reset_index()
        )

    miss = missing_true.groupby("bucket_3h").size().reset_index(name="missing_count") if not missing_true.empty else pd.DataFrame(columns=["bucket_3h","missing_count"])
    late = late_sync.groupby("bucket_3h").size().reset_index(name="late_sync_count") if not late_sync.empty else pd.DataFrame(columns=["bucket_3h","late_sync_count"])

    summary = summary.merge(miss, on="bucket_3h", how="outer").merge(late, on="bucket_3h", how="outer")

    all_buckets = pd.date_range(start=report_start, end=report_end, freq="3H", inclusive="left").tz_convert(report_tz)
    all_df = pd.DataFrame({"bucket_3h": all_buckets})
    summary = all_df.merge(summary, on="bucket_3h", how="left").sort_values("bucket_3h")

    for c in ["matched_count","missing_count","late_sync_count"]:
        summary[c] = summary[c].fillna(0).astype(int)
    for c in ["backend_total","wallet_total","diff_total","abs_diff_total"]:
        summary[c] = summary[c].fillna(0.0)

    return summary

def reconcile_exact(
    backend_df: pd.DataFrame,
    wallet_df: pd.DataFrame,
    backend_ts_col: str,
    backend_tz: str,
    backend_id_col: str,
    backend_amount_col: str,
    wallet_ts_col: str,
    wallet_tz: str,
    wallet_id_col: str,
    wallet_amount_col: str,
    report_tz: str,
    report_start: pd.Timestamp,
    report_end: pd.Timestamp,
    tolerance_minutes: int = 15,
) -> ReconResult:
    b = backend_df.copy()
    w = wallet_df.copy()

    b["txn_id"] = _clean_id(b[backend_id_col])
    w["txn_id"] = _clean_id(w[wallet_id_col])

    b["ts_utc"] = _to_utc(b[backend_ts_col], backend_tz)
    w["ts_utc"] = _to_utc(w[wallet_ts_col], wallet_tz)

    b["ts_report_backend"] = b["ts_utc"].dt.tz_convert(report_tz)
    w["ts_report_wallet"] = w["ts_utc"].dt.tz_convert(report_tz)

    b["amount_backend"] = _safe_float(b[backend_amount_col])
    w["amount_wallet"] = _safe_float(w[wallet_amount_col]).abs()

    b_win = b[(b["ts_report_backend"] >= report_start) & (b["ts_report_backend"] < report_end)].copy()
    # Add a stable row id so if the wallet report contains duplicate Tracking IDs
    # (same txn_id repeated), the join does not duplicate backend rows.
    b_win["__row_id"] = np.arange(len(b_win), dtype=np.int64)

    merged = b_win.merge(
        w[["txn_id","ts_report_wallet","amount_wallet"]],
        on="txn_id",
        how="left",
    )

    merged["delay_min"] = (merged["ts_report_backend"] - merged["ts_report_wallet"]).dt.total_seconds() / 60
    merged["amount_diff"] = merged["amount_backend"] - merged["amount_wallet"]

    # De-duplicate join explosions: if wallet has multiple rows with the same
    # Tracking ID, keep the wallet row that is closest in time (and then closest
    # in amount) for each backend payout row.
    if merged["__row_id"].duplicated().any():
        merged = (
            merged.assign(
                _abs_delay=merged["delay_min"].abs(),
                _abs_amt=merged["amount_diff"].abs(),
            )
            .sort_values(["__row_id", "_abs_delay", "_abs_amt"], na_position="last")
            .drop_duplicates(subset=["__row_id"], keep="first")
            .drop(columns=["_abs_delay", "_abs_amt"])
            .reset_index(drop=True)
        )

    matched = merged[(merged["ts_report_wallet"].notna()) & (merged["delay_min"].abs() <= tolerance_minutes)].copy()
    late_sync = merged[(merged["ts_report_wallet"].notna()) & (merged["delay_min"].abs() > tolerance_minutes)].copy()
    missing_true = merged[merged["ts_report_wallet"].isna()].copy()

    for df in (matched, late_sync, missing_true):
        df["bucket_3h"] = bucket_3h(df["ts_report_backend"])
        df.drop(columns=["__row_id"], errors="ignore", inplace=True)

    summary_3h = _build_summary(matched, late_sync, missing_true, report_start, report_end, report_tz)
    return ReconResult(matched, late_sync, missing_true, summary_3h)

def reconcile_rise_substring(
    backend_df: pd.DataFrame,
    rise_df: pd.DataFrame,
    backend_ts_col: str,
    backend_tz: str,
    backend_id_col: str,
    backend_amount_col: str,
    rise_ts_col: str,
    rise_tz: str,
    rise_desc_col: str,
    rise_amount_col: str,
    report_tz: str,
    report_start: pd.Timestamp,
    report_end: pd.Timestamp,
    tolerance_minutes: int = 15,
) -> ReconResult:
    """
    Rise matching (UPDATED):
    - Match Backend **Payment method Email** (passed via backend_id_col) to Rise email extracted from Rise Description.
    - Amount tolerance: ±$0.10 (compared in cents; avoids float issues).
    - One-to-one matching: a Rise row cannot be matched more than once.
    - If multiple candidates exist for same email, pick smallest amount diff; tie-break by closest timestamp.
    """
    amount_tolerance_usd = 0.10
    tol_cents = int(round(amount_tolerance_usd * 100))

    b = backend_df.copy()
    r = rise_df.copy()

    def _norm_email(x) -> str:
        return str(x).strip().lower()

    def _extract_email(desc: str):
        s = str(desc).lower()
        m1 = re.search(r"paid to\s+([a-z0-9._%+-]+@[a-z0-9.-]+)", s)
        if m1:
            return m1.group(1)
        m2 = re.search(r"([a-z0-9._%+-]+@[a-z0-9.-]+)", s)
        return m2.group(1) if m2 else None

    def _to_cents(series_like) -> pd.Series:
        s = _safe_float(series_like)
        return (s * 100).round().astype("Int64")

    # Keys
    b["match_key"] = b[backend_id_col].apply(_norm_email)
    r["match_key"] = r[rise_desc_col].apply(_extract_email)
    r["match_key"] = r["match_key"].apply(lambda x: _norm_email(x) if (x is not None and pd.notna(x)) else None)

    # Timestamps
    b["ts_utc"] = _to_utc(b[backend_ts_col], backend_tz)
    r["ts_utc"] = _to_utc(r[rise_ts_col], rise_tz)

    b["ts_report_backend"] = b["ts_utc"].dt.tz_convert(report_tz)
    r["ts_report_wallet"] = r["ts_utc"].dt.tz_convert(report_tz)

    # Amounts (Rise amount can be negative; use abs)
    b["amount_backend"] = _safe_float(b[backend_amount_col])
    r["amount_wallet"] = _safe_float(r[rise_amount_col]).abs()

    b["backend_cents"] = _to_cents(b["amount_backend"])
    r["wallet_cents"] = _to_cents(r["amount_wallet"])

    # Window: backend defines the report window; rise allows extra +/- 6h to accommodate timezone/report differences
    b_win = b[(b["ts_report_backend"] >= report_start) & (b["ts_report_backend"] < report_end)].copy()
    r_win = r[(r["ts_report_wallet"] >= report_start - pd.Timedelta(hours=6)) & (r["ts_report_wallet"] < report_end + pd.Timedelta(hours=6))].copy()

    # Build index of rise rows by email within window
    rise_by_email = {}
    for idx, key in r_win["match_key"].items():
        if key:
            rise_by_email.setdefault(key, []).append(idx)

    used_rise = set()
    picked_ts = []
    picked_amt = []
    picked_delay = []
    picked_diff = []
    picked_idx = []

    for _, brow in b_win.iterrows():
        email = brow["match_key"]
        b_cents = brow["backend_cents"]
        b_ts = brow["ts_report_backend"]

        best = None  # (diff_cents, time_diff_sec, ridx)
        for ridx in rise_by_email.get(email, []):
            if ridx in used_rise:
                continue
            w_cents = r_win.at[ridx, "wallet_cents"]
            if pd.isna(b_cents) or pd.isna(w_cents):
                continue
            diff_cents = abs(int(b_cents) - int(w_cents))
            if diff_cents > tol_cents:
                continue
            w_ts = r_win.at[ridx, "ts_report_wallet"]
            if pd.isna(w_ts):
                continue
            tdiff = abs((b_ts - w_ts).total_seconds())
            score = (diff_cents, tdiff, ridx)
            if best is None or score < best:
                best = score

        if best is None:
            picked_ts.append(pd.NaT)
            picked_amt.append(float("nan"))
            picked_delay.append(float("nan"))
            picked_diff.append(float("nan"))
            picked_idx.append(pd.NA)
            continue

        diff_cents, _, ridx = best
        used_rise.add(ridx)

        w_ts = r_win.at[ridx, "ts_report_wallet"]
        w_amt = float(r_win.at[ridx, "amount_wallet"])

        delay_min = (b_ts - w_ts).total_seconds() / 60.0
        amt_diff = float(brow["amount_backend"]) - w_amt

        picked_ts.append(w_ts)
        picked_amt.append(w_amt)
        picked_delay.append(delay_min)
        picked_diff.append(amt_diff)
        picked_idx.append(ridx)

    b_win["ts_report_wallet"] = pd.Series(picked_ts, index=b_win.index)
    b_win["amount_wallet"] = pd.Series(picked_amt, index=b_win.index, dtype="float")
    b_win["delay_min"] = pd.Series(picked_delay, index=b_win.index, dtype="float")
    b_win["amount_diff"] = pd.Series(picked_diff, index=b_win.index, dtype="float")
    b_win["_rise_index"] = pd.Series(picked_idx, index=b_win.index)

    merged = b_win

    matched = merged[(merged["ts_report_wallet"].notna()) & (merged["delay_min"].abs() <= tolerance_minutes)].copy()
    late_sync = merged[(merged["ts_report_wallet"].notna()) & (merged["delay_min"].abs() > tolerance_minutes)].copy()
    missing_true = merged[merged["ts_report_wallet"].isna()].copy()

    for df in (matched, late_sync, missing_true):
        df["bucket_3h"] = bucket_3h(df["ts_report_backend"])

    summary_3h = _build_summary(matched, late_sync, missing_true, report_start, report_end, report_tz)
    return ReconResult(matched, late_sync, missing_true, summary_3h)


def rise_wallet_payout_not_in_backend(
    backend_df: pd.DataFrame,
    rise_df: pd.DataFrame,
    *,
    backend_ts_col: str,
    backend_tz: str,
    backend_email_col: str,
    rise_ts_col: str,
    rise_tz: str,
    rise_desc_col: str,
    rise_amt_col: str,
    report_start: pd.Timestamp,
    report_end: pd.Timestamp,
    report_tz: str,
    tolerance_minutes: int = 15,
    amount_tolerance_usd: float = 0.10,
) -> pd.DataFrame:
    """Find Rise wallet *payout-like* transactions (have an extracted email) that don't match any backend payout.

    Matching rules (one-to-one, greedy):
    - Same email (backend "Payment method Email" vs extracted email from Rise description)
    - Amount within `amount_tolerance_usd` (compared in cents)
    - Timestamp within `tolerance_minutes` in the selected report timezone

    Returned dataframe is filtered to the selected report window (Rise timestamp in report_tz).
    """

    if backend_df is None or rise_df is None or len(backend_df) == 0 or len(rise_df) == 0:
        return pd.DataFrame()

    b = backend_df.copy()
    r = rise_df.copy()

    # Prepare backend
    b["ts_backend_utc"] = _to_utc(b[backend_ts_col], backend_tz)
    b["ts_report_backend"] = b["ts_backend_utc"].dt.tz_convert(report_tz)
    b["_email"] = b[backend_email_col].astype(str).str.strip().str.lower()

    # Prepare rise
    r["ts_wallet_utc"] = _to_utc(r[rise_ts_col], rise_tz)
    r["ts_report_wallet"] = r["ts_wallet_utc"].dt.tz_convert(report_tz)
    r["_email"] = _extract_rise_email_from_desc(r[rise_desc_col])

    # Filter to selected report window *by backend date logic*: we show wallet rows that fall in the report window.
    b_win = b[(b["ts_report_backend"] >= report_start) & (b["ts_report_backend"] < report_end)].copy()
    r_win = r[(r["ts_report_wallet"] >= report_start) & (r["ts_report_wallet"] < report_end)].copy()

    # Keep only payout-like Rise rows (email found)
    r_win = r_win[r_win["_email"].notna() & (r_win["_email"].astype(str).str.len() > 0)].copy()
    if len(b_win) == 0 or len(r_win) == 0:
        return pd.DataFrame()

    # Amounts in cents
    b_win["_amt_cents"] = (pd.to_numeric(b_win[backend_amt_col], errors="coerce").fillna(0.0) * 100).round().astype("int64")
    r_win["_amt_cents"] = (pd.to_numeric(r_win[rise_amt_col], errors="coerce").fillna(0.0) * 100).round().astype("int64")

    tol_cents = int(round(amount_tolerance_usd * 100))

    # Index backend candidates by email
    b_by_email = {}
    for idx, row in b_win.iterrows():
        em = row.get("_email")
        if not em:
            continue
        b_by_email.setdefault(em, []).append(idx)

    used_backend = set()
    missing_rows = []

    # Greedy matching: sort wallet rows by time
    r_win = r_win.sort_values("ts_report_wallet")

    for ridx, rrow in r_win.iterrows():
        em = rrow.get("_email")
        if not em:
            continue

        candidates = [i for i in b_by_email.get(em, []) if i not in used_backend]
        if not candidates:
            missing_rows.append(ridx)
            continue

        r_amt = int(rrow.get("_amt_cents", 0))
        r_ts = rrow.get("ts_report_wallet")

        best = None
        best_score = None

        for bidx in candidates:
            b_amt = int(b_win.at[bidx, "_amt_cents"])
            if abs(b_amt - r_amt) > tol_cents:
                continue
            b_ts = b_win.at[bidx, "ts_report_backend"]
            dt_min = abs((b_ts - r_ts).total_seconds()) / 60.0
            if dt_min > tolerance_minutes:
                continue

            # Score: amount diff then time diff
            score = (abs(b_amt - r_amt), dt_min)
            if best_score is None or score < best_score:
                best = bidx
                best_score = score

        if best is None:
            missing_rows.append(ridx)
        else:
            used_backend.add(best)

    out = r_win.loc[missing_rows].copy() if missing_rows else pd.DataFrame()
    if len(out) == 0:
        return out

    # Friendly columns
    out.rename(
        columns={
            rise_ts_col: "wallet_timestamp",
            rise_amt_col: "wallet_amount",
            rise_desc_col: "wallet_description",
        },
        inplace=True,
    )
    out["wallet_email"] = out["_email"]
    out["report_time"] = out["ts_report_wallet"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    keep = [
        "report_time",
        "wallet_email",
        "wallet_amount",
        "wallet_description",
    ]
    return out[[c for c in keep if c in out.columns]].reset_index(drop=True)
