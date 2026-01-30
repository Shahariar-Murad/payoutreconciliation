
import pandas as pd
import streamlit as st
from datetime import datetime, date, time, timedelta
import plotly.express as px
import numpy as np

from recon import (
    reconcile_exact,
    reconcile_rise_substring,
    plan_category,
    is_automation,
    rise_wallet_payout_not_in_backend,
)

st.set_page_config(page_title="Payout Recon Platform", layout="wide")
st.title("Payout Reconciliation Platform")

st.markdown("""
<style>
  .share-card {background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
               padding: 14px 16px; border-radius: 14px; margin: 10px 0 16px 0;}
  .share-title {font-size: 22px; font-weight: 700; margin-bottom: 6px;}
  .share-sub {color: rgba(255,255,255,0.70); font-size: 13px; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)


def format_range(ts):
    return f"{ts.strftime('%I:%M %p')} - {(ts + pd.Timedelta(hours=3)).strftime('%I:%M %p')}"

with st.sidebar:
    st.header("Upload 3 files")
    backend_file = st.file_uploader("Backend (Payout Wallet CSV)", type=["csv"])
    crypto_file = st.file_uploader("Crypto wallet report CSV", type=["csv"])
    rise_file = st.file_uploader("Rise report CSV", type=["csv"])

    # Report timezone controls *all* filtering + display (date range, quick windows, charts/tables).
    # Provide common options + a custom override.
    tz_choices = {
        "UTC+6 (Asia/Dhaka)": "Asia/Dhaka",
        "UTC+2 (Etc/GMT-2)": "Etc/GMT-2",  # NOTE: Etc/GMT-2 == UTC+2 (POSIX sign is reversed)
        "UTC (UTC)": "UTC",
        "Custom…": None,
    }
    tz_label = st.selectbox("Report timezone", list(tz_choices.keys()), index=0)
    report_tz = tz_choices[tz_label] or st.text_input("Custom report timezone (IANA)", value="Asia/Dhaka")

    st.header(f"Report date range ({report_tz})")
    dr = st.date_input(
        "Select start and end date",
        value=(date.today() - timedelta(days=1), date.today()),
    )

    # Team-friendly presets: often they reconcile from previous day evening to today morning,
    # then review 3-hour slots within the same day.
    st.subheader("Quick time window (optional)")
    st.caption("Preset windows use the END date above as 'today'.")
    quick_window = st.selectbox(
        "Choose a preset window",
        [
            "Custom (use full selected date range)",
            "Shift: Prev day 06:00 PM → Today 09:00 AM",
            "Today 09:00 AM → 12:00 PM",
            "Today 12:00 PM → 03:00 PM",
            "Today 03:00 PM → 06:00 PM",
        ],
        index=0,
    )

    st.header("Source timezones (naive)")
    backend_tz = st.text_input("Backend timezone", value="Etc/GMT-2")  # UTC+2
    crypto_tz = st.text_input("Crypto report timezone", value="UTC")   # data is GMT+00
    rise_tz = st.text_input("Rise report timezone", value="Asia/Dhaka") # GMT+6

    tol = st.number_input("Max wallet→backend delay (minutes)", min_value=0, max_value=120, value=15)

if not backend_file:
    st.info("Upload at least the Backend file to run reconciliation.")
    st.stop()

run_crypto = crypto_file is not None
run_rise = rise_file is not None
if (not run_crypto) and (not run_rise):
    st.info("Upload either Crypto wallet report or Rise report (or both).")
    st.stop()

start_date, end_date = dr

# Default: whole selected date range (inclusive of end_date)
report_start = pd.Timestamp(datetime.combine(start_date, time(0, 0)), tz=report_tz)
report_end = pd.Timestamp(datetime.combine(end_date + timedelta(days=1), time(0, 0)), tz=report_tz)

# Preset windows (all in report_tz, anchored to END date as "today")
if quick_window != "Custom (use full selected date range)":
    today = end_date
    prev = today - timedelta(days=1)
    if quick_window == "Shift: Prev day 06:00 PM → Today 09:00 AM":
        report_start = pd.Timestamp(datetime.combine(prev, time(18, 0)), tz=report_tz)
        report_end = pd.Timestamp(datetime.combine(today, time(9, 0)), tz=report_tz)
    elif quick_window == "Today 09:00 AM → 12:00 PM":
        report_start = pd.Timestamp(datetime.combine(today, time(9, 0)), tz=report_tz)
        report_end = pd.Timestamp(datetime.combine(today, time(12, 0)), tz=report_tz)
    elif quick_window == "Today 12:00 PM → 03:00 PM":
        report_start = pd.Timestamp(datetime.combine(today, time(12, 0)), tz=report_tz)
        report_end = pd.Timestamp(datetime.combine(today, time(15, 0)), tz=report_tz)
    elif quick_window == "Today 03:00 PM → 06:00 PM":
        report_start = pd.Timestamp(datetime.combine(today, time(15, 0)), tz=report_tz)
        report_end = pd.Timestamp(datetime.combine(today, time(18, 0)), tz=report_tz)

backend = pd.read_csv(backend_file)
crypto = pd.read_csv(crypto_file) if run_crypto else None
rise = pd.read_csv(rise_file) if run_rise else None

backend["_ptype"] = plan_category(backend.get("Plan", pd.Series([""]*len(backend))))
backend["_auto"] = is_automation(backend.get("Internal Status", pd.Series([""]*len(backend))))

pm = backend.get("Payment Method", pd.Series([""]*len(backend))).astype(str).str.lower()
backend_crypto = backend[pm.isin(["usdt","usdc"])].copy() if run_crypto else None
backend_rise = backend[pm.isin(["riseworks","risework","rise"])].copy() if run_rise else None

crypto_res = None
if run_crypto:
    crypto_res = reconcile_exact(
        backend_df=backend_crypto,
        wallet_df=crypto,
        backend_ts_col="Disbursed Time",
        backend_tz=backend_tz,
        backend_id_col="Transaction ID",
        backend_amount_col="Disbursement Amount",
        wallet_ts_col="Created",
        wallet_tz=crypto_tz,
        wallet_id_col="Tracking ID",
        wallet_amount_col="Amount",
        report_tz=report_tz,
        report_start=report_start,
        report_end=report_end,
        tolerance_minutes=int(tol),
    )

rise_res = None
if run_rise:
    rise_res = reconcile_rise_substring(
    backend_df=backend_rise,
    rise_df=rise,
    backend_ts_col="Disbursed Time",
    backend_tz=backend_tz,
    backend_id_col="Payment method Email" ,
    backend_amount_col="Disbursement Amount",
    rise_ts_col="Date",
    rise_tz=rise_tz,
    rise_desc_col="Description",
    rise_amount_col="Amount",
    report_tz=report_tz,
    report_start=report_start,
    report_end=report_end,
    tolerance_minutes=int(tol),
)

tab1, tab2, tab3 = st.tabs(["Payout reconciliation", "Breakdown", "Disbursement totals"])

with tab1:
    st.subheader("Overview")
    a,b,c = st.columns(3)
    # IMPORTANT: These headline counts should align with the Breakdown tab,
    # which is "Matched + Late Sync" and is bucketed by backend time.
    if crypto_res is not None:
        crypto_matched = len(pd.concat([crypto_res.matched, crypto_res.late_sync], ignore_index=True))
    else:
        crypto_matched = 0

    if rise_res is not None:
        rise_matched = len(pd.concat([rise_res.matched, rise_res.late_sync], ignore_index=True))
    else:
        rise_matched = 0
    true_missing = (len(crypto_res.missing_true) if crypto_res is not None else 0) + (len(rise_res.missing_true) if rise_res is not None else 0)
    a.metric("Crypto matched + late sync", crypto_matched)
    b.metric("Rise matched + late sync", rise_matched)
    c.metric("True missing (all)", true_missing)

    # Missing counts (Backend present, Wallet missing)
    m1, m2 = st.columns(2)
    crypto_missing = len(crypto_res.missing_true) if (run_crypto and crypto_res is not None) else 0
    rise_missing = len(rise_res.missing_true) if (run_rise and rise_res is not None) else 0
    m1.metric("Crypto missing (backend not in wallet)", crypto_missing)
    m2.metric("Rise missing (backend not in wallet)", rise_missing)

    # Wallet→Backend mismatch (payout present in wallet but missing in backend)
    w1, w2 = st.columns(2)
    crypto_wallet_not_in_backend_count = 0
    rise_wallet_not_in_backend_count = 0
    try:
        if (crypto_file is not None) and (crypto_df is not None) and len(crypto_df) > 0 and (backend_df is not None) and len(backend_df) > 0:
            c = crypto_df.copy()
            if "Tracking ID" in c.columns:
                c = c[c["Tracking ID"].notna()].copy()
            # Parse times
            c_ts = pd.to_datetime(c.get(CRYPTO_TS_COL), errors="coerce")
            b_ts = pd.to_datetime(backend_crypto.get("Disbursed Time"), errors="coerce")
            # Normalize to UTC
            if getattr(c_ts.dt, "tz", None) is None:
                c_utc = c_ts.dt.tz_localize(crypto_tz, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
            else:
                c_utc = c_ts.dt.tz_convert("UTC")
            if getattr(b_ts.dt, "tz", None) is None:
                b_utc = b_ts.dt.tz_localize(backend_tz, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
            else:
                b_utc = b_ts.dt.tz_convert("UTC")
            # Apply report window (report_tz) on backend date basis
            c_report = c_utc.dt.tz_convert(report_tz)
            b_report = b_utc.dt.tz_convert(report_tz)
            c = c.assign(__ts_report=c_report)
            b = backend_crypto.assign(__ts_report=b_report)
            c = c[(c["__ts_report"] >= report_start) & (c["__ts_report"] < report_end)].copy()
            b = b[(b["__ts_report"] >= report_start) & (b["__ts_report"] < report_end)].copy()
            if (not c.empty) and (not b.empty) and ("Tracking ID" in c.columns) and ("Transaction ID" in b.columns):
                b_ids = set(b["Transaction ID"].astype(str).str.strip())
                crypto_wallet_not_in_backend_count = int((~c["Tracking ID"].astype(str).str.strip().isin(b_ids)).sum())
            else:
                crypto_wallet_not_in_backend_count = 0
    except Exception:
        crypto_wallet_not_in_backend_count = 0
    try:
        if (rise_file is not None) and (rise_df is not None) and len(rise_df) > 0 and (backend_df is not None) and len(backend_df) > 0:
            _rmiss = rise_wallet_payout_not_in_backend(
                backend_rise,
                rise,
                backend_ts_col="Disbursed Time",
                backend_tz=backend_tz,
                backend_email_col="Payment method Email",
                backend_amt_col="Disbursement Amount",
                rise_ts_col=RISE_TS_COL,
                rise_tz=rise_tz,
                rise_desc_col="Description",
                rise_amt_col=RISE_AMT_COL,
                report_tz=report_tz,
                report_start=report_start,
                report_end=report_end,
                tolerance_minutes=int(tol),
                amount_tolerance_usd=0.10,
            )
            rise_wallet_not_in_backend_count = int(len(_rmiss))
    except Exception:
        rise_wallet_not_in_backend_count = 0

    with w1:
        st.metric("Crypto wallet payout not in backend", crypto_wallet_not_in_backend_count)
    with w2:
        st.metric("Rise wallet payout not in backend", rise_wallet_not_in_backend_count)

    st.subheader("Missing transaction details")
    with st.expander("Show missing details", expanded=False):
        st.caption("These are Backend payouts that were not found in the selected wallet report (after applying the 15-minute tolerance).")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Crypto missing (Backend present, Wallet missing)**")
            cm = crypto_res.missing_true.copy() if crypto_res is not None else pd.DataFrame()
            if cm.empty:
                st.write("No missing rows ✅")
            else:
                show_cols = [c for c in ["Disbursed Time","Transaction ID","Disbursement Amount","Payment Method","Plan","Internal Status","Customer Email","Login","Id"] if c in cm.columns]
                st.dataframe(cm[show_cols + [c for c in ["txn_id","ts_report_backend","amount_backend"] if c in cm.columns]].head(200), use_container_width=True, height=220)
        with m2:
            st.markdown("**Rise missing (Backend present, Wallet missing)**")
            rm = rise_res.missing_true.copy() if rise_res is not None else pd.DataFrame()
            if rm.empty:
                st.write("No missing rows ✅")
            else:
                show_cols = [c for c in ["Disbursed Time","Payment method ID","Disbursement Amount","Payment Method","Plan","Internal Status","Customer Email","Login","Id"] if c in rm.columns]
                st.dataframe(rm[show_cols + [c for c in ["txn_id","ts_report_backend","amount_backend"] if c in rm.columns]].head(200), use_container_width=True, height=220)


    st.subheader("3-hour payout counts (backend) — Rise vs Crypto")

    def counts(df):
        if df is None or df.empty:
            return pd.DataFrame(columns=["bucket_3h","count"])
        ts = pd.to_datetime(df["Disbursed Time"], errors="coerce")
        ts = ts.dt.tz_localize(backend_tz).dt.tz_convert(report_tz)
        win = df.copy()
        win["_ts"] = ts
        win = win[(win["_ts"]>=report_start)&(win["_ts"]<report_end)]
        win["bucket_3h"] = win["_ts"].dt.floor("3H")
        return win.groupby("bucket_3h").size().reset_index(name="count")

    cc = counts(backend_crypto).rename(columns={"count":"crypto_count"})
    rc = counts(backend_rise).rename(columns={"count":"rise_count"})
    buckets = pd.DataFrame({"bucket_3h": pd.date_range(start=report_start, end=report_end, freq="3H", inclusive="left").tz_convert(report_tz)})
    counts_3h = buckets.merge(cc,on="bucket_3h",how="left").merge(rc,on="bucket_3h",how="left").fillna(0)
    counts_3h["Time Range"] = counts_3h["bucket_3h"].apply(format_range)

    value_cols = []
    if run_rise: value_cols.append("rise_count")
    if run_crypto: value_cols.append("crypto_count")

    if len(value_cols) == 0:
        st.write("Upload Crypto or Rise report to see chart.")
    else:
        chart_df = counts_3h.melt(id_vars=["bucket_3h","Time Range"], value_vars=value_cols, var_name="Channel", value_name="Count")
        chart_df["Channel"] = chart_df["Channel"].replace({"rise_count":"Rise","crypto_count":"Crypto"})
        chart_df["Date"] = chart_df["bucket_3h"].dt.strftime("%Y-%m-%d")
        times = sorted(chart_df["Time Range"].unique().tolist())
        sel_times = st.multiselect("Filter 3-hour slot (optional)", options=times, default=[])

        fdf = chart_df.copy()
        if sel_times:
            fdf = fdf[fdf["Time Range"].isin(sel_times)].copy()
        fig = px.bar(fdf, x="Time Range", y="Count", color="Channel", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
        st.session_state["__sel_times"] = sel_times
    # --- 3-hour summaries (filtered) ---
    if run_rise:
        st.subheader("Rise 3-hour summary")
        rs = rise_res.summary_3h.copy()
        rs["Date"] = rs["bucket_3h"].dt.strftime("%Y-%m-%d")
        rs["Time Range"] = rs["bucket_3h"].apply(format_range)
        rs = rs[["Date","Time Range","matched_count","late_sync_count","missing_count","backend_total","wallet_total","diff_total","abs_diff_total"]]
        sel_times = st.session_state.get("__sel_times")
        if sel_times:
            rs = rs[rs["Time Range"].isin(sel_times)]
        st.markdown('<div class="share-card"><div class="share-title">Rise 3-hour summary</div><div class="share-sub">Filtered view — ready for screenshot.</div></div>', unsafe_allow_html=True)
        st.dataframe(rs, use_container_width=True, height=260)

        # --- Detail transactions (filtered by selected 3-hour slot) ---
        def _detail_block(res, label):
            sel_times = st.session_state.get("__sel_times") or []
            parts = []
            if res is None:
                return
            if hasattr(res, "matched") and res.matched is not None and not res.matched.empty:
                parts.append(res.matched.assign(_status="Matched"))
            if hasattr(res, "late_sync") and res.late_sync is not None and not res.late_sync.empty:
                parts.append(res.late_sync.assign(_status="Late Sync"))
            if hasattr(res, "missing_true") and res.missing_true is not None and not res.missing_true.empty:
                parts.append(res.missing_true.assign(_status="Missing"))

            if not parts:
                st.info("No detail rows.")
                return

            det = pd.concat(parts, ignore_index=True)
            # Build Time Range from the same bucket_3h used in summary
            if "bucket_3h" in det.columns:
                det["Time Range"] = det["bucket_3h"].apply(format_range)
                det["Date"] = det["bucket_3h"].dt.strftime("%Y-%m-%d")
            elif "ts_report_backend" in det.columns:
                det["_b"] = det["ts_report_backend"].dt.floor("3H")
                det["Time Range"] = det["_b"].apply(format_range)
                det["Date"] = det["_b"].dt.strftime("%Y-%m-%d")
                det.drop(columns=["_b"], inplace=True, errors="ignore")

            if sel_times and "Time Range" in det.columns:
                det = det[det["Time Range"].isin(sel_times)]

            st.markdown(f'<div class="share-card"><div class="share-title">{label} transaction details</div><div class="share-sub">This table follows the selected 3-hour slot filter. If no slot is selected, it shows the whole day.</div></div>', unsafe_allow_html=True)

            # Missing (detail)
            miss = det[det["_status"]=="Missing"].copy()
            with st.expander("Show missing details", expanded=False):
                st.dataframe(miss, use_container_width=True, height=220)
                st.download_button(f"Download {label} missing CSV", data=miss.to_csv(index=False).encode("utf-8"), file_name=f"{label.lower()}_missing.csv", mime="text/csv")

            with st.expander("Show matched + late sync details"):
                nonmiss = det[det["_status"]!="Missing"].copy()
                st.dataframe(nonmiss, use_container_width=True, height=260)
                st.download_button(f"Download {label} matched+late CSV", data=nonmiss.to_csv(index=False).encode("utf-8"), file_name=f"{label.lower()}_matched_late.csv", mime="text/csv")

        _detail_block(rise_res, "Rise")

        # Rise wallet→Backend mismatch (Rise payouts in wallet but missing in backend)
