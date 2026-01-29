
import pandas as pd
import streamlit as st
from datetime import datetime, date, time, timedelta
import plotly.express as px
import numpy as np

from recon import reconcile_exact, reconcile_rise_substring, plan_category, is_automation

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

    if run_crypto:
        st.subheader("Crypto 3-hour summary")
        cs = crypto_res.summary_3h.copy()
        cs["Date"] = cs["bucket_3h"].dt.strftime("%Y-%m-%d")
        cs["Time Range"] = cs["bucket_3h"].apply(format_range)
        cs = cs[["Date","Time Range","matched_count","late_sync_count","missing_count","backend_total","wallet_total","diff_total","abs_diff_total"]]
        sel_times = st.session_state.get("__sel_times")
        if sel_times:
            cs = cs[cs["Time Range"].isin(sel_times)]
        st.markdown('<div class="share-card"><div class="share-title">Crypto 3-hour summary</div><div class="share-sub">Filtered view — ready for screenshot.</div></div>', unsafe_allow_html=True)
        st.dataframe(cs, use_container_width=True, height=260)

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

            # Missing details (hidden by default)
            miss = det[det["_status"]=="Missing"].copy()
            with st.expander("Show missing details", expanded=False):
                if len(miss) == 0:
                    st.success("No missing rows ✅")
                else:
                    st.markdown("**Missing (detail)**")
                    st.dataframe(miss, use_container_width=True, height=220)
                    st.download_button(
                        f"Download {label} missing CSV",
                        data=miss.to_csv(index=False).encode("utf-8"),
                        file_name=f"{label.lower()}_missing.csv",
                        mime="text/csv",
                    )

            with st.expander("Show matched + late sync details"):
                nonmiss = det[det["_status"]!="Missing"].copy()
                st.dataframe(nonmiss, use_container_width=True, height=260)
                st.download_button(f"Download {label} matched+late CSV", data=nonmiss.to_csv(index=False).encode("utf-8"), file_name=f"{label.lower()}_matched_late.csv", mime="text/csv")

        _detail_block(crypto_res, "Crypto")
with tab2:
    st.subheader("Breakdown (Matched + Late Sync only)")
    parts = []
    if crypto_res is not None:
        parts += [crypto_res.matched.assign(channel="Crypto"), crypto_res.late_sync.assign(channel="Crypto")]
    if rise_res is not None:
        parts += [rise_res.matched.assign(channel="Rise"), rise_res.late_sync.assign(channel="Rise")]
    rec = pd.concat(parts, ignore_index=True) if len(parts) else pd.DataFrame()

    if rec.empty:
        st.info("No reconciled rows in the selected range.")
        st.stop()

    rec["_ptype"] = plan_category(rec.get("Plan", pd.Series([""]*len(rec))))
    rec["_auto"] = is_automation(rec.get("Internal Status", pd.Series([""]*len(rec))))

    summary = (
        rec.groupby(["_ptype","channel"])
        .agg(
            # txn_id may be null for Rise rows (email-based matching), so use row-count.
            Count=("amount_backend","size"),
            Total_Sum=("amount_backend","sum"),
            Automation_Count=("_auto","sum"),
            Automation_Sum=("amount_backend", lambda s: float(s[rec.loc[s.index,"_auto"]].sum())),
        )
        .reset_index()
        .rename(columns={"_ptype":"Payout Type","channel":"Channel"})
    )

    st.dataframe(summary.sort_values(["Payout Type","Channel"]), use_container_width=True, height=260)

    # Totals by payout type (CFD vs Futures) - across Rise + Crypto (Matched + Late Sync)
    totals = (
        rec.groupby("_ptype")
        .agg(Count=("amount_backend", "size"), Total_Sum=("amount_backend", "sum"))
        .reset_index()
        .rename(columns={"_ptype": "Payout Type"})
    )
    c1, c2, c3, c4 = st.columns(4)
    cfd_row = totals[totals["Payout Type"] == "CFD"]
    fut_row = totals[totals["Payout Type"] == "Futures"]

    c1.metric("CFD count", int(cfd_row["Count"].iloc[0]) if not cfd_row.empty else 0)
    c2.metric("CFD sum", f"{float(cfd_row['Total_Sum'].iloc[0]) if not cfd_row.empty else 0.0:,.2f}")
    c3.metric("Futures count", int(fut_row["Count"].iloc[0]) if not fut_row.empty else 0)
    c4.metric("Futures sum", f"{float(fut_row['Total_Sum'].iloc[0]) if not fut_row.empty else 0.0:,.2f}")

    st.markdown("")
    st.subheader("CFD vs Futures share (by amount)")
    pie_df = totals.copy()
    pie_df = pie_df[pie_df["Total_Sum"] > 0].copy()
    if pie_df.empty:
        st.write("No data to chart.")
    else:
        fig_pie = px.pie(pie_df, names="Payout Type", values="Total_Sum", hole=0.35)
        st.plotly_chart(fig_pie, use_container_width=True)



# ---------------- Tab 3: Disbursement totals ----------------
with tab3:
    st.header("Disbursement totals (All vs Payout vs Others)")
    st.caption(
        "All = total amount in the wallet report within the selected date range. "
        "Payout = amount in the wallet report that matches backend payouts (Matched + Late Sync). "
        "Others = All − Payout."
    )

    def _parse_local(df: pd.DataFrame, ts_col: str, src_tz: str) -> pd.Series:
        """Parse timestamps and return them converted to the report timezone.

        - If the source column is **naive**, we interpret it as `src_tz` (tz_localize).
        - If it's already **tz-aware** (e.g., strings like `2026-01-23 10:38:46+00:00`),
          we keep that timezone and just convert to the report timezone.
        """
        s = pd.to_datetime(df[ts_col], errors="coerce")
        if getattr(s.dt, "tz", None) is None:
            # localize naive -> source timezone
            s = s.dt.tz_localize(src_tz, nonexistent="shift_forward", ambiguous="NaT")
        # convert -> report timezone
        return s.dt.tz_convert(report_tz)

    def _sum_amount_in_range(df: pd.DataFrame, ts_col: str, amount_col: str, src_tz: str) -> float:
        if df is None or len(df) == 0:
            return 0.0
        if ts_col not in df.columns or amount_col not in df.columns:
            return 0.0
        t = _parse_local(df, ts_col, src_tz)
        # `report_end` is already the **exclusive** end boundary (start of the next day) in report timezone.
        # Using +1 day here would unintentionally include an extra day.
        mask = (t >= report_start) & (t < report_end)
        amt = pd.to_numeric(df.loc[mask, amount_col], errors="coerce").fillna(0.0)
        return float(np.nansum(np.abs(amt)))

    def _sum_payout_wallet_amount(res, report_start, report_end) -> float:
        """Sum *wallet-side* payout amount within the selected report date range.

        Why filter on wallet timestamp?
        Reconciliation is primarily driven by backend timestamps. A wallet transaction can
        land slightly outside the selected date range (e.g., late sync / timezone shift).
        For the *Disbursement totals* tab we want:
            All (wallet in range) >= Payout (wallet in range matched to backend)
        so we must restrict payout amounts to wallet rows whose wallet timestamp is within
        the same [report_start, report_end) window (in report_tz).
        """
        if res is None:
            return 0.0

        frames = []
        for name in ("matched", "late_sync"):
            d = getattr(res, name, None)
            if isinstance(d, pd.DataFrame) and len(d) > 0:
                frames.append(d)
        if not frames:
            return 0.0

        df = pd.concat(frames, ignore_index=True)
        if "amount_wallet" not in df.columns:
            return 0.0

        # Filter by wallet timestamp (already converted to report_tz in recon.py)
        if "ts_report_wallet" in df.columns and report_start is not None and report_end is not None:
            wts = pd.to_datetime(df["ts_report_wallet"], errors="coerce")
            # If tz-aware, comparisons work; if tz-naive, localize to report_tz
            if hasattr(wts.dt, "tz") and wts.dt.tz is None:
                try:
                    wts = wts.dt.tz_localize(report_tz)
                except Exception:
                    pass
            df = df[(wts >= report_start) & (wts < report_end)].copy()

        amt = pd.to_numeric(df["amount_wallet"], errors="coerce").fillna(0.0)
        return float(np.nansum(np.abs(amt)))

    # Wallet-report columns (these are the columns used by reconciliation)
    CRYPTO_TS_COL, CRYPTO_AMT_COL = "Created", "Amount"
    RISE_TS_COL, RISE_AMT_COL = "Date", "Amount"

    # Compute totals
    crypto_all = _sum_amount_in_range(crypto if run_crypto else None, CRYPTO_TS_COL, CRYPTO_AMT_COL, crypto_tz)
    rise_all = _sum_amount_in_range(rise if run_rise else None, RISE_TS_COL, RISE_AMT_COL, rise_tz)

    crypto_payout = _sum_payout_wallet_amount(crypto_res, report_start, report_end) if run_crypto else 0.0
    rise_payout = _sum_payout_wallet_amount(rise_res, report_start, report_end) if run_rise else 0.0

    crypto_other = max(0.0, crypto_all - crypto_payout)
    rise_other = max(0.0, rise_all - rise_payout)

    # Cards
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Crypto")
        a, b, c = st.columns(3)
        a.metric("All", f"{crypto_all:,.2f}")
        b.metric("Payout", f"{crypto_payout:,.2f}")
        c.metric("Others disbursed", f"{crypto_other:,.2f}")
    with c2:
        st.subheader("Rise")
        a, b, c = st.columns(3)
        a.metric("All", f"{rise_all:,.2f}")
        b.metric("Payout", f"{rise_payout:,.2f}")
        c.metric("Others disbursed", f"{rise_other:,.2f}")

    # Table
    totals_tbl = pd.DataFrame(
        [
            {"Channel": "Crypto", "All": crypto_all, "Payout": crypto_payout, "Others disbursed": crypto_other},
            {"Channel": "Rise", "All": rise_all, "Payout": rise_payout, "Others disbursed": rise_other},
        ]
    )
    st.dataframe(totals_tbl, use_container_width=True, hide_index=True)

    # Simple stacked bar for quick comparison
    chart_df = totals_tbl.set_index("Channel")[["Payout", "Others disbursed"]]
    st.bar_chart(chart_df)


# --- Wallet payout (Tracking ID present) but NOT found in backend ---
# This helps detect wrong/unknown payouts disbursed from wallet
with st.expander("Show wallet payout not found in backend", expanded=False):
    try:
        # Wallet payouts are rows with Tracking ID present
        _cw = crypto.copy()
        _cw["_tracking_id_norm"] = _cw["Tracking ID"].astype(str).str.strip().str.upper()
        _cw = _cw[_cw["_tracking_id_norm"].notna() & (_cw["_tracking_id_norm"] != "")]

        # Convert wallet timestamp to report timezone and filter to the same window
        _cw["_ts_report"] = pd.to_datetime(_cw["Created"], errors="coerce", utc=True).dt.tz_convert(report_tz)
        _cw = _cw[(_cw["_ts_report"] >= report_start) & (_cw["_ts_report"] < report_end)].copy()

        # Backend payouts in window (already used for crypto reconciliation)
        _b = backend_crypto.copy()
        _b["_txn_id_norm"] = _b["Transaction ID"].astype(str).str.strip().str.upper()
        _b = _b[_b["_txn_id_norm"].notna() & (_b["_txn_id_norm"] != "")]
        _backend_ids = set(_b["_txn_id_norm"].unique().tolist())

        wallet_payout_not_in_backend = _cw[~_cw["_tracking_id_norm"].isin(_backend_ids)].copy()

        st.write("These are wallet transactions with Tracking ID (payout) but no matching Transaction ID found in backend for the selected window.")
        st.metric("Count", int(len(wallet_payout_not_in_backend)))
        st.metric("Total amount", f"{pd.to_numeric(wallet_payout_not_in_backend['Amount'], errors='coerce').abs().sum():,.2f}")

        st.dataframe(wallet_payout_not_in_backend, use_container_width=True, height=260)
        st.download_button(
            "Download wallet payout not in backend (CSV)",
            data=wallet_payout_not_in_backend.to_csv(index=False).encode("utf-8"),
            file_name="crypto_wallet_payout_not_in_backend.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Could not build wallet→backend mismatch table: {e}")

