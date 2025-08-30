import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import nfl_data_py as nfl

st.set_page_config(page_title="Fantasy Lineup Picker", layout="wide")
st.title("🏈 Weekly Fantasy Lineup Picker (ESPN-style) — Hosted")

# -----------------------------
# Query params helpers (shareable link)
# -----------------------------
def get_qp():
    try:
        return st.experimental_get_query_params()
    except Exception:
        return {}

def set_qp(**kwargs):
    try:
        st.experimental_set_query_params(**kwargs)
    except Exception:
        pass

# -----------------------------
# Inputs (left sidebar)
# -----------------------------
st.sidebar.header("1) Week & Data Source")
year = st.sidebar.number_input("Season", min_value=2020, max_value=2030, value=2024, step=1)
week = st.sidebar.number_input("Week", min_value=1, max_value=18, value=1, step=1)
data_source = st.sidebar.radio("Use projections for:", ["K & DST only (safe)", "All positions (beta)"], index=0)

st.sidebar.header("2) Scoring (ESPN preset)")
preset = st.sidebar.selectbox("Scoring preset", ["ESPN (PPR)", "Custom"], index=0)

SC_DEFAULT = {
    "pass_yd_pt": 1.0/25.0,
    "pass_td": 4.0,
    "rush_yd_pt": 1.0/10.0,
    "rush_td": 6.0,
    "rec_yd_pt": 1.0/10.0,
    "rec_td": 6.0,
    "reception": 1.0,
    "two_pt": 2.0,
    "xp": 1.0,
    "fg": 3.0,
    "dst_base": 10.0,
    "dst_sack": 1.0,
    "dst_int": 2.0,
    "dst_fr": 2.0,
    "dst_td": 6.0,
}

if preset == "Custom":
    st.sidebar.caption("Enter your league’s exact numbers.")
    sc = {
        "pass_yd_pt": st.sidebar.number_input("Points per passing yard", value=SC_DEFAULT["pass_yd_pt"], step=0.01, format="%.4f"),
        "pass_td":    st.sidebar.number_input("Points per passing TD", value=SC_DEFAULT["pass_td"], step=0.5),
        "rush_yd_pt": st.sidebar.number_input("Points per rushing yard", value=SC_DEFAULT["rush_yd_pt"], step=0.01, format="%.4f"),
        "rush_td":    st.sidebar.number_input("Points per rushing TD", value=SC_DEFAULT["rush_td"], step=0.5),
        "rec_yd_pt":  st.sidebar.number_input("Points per receiving yard", value=SC_DEFAULT["rec_yd_pt"], step=0.01, format="%.4f"),
        "rec_td":     st.sidebar.number_input("Points per receiving TD", value=SC_DEFAULT["rec_td"], step=0.5),
        "reception":  st.sidebar.number_input("Points per reception", value=SC_DEFAULT["reception"], step=0.5),
        "two_pt":     st.sidebar.number_input("Points per 2-pt conversion", value=SC_DEFAULT["two_pt"], step=0.5),
        "xp":         st.sidebar.number_input("K: points per XP", value=SC_DEFAULT["xp"], step=0.5),
        "fg":         st.sidebar.number_input("K: points per FG", value=SC_DEFAULT["fg"], step=0.5),
        "dst_base":   st.sidebar.number_input("DST: base points", value=SC_DEFAULT["dst_base"], step=1.0),
        "dst_sack":   st.sidebar.number_input("DST: points per sack", value=SC_DEFAULT["dst_sack"], step=0.5),
        "dst_int":    st.sidebar.number_input("DST: points per INT", value=SC_DEFAULT["dst_int"], step=0.5),
        "dst_fr":     st.sidebar.number_input("DST: points per fumble recovery", value=SC_DEFAULT["dst_fr"], step=0.5),
        "dst_td":     st.sidebar.number_input("DST: points per defensive/return TD", value=SC_DEFAULT["dst_td"], step=0.5),
    }
else:
    sc = SC_DEFAULT.copy()

st.sidebar.header("3) Lineup slots (ESPN default)")
slots = {
    "QB": st.sidebar.number_input("QB", 0, 2, 1),
    "RB": st.sidebar.number_input("RB", 0, 5, 2),
    "WR": st.sidebar.number_input("WR", 0, 5, 2),
    "TE": st.sidebar.number_input("TE", 0, 3, 1),
    "FLEX": st.sidebar.number_input("FLEX (RB/WR/TE)", 0, 3, 1),
    "K": st.sidebar.number_input("K", 0, 2, 1),
    "DST": st.sidebar.number_input("D/ST", 0, 2, 1),
}

st.sidebar.header("4) Your Roster (optional)")
roster_text = st.sidebar.text_area("Paste names (one per line). Leave blank to use full pool.", height=140, placeholder="Justin Jefferson\nBijan Robinson\n...")
only_mine = st.sidebar.checkbox("Only pick from my players", value=False)

# Share link
st.sidebar.markdown("---")
if st.sidebar.button("Copy shareable league link"):
    # encode a minimal set into query params
    qp = dict(
        year=str(year), week=str(week),
        es=True if preset == "ESPN (PPR)" else False,
        qb=slots["QB"], rb=slots["RB"], wr=slots["WR"], te=slots["TE"],
        fx=slots["FLEX"], k=slots["K"], dst=slots["DST"]
    )
    set_qp(**qp)
    st.sidebar.success("URL updated. Copy from your browser address bar and share.")

# -----------------------------
# Data helpers
# -----------------------------
@st.cache_data(show_spinner=True)
def weekly_raw(years):
    df = nfl.import_weekly_data(years)
    if "player_name" not in df.columns and "player_display_name" in df.columns:
        df = df.rename(columns={"player_display_name": "player_name"})
    return df

def _series(df, *names, default=0.0):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n]).fillna(0.0)
    return pd.Series(default, index=df.index, dtype="float64")

def compute_points_from_stats(df: pd.DataFrame, scoring: dict) -> pd.Series:
    pts = (
        _series(df, "passing_yards", "pass_yds") * scoring["pass_yd_pt"]
        + _series(df, "passing_tds", "pass_td") * scoring["pass_td"]
        + _series(df, "rushing_yards", "rush_yds") * scoring["rush_yd_pt"]
        + _series(df, "rushing_tds", "rush_td") * scoring["rush_td"]
        + _series(df, "receiving_yards", "rec_yds") * scoring["rec_yd_pt"]
        + _series(df, "receiving_tds", "rec_td") * scoring["rec_td"]
        + _series(df, "receptions", "rec") * scoring["reception"]
        + _series(df, "two_pt_conv", "two_point_conversions", "two_pt_pass", "two_pt_rush", "two_pt_rec") * scoring["two_pt"]
    )
    return pts

def baselines_ppg(df_weekly: pd.DataFrame, scoring: dict, season: int) -> pd.DataFrame:
    d = df_weekly.copy()
    d["my_points"] = compute_points_from_stats(d, scoring)
    g = d[d["season"] == season].groupby(["player_id","player_name","recent_team","position"], dropna=False)
    agg = g.agg(games=("player_id","count"), points=("my_points","sum")).reset_index()
    agg["PPG"] = agg["points"] / agg["games"].replace(0, np.nan)
    agg["position"] = agg["position"].str.upper().str.replace(" ", "", regex=False)
    return agg

def fetch_fp_table(url: str) -> pd.DataFrame:
    html = requests.get(url, headers={"User-Agent": "lineup-picker/1.0"}, timeout=30).text
    soup = BeautifulSoup(html, "html5lib")
    table = soup.find("table")
    if not table:
        return pd.DataFrame()
    rows = table.find_all("tr")
    header = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
    out = []
    for r in rows[1:]:
        tds = [td.get_text(" ", strip=True) for td in r.find_all("td")]
        if not tds or len(tds) < 2:
            continue
        row = {}
        for i, h in enumerate(header[:len(tds)]):
            row[h] = tds[i]
        out.append(row)
    return pd.DataFrame(out)

def get_k_proj(week: int, sc: dict) -> pd.DataFrame:
    url = f"https://www.fantasypros.com/nfl/projections/k.php?week={week}&scoring=PPR"
    df = fetch_fp_table(url)
    if df.empty:
        return df
    if "Player" in df.columns:
        df["player_name"] = df["Player"].str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()
    else:
        df["player_name"] = df.iloc[:,0].astype(str)
    def find(cands):
        for c in df.columns:
            if any(x.lower() in c.lower() for x in cands):
                return c
        return None
    fgm = find(["FG","Field Goals Made","FGM"])
    xpm = find(["XP","Extra Points Made","XPM"])
    def num(x):
        try: return float(str(x).replace("%",""))
        except: return 0.0
    FGM = df[fgm].map(num) if fgm else 0.0
    XPM = df[xpm].map(num) if xpm else 0.0
    df["position"] = "K"
    df["PPG"] = FGM*sc["fg"] + XPM*sc["xp"]
    return df[["player_name","position","PPG"]]

def get_dst_proj(week: int, sc: dict) -> pd.DataFrame:
    url = f"https://www.fantasypros.com/nfl/projections/dst.php?week={week}&scoring=PPR"
    df = fetch_fp_table(url)
    if df.empty:
        return df
    if "Team" in df.columns:
        df["player_name"] = df["Team"].str.strip()
    else:
        df["player_name"] = df.iloc[:,0].astype(str).str.strip()
    def find(cands):
        for c in df.columns:
            if any(x.lower() in c.lower() for x in cands):
                return c
        return None
    sack_c = find(["Sack"])
    int_c  = find(["Int"])
    fr_c   = find(["Fum Rec","Fumble Rec","FR"])
    td_c   = find(["TD"])
    def num(s):
        try: return float(str(s))
        except: return 0.0
    sacks = df[sack_c].map(num) if sack_c else 0.0
    ints  = df[int_c].map(num) if int_c else 0.0
    fr    = df[fr_c].map(num) if fr_c else 0.0
    tds   = df[td_c].map(num) if td_c else 0.0
    df["position"] = "DST"
    df["PPG"] = sc["dst_base"] + sacks*sc["dst_sack"] + ints*sc["dst_int"] + fr*sc["dst_fr"] + tds*sc["dst_td"]
    return df[["player_name","position","PPG"]]

def get_skill_proj(week: int, sc: dict) -> pd.DataFrame:
    pos_urls = {
        "QB": f"https://www.fantasypros.com/nfl/projections/qb.php?week={week}&scoring=PPR",
        "RB": f"https://www.fantasypros.com/nfl/projections/rb.php?week={week}&scoring=PPR",
        "WR": f"https://www.fantasypros.com/nfl/projections/wr.php?week={week}&scoring=PPR",
        "TE": f"https://www.fantasypros.com/nfl/projections/te.php?week={week}&scoring=PPR",
    }
    frames = []
    for pos, url in pos_urls.items():
        df = fetch_fp_table(url)
        if df.empty: 
            continue
        if "Player" in df.columns:
            df["player_name"] = df["Player"].str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()
        else:
            df["player_name"] = df.iloc[:,0].astype(str)
        df["position"] = pos
        def find(cands):
            for c in df.columns:
                if any(x.lower() in c.lower() for x in cands):
                    return c
            return None
        def num(s):
            try: return float(str(s))
            except: return 0.0
        pass_yd = num(df[find(["Pass Yds","Passing Yds"])] if find(["Pass Yds","Passing Yds"]) else 0.0)
        frames.append(df[["player_name","position"]])
        # compute via stat cols if present
        pyc = find(["Pass Yds","Passing Yds"]); ptd=find(["Pass TD"])
        ryc = find(["Rush Yds"]); rtd=find(["Rush TD"])
        recyc = find(["Rec Yds","Receiving Yds"]); rectd=find(["Rec TD"]); recs=find(["Receptions","Rec"])
        twopt = find(["2PT"])
        def col_or_zero(c): return df[c].map(num) if c else 0.0
        pts = ( col_or_zero(pyc)*sc["pass_yd_pt"]
                + col_or_zero(ptd)*sc["pass_td"]
                + col_or_zero(ryc)*sc["rush_yd_pt"]
                + col_or_zero(rtd)*sc["rush_td"]
                + col_or_zero(recyc)*sc["rec_yd_pt"]
                + col_or_zero(rectd)*sc["rec_td"]
                + col_or_zero(recs)*sc["reception"]
                + col_or_zero(twopt)*sc["two_pt"]
              )
        out = df[["player_name","position"]].copy()
        out["PPG"] = pts
        frames[-1] = out
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["player_name","position","PPG"])

# -----------------------------
# Build the player pool
# -----------------------------
with st.spinner("Loading data…"):
    raw = weekly_raw([year])
    base = baselines_ppg(raw, sc, year)

    k_proj = get_k_proj(week, sc)
    dst_proj = get_dst_proj(week, sc)
    skill_proj = get_skill_proj(week, sc) if data_source.startswith("All") else pd.DataFrame()

skill_positions = ["QB","RB","WR","TE"]
if not skill_proj.empty:
    skill = skill_proj
else:
    skill = base[base["position"].isin(skill_positions)][["player_name","position","PPG"]].copy()

frames = [skill]
if not k_proj.empty: frames.append(k_proj)
if not dst_proj.empty: frames.append(dst_proj)
pool = pd.concat(frames, ignore_index=True).dropna(subset=["player_name","position"])

if only_mine:
    my_names = {n.strip().lower() for n in roster_text.splitlines() if n.strip()}
    pool = pool[pool["player_name"].str.lower().isin(my_names)]

# -----------------------------
# Lineup optimization
# -----------------------------
def pick_best(pool: pd.DataFrame, slots: dict):
    df = pool.sort_values("PPG", ascending=False).reset_index(drop=True)
    need = slots.copy()
    chosen = []

    def take(pos, n):
        nonlocal df, chosen
        if n <= 0: return
        got = df[df["position"] == pos].head(n).copy()
        got = got.assign(slot=pos)
        chosen.append(got)
        df = df.drop(got.index)

    # fixed
    for pos in ["QB","K","DST"]:
        take(pos, need.get(pos,0))

    # base RB/WR/TE without flex
    for pos in ["RB","WR","TE"]:
        base_n = need.get(pos,0)
        # hold back for flex competition
        base_n = max(0, base_n - need.get("FLEX",0))
        take(pos, base_n)

    remaining = df.copy()
    # FLEX from RB/WR/TE
    flex_n = need.get("FLEX",0)
    if flex_n > 0:
        candidates = remaining[remaining["position"].isin(["RB","WR","TE"])].head(flex_n).copy()
        if not candidates.empty:
            candidates["slot"] = "FLEX"
            chosen.append(candidates)
            remaining = remaining.drop(candidates.index)

    out = pd.concat(chosen, ignore_index=True) if chosen else pd.DataFrame(columns=["player_name","position","PPG","slot"])
    # number RB/WR/TE
    def add_num(pos):
        idx = out.index[out["slot"] == pos].tolist()
        for i, j in enumerate(sorted(idx, key=lambda k: -out.loc[k,"PPG"]), start=1):
            out.loc[j,"slot"] = f"{pos}{i}"
    for p in ["RB","WR","TE"]:
        add_num(p)

    order = ["QB","RB1","RB2","WR1","WR2","TE","FLEX","K","DST"]
    out["slot_order"] = out["slot"].apply(lambda s: order.index(s) if s in order else 99)
    out = out.sort_values(["slot_order","PPG"], ascending=[True, False]).drop(columns=["slot_order"])

    bench = remaining.sort_values("PPG", ascending=False).head(20)
    return out, bench

picked, bench = pick_best(pool, slots)

# -----------------------------
# Output
# -----------------------------
left, right = st.columns([2,1])
with left:
    st.subheader("✅ Your Starting Lineup")
    if picked.empty:
        st.error("No players available for required slots. Paste your roster or uncheck 'Only pick from my players'.")
    else:
        st.dataframe(picked[["slot","player_name","position","PPG"]].reset_index(drop=True), use_container_width=True)
        st.download_button("Download Lineup CSV",
                           picked[["slot","player_name","position","PPG"]].to_csv(index=False).encode(),
                           file_name=f"lineup_week{week}.csv", mime="text/csv")

with right:
    st.subheader("🪑 Bench Suggestions")
    if bench.empty:
        st.caption("No bench suggestions.")
    else:
        st.dataframe(bench[["player_name","position","PPG"]].reset_index(drop=True), use_container_width=True)

st.markdown("---")
st.caption("Free sources: nflverse (via nfl-data-py) for last-season per-game; FantasyPros for week projections (K/DST always, skill optional), parsed with BeautifulSoup+html5lib. No lxml.")

