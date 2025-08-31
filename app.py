import streamlit as st
import pandas as pd
import numpy as np
import requests, os, re, base64
from pathlib import Path
from bs4 import BeautifulSoup
import nfl_data_py as nfl

st.set_page_config(page_title="Fantasy Lineup Picker", layout="wide")
st.title("🏈 Weekly Fantasy Lineup Picker (ESPN-style) — Fan Mode")

# -----------------------------
# Query params (shareable link)
# -----------------------------
def get_qp():
    try:
        return st.experimental_get_query_params()
    except:
        return {}

def set_qp(**kwargs):
    try:
        st.experimental_set_query_params(**kwargs)
    except:
        pass

# -----------------------------
# Sidebar inputs (Steps 1–3)
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
qb_n = st.sidebar.number_input("QB", 0, 2, 1, key="slot_qb")
rb_n = st.sidebar.number_input("RB", 0, 5, 2, key="slot_rb")
wr_n = st.sidebar.number_input("WR", 0, 5, 2, key="slot_wr")
te_n = st.sidebar.number_input("TE", 0, 3, 1, key="slot_te")
fx_n = st.sidebar.number_input("FLEX (RB/WR/TE)", 0, 3, 1, key="slot_flex")
k_n  = st.sidebar.number_input("K", 0, 2, 1, key="slot_k")
dst_n= st.sidebar.number_input("D/ST", 0, 2, 1, key="slot_dst")
slots = {"QB": qb_n, "RB": rb_n, "WR": wr_n, "TE": te_n, "FLEX": fx_n, "K": k_n, "DST": dst_n}

# -----------------------------
# Sidebar: Your roster (Step 4)
# -----------------------------
qp = get_qp()
roster_from_qp = ""
only_flag = False
if "r" in qp and qp["r"]:
    try:
        roster_from_qp = base64.urlsafe_b64decode(qp["r"][0].encode()).decode("utf-8", "ignore")
    except Exception:
        try:
            import urllib.parse as up
            roster_from_qp = up.unquote_plus(qp["r"][0])
        except Exception:
            roster_from_qp = ""
if "mine" in qp and qp["mine"]:
    only_flag = qp["mine"][0].lower() in ("1","true","yes","on")

st.sidebar.header("4) Your Roster (optional)")
roster_text = st.sidebar.text_area(
    "Paste names (one per line). Leave blank to use full pool.",
    height=140,
    value=roster_from_qp if roster_from_qp else "",
    placeholder="Lamar Jackson\nBijan Robinson\nRavens D/ST\n..."
)
only_mine = st.sidebar.checkbox("Only pick from my players", value=only_flag)

# Share link (includes roster if you want)
st.sidebar.markdown("---")
if st.sidebar.button("Copy shareable league link (includes roster)"):
    renc = base64.urlsafe_b64encode(roster_text.encode()).decode() if roster_text.strip() else None
    qp_out = dict(
        year=str(year), week=str(week), es=(preset=="ESPN (PPR)"),
        qb=slots["QB"], rb=slots["RB"], wr=slots["WR"], te=slots["TE"],
        fx=slots["FLEX"], k=slots["K"], dst=slots["DST"],
    )
    if renc: qp_out["r"] = renc
    if only_mine: qp_out["mine"] = "1"
    set_qp(**qp_out)
    st.sidebar.success("URL updated. Copy from the address bar and share.")

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
            return pd.to_numeric(df[n], errors="coerce").fillna(0.0)
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

# Cached projections from repo (via GitHub Action)
@st.cache_data(show_spinner=False)
def load_cached_projections(week: int) -> pd.DataFrame:
    path = Path(__file__).parent / "data" / f"projections_week_{week}.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            if "position" not in df.columns:
                return pd.DataFrame()
            if "player_name" not in df.columns:
                if "Player" in df.columns:
                    df["player_name"] = df["Player"].astype(str)
                elif "Team" in df.columns:
                    df["player_name"] = df["Team"].astype(str)
                else:
                    df["player_name"] = df.iloc[:,0].astype(str)
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# Simple table scraper (no lxml)
def fetch_fp_table(url: str) -> pd.DataFrame:
    try:
        html = requests.get(url, headers={"User-Agent": "lineup-picker/1.0"}, timeout=30).text
        soup = BeautifulSoup(html, "html5lib")
        table = soup.find("table")
        if not table: return pd.DataFrame()
        rows = table.find_all("tr")
        if not rows: return pd.DataFrame()
        header = [th.get_text(strip=True) for th in rows[0].find_all(["th","td"])]
        out=[]
        for r in rows[1:]:
            tds=[td.get_text(" ", strip=True) for td in r.find_all("td")]
            if not tds: continue
            row={}
            for i,h in enumerate(header[:len(tds)]):
                row[h]=tds[i]
            out.append(row)
        return pd.DataFrame(out)
    except Exception:
        return pd.DataFrame()

def k_from_df(df: pd.DataFrame, sc: dict) -> pd.DataFrame:
    if df.empty: return df
    if "player_name" not in df.columns:
        if "Player" in df.columns:
            df["player_name"] = df["Player"].astype(str).str.replace(r"\s*\(.*?\)","", regex=True).str.strip()
        else:
            df["player_name"] = df.iloc[:,0].astype(str)
    def find(cands):
        for c in df.columns:
            if any(x.lower() in c.lower() for x in cands):
                return c
        return None
    def num(x):
        try: return float(str(x).replace("%",""))
        except: return 0.0
    fgm = find(["FG","Field Goals Made","FGM"])
    xpm = find(["XP","Extra Points Made","XPM"])
    FGM = df[fgm].map(num) if fgm else 0.0
    XPM = df[xpm].map(num) if xpm else 0.0
    out = pd.DataFrame({"player_name": df["player_name"]})
    out["position"] = "K"
    out["PPG"] = FGM*sc["fg"] + XPM*sc["xp"]
    return out

def dst_from_df(df: pd.DataFrame, sc: dict) -> pd.DataFrame:
    if df.empty: return df
    if "player_name" not in df.columns:
        if "Team" in df.columns:
            df["player_name"] = df["Team"].astype(str).str.strip()
        else:
            df["player_name"] = df.iloc[:,0].astype(str).str.strip()
    def find(cands):
        for c in df.columns:
            if any(x.lower() in c.lower() for x in cands):
                return c
        return None
    def num(x):
        try: return float(str(x))
        except: return 0.0
    sacks = df[find(["Sack"])].map(num) if find(["Sack"]) else 0.0
    ints  = df[find(["Int"])].map(num) if find(["Int"]) else 0.0
    fr    = df[find(["Fum Rec","Fumble Rec","FR"])].map(num) if find(["Fum Rec","Fumble Rec","FR"]) else 0.0
    tds   = df[find(["TD"])].map(num) if find(["TD"]) else 0.0
    out = pd.DataFrame({"player_name": df["player_name"]})
    out["position"] = "DST"
    out["PPG"] = sc["dst_base"] + sacks*sc["dst_sack"] + ints*sc["dst_int"] + fr*sc["dst_fr"] + tds*sc["dst_td"]
    return out

def skill_from_df(df: pd.DataFrame, pos: str, sc: dict) -> pd.DataFrame:
    if df.empty: return df
    if "player_name" not in df.columns:
        if "Player" in df.columns:
            df["player_name"] = df["Player"].astype(str).str.replace(r"\s*\(.*?\)","", regex=True).str.strip()
        else:
            df["player_name"] = df.iloc[:,0].astype(str)
    def find(cands):
        for c in df.columns:
            if any(x.lower() in c.lower() for x in cands):
                return c
        return None
    def num(x):
        try: return float(str(x))
        except: return 0.0
    py = skill = 0.0
    pyc = find(["Pass Yds","Passing Yds"]); ptd=find(["Pass TD"])
    ryc = find(["Rush Yds"]); rtd=find(["Rush TD"])
    recyc = find(["Rec Yds","Receiving Yds"]); rectd=find(["Rec TD"]); recs=find(["Receptions","Rec"])
    twopt = find(["2PT"])
    def col_or_zero(c): return df[c].map(num) if c else 0.0
    pts = ( col_or_zero(pyc)*sc["pass_yd_pt"] + col_or_zero(ptd)*sc["pass_td"]
            + col_or_zero(ryc)*sc["rush_yd_pt"] + col_or_zero(rtd)*sc["rush_td"]
            + col_or_zero(recyc)*sc["rec_yd_pt"] + col_or_zero(rectd)*sc["rec_td"]
            + col_or_zero(recs)*sc["reception"] + col_or_zero(twopt)*sc["two_pt"] )
    out = pd.DataFrame({"player_name": df["player_name"]})
    out["position"] = pos
    out["PPG"] = pts
    return out

def get_cached_or_web_projections(week: int, sc: dict, want_skill: bool):
    cached = load_cached_projections(week)
    skill = pd.DataFrame(columns=["player_name","position","PPG"])
    k = pd.DataFrame(columns=["player_name","position","PPG"])
    dst = pd.DataFrame(columns=["player_name","position","PPG"])

    if not cached.empty:
        cached_pos = cached.copy()
        if "position" in cached_pos.columns:
            k = k_from_df(cached_pos[cached_pos["position"]=="K"].copy(), sc)
            dst = dst_from_df(cached_pos[cached_pos["position"]=="DST"].copy(), sc)
            if want_skill:
                frames=[]
                for pos in ["QB","RB","WR","TE"]:
                    frames.append(skill_from_df(cached_pos[cached_pos["position"]==pos].copy(), pos, sc))
                skill = pd.concat(frames, ignore_index=True) if frames else skill

    if k.empty:
        df = fetch_fp_table(f"https://www.fantasypros.com/nfl/projections/k.php?week={week}&scoring=PPR")
        k = k_from_df(df, sc)
    if dst.empty:
        df = fetch_fp_table(f"https://www.fantasypros.com/nfl/projections/dst.php?week={week}&scoring=PPR")
        dst = dst_from_df(df, sc)
    if want_skill and skill.empty:
        frames=[]
        for pos, url in {
            "QB": f"https://www.fantasypros.com/nfl/projections/qb.php?week={week}&scoring=PPR",
            "RB": f"https://www.fantasypros.com/nfl/projections/rb.php?week={week}&scoring=PPR",
            "WR": f"https://www.fantasypros.com/nfl/projections/wr.php?week={week}&scoring=PPR",
            "TE": f"https://www.fantasypros.com/nfl/projections/te.php?week={week}&scoring=PPR",
        }.items():
            frames.append(skill_from_df(fetch_fp_table(url), pos, sc))
        skill = pd.concat(frames, ignore_index=True) if frames else skill

    return skill, k, dst

# -----------------------------
# Build the player pool
# -----------------------------
with st.spinner("Loading data…"):
    raw = weekly_raw([year])
    base = baselines_ppg(raw, sc, year)

want_skill = data_source.startswith("All")
skill_proj, k_proj, dst_proj = get_cached_or_web_projections(week, sc, want_skill)

skill_positions = ["QB","RB","WR","TE"]
if want_skill and not skill_proj.empty:
    skill = skill_proj
else:
    skill = base[base["position"].isin(skill_positions)][["player_name","position","PPG"]].copy()

frames = [skill]
if not k_proj.empty: frames.append(k_proj)
if not dst_proj.empty: frames.append(dst_proj)
pool = pd.concat(frames, ignore_index=True).dropna(subset=["player_name","position"])

# If user wants only their roster
def normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s/]", " ", s)
    s = s.replace("d/st", "dst").replace("d st","dst").replace("defense","dst").replace("special teams","dst")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def name_keys(full: str):
    n = normalize_name(full)
    toks = n.split()
    keys = set()
    keys.add(" ".join(toks))
    keys.add("".join(toks))
    if len(toks) >= 2:
        first, last = toks[0], toks[-1]
        keys.add(f"{first[0]} {last}")
        keys.add(f"{first[0]}{last}")
        keys.add(last)
    for t in toks:
        keys.add(t)
    return list(keys)

pool_names = pool["player_name"].astype(str).tolist()
pool_keys = {}
last_counts = {}
for name in pool_names:
    toks = normalize_name(name).split()
    if len(toks) >= 2:
        last = toks[-1]
        last_counts[last] = last_counts.get(last, 0) + 1
for name in pool_names:
    for k in name_keys(name):
        if re.fullmatch(r"[a-z]+", k) and k in last_counts and last_counts[k] > 1:
            continue
        pool_keys.setdefault(k, set()).add(name)

def match_roster(roster_lines):
    if not roster_lines:
        return pool.copy(), []
    wanted = []
    unmatched = []
    seen = set()
    for rawline in roster_lines:
        q = normalize_name(rawline)
        if not q: 
            continue
        candidates = set()
        for k in name_keys(q):
            if k in pool_keys:
                candidates |= pool_keys[k]
        if not candidates:
            for pname in pool_names:
                if normalize_name(pname).find(q) >= 0:
                    candidates.add(pname)
        if candidates:
            sub = pool[pool["player_name"].isin(candidates)].sort_values("PPG", ascending=False)
            if not sub.empty:
                best = sub.iloc[0]["player_name"]
                if best not in seen:
                    wanted.append(best)
                    seen.add(best)
            else:
                unmatched.append(rawline)
        else:
            unmatched.append(rawline)
    filtered = pool[pool["player_name"].isin(wanted)]
    return filtered, unmatched

if only_mine:
    roster_list = [x.strip() for x in roster_text.splitlines() if x.strip()]
    pool, unmatched = match_roster(roster_list)
else:
    unmatched = []

# -----------------------------
# Fan Mode controls (Step 5)
# -----------------------------
st.sidebar.header("5) Fan Mode")
opt_mode = st.sidebar.radio("Optimize for", ["Balanced","Floor","Ceiling"], index=0, help="Balanced = PPG; Floor/Ceiling use recent-game volatility.")
lock_names = st.sidebar.multiselect("Lock starters (optional)", sorted(pool["player_name"].unique().tolist()))
exclude_names = st.sidebar.multiselect("Exclude players (optional)", [])

# Compute recent volatility (std) and last-3 trend (safe fallback if missing)
def compute_player_volatility(raw_df, scoring, season):
    try:
        d = raw_df[raw_df["season"] == season].copy()
        d["pts"] = compute_points_from_stats(d, scoring)
        # std & last3
        grp = d.groupby("player_name", dropna=False)
        stats = grp["pts"].agg(ppg="mean", std="std", games="count").reset_index()
        # last 3 by week
        last3 = (d.sort_values(["player_name","week"])
                   .groupby("player_name")["pts"]
                   .apply(lambda s: s.tail(3).mean() if len(s) else np.nan)).reset_index().rename(columns={"pts":"last3"})
        out = stats.merge(last3, on="player_name", how="left")
        return out
    except Exception:
        return pd.DataFrame(columns=["player_name","ppg","std","games","last3"])

vol = compute_player_volatility(raw, sc, year)
vol = vol.fillna({"std":0.0, "last3":np.nan})

# Merge vol into pool and compute OPT score
pool = pool.merge(vol[["player_name","std","last3"]], on="player_name", how="left")
pool["std"] = pool["std"].fillna(0.0)
if opt_mode == "Balanced":
    pool["OPT"] = pool["PPG"]
elif opt_mode == "Ceiling":
    pool["OPT"] = pool["PPG"] + 0.5*pool["std"]
else:  # Floor
    pool["OPT"] = pool["PPG"] - 0.5*pool["std"]

# Apply excludes
if exclude_names:
    pool = pool[~pool["player_name"].isin(set(exclude_names))]

# -----------------------------
# Lineup optimization (respects locks)
# -----------------------------
def assign_locked(chosen_list, df, slots_left, locked_df):
    """Place locked players into their slots; prefer base slot, then FLEX for RB/WR/TE."""
    for _, r in locked_df.iterrows():
        pos = r["position"]
        # Base slot first
        if slots_left.get(pos,0) > 0:
            r2 = r.copy()
            r2["slot"] = pos
            chosen_list.append(r2.to_frame().T)
            slots_left[pos] -= 1
            # remove from pool
            df.drop(df[(df["player_name"]==r["player_name"]) & (df["position"]==pos)].index, inplace=True, errors="ignore")
        elif pos in ("RB","WR","TE") and slots_left.get("FLEX",0) > 0:
            r2 = r.copy()
            r2["slot"] = "FLEX"
            chosen_list.append(r2.to_frame().T)
            slots_left["FLEX"] -= 1
            df.drop(df[(df["player_name"]==r["player_name"]) & (df["position"]==pos)].index, inplace=True, errors="ignore")
        else:
            raise ValueError(f"Too many locked at {pos}; no slot left.")

def pick_best(pool_df: pd.DataFrame, slots_dict: dict, score_col="OPT", locked_names=None):
    df = pool_df.sort_values(score_col, ascending=False).reset_index(drop=True).copy()
    slots_left = {k:int(v) for k,v in slots_dict.items()}
    chosen = []

    # Handle locks
    if locked_names:
        locked = df[df["player_name"].isin(set(locked_names))].copy()
        try:
            assign_locked(chosen, df, slots_left, locked)
        except ValueError as e:
            return pd.DataFrame(), pd.DataFrame(), str(e)

    def take(pos, n):
        nonlocal df, chosen
        if n <= 0: return
        got = df[df["position"] == pos].head(n).copy()
        if not got.empty:
            got = got.assign(slot=pos)
            chosen.append(got)
            df = df.drop(got.index)

    # Fill remaining base slots
    for p in ["QB","RB","WR","TE","K","DST"]:
        take(p, slots_left.get(p,0))

    # FLEX from remaining RB/WR/TE
    flex_n = slots_left.get("FLEX",0)
    if flex_n > 0:
        flex_pool = df[df["position"].isin(["RB","WR","TE"])].head(flex_n).copy()
        if not flex_pool.empty:
            flex_pool["slot"] = "FLEX"
            chosen.append(flex_pool)
            df = df.drop(flex_pool.index)

    out = pd.concat(chosen, ignore_index=True) if chosen else pd.DataFrame(columns=["player_name","position","PPG","slot",score_col])

    # Number RB/WR/TE
    def number_slot(df_slots: pd.DataFrame, base: str):
        idx = df_slots.index[df_slots["slot"] == base].tolist()
        if not idx: return
        idx_sorted = sorted(idx, key=lambda j: -df_slots.loc[j, score_col])
        for i, j in enumerate(idx_sorted, start=1):
            df_slots.loc[j, "slot"] = f"{base}{i}"

    for base in ["RB","WR","TE"]:
        number_slot(out, base)

    order = ["QB","RB1","RB2","WR1","WR2","TE1","TE","FLEX","K","DST"]
    out["slot_order"] = out["slot"].apply(lambda s: order.index(s) if s in order else 99)
    out = out.sort_values(["slot_order",score_col], ascending=[True, False]).drop(columns=["slot_order"])

    taken = set(out["player_name"]) if not out.empty else set()
    bench = df[~df["player_name"].isin(taken)].sort_values(score_col, ascending=False).head(20)
    return out, bench, None

picked, bench, lock_err = pick_best(pool, slots, score_col="OPT", locked_names=lock_names)

# -----------------------------
# Output
# -----------------------------
left, right = st.columns([2,1])

with left:
    st.subheader("✅ Your Starting Lineup")
    if lock_err:
        st.error(lock_err)
    elif picked.empty:
        st.error("No players available for required slots. Paste your roster or uncheck 'Only pick from my players'.")
    else:
        show = picked[["slot","player_name","position","PPG","OPT"]].rename(columns={"OPT":"Score"})
        st.dataframe(show.reset_index(drop=True), use_container_width=True)

        # Reason notes under each starter (PPG, last-3)
        st.markdown("**Why these picks**")
        for _, r in picked.iterrows():
            last3 = pool.loc[pool["player_name"]==r["player_name"], "last3"]
            last3_val = None if last3.empty else last3.iloc[0]
            bullets = []
            bullets.append(f"PPG: {r['PPG']:.2f}")
            if pd.notna(last3_val):
                bullets.append(f"Last 3 avg: {last3_val:.2f}")
            if opt_mode != "Balanced":
                bullets.append(f"Mode: {opt_mode}")
            st.markdown(f"- **{r['slot']} — {r['player_name']}** ({r['position']}): " + " • ".join(bullets))

        st.download_button("Download Lineup CSV",
                           show.to_csv(index=False).encode(),
                           file_name=f"lineup_week{week}.csv", mime="text/csv")

with right:
    st.subheader("🪑 Bench Suggestions")
    if bench.empty:
        st.caption("No bench suggestions.")
    else:
        st.dataframe(bench[["player_name","position","PPG","OPT"]].rename(columns={"OPT":"Score"}).reset_index(drop=True),
                     use_container_width=True)

# Unmatched roster names
if only_mine and unmatched:
    st.markdown("----")
    st.warning(f"Unmatched roster names: {', '.join(unmatched)}")
    st.caption("Tip: try full name or last name (unique); for defenses use 'Ravens D/ST' or 'Ravens'.")

st.markdown("---")
st.caption("Data: nflverse (nfl_data_py) for last-season per-game; FantasyPros weekly projections cached via GitHub Actions (K/DST always, skill optional). No lxml. Fan Mode adds Locks/Excludes, Floor/Ceiling, and simple reasons.")
