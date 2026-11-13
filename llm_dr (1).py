"""
LLM-DR re-implementation (independent, from the paper's description)
====================================================================
Persona-grounded LLM rider agents in a shared on-demand delivery simulation.

Pipeline
  1. Persona extraction  : per-courier features -> log1p -> z-score -> KMeans(k=4)
  2. Environment         : 10-min interval batching, 2 km broadcast, K=3 riders/order,
                           B=6 offers/rider, conflict resolution by mean pickup distance
  3. Agents              : LLM (any OpenAI-compatible endpoint, e.g. local vLLM) or
                           rule baselines (nearest / highest fee / fee-per-km / persona heuristic)
  4. Evaluation          : Table-I style delivery metrics + JSD / Wasserstein / KS vs empirical



Usage
  python llm_dr.py --policy nearest                       # synthetic data, baseline
  python llm_dr.py --data orders.csv --policy llm \
        --base_url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse, json, math, os, pickle, re, threading, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from scipy.spatial import ConvexHull
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Config (paper values)
# ----------------------------------------------------------------------------
DT = 10            # minutes per decision interval
HORIZON = 30       # pending-order look-back h (minutes)
RHO = 2.0          # broadcast radius (km)
K_RIDERS = 3       # candidate riders per order
B_OFFERS = 6       # max offers per rider
SERVICE_S = 120    # service time per order (seconds)
T_START, T_END = 8 * 60, 20 * 60
CIRCUITY, SPEED_KMH = 1.3, 15.0   # free routing surrogate
PERSONAS = ["Full-time Workhorse", "Lunch-Peak Specialist", "Super Stacker", "Dinner-Peak Core"]


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


AMAP_KEY = os.environ.get("AMAP_KEY")
CACHE_FILE = "amap_cache.pkl"
_cache = pickle.load(open(CACHE_FILE, "rb")) if os.path.exists(CACHE_FILE) else {}
_cache_lock, _rate_lock, _last = threading.Lock(), threading.Lock(), [0.0]
AMAP_STATS = dict(requests=0, cache_hits=0)


def save_cache():
    with _cache_lock:
        pickle.dump(_cache, open(CACHE_FILE, "wb"))


def route(a, b):
    """(lon,lat)->(lon,lat): returns (road km, minutes). Amap cycling API with a persistent cache.
    Falls back to haversine only when no AMAP_KEY is set (never mixes sources mid-run)."""
    if not AMAP_KEY:
        d = float(haversine(a[0], a[1], b[0], b[1])) * CIRCUITY
        return d, d / SPEED_KMH * 60.0
    key = (round(a[0], 5), round(a[1], 5), round(b[0], 5), round(b[1], 5))
    with _cache_lock:
        if key in _cache:
            AMAP_STATS["cache_hits"] += 1
            return _cache[key]
    with _rate_lock:                       # stay under the free QPS limit
        time.sleep(max(0, 0.4 - (time.time() - _last[0]))); _last[0] = time.time()
    r = requests.get("https://restapi.amap.com/v5/direction/bicycling",
                     params=dict(key=AMAP_KEY, origin=f"{a[0]:.6f},{a[1]:.6f}",
                                 destination=f"{b[0]:.6f},{b[1]:.6f}", show_fields="cost"),
                     timeout=10).json()
    if r.get("status") != "1":             # e.g. quota exhausted: stop, keep cache, resume later
        save_cache()
        raise RuntimeError(f"Amap error: {r.get('info')} ({r.get('infocode')}). Cache saved; rerun to resume.")
    p = r["route"]["paths"][0]
    out = (float(p["distance"]) / 1000, float(p["cost"]["duration"]) / 60)
    with _cache_lock:
        _cache[key] = out; AMAP_STATS["requests"] += 1
    return out


def reward(od_km):
    return 3.0 if od_km <= 1 else 6.0 if od_km <= 3 else 10.0 if od_km <= 5 else 15.0


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
REQUIRED = ["order_id", "courier_id", "create_min", "deadline_min",
            "p_lon", "p_lat", "d_lon", "d_lat"]


def synthetic_orders(n_couriers=240, seed=0):
    """Beijing-like synthetic orders with 4 latent courier archetypes."""
    rng = np.random.default_rng(seed)
    rows, oid = [], 0
    for c in range(n_couriers):
        kind = rng.choice(4, p=[.35, .2, .3, .15])
        n = int(rng.integers(8, 25))
        cx, cy = 116.40 + rng.normal(0, .08), 39.92 + rng.normal(0, .06)
        sd = [.02, .01, .03, .015][kind]
        for _ in range(n):
            if kind == 1:   t = rng.normal(12.2, .6) * 60
            elif kind == 3: t = rng.normal(17.8, .8) * 60
            elif kind == 2: t = rng.choice([11.5, 14, 17.5]) * 60 + rng.normal(0, 40)
            else:           t = rng.uniform(8, 20) * 60
            t = float(np.clip(t, T_START, T_END - 1))
            p = (cx + rng.normal(0, sd), cy + rng.normal(0, sd))
            off = [.02, .01, .035, .015][kind] * rng.uniform(.3, 1.4)
            ang = rng.uniform(0, 2 * np.pi)
            rows.append((oid, c, t, t + rng.uniform(35, 60), p[0], p[1],
                         p[0] + off * np.cos(ang), p[1] + off * np.sin(ang)))
            oid += 1
    return pd.DataFrame(rows, columns=REQUIRED)


def load_orders(path, colmap=None):
    """colmap: dict {your_column: required_name}. Times may be datetimes or minutes-from-midnight."""
    df = pd.read_csv(path)
    if colmap:
        df = df.rename(columns=colmap)
    missing = [c for c in REQUIRED if c not in df.columns]
    assert not missing, f"missing columns {missing}; pass a column map"
    for c in ("create_min", "deadline_min"):
        if not np.issubdtype(df[c].dtype, np.number):
            ts = pd.to_datetime(df[c])
            df[c] = ts.dt.hour * 60 + ts.dt.minute + ts.dt.second / 60
    return df[REQUIRED].copy()


# ----------------------------------------------------------------------------
# 1. Persona extraction
# ----------------------------------------------------------------------------
def courier_features(df):
    feats = []
    for cid, g in df.groupby("courier_id"):
        t = g.create_min.values
        od = haversine(g.p_lon, g.p_lat, g.d_lon, g.d_lat).values
        bins = (t // 10).astype(int)
        stacked = np.mean(pd.Series(bins).map(pd.Series(bins).value_counts()).values > 1)
        pts = np.c_[g.p_lon, g.p_lat]
        try: area = ConvexHull(pts * [85.0, 111.0]).volume if len(pts) >= 3 else 0.0
        except Exception: area = 0.0
        feats.append(dict(courier_id=cid, n_orders=len(g), work_hr=(t.max() - t.min()) / 60,
                          lunch_ratio=np.mean((t >= 660) & (t < 840)),
                          dinner_ratio=np.mean((t >= 1020) & (t < 1200)),
                          active_bins=len(set(bins)), mean_od_km=od.mean(),
                          stacked_ratio=stacked, area_km2=area))
    return pd.DataFrame(feats).set_index("courier_id")


def extract_personas(df, k=4, seed=0):
    F = courier_features(df)
    skew = ["n_orders", "work_hr", "active_bins", "mean_od_km", "area_km2"]
    X = F.copy(); X[skew] = np.log1p(X[skew])
    Z = StandardScaler().fit_transform(X)
    km = KMeans(k, init="k-means++", n_init=80, max_iter=300, tol=1e-5, random_state=seed).fit(Z)
    sil = silhouette_score(Z, km.labels_)
    aris = [adjusted_rand_score(km.labels_, KMeans(k, n_init=10, random_state=s).fit_predict(Z))
            for s in range(1, 11)]
    print(f"[personas] silhouette={sil:.3f}  mean ARI over 10 reseeds={np.mean(aris):.3f}")
    C = pd.DataFrame(km.cluster_centers_, columns=F.columns)
    # name clusters from centroid patterns (each persona used once)
    score = {"Super Stacker": C.stacked_ratio + C.mean_od_km + C.area_km2,
             "Lunch-Peak Specialist": C.lunch_ratio - C.work_hr,
             "Dinner-Peak Core": C.dinner_ratio - C.work_hr,
             "Full-time Workhorse": C.work_hr + C.active_bins}
    names, used = {}, set()
    for p in ["Super Stacker", "Lunch-Peak Specialist", "Dinner-Peak Core", "Full-time Workhorse"]:
        for j in score[p].sort_values(ascending=False).index:
            if j not in used: names[j] = p; used.add(j); break
    F["persona"] = [names[l] for l in km.labels_]
    stats = F.groupby("persona").mean(numeric_only=True).round(2)
    print(F.persona.value_counts().to_string()); print(stats.to_string())
    return F, stats


def _norm_persona(name):
    n = str(name).lower()
    for key, p in [("full", "Full-time Workhorse"), ("lunch", "Lunch-Peak Specialist"),
                   ("stack", "Super Stacker"), ("dinner", "Dinner-Peak Core")]:
        if key in n: return p
    return name


def personas_from_release(pdir, seed=0):
    """Cluster the 15 released rider features (computed by the dataset authors from Amap-routed waves)
    with our own preprocessing + KMeans, then verify against the released standardized features,
    cluster-quality numbers and assignments."""
    from scipy.optimize import linear_sum_assignment
    feat = pd.read_csv(f"{pdir}/rider_persona_features.csv")
    std = pd.read_csv(f"{pdir}/rider_persona_features_standardized.csv")
    asg = pd.read_csv(f"{pdir}/rider_persona_assignments.csv")
    cols = [c for c in std.columns if c != "courier_id"]
    feat = feat.set_index("courier_id").loc[std.courier_id.values]
    Z, logged = [], []
    for c in cols:                               # detect which features were log1p-transformed
        x = feat[c].astype(float).values; t = std[c].values
        zr = (x - x.mean()) / (x.std() + 1e-12)
        xl = np.log1p(np.clip(x, 0, None)); zl = (xl - xl.mean()) / (xl.std() + 1e-12)
        use_log = np.abs(zl - t).mean() < np.abs(zr - t).mean()
        Z.append(zl if use_log else zr); logged.append(c) if use_log else None
    Z = np.column_stack(Z)
    print(f"[personas] log1p detected on: {logged}")
    print(f"[personas] max |my z - released z| = {np.abs(Z - std[cols].values).max():.4f}")
    km = KMeans(k := 4, init="k-means++", n_init=80, max_iter=300, tol=1e-5, random_state=seed).fit(Z)
    sil = silhouette_score(Z, km.labels_)
    runs = [KMeans(k, n_init=10, random_state=100 + s).fit_predict(Z) for s in range(40)]
    aris = [adjusted_rand_score(runs[i], runs[j]) for i in range(40) for j in range(i + 1, 40)]
    print(f"[personas] silhouette={sil:.3f} (paper 0.108)  stability ARI={np.mean(aris):.3f} (paper 0.927)")
    rel = asg.set_index("courier_id").persona.map(_norm_persona).loc[std.courier_id.values].values
    names = list(PERSONAS)
    M = np.array([[np.sum((km.labels_ == j) & (rel == p)) for p in names] for j in range(k)])
    r, c = linear_sum_assignment(-M)
    cl2p = {j: names[cc] for j, cc in zip(r, c)}
    mine = np.array([cl2p[l] for l in km.labels_])
    print(f"[personas] agreement with released assignments: ARI={adjusted_rand_score(rel, mine):.3f}, "
          f"exact match={np.mean(rel == mine):.1%}")
    F = pd.DataFrame({"persona": mine}, index=std.courier_id.astype(str).values)
    print(F.persona.value_counts().to_string(), "\n(paper: 281 / 112 / 219 / 98)")
    summ = pd.read_csv(f"{pdir}/persona_cluster_summary.csv")
    summ["persona"] = summ.persona.map(_norm_persona)
    stats = summ.set_index("persona").round(2)
    return F, stats


DESCRIPTIONS = {
    "Full-time Workhorse": "You are an experienced full-time rider. You value stable daily earnings, steady order completion and broad spatial coverage across the day.",
    "Lunch-Peak Specialist": "You focus on the 11:00-14:00 rush, preferring short, dense, nearby orders and compact coverage.",
    "Super Stacker": "You are an efficiency master who maximizes income through complex multi-order trips. A simple single order is less attractive to you.",
    "Dinner-Peak Core": "You work mainly in the evening and prefer direct, manageable orders and simple waves; you avoid complex bundles.",
}


# ----------------------------------------------------------------------------
# 2. Agents
# ----------------------------------------------------------------------------
def fmt_t(m): return f"{int(m // 60):02d}:{int(m % 60):02d}"


def build_prompt(r, cands, t, stats):
    lines = [
        "# ROLE AND GOAL",
        "You are an AI agent acting as a delivery rider in a realistic simulation of Beijing's on-demand delivery market.",
        "Make a human-like decision according to your rider persona and current operating context.",
        "# PERSONA PROFILE", f"Persona: {r['persona']}", f"Description: {DESCRIPTIONS[r['persona']]}",
        f"Empirical statistics: {json.dumps(stats.get(r['persona'], {}))}",
        "# CURRENT STATE", f"Current time: {fmt_t(t)}",
        f"Current location: POINT ({r['loc'][0]:.5f} {r['loc'][1]:.5f})",
        f"Today's earnings so far: CNY {r['earn']:.2f}", f"Today's mileage so far: {r['dist']:.2f}km",
        "# MEMORY"]
    if not r["mem"]: lines.append("No completed orders yet.")
    else:
        lines.append(f"You have completed {r['n_done']} orders, earned CNY {r['earn']:.2f}, traveled {r['dist']:.2f} km.")
        lines += [f"- Order {m['id']}: fee {m['fee']}, duration {m['dur']:.0f}s, {'on time' if m['ok'] else 'late'}."
                  for m in r["mem"][-5:]]
    lines.append("# AVAILABLE ORDERS")
    for i, o in enumerate(cands, 1):
        lines += [f"{i}. Order ID: {o['id']}", f"- Fee: CNY {o['fee']}",
                  f"- Route-tool distance to pickup: {o['d_pick']:.2f}km",
                  f"- Route-tool distance pickup-to-delivery: {o['d_del']:.2f}km",
                  f"- Route-tool travel time: {o['t_pick']:.1f}min to pickup; {o['t_del']:.1f}min pickup-to-delivery",
                  f"- Pickup: POINT ({o['p'][0]:.5f} {o['p'][1]:.5f})",
                  f"- Delivery: POINT ({o['d'][0]:.5f} {o['d'][1]:.5f})",
                  f"- Promised delivery time: {fmt_t(o['ddl'])}"]
    lines += ["# DECISION WORKFLOW", "Think through four concise steps:", "1. Perception: summarize the offers.",
              "2. Recall & Evaluation: compare the offers with the persona.",
              "3. Planning & Bundling: decide whether a single order, a bundle, or rejection is best.",
              "4. Decision: list the accepted order IDs.", "# OUTPUT FORMAT", "Return only this JSON object:",
              '{"accepted_order_ids": [integer order IDs chosen from AVAILABLE ORDERS, or []], '
              '"reasoning_steps": ["Perception: ...","Recall & Evaluation: ...","Planning & Bundling: ...","Decision: ..."]}']
    return "\n".join(lines)


def first_json(text):
    s = text.find("{")
    while s != -1:
        depth = 0
        for i in range(s, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                try: return json.loads(text[s:i + 1])
                except Exception: break
        s = text.find("{", s + 1)
    return None


class LLMPolicy:
    def __init__(self, base_url, model, stats, api_key="EMPTY", temperature=0.2, max_tokens=1024, log=None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.stats, self.temp, self.max_tokens = model, stats, temperature, max_tokens
        self.log = log if log is not None else []

    def chat(self, prompt):
        r = self.client.chat.completions.create(
            model=self.model, temperature=self.temp, max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": "You are a delivery rider agent. Return concise strict JSON only."},
                      {"role": "user", "content": prompt}])
        return r.choices[0].message.content

    def __call__(self, r, cands, t):
        prompt = build_prompt(r, cands, t, self.stats)
        for attempt in range(2):                      # one retry, then treat as rejection
            try: obj = first_json(self.chat(prompt))
            except Exception as e: obj = None
            if obj and isinstance(obj.get("accepted_order_ids"), list):
                valid = {o["id"] for o in cands}
                ids = []
                for x in obj["accepted_order_ids"]:   # V(.): drop invalid / duplicate ids
                    try: x = int(x)
                    except Exception: continue
                    if x in valid and x not in ids: ids.append(x)
                self.log.append(dict(rider=r["id"], t=t, ids=ids, why=obj.get("reasoning_steps")))
                return ids
        return []


def nearest_policy(r, c, t): return [min(c, key=lambda o: o["d_pick"])["id"]]
def highest_fee_policy(r, c, t): return [max(c, key=lambda o: (o["fee"], -o["d_pick"]))["id"]]
def fee_dist_policy(r, c, t):
    ranked = sorted(c, key=lambda o: -o["fee"] / (o["d_pick"] + o["d_del"] + 1e-6))
    return [o["id"] for o in ranked[:2] if o["fee"] / (o["d_pick"] + o["d_del"] + 1e-6) >= 1.0]


def persona_heuristic(r, c, t):
    p = r["persona"]
    if p == "Lunch-Peak Specialist":
        return [min(c, key=lambda o: o["d_pick"])["id"]] if 630 <= t < 840 else []
    if p == "Dinner-Peak Core":
        return [min(c, key=lambda o: o["d_pick"])["id"]] if 990 <= t < 1200 else []
    n = 3 if p == "Super Stacker" else 2 if p == "Full-time Workhorse" else 1
    return [o["id"] for o in sorted(c, key=lambda o: o["d_pick"])[:n]]


BASELINES = dict(nearest=nearest_policy, highest_fee=highest_fee_policy,
                 fee_dist=fee_dist_policy, persona_heuristic=persona_heuristic)


# ----------------------------------------------------------------------------
# 3. Environment
# ----------------------------------------------------------------------------
class DeliverySim:
    def __init__(self, orders, riders_df, policy, per_persona=15, seed=0, workers=8):
        self.O = orders.sort_values("create_min").reset_index(drop=True)
        self.policy, self.workers = policy, workers
        rng = np.random.default_rng(seed)
        pick = self.O[["p_lon", "p_lat"]].values
        riders = []
        for p in PERSONAS:
            for _ in range(per_persona):
                loc = pick[rng.integers(len(pick))] + rng.normal(0, 3e-4, 2)   # small perturbation
                riders.append(dict(id=len(riders), persona=p, loc=tuple(loc), busy_until=None, wave=None,
                                   earn=0.0, dist=0.0, mem=[], n_done=0, n_dec=0))
        self.R = riders
        self.status = np.zeros(len(self.O), dtype=np.int8)      # 0 pending, 1 assigned
        self.waves = []

    def _complete(self, t):
        for r in self.R:
            if r["busy_until"] is not None and r["busy_until"] <= t:
                w = r["wave"]
                r["loc"], r["earn"], r["dist"] = w["end_loc"], r["earn"] + w["reward"], r["dist"] + w["dist"]
                r["n_done"] += w["n"]
                r["mem"] += [dict(id=i, fee=f, dur=w["dur"] * 60, ok=ok)
                             for i, f, ok in zip(w["oids"], w["fees"], w["ok"])]
                r["mem"] = r["mem"][-5:]
                r["busy_until"], r["wave"] = None, None

    def _offers(self, t):
        O = self.O
        elig = np.where((self.status == 0) & (O.deadline_min.values > t) &
                        (O.create_min.values >= t - HORIZON) & (O.create_min.values < t + DT))[0]
        idle = [r for r in self.R if r["busy_until"] is None]
        if not len(elig) or not idle: return {}
        il = np.array([r["loc"] for r in idle])
        offers = {r["id"]: [] for r in idle}
        for oi in elig:
            row = O.iloc[oi]
            d = haversine(il[:, 0], il[:, 1], row.p_lon, row.p_lat)
            for j in np.argsort(d)[:K_RIDERS]:
                if d[j] <= RHO: offers[idle[j]["id"]].append(oi)
        out = {}
        for rid, ois in offers.items():
            if not ois: continue
            r = self.R[rid]; cands = []
            for oi in ois:
                row = O.iloc[oi]
                dp, tp = route(r["loc"], (row.p_lon, row.p_lat))
                dd, td = route((row.p_lon, row.p_lat), (row.d_lon, row.d_lat))
                od = float(haversine(row.p_lon, row.p_lat, row.d_lon, row.d_lat))
                cands.append(dict(id=int(row.order_id), idx=oi, fee=reward(od), od=od, d_pick=dp, d_del=dd,
                                  t_pick=tp, t_del=td, ddl=float(row.deadline_min), p=(row.p_lon, row.p_lat),
                                  d=(row.d_lon, row.d_lat)))
            cands.sort(key=lambda o: (o["ddl"], o["d_pick"], -o["fee"]))
            out[rid] = cands[:B_OFFERS]
        return out

    def step(self, t):
        self._complete(t)
        offers = self._offers(t)
        items = list(offers.items())
        if self.workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(self.workers) as ex:
                decs = list(ex.map(lambda kv: self.policy(self.R[kv[0]], kv[1], t), items))
        else:
            decs = [self.policy(self.R[k], v, t) for k, v in items]
        bids = []
        for (rid, cands), ids in zip(items, decs):
            self.R[rid]["n_dec"] += 1
            valid = {o["id"]: o for o in cands}
            chosen = [valid[i] for i in dict.fromkeys(ids) if i in valid]
            if chosen:
                md = np.mean([o["d_pick"] for o in chosen])
                bids.append((md, rid, chosen))
        bids.sort(key=lambda b: (b[0], b[1]))                 # priority: shorter mean pickup distance
        proposals = len(bids); lost_all = lost_some = 0
        for _, rid, chosen in bids:
            avail = [o for o in chosen if self.status[o["idx"]] == 0]
            if not avail: lost_all += 1; continue
            lost_some += len(avail) < len(chosen)
            self._assign(self.R[rid], avail, t)
        return proposals, lost_all, lost_some

    def _assign(self, r, orders, t):
        wp = [r["loc"]] + [o["p"] for o in orders] + [o["d"] for o in orders]   # pickups then drop-offs
        dist = tm = 0.0
        for a, b in zip(wp[:-1], wp[1:]):
            d, m = route(a, b); dist += d; tm += m
        tm += len(orders) * SERVICE_S / 60
        fin = t + tm
        for o in orders: self.status[o["idx"]] = 1
        r["busy_until"] = fin
        r["wave"] = dict(n=len(orders), oids=[o["id"] for o in orders], fees=[o["fee"] for o in orders],
                         reward=sum(o["fee"] for o in orders), dist=dist, dur=tm, end_loc=wp[-1],
                         ok=[fin <= o["ddl"] for o in orders])
        self.waves.append(dict(rider=r["id"], persona=r["persona"], start=t, end=fin, n=len(orders),
                               dist=dist, reward=r["wave"]["reward"], ontime=float(np.mean(r["wave"]["ok"])),
                               od_km=[o["od"] for o in orders], oids=r["wave"]["oids"],
                               create=[float(self.O.create_min.iloc[o["idx"]]) for o in orders]))

    def run(self):
        stats = np.zeros(3)
        t = T_START
        while t <= T_END:
            stats += self.step(t); t += DT
        self._complete(1e9)                                    # finish carry-over work
        self.proposal_stats = dict(proposals=int(stats[0]), lost_all=int(stats[1]), lost_subset=int(stats[2]))
        return pd.DataFrame(self.waves)


# ----------------------------------------------------------------------------
# 4. Evaluation
# ----------------------------------------------------------------------------
def table1(sim, waves):
    rows, allp = [], []
    for p in PERSONAS:
        per = []
        for r in [r for r in sim.R if r["persona"] == p]:
            rw = waves[waves.rider == r["id"]] if len(waves) else waves
            n = int(rw.n.sum()) if len(rw) else 0
            per.append(dict(dec=r["n_dec"], orders=n, earn=float(rw.reward.sum()) if len(rw) else 0.0,
                            ontime=float(np.average(rw.ontime, weights=rw.n)) if n else np.nan,
                            bundle=float(rw.n.mean()) if len(rw) else np.nan,
                            stacked=float((rw.n >= 2).mean()) if len(rw) else np.nan))
        allp += per
        rows.append(pd.Series(pd.DataFrame(per).mean(), name=p))
    rows.append(pd.Series(pd.DataFrame(allp).mean(), name="ALL 60 riders"))
    return pd.DataFrame(rows).round(3)


def jsd_hourly(a_min, b_min):
    ha = np.histogram(np.asarray(a_min) / 60, bins=np.arange(8, 21))[0] + 1e-9
    hb = np.histogram(np.asarray(b_min) / 60, bins=np.arange(8, 21))[0] + 1e-9
    return float(jensenshannon(ha / ha.sum(), hb / hb.sum()) ** 2)


def validation(waves, orders, F, emp_waves=None):
    """Per-persona similarity of simulated served orders vs empirical orders of that persona."""
    od_all = haversine(orders.p_lon, orders.p_lat, orders.d_lon, orders.d_lat)
    orders = orders.assign(od=od_all, persona=orders.courier_id.astype(str).map(F.persona))
    if emp_waves is not None:
        emp_waves = emp_waves.assign(persona=emp_waves.courier_id.astype(str).map(F.persona))
    rows = []
    for p in PERSONAS:
        e = orders[orders.persona == p]
        w = waves[waves.persona == p]
        if not len(w) or not len(e): continue
        s_od = np.concatenate(w.od_km.values); s_t = np.concatenate(w.create.values)
        row = dict(persona=p, order_time_JSD=jsd_hourly(s_t, e.create_min),
                   OD_W1_km=wasserstein_distance(s_od, e.od), OD_KS=ks_2samp(s_od, e.od).statistic)
        if emp_waves is not None:
            ew = emp_waves[emp_waves.persona == p].wave_total_nav_distance_m.dropna() / 1000
            sw = w.dist.values
            if len(ew) and len(sw):
                row.update(Wave_W1_km=wasserstein_distance(sw, ew), Wave_KS=ks_2samp(sw, ew).statistic)
        rows.append(row)
    return pd.DataFrame(rows).set_index("persona").round(3)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--policy", default="nearest",
                    choices=list(BASELINES) + ["llm"])
    ap.add_argument("--base_url", default="http://localhost:8000/v1"); ap.add_argument("--model", default="")
    ap.add_argument("--per_persona", type=int, default=15); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--persona_dir"); ap.add_argument("--waves_csv")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--out", default="results")
    a = ap.parse_args()

    df = load_orders(a.data) if a.data else synthetic_orders(seed=a.seed)
    df = df[(df.create_min >= T_START) & (df.create_min < T_END)]
    print(f"orders in window: {len(df)}, couriers: {df.courier_id.nunique()}")
    df = df.assign(courier_id=df.courier_id.astype(str))
    F, stats = personas_from_release(a.persona_dir, a.seed) if a.persona_dir else extract_personas(df, seed=a.seed)
    ew = pd.read_csv(a.waves_csv) if a.waves_csv else None
    sdict = {p: stats.loc[p].to_dict() for p in stats.index}

    log = []
    pol = LLMPolicy(a.base_url, a.model, sdict, log=log) if a.policy == "llm" else BASELINES[a.policy]
    sim = DeliverySim(df, F, pol, per_persona=a.per_persona, seed=a.seed,
                      workers=a.workers if a.policy == "llm" else 1)
    t0 = time.time(); waves = sim.run()
    print(f"simulation done in {time.time() - t0:.0f}s; completed {int(waves.n.sum())}/{len(df)} orders "
          f"({waves.n.sum() / len(df):.1%}); {sim.proposal_stats}")
    T1 = table1(sim, waves); print("\n=== Table I analogue ===\n", T1.to_string())
    V = validation(waves, df, F, ew); print("\n=== Validation vs empirical ===\n", V.to_string())
    import os; os.makedirs(a.out, exist_ok=True)
    T1.to_csv(f"{a.out}/table1_{a.policy}.csv"); V.to_csv(f"{a.out}/validation_{a.policy}.csv")
    waves.drop(columns=["od_km", "create"]).to_csv(f"{a.out}/waves_{a.policy}.csv", index=False)
    save_cache(); print('Amap stats:', AMAP_STATS)
    if log: json.dump(log, open(f"{a.out}/llm_decisions.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
