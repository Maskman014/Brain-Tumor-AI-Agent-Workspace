"""Shared logic for the Streamlit version (same data model as app.py: users + scans in SQLite)."""
import base64
import csv
import datetime
import io
import json
import os
import re
import sqlite3
import time
import uuid

from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash

from model_utils import load_model, predict

APP_NAME = "Brain Tumor Detection"
APP_TAG = "AI-assisted MRI screening"
LOW_CONF = 0.75          # below this a result is shown as "inconclusive"
PER_PAGE = 8

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_resnet_90.pth")
UPLOADS = os.path.join(BASE, "uploads")
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE, "neuroscan.db"))
os.makedirs(UPLOADS, exist_ok=True)


def read_json(name, default):
    p = os.path.join(MODEL_DIR, name)
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


NORMAL_KEYS = {"no", "notumor", "no_tumor", "normal", "healthy", "none"}


def pretty(k):
    kl = k.lower().replace(" ", "_")
    if kl in NORMAL_KEYS:
        return "Normal"
    if kl == "yes":
        return "Tumor"
    if kl == "pituitary":
        return "Pituitary tumor"
    return k.replace("_", " ").strip().title()


RAW = read_json("classes.json", ["no", "yes"])
if isinstance(RAW, dict):
    RAW = RAW.get("classes", ["no", "yes"])
CLASSES = [pretty(c) for c in RAW]

INFO = {
    "Normal": "The model found no tumor pattern in this slice. A normal AI result is not a clearance - only a doctor can rule out a tumor.",
    "Tumor": "The model found patterns similar to the tumor images it was trained on. Please have a doctor review the scan.",
    "Glioma": "Gliomas arise from glial (supporting) cells of the brain. They range from slow-growing to aggressive; diagnosis and treatment are decided by specialists using imaging and biopsy.",
    "Meningioma": "Meningiomas grow from the meninges, the membranes covering the brain. Most are benign and slow-growing; some are only monitored, others treated.",
    "Pituitary tumor": "Pituitary tumors develop in the pituitary gland at the base of the brain, which controls hormones. Most are benign and can affect hormones or vision.",
}

DEFAULT_METRICS = {"classes": ["no", "yes"], "cm": [[19, 1], [1, 30]], "n_train": 202, "n_val": 0,
                   "n_test": 51, "total": 253, "epochs": 10, "test_name": "validation",
                   "dataset": "Brain MRI Images for Brain Tumor Detection"}


def model_metrics():
    saved = read_json("metrics.json", {})
    m = {**DEFAULT_METRICS, **saved}
    cm, names = m["cm"], [pretty(c) for c in m["classes"]]
    k, n = len(cm), sum(map(sum, cm)) or 1
    rows = []
    for i in range(k):
        tp, col, row = cm[i][i], sum(cm[r][i] for r in range(k)), sum(cm[i])
        p, r = (tp / col if col else 0), (tp / row if row else 0)
        rows.append((names[i], p, r, 2 * p * r / (p + r) if p + r else 0, row))
    return {**m, "names": names, "k": k, "n": n, "rows": rows, "train": m["n_train"],
            "test_name": "test" if m.get("n_test") and "test_name" not in saved else m["test_name"],
            "acc": round(100 * sum(cm[i][i] for i in range(k)) / n, 1)}


METRICS = model_metrics()


def get_model():
    """Returns (model, error_message)."""
    if not os.path.exists(MODEL_PATH):
        return None, "Model not found. Add model/brain_tumor_resnet_90.pth (and classes.json) to the repository."
    try:
        return load_model(MODEL_PATH, len(CLASSES)), None
    except Exception as e:
        return None, f"The model file doesn't match model/classes.json ({len(CLASSES)} classes): {e}"


# ---------------------------------------------------------------- database
def db():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


with db() as _c:
    _c.executescript("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE COLLATE NOCASE,
        name TEXT, pw TEXT, created TEXT);
    CREATE TABLE IF NOT EXISTS scans(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, uid TEXT, original_name TEXT,
        patient TEXT, label TEXT, confidence REAL, p_normal REAL, p_tumor REAL,
        uncertain INTEGER, ms INTEGER, created TEXT, probs TEXT);
    CREATE INDEX IF NOT EXISTS ix_scans_user ON scans(user_id, id);""")


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return s


# ---------------------------------------------------------------- accounts
def register(name, username, pw):
    """Returns (user_id, error)."""
    name, username = name.strip()[:40], username.strip()
    if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", username):
        return None, "Username must be 3-30 letters, numbers or underscores."
    if len(pw) < 6:
        return None, "Password must be at least 6 characters."
    with db() as c:
        if c.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            return None, "That username is taken."
        cur = c.execute("INSERT INTO users(username,name,pw,created) VALUES(?,?,?,?)",
                        (username, name, generate_password_hash(pw), now()))
        return cur.lastrowid, None


def login(username, pw):
    with db() as c:
        r = c.execute("SELECT * FROM users WHERE username=?", (username.strip(),)).fetchone()
    return r["id"] if r and check_password_hash(r["pw"], pw) else None


def get_user(uid):
    with db() as c:
        return c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()


def change_password(uid, old, new):
    u = get_user(uid)
    if not check_password_hash(u["pw"], old):
        return "Current password is incorrect."
    if len(new) < 6:
        return "New password must be at least 6 characters."
    with db() as c:
        c.execute("UPDATE users SET pw=? WHERE id=?", (generate_password_hash(new), uid))
    return None


# ---------------------------------------------------------------- scans
def scan_dict(r):
    probs = json.loads(r["probs"]) if r["probs"] else {"Normal": r["p_normal"], "Tumor": r["p_tumor"]}
    label = r["label"]
    is_tumor = label != "Normal"
    return {
        "id": r["id"], "patient": r["patient"] or "", "name": r["original_name"], "label": label,
        "is_tumor": is_tumor, "headline": "Tumor detected" if is_tumor else "No tumor detected",
        "kind": label if label not in ("Normal", "Tumor") else "", "info": INFO.get(label, ""),
        "confidence": round(r["confidence"] * 100, 1), "uncertain": bool(r["uncertain"]),
        "probs": [{"name": k, "pct": round(v * 100, 1)} for k, v in sorted(probs.items(), key=lambda kv: -kv[1])],
        "seconds": round((r["ms"] or 0) / 1000, 2), "created": fmt(r["created"]),
        "orig": os.path.join(UPLOADS, f"{r['uid']}_orig.png"), "cam": os.path.join(UPLOADS, f"{r['uid']}_cam.png"),
    }


def save_scan(uid, filename, img, patient, model):
    """Analyse one PIL image, store it, return the scan dict."""
    if min(img.size) < 64:
        raise ValueError("Image is too small to analyse.")
    img = img.copy()
    img.thumbnail((1024, 1024))
    t0 = time.time()
    idx, probs, cam = predict(model, img)
    ms = int((time.time() - t0) * 1000)
    u = uuid.uuid4().hex
    img.convert("RGB").save(os.path.join(UPLOADS, f"{u}_orig.png"))
    cam.save(os.path.join(UPLOADS, f"{u}_cam.png"))
    conf = float(probs[idx])
    p_normal = float(sum(p for n, p in zip(CLASSES, probs) if n == "Normal"))
    with db() as c:
        cur = c.execute(
            "INSERT INTO scans(user_id,uid,original_name,patient,label,confidence,p_normal,p_tumor,uncertain,ms,created,probs)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (uid, u, os.path.basename(filename)[:80], patient.strip()[:60], CLASSES[idx], conf, p_normal,
             1 - p_normal, int(conf < LOW_CONF), ms, now(), json.dumps({n: float(p) for n, p in zip(CLASSES, probs)})))
        return scan_dict(c.execute("SELECT * FROM scans WHERE id=?", (cur.lastrowid,)).fetchone())


def history(uid, q="", label="", page=1):
    where, args = ["user_id=?"], [uid]
    if q:
        where.append("(patient LIKE ? OR original_name LIKE ?)")
        args += [f"%{q}%"] * 2
    if label == "tumor":
        where.append("label!='Normal'")
    elif label:
        where.append("label=?")
        args.append(label)
    w = " AND ".join(where)
    with db() as c:
        total = c.execute(f"SELECT COUNT(*) FROM scans WHERE {w}", args).fetchone()[0]
        pages = max(1, -(-total // PER_PAGE))
        page = min(max(1, page), pages)
        rows = c.execute(f"SELECT * FROM scans WHERE {w} ORDER BY id DESC LIMIT ? OFFSET ?",
                         args + [PER_PAGE, (page - 1) * PER_PAGE]).fetchall()
        labels = [r[0] for r in c.execute("SELECT DISTINCT label FROM scans WHERE user_id=? ORDER BY label", (uid,))]
    return [scan_dict(r) for r in rows], total, pages, page, labels


def _remove_files(u):
    for k in ("orig", "cam"):
        p = os.path.join(UPLOADS, f"{u}_{k}.png")
        if os.path.exists(p):
            os.remove(p)


def delete_scan(uid, scan_id):
    with db() as c:
        r = c.execute("SELECT uid FROM scans WHERE id=? AND user_id=?", (scan_id, uid)).fetchone()
        if r:
            _remove_files(r["uid"])
            c.execute("DELETE FROM scans WHERE id=?", (scan_id,))


def clear_scans(uid):
    with db() as c:
        for r in c.execute("SELECT uid FROM scans WHERE user_id=?", (uid,)).fetchall():
            _remove_files(r["uid"])
        c.execute("DELETE FROM scans WHERE user_id=?", (uid,))


def stats(uid):
    with db() as c:
        s = c.execute("""SELECT COUNT(*) total, SUM(label!='Normal') tumor, SUM(label='Normal') normal,
            SUM(uncertain) unc, AVG(confidence) avg FROM scans WHERE user_id=?""", (uid,)).fetchone()
        days = [datetime.date.today() - datetime.timedelta(days=i) for i in range(6, -1, -1)]
        rows = c.execute("""SELECT substr(created,1,10) d, (label='Normal') n, COUNT(*) c FROM scans
            WHERE user_id=? AND substr(created,1,10)>=? GROUP BY substr(created,1,10), (label='Normal')""",
                         (uid, days[0].isoformat())).fetchall()
        by_type = [(r["label"], r["c"]) for r in c.execute(
            "SELECT label, COUNT(*) c FROM scans WHERE user_id=? GROUP BY label ORDER BY c DESC", (uid,))]
        recent = [scan_dict(r) for r in c.execute("SELECT * FROM scans WHERE user_id=? ORDER BY id DESC LIMIT 5", (uid,))]
    cnt = {(r["d"], r["n"]): r["c"] for r in rows}
    series = [{"Day": d.strftime("%a %d"), "Tumor": cnt.get((d.isoformat(), 0), 0),
               "Normal": cnt.get((d.isoformat(), 1), 0)} for d in days]
    total = s["total"] or 0
    return {"total": total, "tumor": s["tumor"] or 0, "normal": s["normal"] or 0, "unc": s["unc"] or 0,
            "avg": round((s["avg"] or 0) * 100, 1), "series": series, "by_type": by_type, "recent": recent}


# ---------------------------------------------------------------- exports
def _safe(v):  # CSV formula-injection guard
    v = str(v or "")
    return "'" + v if v and v[0] in "=+-@" else v


def csv_text(uid):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "date", "patient", "file", "result", "confidence_%", "inconclusive", "all_probabilities_%"])
    with db() as c:
        for r in c.execute("SELECT * FROM scans WHERE user_id=? ORDER BY id DESC", (uid,)):
            d = scan_dict(r)
            w.writerow([r["id"], r["created"], _safe(r["patient"]), _safe(r["original_name"]), r["label"], d["confidence"],
                        "yes" if r["uncertain"] else "no", "; ".join(f"{p['name']} {p['pct']}" for p in d["probs"])])
    return buf.getvalue()


def _b64(path):
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except OSError:
        return ""


def report_html(s):
    """Self-contained printable report (open it and use Print -> Save as PDF)."""
    esc = lambda t: str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    probs = "".join(f"<tr><td>{esc(p['name'])}</td><td>{p['pct']}%</td></tr>" for p in s["probs"])
    note = "<p><b>Inconclusive:</b> confidence is low, treat this result with extra caution.</p>" if s["uncertain"] else ""
    return f"""<!doctype html><meta charset="utf-8"><title>Scan report #{s['id']}</title>
<style>body{{font:15px/1.5 Arial,sans-serif;max-width:820px;margin:30px auto;color:#111}}
img{{width:48%;border:1px solid #ccc;border-radius:6px}}table{{border-collapse:collapse}}td{{border:1px solid #ccc;padding:5px 12px}}
.r{{font-size:22px;font-weight:700;color:{'#b42318' if s['is_tumor'] else '#067647'}}}small{{color:#555}}</style>
<h1>{APP_NAME} - scan report</h1><p>Report #{s['id']} &middot; {esc(s['created'])} &middot; File: {esc(s['name'])}
{'&middot; Patient / label: ' + esc(s['patient']) if s['patient'] else ''}</p>
<p class="r">{s['headline']}{(' (' + esc(s['kind']) + ')') if s['kind'] else ''} - {s['confidence']}% confidence</p>{note}
<p>{esc(s['info'])}</p><p><img src="{_b64(s['orig'])}"> <img src="{_b64(s['cam'])}"></p>
<table>{probs}</table>
<p><small>Educational project - not a medical device. This result must not be used for diagnosis or treatment decisions.
Always consult a qualified doctor.</small></p>"""
