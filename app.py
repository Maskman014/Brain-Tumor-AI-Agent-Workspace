import csv
import datetime
import io
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from functools import wraps

from flask import (Flask, Response, abort, flash, g, jsonify, redirect, render_template,
                   request, send_from_directory, session, url_for)
from PIL import Image, UnidentifiedImageError
from werkzeug.security import check_password_hash, generate_password_hash

from model_utils import load_model, predict

APP_NAME = "Brain Tumor Detection"
APP_TAG = "AI-assisted MRI screening"
LOW_CONF = 0.75          
PER_PAGE = 8

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_resnet_90.pth")
UPLOADS = os.path.join(BASE, "uploads")            
DB_PATH = os.path.join(BASE, "neuroscan.db")
os.makedirs(UPLOADS, exist_ok=True)

def read_json(name, default):
    p = os.path.join(MODEL_DIR, name)
    try:
        return json.load(open(p)) if os.path.exists(p) else default
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
    "Normal": "The model found no tumor pattern in this slice. A normal AI result is not a clearance — only a doctor can rule out a tumor.",
    "Tumor": "The model found patterns similar to the tumor images it was trained on. Please have a doctor review the scan.",
    "Glioma": "Gliomas arise from glial (supporting) cells of the brain. They range from slow-growing to aggressive; diagnosis and treatment are decided by specialists using imaging and biopsy.",
    "Meningioma": "Meningiomas grow from the meninges, the membranes covering the brain. Most are benign and slow-growing; some are only monitored, others treated.",
    "Pituitary tumor": "Pituitary tumors develop in the pituitary gland at the base of the brain, which controls hormones. Most are benign and can affect hormones or vision.",
}

DEFAULT_METRICS = {"classes": ["no", "yes"], "cm": [[19, 1], [1, 30]], "n_train": 202, "n_val": 0,
                   "n_test": 51, "total": 253, "epochs": 10, "test_name": "validation", "dataset": "Brain MRI Images for Brain Tumor Detection"}

def model_metrics():
    m = {**DEFAULT_METRICS, **read_json("metrics.json", {})}
    cm, names = m["cm"], [pretty(c) for c in m["classes"]]
    k, n = len(cm), sum(map(sum, cm)) or 1
    rows = []
    for i in range(k):
        tp, col, row = cm[i][i], sum(cm[r][i] for r in range(k)), sum(cm[i])
        p, r = (tp / col if col else 0), (tp / row if row else 0)
        rows.append((names[i], p, r, 2 * p * r / (p + r) if p + r else 0, row))
    return {**m, "names": names, "k": k, "n": n, "rows": rows, "train": m["n_train"],
            "test_name": "test" if m.get("n_test") and "test_name" not in read_json("metrics.json", {}) else m["test_name"],
            "acc": round(100 * sum(cm[i][i] for i in range(k)) / n, 1)}

METRICS = model_metrics()

if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found. Put brain_tumor_resnet_90.pth inside the 'model' folder.")
try:
    model = load_model(MODEL_PATH, len(CLASSES))
except RuntimeError:
    raise SystemExit("The model file doesn't match model/classes.json.")

def get_secret_key():
    sk = os.environ.get("SECRET_KEY")
    if sk:
        return sk
    p = os.path.join(BASE, "secret.key")
    if os.path.exists(p):
        return open(p).read().strip()
    generated_key = secrets.token_hex(32)
    return generated_key

app = Flask(__name__)
app.config.update(SECRET_KEY=get_secret_key(), MAX_CONTENT_LENGTH=8 * 1024 * 1024,
                   SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_HTTPONLY=True,
                   PERMANENT_SESSION_LIFETIME=datetime.timedelta(days=7))

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d:
        d.close()

with sqlite3.connect(DB_PATH) as c:
    c.executescript("""
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

@app.template_filter("fmt")
def fmt(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return s

@app.before_request
def load_user():
    g.user = None
    if request.endpoint != "static" and session.get("uid"):
        g.user = db().execute("SELECT * FROM users WHERE id=?", (session["uid"],)).fetchone()

@app.context_processor
def inject():
    return {"user": g.get("user"), "endpoint": request.endpoint, "APP_NAME": APP_NAME, "APP_TAG": APP_TAG}

def login_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if not g.user:
            if request.path.startswith("/api/"):
                return jsonify(error="Please log in again."), 401
            return redirect(url_for("login", next=request.full_path.rstrip("?")))
        return f(*a, **k)
    return wrapper

def safe_next(n):
    return n if n and n.startswith("/") and not n.startswith("//") and "\\" not in n else url_for("dashboard")

@app.route("/register", methods=["GET", "POST"])
def register():
    if g.user:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()[:40]
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "")
        if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", u):
            flash("Username must be 3-30 letters, numbers or underscores.", "err")
        elif len(p) < 6:
            flash("Password must be at least 6 characters.", "err")
        elif db().execute("SELECT 1 FROM users WHERE username=?", (u,)).fetchone():
            flash("That username is taken.", "err")
        else:
            cur = db().execute("INSERT INTO users(username,name,pw,created) VALUES(?,?,?,?)",
                               (u, name, generate_password_hash(p), now()))
            db().commit()
            session.clear()
            session["uid"] = cur.lastrowid
            session.permanent = True
            flash(f"Welcome to {APP_NAME}!", "ok")
            return redirect(url_for("analyze"))
    return render_template("auth.html", mode="register")

@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("dashboard"))
    nxt = request.values.get("next", "")
    if request.method == "POST":
        r = db().execute("SELECT * FROM users WHERE username=?", (request.form.get("username", "").strip(),)).fetchone()
        if r and check_password_hash(r["pw"], request.form.get("password", "")):
            session.clear()
            session["uid"] = r["id"]
            session.permanent = True
            return redirect(safe_next(nxt))
        flash("Invalid username or password.", "err")
    return render_template("auth.html", mode="login", next=nxt)

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/")
def home():
    return render_template("home.html", m=METRICS)

@app.route("/learn")
def learn():
    return render_template("learn.html", m=METRICS,
                           info={k: v for k, v in INFO.items() if k not in ("Normal", "Tumor")})

@app.route("/about")
def about():
    return render_template("about.html", m=METRICS)

def scan_dict(r):
    probs = json.loads(r["probs"]) if r["probs"] else {"Normal": r["p_normal"], "Tumor": r["p_tumor"]}
    label = r["label"]
    is_tumor = label != "Normal"
    return {
        "id": r["id"], "patient": r["patient"], "name": r["original_name"], "label": label,
        "is_tumor": is_tumor, "tumor": is_tumor,
        "headline": "Tumor detected" if is_tumor else "No tumor detected",
        "kind": label if label not in ("Normal", "Tumor") else "",
        "info": INFO.get(label, ""),
        "confidence": round(r["confidence"] * 100, 1), "uncertain": bool(r["uncertain"]),
        "probs": [{"name": k, "pct": round(v * 100, 1), "normal": k == "Normal", "tumor": k != "Normal"}
                  for k, v in sorted(probs.items(), key=lambda kv: -kv[1])],
        "seconds": round((r["ms"] or 0) / 1000, 2), "created": fmt(r["created"]),
        "original_url": url_for("media", scan_id=r["id"], kind="orig"),
        "heatmap_url": url_for("media", scan_id=r["id"], kind="cam"),
        "report_url": url_for("report", scan_id=r["id"]),
    }

def owned(scan_id):
    r = db().execute("SELECT * FROM scans WHERE id=? AND user_id=?", (scan_id, g.user["id"])).fetchone()
    if not r:
        abort(404)
    return r

def remove_files(uid):
    for k in ("orig", "cam"):
        p = os.path.join(UPLOADS, f"{uid}_{k}.png")
        if os.path.exists(p):
            os.remove(p)

@app.route("/analyze")
@login_required
def analyze():
    return render_template("analyze.html")

@app.route("/dashboard")
@login_required
def dashboard():
    uid = g.user["id"]
import csv
import datetime
import io
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from functools import wraps

from flask import (Flask, Response, abort, flash, g, jsonify, redirect, render_template,
                   request, send_from_directory, session, url_for)
from PIL import Image, UnidentifiedImageError
from werkzeug.security import check_password_hash, generate_password_hash

from model_utils import load_model, predict

APP_NAME = "Brain Tumor Detection"
APP_TAG = "AI-assisted MRI screening"
LOW_CONF = 0.75          
PER_PAGE = 8

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_resnet_90.pth")
UPLOADS = os.path.join(BASE, "uploads")            
DB_PATH = os.path.join(BASE, "neuroscan.db")
os.makedirs(UPLOADS, exist_ok=True)

def read_json(name, default):
    p = os.path.join(MODEL_DIR, name)
    try:
        return json.load(open(p)) if os.path.exists(p) else default
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
    "Normal": "The model found no tumor pattern in this slice. A normal AI result is not a clearance — only a doctor can rule out a tumor.",
    "Tumor": "The model found patterns similar to the tumor images it was trained on. Please have a doctor review the scan.",
    "Glioma": "Gliomas arise from glial (supporting) cells of the brain. They range from slow-growing to aggressive; diagnosis and treatment are decided by specialists using imaging and biopsy.",
    "Meningioma": "Meningiomas grow from the meninges, the membranes covering the brain. Most are benign and slow-growing; some are only monitored, others treated.",
    "Pituitary tumor": "Pituitary tumors develop in the pituitary gland at the base of the brain, which controls hormones. Most are benign and can affect hormones or vision.",
}

DEFAULT_METRICS = {"classes": ["no", "yes"], "cm": [[19, 1], [1, 30]], "n_train": 202, "n_val": 0,
                   "n_test": 51, "total": 253, "epochs": 10, "test_name": "validation", "dataset": "Brain MRI Images for Brain Tumor Detection"}

def model_metrics():
    m = {**DEFAULT_METRICS, **read_json("metrics.json", {})}
    cm, names = m["cm"], [pretty(c) for c in m["classes"]]
    k, n = len(cm), sum(map(sum, cm)) or 1
    rows = []
    for i in range(k):
        tp, col, row = cm[i][i], sum(cm[r][i] for r in range(k)), sum(cm[i])
        p, r = (tp / col if col else 0), (tp / row if row else 0)
        rows.append((names[i], p, r, 2 * p * r / (p + r) if p + r else 0, row))
    return {**m, "names": names, "k": k, "n": n, "rows": rows, "train": m["n_train"],
            "test_name": "test" if m.get("n_test") and "test_name" not in read_json("metrics.json", {}) else m["test_name"],
            "acc": round(100 * sum(cm[i][i] for i in range(k)) / n, 1)}

METRICS = model_metrics()

if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found. Put brain_tumor_resnet_90.pth inside the 'model' folder.")
try:
    model = load_model(MODEL_PATH, len(CLASSES))
except RuntimeError:
    raise SystemExit("The model file doesn't match model/classes.json.")

def get_secret_key():
    sk = os.environ.get("SECRET_KEY")
    if sk:
        return sk
    p = os.path.join(BASE, "secret.key")
    if os.path.exists(p):
        return open(p).read().strip()
    generated_key = secrets.token_hex(32)
    return generated_key

app = Flask(__name__)
app.config.update(SECRET_KEY=get_secret_key(), MAX_CONTENT_LENGTH=8 * 1024 * 1024,
                   SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_HTTPONLY=True,
                   PERMANENT_SESSION_LIFETIME=datetime.timedelta(days=7))

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d:
        d.close()

with sqlite3.connect(DB_PATH) as c:
    c.executescript("""
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

@app.template_filter("fmt")
def fmt(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return s

@app.before_request
def load_user():
    g.user = None
    if request.endpoint != "static" and session.get("uid"):
        g.user = db().execute("SELECT * FROM users WHERE id=?", (session["uid"],)).fetchone()

@app.context_processor
def inject():
    return {"user": g.get("user"), "endpoint": request.endpoint, "APP_NAME": APP_NAME, "APP_TAG": APP_TAG}

def login_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if not g.user:
            if request.path.startswith("/api/"):
                return jsonify(error="Please log in again."), 401
            return redirect(url_for("login", next=request.full_path.rstrip("?")))
        return f(*a, **k)
    return wrapper

def safe_next(n):
    return n if n and n.startswith("/") and not n.startswith("//") and "\\" not in n else url_for("dashboard")

@app.route("/register", methods=["GET", "POST"])
def register():
    if g.user:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()[:40]
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "")
        if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", u):
            flash("Username must be 3-30 letters, numbers or underscores.", "err")
        elif len(p) < 6:
            flash("Password must be at least 6 characters.", "err")
        elif db().execute("SELECT 1 FROM users WHERE username=?", (u,)).fetchone():
            flash("That username is taken.", "err")
        else:
            cur = db().execute("INSERT INTO users(username,name,pw,created) VALUES(?,?,?,?)",
                               (u, name, generate_password_hash(p), now()))
            db().commit()
            session.clear()
            session["uid"] = cur.lastrowid
            session.permanent = True
            flash(f"Welcome to {APP_NAME}!", "ok")
            return redirect(url_for("analyze"))
    return render_template("auth.html", mode="register")

@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("dashboard"))
    nxt = request.values.get("next", "")
    if request.method == "POST":
        r = db().execute("SELECT * FROM users WHERE username=?", (request.form.get("username", "").strip(),)).fetchone()
        if r and check_password_hash(r["pw"], request.form.get("password", "")):
            session.clear()
            session["uid"] = r["id"]
            session.permanent = True
            return redirect(safe_next(nxt))
        flash("Invalid username or password.", "err")
    return render_template("auth.html", mode="login", next=nxt)

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/")
def home():
    return render_template("home.html", m=METRICS)

@app.route("/learn")
def learn():
    return render_template("learn.html", m=METRICS,
                           info={k: v for k, v in INFO.items() if k not in ("Normal", "Tumor")})

@app.route("/about")
def about():
    return render_template("about.html", m=METRICS)

def scan_dict(r):
    probs = json.loads(r["probs"]) if r["probs"] else {"Normal": r["p_normal"], "Tumor": r["p_tumor"]}
    label = r["label"]
    is_tumor = label != "Normal"
    return {
        "id": r["id"], "patient": r["patient"], "name": r["original_name"], "label": label,
        "is_tumor": is_tumor, "tumor": is_tumor,
        "headline": "Tumor detected" if is_tumor else "No tumor detected",
        "kind": label if label not in ("Normal", "Tumor") else "",
        "info": INFO.get(label, ""),
        "confidence": round(r["confidence"] * 100, 1), "uncertain": bool(r["uncertain"]),
        "probs": [{"name": k, "pct": round(v * 100, 1), "normal": k == "Normal", "tumor": k != "Normal"}
                  for k, v in sorted(probs.items(), key=lambda kv: -kv[1])],
        "seconds": round((r["ms"] or 0) / 1000, 2), "created": fmt(r["created"]),
        "original_url": url_for("media", scan_id=r["id"], kind="orig"),
        "heatmap_url": url_for("media", scan_id=r["id"], kind="cam"),
        "report_url": url_for("report", scan_id=r["id"]),
    }

def owned(scan_id):
    r = db().execute("SELECT * FROM scans WHERE id=? AND user_id=?", (scan_id, g.user["id"])).fetchone()
    if not r:
        abort(404)
    return r

def remove_files(uid):
    for k in ("orig", "cam"):
        p = os.path.join(UPLOADS, f"{uid}_{k}.png")
        if os.path.exists(p):
            os.remove(p)

@app.route("/analyze")
@login_required
def analyze():
    return render_template("analyze.html")

@app.route("/dashboard")
@login_required
def dashboard():
    uid = g.user["id"]
    # Perfected computation logic for tracking metrics loop bounds
    peak = max([x["tumor"] + x["normal"] for x in series] +[])
    for x in series:
        x["total"] = x["tumor"] + x["normal"]
        x["h"] = round(x["total"] / peak * 110)
        
    by_type = [dict(label=r["label"], c=r["c"], pct=round(r["c"] / total * 100)) if total else dict(label=r["label"], c=r["c"], pct=0) for r in db().execute(
        "SELECT label, COUNT(*) c FROM scans WHERE user_id=? GROUP BY label ORDER BY c DESC", (uid,))]
    recent = [scan_dict(r) for r in db().execute(
        "SELECT * FROM scans WHERE user_id=? ORDER BY id DESC LIMIT 5", (uid,))]
    return render_template("dashboard.html", total=total, tumor=tumor, normal=normal,
                           unc=s["unc"] or 0, avg=round((s["avg"] or 0) * 100, 1),
                           tp=round(tumor / total * 100, 1) if total else 0, series=series,
                           by_type=by_type, recent=recent, multi=len(CLASSES) > 2,
                           types=[{**t, "tumor": t["label"] != "Normal"} for t in by_type])

def history_filter():
    q = request.args.get("q", "").strip()[:60]
    label = request.args.get("label", "")[:30]
    where, args = ["user_id=?"], [g.user["id"]]
    if q:
        where.append("(patient LIKE ? OR original_name LIKE ?)")
        args += [f"%{q}%"] * 2
    if label == "tumor":
        where.append("label!='Normal'")
    elif label:
        where.append("label=?")
        args.append(label)
    return q, label, " AND ".join(where), args

@app.route("/history")
@login_required
def history():
    q, label, w, args = history_filter()
    page = max(1, request.args.get("page", 1, type=int))
    total = db().execute(f"SELECT COUNT(*) FROM scans WHERE {w}", args).fetchone()[0]
    rows = db().execute(f"SELECT * FROM scans WHERE {w} ORDER BY id DESC LIMIT ? OFFSET ?",
                        args + [PER_PAGE, (page - 1) * PER_PAGE]).fetchall()
    labels = [r[0] for r in db().execute("SELECT DISTINCT label FROM scans WHERE user_id=? ORDER BY label", (g.user["id"],))]
    opts = [("", "All results"), ("tumor", "Any tumor")] + [(l, l) for l in labels]
    return render_template("history.html", scans=[scan_dict(r) for r in rows], q=q, label=label, labels=labels, opts=opts,
                           page=page, pages=max(1, -(-total // PER_PAGE)), total=total)

@app.route("/history.csv")
@login_required
def history_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "date", "patient", "file", "result", "confidence_%", "inconclusive", "all_probabilities_%"])
    safe = lambda v: "'" + v if v and v[0] in "=+-@" else v
    for r in db().execute("SELECT * FROM scans WHERE user_id=? ORDER BY id DESC", (g.user["id"],)):
        d = scan_dict(r)
        w.writerow([r["id"], r["created"], safe(r["patient"]), safe(r["original_name"]), r["label"], d["confidence"],
                    "yes" if r["uncertain"] else "no", "; ".join(f"{p['name']} {p['pct']}" for p in d["probs"])])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=brain_tumor_history.csv"})

@app.route("/report/<int:scan_id>")
@login_required
def report(scan_id):
    return render_template("report.html", s=scan_dict(owned(scan_id)))

@app.route("/media/<int:scan_id>/<kind>")
@login_required
def media(scan_id, kind):
    if kind not in ("orig", "cam"):
        abort(404)
    r = owned(scan_id)
    resp = send_from_directory(UPLOADS, f"{r['uid']}_{kind}.png")
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp

@app.route("/account", methods=["GET", "POST"])
@login_required
def account():
    if request.method == "POST":
        act = request.form.get("action")
        if act == "password":
            new = request.form.get("new", "")
            if not check_password_hash(g.user["pw"], request.form.get("old", "")):
                flash("Current password is incorrect.", "err")
            elif len(new) < 6:
                flash("New password must be at least 6 characters.", "err")
            else:
                db().execute("UPDATE users SET pw=? WHERE id=?", (generate_password_hash(new), g.user["id"]))
                db().commit()
                flash("Password updated.", "ok")
        elif act == "clear":
            for r in db().execute("SELECT uid FROM scans WHERE user_id=?", (g.user["id"],)).fetchall():
                remove_files(r["uid"])
            db().execute("DELETE FROM scans WHERE user_id=?", (g.user["id"],))
            db().commit()
            flash("All your scans were deleted.", "ok")
        return redirect(url_for("account"))
    n = db().execute("SELECT COUNT(*) FROM scans WHERE user_id=?", (g.user["id"],)).fetchone()[0]
    return render_template("account.html", n=n)

@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify(error="Please choose an MRI image."), 400
    try:
        img = Image.open(f.stream)
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return jsonify(error="That file is not a valid image (use JPG or PNG)."), 400
    if min(img.size) < 64:
        return jsonify(error="Image is too small to analyse."), 400
    img.thumbnail((1024, 1024))
    t0 = time.time()
    idx, probs, cam = predict(model, img)
    ms = int((time.time() - t0) * 1000)
    uid = uuid.uuid4().hex
    img.convert("RGB").save(os.path.join(UPLOADS, f"{uid}_orig.png"))
    cam.save(os.path.join(UPLOADS, f"{uid}_cam.png"))
    conf = float(probs[idx])
    p_normal = float(sum(p for n, p in zip(CLASSES, probs) if n == "Normal"))
    cur = db().execute(
        "INSERT INTO scans(user_id,uid,original_name,patient,label,confidence,p_normal,p_tumor,uncertain,ms,created,probs)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (g.user["id"], uid, os.path.basename(f.filename)[:80], request.form.get("patient", "").strip()[:60],
         CLASSES[idx], conf, p_normal, 1 - p_normal, int(conf < LOW_CONF), ms, now(),
         json.dumps({n: float(p) for n, p in zip(CLASSES, probs)})))
    db().commit()
    return jsonify(scan_dict(db().execute("SELECT * FROM scans WHERE id=?", (cur.lastrowid,)).fetchone()))

@app.route("/api/delete/<int:scan_id>", methods=["POST"])
@login_required
def api_delete(scan_id):
    r = owned(scan_id)
    remove_files(r["uid"])
    db().execute("DELETE FROM scans WHERE id=?", (scan_id,))
    db().commit()
    return jsonify(ok=True)

@app.errorhandler(413)
def too_big(_):
    return jsonify(error="File too large (max 8 MB)."), 413

@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404, msg="That page doesn't exist."), 404

@app.errorhandler(500)
def server_error(_):
    return render_template("error.html", code=500, msg="Something went wrong on our side."), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    try:
        from waitress import serve
    except ImportError:
        app.run(host="0.0.0.0", port=port)
    else:
        print(f"{APP_NAME} running at http://127.0.0.1:{port}  (Ctrl+C to stop)")
        serve(app, host="0.0.0.0", port=port, threads=4)
