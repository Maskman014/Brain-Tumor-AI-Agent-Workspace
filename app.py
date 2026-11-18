import os
import re
import csv
import io
import json
import time
import uuid
import sqlite3
import datetime
from PIL import Image, UnidentifiedImageError
from werkzeug.security import generate_password_hash, check_password_hash
import streamlit as st

# --- Import Core Deep Learning Hooks ---
from model_utils import load_model, predict

# --- Global Configurations ---
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

# --- Visual Core Shell Theme Configuration ---
st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="🧠")

# --- Persistent DB Session Storage Handlers ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Database Initialization
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

# --- Resource Optimization Singleton ---
@st.cache_resource
def get_optimized_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model weights binary package missing at target path storage configuration destinations.")
        st.stop()
    return load_model(MODEL_PATH, len(CLASSES))

model = get_optimized_model()

# --- Shared UI Core Helpers ---
def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def fmt(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return s

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
        "uid": r["uid"]
    }

# --- State Engine Persistence Layout ---
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_fullname" not in st.session_state:
    st.session_state.user_fullname = None

# --- Top App Navigation Bar Header Layout ---
st.title(f"🧠 {APP_NAME}")
st.caption(f"*{APP_TAG}*")

# Navigation Routing tabs interface setup
if st.session_state.user_id is None:
    menu = ["Login", "Register"]
else:
    menu = ["AI Workspace", "Dashboard Metrics", "Screening History", "User Account Options"]

choice = st.sidebar.radio("Navigation Menu Control Panel", menu)

# --- Authentication Workspaces Router Context ---
if choice == "Register":
    st.subheader("Create a Free Clinical Workspace Account")
    reg_name = st.text_input("Full Professional Name", max_chars=40)
    reg_user = st.text_input("Desired Username System ID")
    reg_pass = st.text_input("Account Secret Password", type="password")
    
    if st.button("Complete Deployment Registration"):
        db = get_db_connection()
        if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", reg_user.strip()):
            st.error("Username must be 3-30 alpha-numeric characters or underscores.")
        elif len(reg_pass) < 6:
            st.error("Security passphrase string must length exceed 6 minimum boundaries.")
        elif db.execute("SELECT 1 FROM users WHERE username=?", (reg_user.strip(),)).fetchone():
            st.error("That administrative clinical database identification ID name profile is taken.")
        else:
            db.execute("INSERT INTO users(username,name,pw,created) VALUES(?,?,?,?)",
                       (reg_user.strip(), reg_name.strip(), generate_password_hash(reg_pass), now()))
            db.commit()
            st.success("Registration parameter blocks validated! Proceed to login tab.")
        db.close()

elif choice == "Login":
    st.subheader("Secure Workspace User Sign-In Portal")
    login_user = st.text_input("Username Identification Profile")
    login_pass = st.text_input("Passphrase Key Verification", type="password")
    
    if st.button("Authenticate Workspace Session Key"):
        db = get_db_connection()
        r = db.execute("SELECT * FROM users WHERE username=?", (login_user.strip(),)).fetchone()
        if r and check_password_hash(r["pw"], login_pass):
            st.session_state.user_id = r["id"]
            st.session_state.user_fullname = r["name"] if r["name"] else r["username"]
            st.success(f"Welcome back, operator {st.session_state.user_fullname}!")
            st.rerun()
        else:
            st.error("Invalid secure operational validation credentials matching indices failed.")
        db.close()

# --- Logged-In Core Operations ---
elif choice == "AI Workspace":
    st.subheader(f"👨‍⚕️ Screening Analysis Deck — Operator: {st.session_state.user_fullname}")
    
    patient = st.text_input("Patient Reference Record Tracking Identifier", max_chars=60).strip()
    f = st.file_uploader("Upload Target Diagnostic Structural Brain MRI Slice Sequence Node File", type=["png", "jpg", "jpeg"])
    
    if f is not None:
        try:
            img = Image.open(f)
            img.load()
        except Exception:
            st.error("That target asset input file format cannot handle standard image array maps payload parsing configurations.")
        else:
            if min(img.size) < 64:
                st.error("Target scan array resolution metrics below structural execution safety thresholds.")
            else:
                img.thumbnail((1024, 1024))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original Processing Sequence Source", use_container_width=True)
                
                if st.button("Run Diagnostic Agent Core Diagnostics Inference Model Pipelines"):
                    with st.spinner("Executing dense layer gradient activation processing layers..."):
                        t0 = time.time()
                        idx, probs, cam = predict(model, img)
                        ms = int((time.time() - t0) * 1000)
                        
                        uid = uuid.uuid4().hex
                        img.convert("RGB").save(os.path.join(UPLOADS, f"{uid}_orig.png"))
                        cam.save(os.path.join(UPLOADS, f"{uid}_cam.png"))
                        
                        conf = float(probs[idx])
                        p_normal = float(sum(p for n, p in zip(CLASSES, probs) if n == "Normal"))
                        
                        db = get_db_connection()
                        db.execute(
                            "INSERT INTO scans(user_id,uid,original_name,patient,label,confidence,p_normal,p_tumor,uncertain,ms,created,probs)"
                            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (st.session_state.user_id, uid, os.path.basename(f.name)[:80], patient,
                             CLASSES[idx], conf, p_normal, 1 - p_normal, int(conf < LOW_CONF), ms, now(),
                             json.dumps({n: float(p) for n, p in zip(CLASSES, probs)})))
                        db.commit()
                        db.close()
                        
                        with col2:
import os
import re
import csv
import io
import json
import time
import uuid
import sqlite3
import datetime
from PIL import Image, UnidentifiedImageError
from werkzeug.security import generate_password_hash, check_password_hash
import streamlit as st

# --- Import Core Deep Learning Hooks ---
from model_utils import load_model, predict

# --- Global Configurations ---
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

# --- Visual Core Shell Theme Configuration ---
st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="🧠")

# --- Persistent DB Session Storage Handlers ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Database Initialization
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

# --- Resource Optimization Singleton ---
@st.cache_resource
def get_optimized_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model weights binary package missing at target path storage configuration destinations.")
        st.stop()
    return load_model(MODEL_PATH, len(CLASSES))

model = get_optimized_model()

# --- Shared UI Core Helpers ---
def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def fmt(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return s

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
        "uid": r["uid"]
    }

# --- State Engine Persistence Layout ---
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_fullname" not in st.session_state:
    st.session_state.user_fullname = None

# --- Navigation Setup ---
if st.session_state.user_id is None:
    menu = ["Login", "Register"]
else:
    menu = ["AI Workspace", "Dashboard Metrics", "Screening History", "User Account Options"]

choice = st.sidebar.radio("Navigation Menu Control Panel", menu)

# --- Authentication Workspaces Router Context ---
if choice == "Register":
    st.subheader("Create a Free Clinical Workspace Account")
    reg_name = st.text_input("Full Professional Name", max_chars=40)
    reg_user = st.text_input("Desired Username System ID")
    reg_pass = st.text_input("Account Secret Password", type="password")
    
    if st.button("Complete Deployment Registration"):
        db = get_db_connection()
        if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", reg_user.strip()):
            st.error("Username must be 3-30 alpha-numeric characters or underscores.")
        elif len(reg_pass) < 6:
            st.error("Security passphrase string must length exceed 6 minimum boundaries.")
        elif db.execute("SELECT 1 FROM users WHERE username=?", (reg_user.strip(),)).fetchone():
            st.error("That administrative clinical database identification ID name profile is taken.")
        else:
            db.execute("INSERT INTO users(username,name,pw,created) VALUES(?,?,?,?)",
                       (reg_user.strip(), reg_name.strip(), generate_password_hash(reg_pass), now()))
            db.commit()
            st.success("Registration parameter blocks validated! Proceed to login tab.")
        db.close()

elif choice == "Login":
    st.subheader("Secure Workspace User Sign-In Portal")
    login_user = st.text_input("Username Identification Profile")
    login_pass = st.text_input("Passphrase Key Verification", type="password")
    
    if st.button("Authenticate Workspace Session Key"):
        db = get_db_connection()
        r = db.execute("SELECT * FROM users WHERE username=?", (login_user.strip(),)).fetchone()
        if r and check_password_hash(r["pw"], login_pass):
            st.session_state.user_id = r["id"]
            st.session_state.user_fullname = r["name"] if r["name"] else r["username"]
            st.success(f"Welcome back, operator {st.session_state.user_fullname}!")
            st.rerun()
        else:
            st.error("Invalid secure operational validation credentials matching indices failed.")
        db.close()

# --- Logged-In Core Operations ---
elif choice == "AI Workspace":
    st.subheader(f"👨‍⚕️ Screening Analysis Deck — Operator: {st.session_state.user_fullname}")
    
    patient = st.text_input("Patient Reference Record Tracking Identifier", max_chars=60).strip()
    f = st.file_uploader("Upload Target Diagnostic Structural Brain MRI Slice Sequence Node File", type=["png", "jpg", "jpeg"])
    
    if f is not None:
        try:
            img = Image.open(f)
            img.load()
        except Exception:
            st.error("That target asset input file format cannot handle standard image array maps payload parsing configurations.")
        else:
            if min(img.size) < 64:
                st.error("Target scan array resolution metrics below structural execution safety thresholds.")
            else:
                img.thumbnail((1024, 1024))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original Processing Sequence Source", use_container_width=True)
                
                if st.button("Run Diagnostic Agent Core Pipelines"):
                    with st.spinner("Executing dense layer gradient activation processing layers..."):
                        t0 = time.time()
                        idx, probs, cam = predict(model, img)
                        ms = int((time.time() - t0) * 1000)
                        
                        uid = uuid.uuid4().hex
                        img.convert("RGB").save(os.path.join(UPLOADS, f"{uid}_orig.png"))
                        cam.save(os.path.join(UPLOADS, f"{uid}_cam.png"))
                        
                        conf = float(probs[idx])
                        p_normal = float(sum(p for n, p in zip(CLASSES, probs) if n == "Normal"))
                        
                        db = get_db_connection()
                        db.execute(
                            "INSERT INTO scans(user_id,uid,original_name,patient,label,confidence,p_normal,p_tumor,uncertain,ms,created,probs)"
                            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (st.session_state.user_id, uid, os.path.basename(f.name)[:80], patient,
                             CLASSES[idx], conf, p_normal, 1 - p_normal, int(conf < LOW_CONF), ms, now(),
                             json.dumps({n: float(p) for n, p in zip(CLASSES, probs)})))
                        db.commit()
                        db.close()
                        
                        with col2:
                            st.image(cam, caption="Class Activation Mapping (CAM) Spatial Localization Mask", use_container_width=True)
                        
st.write("---")
st.subheader(f"Results Summary: {CLASSES[idx]}")
st.info(f"Confidence Metric Index Matching: {round(conf*100, 2)}% | latency processing response frame output: {ms} ms")
st.write(f"Clinical Advisory Notice Notes: {INFO.get(CLASSES[idx], '')}")
elif choice == "Dashboard Metrics":
st.subheader("📈 Statistical Overview & Classification Metrics Engine")
uid = st.session_state.user_id
db = get_db_connection()
scans_data = db.execute("SELECT * FROM scans WHERE user_id=?", (uid,)).fetchall()
total = len(scans_data)
if total == 0:
st.info("No recorded diagnostics tracking array sequences instances found in active local runtime tables layout scopes.")
else:
tumor_tuple = db.execute("SELECT COUNT(*) FROM scans WHERE user_id=? AND label!='Normal'", (uid,)).fetchone()
tumor = tumor_tuple[0] if tumor_tuple else 0
normal = total - tumor
c1, c2, c3 = st.columns(3)
c1.metric("Total Automated Evaluations Run", total)
c2.metric("Detected Tumor Pattern Occurrences", tumor)
c3.metric("Normal Clearance Diagnostics Baselines", normal)
st.write("### Data Breakdown Category Profile Arrays Distribution Matrix")
by_type = db.execute("SELECT label, COUNT(*) as c FROM scans WHERE user_id=? GROUP BY label ORDER BY c DESC", (uid,)).fetchall()
for r in by_type:
pct = round((r['c'] / total) * 100, 1)
st.write(f"{r['label']} Profile Matches: {r['c']} records ({pct}%)")
st.progress(int(r['c'] / total * 100))
db.close()
elif choice == "Screening History":
st.subheader("📋 Filterable Historical Screening Logs & Manifest Master Archive")
uid = st.session_state.user_id
db = get_db_connection()
q = st.text_input("Search Engine Index Text Query (Patient Name / Source Document Filename)").strip()
sql = "SELECT * FROM scans WHERE user_id=?"
params = [uid]
if q:
sql += " AND (patient LIKE ? OR original_name LIKE ?)"
params += [f"%{q}%", f"%{q}%"]
sql += " ORDER BY id DESC"
rows = db.execute(sql, params).fetchall()
if not rows:
st.warning("No records matched your search parameters.")
else:
buf = io.StringIO()
w = csv.writer(buf)
w.writerow(["id", "date", "patient", "file", "result", "confidence_%", "inconclusive"])
for r in rows:
w.writerow([r["id"], r["created"], r["patient"], r["original_name"], r["label"], round(r["confidence"]*100,1), r["uncertain"]])
st.download_button(
label="📥 Export Filtered History Context to Master Data CSV Spreadsheet",
data=buf.getvalue(),
file_name="brain_tumor_screening_history.csv",
mime="text/csv"
)
st.write("---")
for r in rows:
d = scan_dict(r)
with st.container():
st.write(f"Patient Record Tracking identification Key ID: {d['patient'] if d['patient'] else 'Unknown Identifier Code'} | Date Timestamp: {d['created']}")
st.write(f"Result Assessment Label Profile Map String matching state: {d['label']} ({d['confidence']}% score index match)")
st.write("---")
db.close()
elif choice == "User Account Options":
st.subheader("⚙️ Account Controls & Database Housekeeping Panel")
uid = st.session_state.user_id
if st.button("Sign Out of Workspace Session Context Registry Profile"):
st.session_state.user_id = None
st.session_state.user_fullname = None
st.rerun()
st.write("---")
st.write("### Data Destruction Protocols")
if st.button("🚨 Wipe and Drop All Local Scan History Tables Instances Databases Permanently", help="This operation drops tracking file vectors from disk spaces securely and cannot be reversed."):
db = get_db_connection()
db.execute("DELETE FROM scans WHERE user_id=?", (uid,))
db.commit()
db.close()
st.success("All analytical instances history tables metadata configurations flushed down context scopes array layers bounds lines securely.")
