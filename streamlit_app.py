"""Brain Tumor Detection - Streamlit version of app.py (same pages, accounts and database).
Run locally:       streamlit run streamlit_app.py
Streamlit Cloud:   set 'Main file path' to streamlit_app.py
"""
import os

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

import core as C

st.set_page_config(page_title=C.APP_NAME, page_icon="🧠", layout="wide")
ss = st.session_state


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return C.get_model()


model, model_err = get_model()
user = C.get_user(ss["uid"]) if ss.get("uid") else None
if ss.get("uid") and not user:        # account vanished (e.g. database reset)
    ss.pop("uid")

GUEST = ["Home", "Learn", "About model", "Log in", "Sign up"]
MEMBER = ["Dashboard", "Analyse scan", "History", "Learn", "About model", "Account"]
pages = MEMBER if user else GUEST
if "goto" in ss:                      # programmatic navigation (set before the radio is created)
    ss["page"] = ss.pop("goto")
if ss.get("page") not in pages:
    ss["page"] = pages[0]


def go(p):
    ss["goto"] = p
    st.rerun()


with st.sidebar:
    st.title("🧠 " + C.APP_NAME)
    st.caption(C.APP_TAG)
    page = st.radio("Menu", pages, key="page", label_visibility="collapsed")
    st.divider()
    if user:
        st.write(f"Signed in as **{user['name'] or user['username']}**")
        if st.button("Log out", width="stretch"):
            ss.clear()
            st.rerun()
    st.caption("Educational project - not a medical device. It must not be used for diagnosis or treatment "
               "decisions. Always consult a qualified doctor.")


def show_scan(s, key):
    """Result card: images, verdict, probabilities, report download, delete."""
    c1, c2, c3 = st.columns([1, 1, 1.3])
    for col, path, cap in ((c1, s["orig"], "Uploaded scan"), (c2, s["cam"], "Grad-CAM heat-map")):
        if os.path.exists(path):
            col.image(path, caption=cap, width="stretch")
        else:
            col.info("Image no longer on the server.")
    with c3:
        title = s["headline"] + (f" - {s['kind']}" if s["kind"] else "")
        if s["uncertain"]:
            st.warning(f"**Inconclusive** - closest match: {s['label']} ({s['confidence']}%)")
        elif s["is_tumor"]:
            st.error(f"**{title}** - {s['confidence']}% confidence")
        else:
            st.success(f"**{title}** - {s['confidence']}% confidence")
        for p in s["probs"]:
            st.progress(min(max(p["pct"] / 100, 0.0), 1.0), text=f"{p['name']}: {p['pct']}%")
        st.caption(s["info"])
        st.caption(f"{s['created']} - {s['name']}" + (f" - {s['patient']}" if s["patient"] else "")
                   + f" - {s['seconds']} s")
        b1, b2 = st.columns(2)
        b1.download_button("Download report", C.report_html(s), f"scan_report_{s['id']}.html", "text/html",
                           key=f"rep{key}{s['id']}", width="stretch")
        if b2.button("Delete", key=f"del{key}{s['id']}", width="stretch"):
            C.delete_scan(user["id"], s["id"])
            st.rerun()


# ------------------------------------------------------------------ Home
if page == "Home":
    st.title("Brain tumor detection from MRI scans")
    st.write("Upload an MRI slice and get an AI prediction with a Grad-CAM heat-map that shows where the "
             "ResNet-18 model looked. Create a free account to keep your scan history.")
    m = C.METRICS
    a, b, c, d = st.columns(4)
    a.metric("Model", "ResNet-18")
    b.metric(f"{m['test_name'].title()} accuracy", f"{m['acc']}%")
    c.metric("Training images", m["total"])
    d.metric("Classes", " / ".join(m["names"]))
    x, y = st.columns(2)
    if x.button("Sign up", type="primary", width="stretch"):
        go("Sign up")
    if y.button("Log in", width="stretch"):
        go("Log in")

# ------------------------------------------------------------------ Sign up / Log in
elif page == "Sign up":
    st.header("Create your account")
    with st.form("reg"):
        name = st.text_input("Your name")
        uname = st.text_input("Username", help="3-30 letters, numbers or underscores")
        pw = st.text_input("Password", type="password", help="At least 6 characters")
        ok = st.form_submit_button("Sign up", type="primary")
    if ok:
        uid, err = C.register(name, uname, pw)
        if err:
            st.error(err)
        else:
            ss["uid"] = uid
            go("Analyse scan")

elif page == "Log in":
    st.header("Log in")
    with st.form("login"):
        uname = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Log in", type="primary")
    if ok:
        uid = C.login(uname, pw)
        if uid:
            ss["uid"] = uid
            go("Dashboard")
        st.error("Invalid username or password.")

# ------------------------------------------------------------------ Dashboard
elif page == "Dashboard":
    st.header("Dashboard")
    s = C.stats(user["id"])
    a, b, c, d, e = st.columns(5)
    a.metric("Scans", s["total"])
    b.metric("Tumor", s["tumor"], f"{round(s['tumor'] / s['total'] * 100, 1) if s['total'] else 0}%", delta_color="off")
    c.metric("Normal", s["normal"])
    d.metric("Inconclusive", s["unc"])
    e.metric("Avg confidence", f"{s['avg']}%")
    if not s["total"]:
        st.info("No scans yet. Open **Analyse scan** to run your first one.")
        if st.button("Analyse a scan", type="primary"):
            go("Analyse scan")
    else:
        l, r = st.columns([1.4, 1])
        with l:
            st.subheader("Last 7 days")
            st.bar_chart(pd.DataFrame(s["series"]).set_index("Day"), color=["#d92d20", "#12b76a"])
        with r:
            st.subheader("Results by type")
            for label, n in s["by_type"]:
                st.progress(n / s["total"], text=f"{label}: {n} ({round(n / s['total'] * 100)}%)")
        st.subheader("Recent scans")
        st.dataframe(pd.DataFrame([{"Date": x["created"], "File": x["name"], "Patient": x["patient"],
                                    "Result": x["label"], "Confidence %": x["confidence"],
                                    "Status": "Inconclusive" if x["uncertain"] else "OK"} for x in s["recent"]]),
                     hide_index=True, width="stretch")

# ------------------------------------------------------------------ Analyse
elif page == "Analyse scan":
    st.header("Analyse scan")
    if model is None:
        st.error(model_err)
        st.stop()
    with st.form("analyse", clear_on_submit=True):
        patient = st.text_input("Patient / label (optional)", max_chars=60)
        files = st.file_uploader("MRI image(s) - JPG or PNG", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        go_btn = st.form_submit_button("Analyse", type="primary")
    if go_btn:
        ss["last"] = []
        if not files:
            st.warning("Please choose at least one MRI image.")
        for f in files:
            if f.size > 8 * 1024 * 1024:
                st.error(f"{f.name}: file too large (max 8 MB).")
                continue
            try:
                img = Image.open(f)
                img.load()
                with st.spinner(f"Analysing {f.name}..."):
                    ss["last"].append(C.save_scan(user["id"], f.name, img, patient, model))
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as e:
                st.error(f"{f.name}: " + (str(e) if isinstance(e, ValueError) else "not a valid image (use JPG or PNG)."))
    for s in ss.get("last", []):
        st.subheader(s["name"])
        show_scan(s, "a")
        st.divider()

# ------------------------------------------------------------------ History
elif page == "History":
    st.header("History")
    f1, f2, f3 = st.columns([2, 1.2, 0.8])
    q = f1.text_input("Search patient or file name", max_chars=60)
    _, _, _, _, labels = C.history(user["id"])
    opts = {"All results": "", "Any tumor": "tumor", **{lb: lb for lb in labels}}
    label = opts[f2.selectbox("Result", list(opts))]
    _, total, pages_n, _, _ = C.history(user["id"], q.strip(), label, 1)
    pg = f3.number_input("Page", 1, pages_n, 1)
    rows, total, pages_n, pg, _ = C.history(user["id"], q.strip(), label, int(pg))
    st.caption(f"{total} scan(s) - page {pg} of {pages_n}")
    if not rows:
        st.info("No scans match.")
    for s in rows:
        icon = "🟠" if s["uncertain"] else ("🔴" if s["is_tumor"] else "🟢")
        with st.expander(f"{icon} {s['label']} - {s['confidence']}% - {s['name']} - {s['created']}"):
            show_scan(s, "h")
    st.download_button("Download all as CSV", C.csv_text(user["id"]), "brain_tumor_history.csv", "text/csv")

# ------------------------------------------------------------------ Learn
elif page == "Learn":
    st.header("Learn")
    st.subheader("Brain tumor types")
    for k, v in C.INFO.items():
        if k not in ("Normal", "Tumor"):
            st.markdown(f"**{k}** - {v}")
    st.markdown("**Tumor / Normal** - This project's binary model only says whether tumor-like patterns are present; "
                "it does not name the tumor type.")
    st.subheader("How the AI works")
    st.write("A ResNet-18 convolutional neural network, pre-trained on ImageNet, was fine-tuned on labelled MRI "
             "images (transfer learning). For a new scan it outputs a probability for each class.")
    st.subheader("Reading the heat-map")
    st.write("Grad-CAM colours the regions that influenced the decision: red/yellow = strong influence, "
             "blue = little. It shows where the model looked, not a medical outline of a tumor.")
    st.subheader("Limits")
    st.write("Small training data, single slices only, no clinical information. Results below "
             f"{int(C.LOW_CONF * 100)}% confidence are marked inconclusive. Always see a doctor.")

# ------------------------------------------------------------------ About
elif page == "About model":
    st.header("About the model")
    m = C.METRICS
    a, b, c, d = st.columns(4)
    a.metric("Architecture", "ResNet-18")
    b.metric(f"{m['test_name'].title()} accuracy", f"{m['acc']}%")
    c.metric("Images (total)", m["total"])
    d.metric("Epochs", m["epochs"])
    st.write("Pre-trained on ImageNet, final layer replaced, fine-tuned with Adam (lr 1e-4) on 224x224 slices with "
             "flip/rotation augmentation. Grad-CAM from the last convolutional block explains each prediction.")
    st.subheader("Per-class results")
    st.dataframe(pd.DataFrame([{"Class": n, "Precision %": round(p * 100, 1), "Recall %": round(r * 100, 1),
                                "F1 %": round(f * 100, 1), "Images": k} for n, p, r, f, k in m["rows"]]),
                 hide_index=True, width="stretch")
    st.subheader("Confusion matrix (rows = actual, columns = predicted)")
    st.dataframe(pd.DataFrame(m["cm"], index=[f"Actual {x}" for x in m["names"]],
                              columns=[f"Predicted {x}" for x in m["names"]]), width="stretch")
    st.warning("Small dataset (253 images) with some duplicates between training and validation, so the accuracy "
               "may be optimistic. Educational use only.")

# ------------------------------------------------------------------ Account
elif page == "Account":
    st.header("Account")
    st.write(f"Username: **{user['username']}** - member since {C.fmt(user['created'])}")
    with st.form("pw"):
        st.subheader("Change password")
        old = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password")
        if st.form_submit_button("Update password"):
            err = C.change_password(user["id"], old, new)
            st.error(err) if err else st.success("Password updated.")
    st.subheader("Your data")
    st.download_button("Download history (CSV)", C.csv_text(user["id"]), "brain_tumor_history.csv", "text/csv")
    sure = st.checkbox("I want to delete all my scans")
    if st.button("Delete all my scans", disabled=not sure):
        C.clear_scans(user["id"])
        ss.pop("last", None)
        st.success("All your scans were deleted.")
