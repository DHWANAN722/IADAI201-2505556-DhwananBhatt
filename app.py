"""
PARKVISION//AI  —  Intelligent Urban Parking Analytics & Space Optimisation Platform
Streamlit dashboard  |  Dhwanan Bhatt (2505556)  |  CRS Artificial Intelligence
"""
import io, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from parkvision.engine import ParkVision, congestion_level

ROOT = Path(__file__).parent
SAMPLES = ROOT / "samples"
REPORTS = ROOT / "reports"

st.set_page_config(page_title="ParkVision AI", page_icon="🅿️", layout="wide", initial_sidebar_state="auto")

# ============================================================== CYBERPUNK THEME
C = {"bg": "#07060f", "panel": "#0f0d1f", "cyan": "#00f0ff", "pink": "#ff2bd6", "yellow": "#f5ff3b",
     "green": "#39ff14", "red": "#ff3860", "violet": "#8a5cff", "text": "#e8e6ff", "muted": "#8d88b8"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');
:root {{ --cyan:{C['cyan']}; --pink:{C['pink']}; --yellow:{C['yellow']}; --green:{C['green']}; --red:{C['red']}; }}
html, body, [data-testid="stAppViewContainer"], .stApp {{
  background: {C['bg']} !important; color:{C['text']}; font-family:'Rajdhani',sans-serif;
}}
.stApp {{
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(255,43,214,.18), transparent 60%),
    radial-gradient(900px 500px at 110% 10%, rgba(0,240,255,.16), transparent 60%),
    linear-gradient(rgba(0,240,255,.05) 1px, transparent 1px) 0 0/100% 38px,
    linear-gradient(90deg, rgba(0,240,255,.05) 1px, transparent 1px) 0 0/38px 100%,
    {C['bg']} !important;
}}
.stApp::after {{ content:""; position:fixed; inset:0; pointer-events:none; z-index:999;
  background: repeating-linear-gradient(0deg, rgba(0,0,0,.10) 0 1px, transparent 1px 3px); mix-blend-mode:multiply; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg,#0c0a1c,#07060f) !important; border-right:1px solid rgba(255,43,214,.35); }}
[data-testid="stSidebar"] * {{ color:{C['text']}; }}
h1,h2,h3,h4 {{ font-family:'Orbitron',sans-serif !important; letter-spacing:.06em; }}
p, li, label, span, div {{ font-size:1.02rem; }}
.hero {{ position:relative; padding:26px 30px; border:1px solid rgba(0,240,255,.45); border-radius:4px;
  background: linear-gradient(120deg, rgba(255,43,214,.10), rgba(0,240,255,.07) 55%, rgba(245,255,59,.05));
  clip-path: polygon(0 0, calc(100% - 28px) 0, 100% 28px, 100% 100%, 28px 100%, 0 calc(100% - 28px));
  box-shadow: 0 0 40px rgba(0,240,255,.12) inset; margin-bottom:18px; }}
.hero .h1 {{ font-family:'Orbitron',sans-serif; margin:4px 0 0; letter-spacing:.04em; font-size:clamp(1.9rem,4.2vw,3.3rem); font-weight:900; line-height:1.05;
  background: linear-gradient(90deg, var(--cyan), var(--pink) 60%, var(--yellow)); -webkit-background-clip:text; background-clip:text; color:transparent;
  text-shadow: 0 0 22px rgba(0,240,255,.25); }}
.hero .tag {{ font-family:'Share Tech Mono',monospace; color:var(--yellow); letter-spacing:.25em; font-size:.8rem; }}
.hero .sub {{ color:{C['muted']}; font-size:1.1rem; margin-top:6px; }}
.glitch {{ animation: flicker 4s infinite; }}
@keyframes flicker {{ 0%,19%,21%,23%,100% {{opacity:1}} 20%,22% {{opacity:.55}} }}
.kpi {{ position:relative; padding:16px 18px 14px; background:{C['panel']}; border:1px solid var(--c);
  clip-path: polygon(0 0, calc(100% - 16px) 0, 100% 16px, 100% 100%, 0 100%);
  box-shadow: 0 0 18px -4px var(--c), inset 0 0 22px -12px var(--c); }}
.kpi .lbl {{ font-family:'Share Tech Mono',monospace; font-size:.78rem; letter-spacing:.2em; color:{C['muted']}; }}
.kpi .val {{ font-family:'Orbitron',sans-serif; font-weight:900; font-size:2.35rem; color:var(--c); text-shadow:0 0 14px var(--c); line-height:1.1; }}
.kpi .foot {{ font-size:.9rem; color:{C['muted']}; }}
.banner {{ padding:18px 22px; border-left:6px solid var(--c); background: linear-gradient(90deg, color-mix(in srgb, var(--c) 18%, transparent), transparent 80%);
  border-radius:2px; margin:6px 0 14px; }}
.banner .t {{ font-family:'Orbitron',sans-serif; font-weight:900; font-size:1.45rem; color:var(--c); text-shadow:0 0 12px var(--c); letter-spacing:.05em; }}
.banner .d {{ font-size:1.12rem; color:{C['text']}; margin-top:4px; }}
.chip {{ display:inline-block; font-family:'Share Tech Mono',monospace; font-size:.8rem; padding:3px 10px; margin:2px 4px 2px 0; border:1px solid var(--c); color:var(--c); border-radius:2px; }}
.section {{ font-family:'Orbitron',sans-serif; font-size:1.02rem; letter-spacing:.2em; color:var(--cyan); margin:22px 0 8px; border-bottom:1px solid rgba(0,240,255,.25); padding-bottom:6px; }}
.section::before {{ content:"// "; color:var(--pink); }}
.frame {{ border:1px solid rgba(255,43,214,.45); padding:6px; background:#050409; box-shadow:0 0 22px -6px rgba(255,43,214,.6); }}
.caption {{ font-family:'Share Tech Mono',monospace; font-size:.8rem; color:{C['muted']}; letter-spacing:.15em; margin-top:4px; }}
.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid rgba(0,240,255,.3); }}
.stTabs [data-baseweb="tab"] {{ height:auto; min-width:max-content; padding:8px 18px !important; background:rgba(15,13,31,.8); border:1px solid rgba(0,240,255,.25); border-bottom:none; padding:8px 18px; color:{C['muted']}; }}
.stTabs [data-baseweb="tab"] p {{ font-family:'Orbitron',sans-serif !important; letter-spacing:.12em; font-size:.82rem !important; white-space:nowrap; }}
.stTabs [data-baseweb="tab-highlight"] {{ display:none; }}
.stTabs [aria-selected="true"] p {{ color:{C['bg']} !important; font-weight:700; }}
.block-container {{ padding-top:2.2rem !important; max-width:1400px; }}
.side-title {{ font-family:'Orbitron',sans-serif; font-weight:900; color:{C['pink']}; letter-spacing:.12em; font-size:1.15rem; text-shadow:0 0 10px rgba(255,43,214,.6); }}
.stTabs [aria-selected="true"] {{ color:{C['bg']} !important; background:linear-gradient(90deg,var(--cyan),var(--pink)) !important; }}
.stButton>button, .stDownloadButton>button {{ font-family:'Orbitron',sans-serif; letter-spacing:.12em; background:transparent; color:var(--cyan);
  border:1px solid var(--cyan); border-radius:2px; box-shadow:0 0 12px -2px var(--cyan); transition:.15s; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ background:var(--cyan); color:{C['bg']}; }}
[data-testid="stFileUploaderDropzone"] {{ background:rgba(0,240,255,.05); border:1px dashed var(--cyan); border-radius:2px; }}
[data-testid="stDataFrame"] {{ border:1px solid rgba(0,240,255,.25); }}
.footer {{ text-align:center; font-family:'Share Tech Mono',monospace; color:{C['muted']}; font-size:.8rem; letter-spacing:.2em; margin-top:40px; }}
</style>
""", unsafe_allow_html=True)


# ============================================================== HELPERS
@st.cache_resource(show_spinner="Booting neural cores…")
def load_engine():
    return ParkVision()


def kpi(label, value, color, foot=""):
    return f'<div class="kpi" style="--c:{color}"><div class="lbl">{label}</div><div class="val">{value}</div><div class="foot">{foot}</div></div>'


def section(t):
    st.markdown(f'<div class="section">{t}</div>', unsafe_allow_html=True)


def plotly_dark(fig, h=300):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Rajdhani, Arial, sans-serif", color=C["text"], size=14))
    return fig


LEVEL_COLOR = {"LOW": C["green"], "MODERATE": C["yellow"], "HIGH": C["red"]}


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ============================================================== SIDEBAR
with st.sidebar:
    st.markdown("<div class='side-title'>⚙ CONTROL DECK</div>", unsafe_allow_html=True)
    st.caption("Tune the vision pipeline")
    mode_label = st.radio("Inference pipeline", ["Hybrid · YOLO + MobileNet", "YOLOv8 detector only", "MobileNet verdict only"],
                          help="Hybrid: YOLOv8 finds every slot, MobileNetV2 re-checks each crop and the two probabilities are fused.")
    mode = {"Hybrid · YOLO + MobileNet": "hybrid", "YOLOv8 detector only": "detector", "MobileNet verdict only": "classifier"}[mode_label]
    conf = st.slider("Slot detection confidence", 0.05, 0.80, 0.25, 0.05, help="Lower = find more slots (more false positives).")
    w_cls = st.slider("MobileNet weight in fusion", 0.0, 1.0, 0.5, 0.1, disabled=(mode != "hybrid"))
    show_ids = st.toggle("Show slot numbers", value=True)
    st.markdown("---")
    st.markdown(f"<span class='chip' style='--c:{C['green']}'>■ EMPTY</span><span class='chip' style='--c:{C['red']}'>■ OCCUPIED</span>", unsafe_allow_html=True)
    st.markdown(f"""<div style='font-size:.92rem;color:{C['muted']};margin-top:10px'>
    <b style='color:{C['cyan']}'>Congestion scale</b><br>
    <span style='color:{C['green']}'>LOW</span> &lt; 40% &nbsp;·&nbsp; <span style='color:{C['yellow']}'>MODERATE</span> 40–75% &nbsp;·&nbsp; <span style='color:{C['red']}'>HIGH</span> &gt; 75%</div>""",
                unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built by **Dhwanan Bhatt** · 2505556\n\nUdgam School for Children · CRS AI")

# ============================================================== HERO
st.markdown("""
<div class="hero">
  <div class="tag glitch">URBANFLOW AI ▸ PARKSMART INITIATIVE ▸ v1.0</div>
  <div class="h1">PARKVISION//AI</div>
  <div class="sub">Slot-level parking occupancy detection, live availability metrics and smart routing advice — from a single camera frame.</div>
</div>""", unsafe_allow_html=True)

engine = load_engine()
tab_scan, tab_slot, tab_lab, tab_about = st.tabs(["◉ LOT SCAN", "◧ SLOT CHECK", "⌬ MODEL LAB", "ⓘ ABOUT"])

# ============================================================== TAB 1 — FULL LOT SCAN
with tab_scan:
    c_up, c_pick = st.columns([3, 2])
    with c_up:
        up = st.file_uploader("Upload a parking-lot image (elevated CCTV view works best)", type=["jpg", "jpeg", "png", "webp"])
    samples = sorted(SAMPLES.glob("*.jpg"))
    with c_pick:
        names = ["—"] + [p.stem.replace("_", " ") for p in samples]
        pick = st.selectbox("…or load a test frame (unseen during training)", names, index=min(6, len(samples)) if samples else 0)

    img_bgr, src_name = None, None
    if up is not None:
        img_bgr = cv2.cvtColor(np.array(Image.open(up).convert("RGB")), cv2.COLOR_RGB2BGR)
        src_name = up.name
    elif pick != "—":
        p = samples[names.index(pick) - 1]
        img_bgr, src_name = cv2.imread(str(p)), p.name

    if img_bgr is None:
        st.info("Upload an image or choose a sample frame to start scanning.")
    else:
        t0 = time.time()
        with st.spinner("Scanning slots…"):
            r = engine.scan(img_bgr, conf=conf, mode=mode, w_cls=w_cls, show_ids=show_ids)
        dt = time.time() - t0
        lvl_c = LEVEL_COLOR[r.congestion]

        # ---- KPIs
        k = st.columns(4)
        k[0].markdown(kpi("TOTAL SLOTS", r.total, C["cyan"], "detected in frame"), unsafe_allow_html=True)
        k[1].markdown(kpi("OCCUPIED", r.occupied, C["red"], "vehicles present"), unsafe_allow_html=True)
        k[2].markdown(kpi("AVAILABLE", r.available, C["green"], "free right now"), unsafe_allow_html=True)
        k[3].markdown(kpi("OCCUPANCY", f"{r.occupancy:.0f}%", lvl_c, f"congestion: {r.congestion}"), unsafe_allow_html=True)

        # ---- Recommendation
        rc = C["red"] if "TRY ANOTHER" in r.recommendation or "NO SLOTS" in r.recommendation else C["green"]
        st.markdown(f'<div class="banner" style="--c:{rc}"><div class="t">▶ {r.recommendation}</div><div class="d">{r.advice}</div></div>',
                    unsafe_allow_html=True)

        # ---- Images
        section("VISUAL SMART OVERLAY")
        a, b = st.columns(2)
        with a:
            st.image(to_rgb(img_bgr), use_container_width=True)
            st.markdown(f'<div class="caption">INPUT ▸ {src_name}</div>', unsafe_allow_html=True)
        with b:
            st.image(to_rgb(r.annotated), use_container_width=True)
            st.markdown(f'<div class="caption">OUTPUT ▸ {r.total} slots · {mode.upper()} · {dt*1000:.0f} ms</div>', unsafe_allow_html=True)

        # ---- Analytics
        section("UTILISATION ANALYTICS")
        g1, g2, g3 = st.columns([1.1, 1, 1.3])
        with g1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=r.occupancy, number={"suffix": "%", "font": {"family": "Orbitron", "color": lvl_c, "size": 44}},
                title={"text": f"CONGESTION · {r.congestion}", "font": {"family": "Orbitron", "size": 14, "color": lvl_c}},
                gauge={"axis": {"range": [0, 100], "tickcolor": C["muted"]}, "bar": {"color": lvl_c, "thickness": .28},
                       "bgcolor": "rgba(0,0,0,0)", "borderwidth": 1, "bordercolor": C["violet"],
                       "steps": [{"range": [0, 40], "color": "rgba(57,255,20,.13)"}, {"range": [40, 75], "color": "rgba(245,255,59,.13)"},
                                 {"range": [75, 100], "color": "rgba(255,56,96,.16)"}]}))
            st.plotly_chart(plotly_dark(fig, 270), use_container_width=True, config={"displayModeBar": False})
        with g2:
            fig = go.Figure(go.Pie(labels=["Available", "Occupied"], values=[r.available, r.occupied], hole=.68,
                                   marker=dict(colors=[C["green"], C["red"]], line=dict(color=C["bg"], width=3)),
                                   textinfo="value", textfont=dict(family="Orbitron, Rajdhani, Arial, sans-serif", size=15, color=C["bg"]), sort=False))
            fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
                              annotations=[dict(text=f"<b>{r.available}</b><br>FREE", showarrow=False, font=dict(family="Orbitron, Rajdhani, Arial, sans-serif", size=20, color=C["green"]))])
            st.plotly_chart(plotly_dark(fig, 270), use_container_width=True, config={"displayModeBar": False})
        with g3:
            zn = list(r.zones)
            fig = go.Figure()
            fig.add_bar(y=zn, x=[r.zones[z]["free"] for z in zn], name="Free", orientation="h", marker_color=C["green"])
            fig.add_bar(y=zn, x=[r.zones[z]["total"] - r.zones[z]["free"] for z in zn], name="Occupied", orientation="h", marker_color=C["red"])
            fig.update_layout(barmode="stack", title=dict(text="ZONE BREAKDOWN (left → right)", font=dict(family="Orbitron, Rajdhani, Arial, sans-serif", size=13, color=C["cyan"])),
                              legend=dict(orientation="h", y=-0.15), xaxis=dict(gridcolor="rgba(141,136,184,.15)"), yaxis=dict(autorange="reversed"))
            st.plotly_chart(plotly_dark(fig, 270), use_container_width=True, config={"displayModeBar": False})

        if r.best_zone:
            chips = "".join(f"<span class='chip' style='--c:{C['green'] if z == r.best_zone else C['muted']}'>{z}: {v['free']} free / {v['total']}</span>"
                            for z, v in r.zones.items())
            st.markdown(f"🧭 <b style='color:{C['yellow']}'>Best zone to head for:</b> <b style='color:{C['green']}'>{r.best_zone}</b> &nbsp; {chips}", unsafe_allow_html=True)

        # ---- Slot table
        section("SLOT-WISE STATUS LOG")
        df = pd.DataFrame([{
            "Slot": s.id, "Status": "OCCUPIED" if s.occupied else "EMPTY", "Zone": s.zone,
            "P(occupied) fused": round(s.p_occ, 3), "YOLO P(occ)": round(s.p_occ_det, 3),
            "MobileNet P(occ)": None if s.p_occ_cls is None else round(s.p_occ_cls, 3),
            "Box (x1,y1,x2,y2)": str(s.box)} for s in r.slots])
        if len(df):
            only = st.segmented_control("Filter", ["ALL", "EMPTY", "OCCUPIED"], default="ALL")
            view = df if only in (None, "ALL") else df[df.Status == only]
            st.dataframe(view, use_container_width=True, hide_index=True, height=280,
                         column_config={"P(occupied) fused": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
            d1, d2, _ = st.columns([1, 1, 2])
            d1.download_button("⬇ SLOT REPORT (CSV)", df.to_csv(index=False).encode(), "parkvision_slots.csv", "text/csv")
            buf = io.BytesIO(); Image.fromarray(to_rgb(r.annotated)).save(buf, format="PNG")
            d2.download_button("⬇ ANNOTATED IMAGE", buf.getvalue(), "parkvision_annotated.png", "image/png")

# ============================================================== TAB 2 — SINGLE SLOT CHECK
with tab_slot:
    st.markdown("Upload **cropped images of individual parking spaces** — MobileNetV2 classifies each one as empty or occupied.")
    ups = st.file_uploader("Slot crops", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="slots")
    crops_dir = ROOT / "samples" / "slots"
    use_demo = st.button("⚡ Try demo slot crops") if crops_dir.exists() else False
    items = []
    if ups:
        items = [(u.name, cv2.cvtColor(np.array(Image.open(u).convert("RGB")), cv2.COLOR_RGB2BGR)) for u in ups]
    elif use_demo:
        items = [(p.name, cv2.imread(str(p))) for p in sorted(crops_dir.glob("*.jpg"))]
    if items:
        probs = engine.classify_crops([im for _, im in items])
        cols = st.columns(6)
        for i, ((name, im), p) in enumerate(zip(items, probs)):
            occ = p >= .5
            col = C["red"] if occ else C["green"]
            with cols[i % 6]:
                st.image(to_rgb(cv2.resize(im, (224, 224))), use_container_width=True)
                st.markdown(f"<div style='text-align:center;font-family:Orbitron;color:{col};text-shadow:0 0 8px {col}'>"
                            f"{'OCCUPIED' if occ else 'EMPTY'}</div><div class='caption' style='text-align:center'>conf {max(p, 1-p):.0%}</div>",
                            unsafe_allow_html=True)
        n_occ = int((probs >= .5).sum())
        occ_pct = 100 * n_occ / len(probs)
        st.markdown(f'<div class="banner" style="--c:{LEVEL_COLOR[congestion_level(occ_pct)]}"><div class="t">{len(probs)} SLOTS · {n_occ} OCCUPIED · {len(probs)-n_occ} FREE</div>'
                    f'<div class="d">Occupancy {occ_pct:.0f}% → congestion {congestion_level(occ_pct)}</div></div>', unsafe_allow_html=True)

# ============================================================== TAB 3 — MODEL LAB
with tab_lab:
    cm_path = REPORTS / "classifier_metrics.json"
    ym_path = REPORTS / "yolo_metrics.json"
    ev_path = REPORTS / "pipeline_eval.json"
    if cm_path.exists():
        m = json.load(open(cm_path))
        y = json.load(open(ym_path)) if ym_path.exists() else {}
        ev = json.load(open(ev_path)) if ev_path.exists() else {}
        k = st.columns(4)
        k[0].markdown(kpi("MOBILENET TEST ACC", f"{m['test_acc']*100:.1f}%", C["cyan"], f"{sum(map(sum, m['confusion_matrix']))} unseen crops"), unsafe_allow_html=True)
        k[1].markdown(kpi("YOLO mAP@50", f"{y.get('mAP50', 0)*100:.1f}%", C["pink"], "slot detection, test frames"), unsafe_allow_html=True)
        k[2].markdown(kpi("HYBRID SLOT ACC", f"{ev.get('hybrid', {}).get('accuracy', 0)*100:.1f}%", C["green"], "end-to-end, test frames"), unsafe_allow_html=True)
        k[3].markdown(kpi("COUNT ERROR", f"±{ev.get('hybrid', {}).get('mae_available', 0):.1f}", C["yellow"], "avg free-slot error / frame"), unsafe_allow_html=True)

        section("MOBILENETV2 · TRAINING")
        a, b = st.columns([1.6, 1])
        h = m["history"]; e = list(range(1, len(h["train_acc"]) + 1))
        fig = go.Figure()
        fig.add_scatter(x=e, y=h["train_acc"], name="train acc", line=dict(color=C["cyan"], width=3))
        fig.add_scatter(x=e, y=h["val_acc"], name="val acc", line=dict(color=C["pink"], width=3))
        fig.add_scatter(x=e, y=h["train_loss"], name="train loss", line=dict(color=C["cyan"], dash="dot"), yaxis="y2")
        fig.add_scatter(x=e, y=h["val_loss"], name="val loss", line=dict(color=C["pink"], dash="dot"), yaxis="y2")
        fig.update_layout(xaxis=dict(title="epoch", gridcolor="rgba(141,136,184,.12)"), yaxis=dict(title="accuracy", gridcolor="rgba(141,136,184,.12)"),
                          yaxis2=dict(title="loss", overlaying="y", side="right", showgrid=False), legend=dict(orientation="h", y=1.12))
        a.plotly_chart(plotly_dark(fig, 340), use_container_width=True, config={"displayModeBar": False})
        cmx = np.array(m["confusion_matrix"])
        fig = go.Figure(go.Heatmap(z=cmx, x=["pred EMPTY", "pred OCCUPIED"], y=["EMPTY", "OCCUPIED"], text=cmx, texttemplate="%{text}",
                                   textfont=dict(family="Orbitron, Rajdhani, Arial, sans-serif", size=22), colorscale=[[0, "#120f26"], [1, C["violet"]]], showscale=False))
        fig.update_layout(title=dict(text="CONFUSION MATRIX (test)", font=dict(family="Orbitron, Rajdhani, Arial, sans-serif", size=13, color=C["cyan"])), yaxis=dict(autorange="reversed"))
        b.plotly_chart(plotly_dark(fig, 340), use_container_width=True, config={"displayModeBar": False})

        if ev:
            section("PIPELINE COMPARISON · UNSEEN TEST FRAMES")
            rows = [{"Pipeline": k_.upper(), "Slot accuracy": f"{v['accuracy']*100:.1f}%", "Slots matched": v["matched"],
                     "Recall (slots found)": f"{v['recall']*100:.1f}%", "MAE free-slot count": round(v["mae_available"], 2)}
                    for k_, v in ev.items() if isinstance(v, dict) and "accuracy" in v]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        for img, cap in [("augmentation_examples.png", "Augmentation used in training: original · flip · rotate ±15° · brighter · darker"),
                         ("sample_crops.png", "Training crops — top: empty, bottom: occupied")]:
            if (REPORTS / img).exists():
                st.image(str(REPORTS / img), caption=cap, width=760)
    else:
        st.warning("Metrics files not found.")

# ============================================================== TAB 4 — ABOUT
with tab_about:
    st.markdown(f"""
### The problem
Drivers in dense cities can spend 10+ minutes circling for a space — adding congestion, fuel burn and frustration.
**ParkVision AI** turns an existing CCTV frame into *slot-level* availability plus a clear go / re-route decision.

### How it works
1. **Input** → a parking-lot image. **Output** → every slot tagged *Occupied* / *Empty*, with totals.
2. **YOLOv8n** (fine-tuned on PKLot) locates every slot in the full frame — works on irregular layouts, no manual slot map needed.
3. **MobileNetV2** (ImageNet transfer learning, 224×224 crops) re-checks each slot; the two probabilities are fused.
4. **Insight layer** → occupancy %, congestion level (<40% low · 40–75% moderate · >75% high), per-zone availability and a recommendation.

### Data
PKLot (Almeida et al., 2015) — UFPR & PUCPR lots, sunny / cloudy / rainy days. A balanced, frame-level 70/15/15 split was used.

### Known limits
Trained on elevated fixed-camera views; street-level photos, night scenes and heavy occlusion reduce accuracy.
""")

st.markdown('<div class="footer">PARKVISION//AI · DHWANAN BHATT · 2505556 · UDGAM SCHOOL FOR CHILDREN · CRS ARTIFICIAL INTELLIGENCE</div>', unsafe_allow_html=True)
