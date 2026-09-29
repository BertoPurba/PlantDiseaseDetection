import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

MODEL_PATH = "models/plant_disease_mobilenetv2.keras"
IMG_SIZE = (160, 160)

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
)


# =========================
# Tema (terang / gelap)
# =========================

THEMES = {
    "light": {
        "ink": "#18271D", "muted": "#5C6E61", "line": "#D6E0D3",
        "paper": "#F2F5EC", "card": "#FFFFFF", "accent": "#2E7D4F",
        "track": "#E1E9DD", "dash": "#A9BDAD",
        "ok": "#2E7D4F", "ok_bg": "#E4F2E6",
        "bad": "#C2500F", "bad_bg": "#FCEBDD",
        "hero_a": "#DDEBD5", "hero_b": "#F2F5EC",
    },
    "dark": {
        "ink": "#E9F1EA", "muted": "#94A99A", "line": "#263A2D",
        "paper": "#0C1510", "card": "#13201A", "accent": "#5FD08A",
        "track": "#20332A", "dash": "#38513F",
        "ok": "#5FD08A", "ok_bg": "#15301F",
        "bad": "#FF9B57", "bad_bg": "#3A2413",
        "hero_a": "#16281D", "hero_b": "#0C1510",
    },
}

st.session_state.setdefault("dark_mode", False)
is_dark = st.session_state["dark_mode"]
T = THEMES["dark" if is_dark else "light"]

theme_vars = "".join(f"--{k.replace('_', '-')}: {v};" for k, v in T.items())
st.markdown(
    f"""<style>
:root {{ {theme_vars} color-scheme: {"dark" if is_dark else "light"}; }}
</style>""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, .stApp, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; color: var(--ink); }
.stApp { background: var(--paper); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 820px; padding-top: 1.6rem; padding-bottom: 4rem; }

/* Paksa warna teks mengikuti tema aplikasi */
.stApp p, .stApp span, .stApp label, .stApp li,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"], [data-testid="stSpinner"] { color: var(--ink); }
[data-testid="stAlert"] p, [data-testid="stAlert"] div { color: var(--ink); }
hr { border-color: var(--line) !important; }

/* Hero */
.hero {
    background: linear-gradient(135deg, var(--hero-a), var(--hero-b));
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
}
.hero .eyebrow { font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); }
.hero h1 { font-family: 'Fraunces', serif; font-weight: 700; font-size: 2.3rem; line-height: 1.1; margin: .35rem 0 .5rem; color: var(--ink); }
.hero p { color: var(--muted) !important; margin: 0; max-width: 30rem; }

/* Tabs */
div[data-baseweb="tab-list"] { border-bottom: 1px solid var(--line); background: transparent; }
button[data-baseweb="tab"] { background: transparent !important; font-weight: 600; }
button[data-baseweb="tab"] p, button[data-baseweb="tab"] span { color: var(--muted) !important; font-weight: 600; }
button[data-baseweb="tab"]:hover p { color: var(--ink) !important; }
button[data-baseweb="tab"][aria-selected="true"] p,
button[data-baseweb="tab"][aria-selected="true"] span { color: var(--accent) !important; }
div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
div[data-baseweb="tab-border"] { background-color: var(--line) !important; }

/* Uploader & kamera */
[data-testid="stFileUploaderDropzone"] {
    background: var(--card); border: 2px dashed var(--dash); border-radius: 18px; padding: 1.6rem;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent); }
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] p { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stCameraInput"] button { background: var(--card); border: 1px solid var(--line); }
[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stCameraInput"] button *,
[data-testid="stFileUploaderFile"] * { color: var(--ink) !important; }

/* Gambar */
[data-testid="stImage"] img { border-radius: 18px; border: 1px solid var(--line); }
[data-testid="stImageCaption"] { color: var(--muted); font-size: .85rem; }

/* Kartu verdict */
.verdict {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 1.4rem 1.5rem;
}
.pill {
    display: inline-block; font-size: .75rem; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; padding: .3rem .7rem; border-radius: 999px;
}
.plant { color: var(--muted); font-weight: 500; margin-top: .9rem; }
.condition { font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.75rem; line-height: 1.15; margin: .1rem 0 1.1rem; }
.gauge-wrap { display: flex; align-items: center; gap: 1rem; }
.gauge {
    width: 84px; height: 84px; border-radius: 50%;
    display: grid; place-items: center; flex: none;
}
.gauge-inner {
    width: 64px; height: 64px; border-radius: 50%; background: var(--card);
    display: grid; place-items: center; font-weight: 700; font-size: 1.05rem;
}
.gauge-label { color: var(--muted); font-size: .92rem; line-height: 1.35; }

/* Top 5 */
.section-title { font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.3rem; margin: 1.8rem 0 .8rem; }
.rank-row {
    display: grid; grid-template-columns: 2rem 1fr 4rem; align-items: center; gap: .8rem;
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: .7rem 1rem; margin-bottom: .5rem;
}
.rank-n { font-weight: 700; color: var(--muted); }
.rank-name { font-weight: 600; font-size: .95rem; }
.rank-name small { display: block; color: var(--muted); font-weight: 400; font-size: .8rem; }
.rank-bar { height: 7px; border-radius: 999px; background: var(--track); margin-top: .4rem; overflow: hidden; }
.rank-fill { height: 100%; border-radius: 999px; }
.rank-pct { text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }
.rank-row.first { border-color: var(--accent); }

.note { color: var(--muted); font-size: .86rem; margin-top: 1.6rem; }

@media (max-width: 640px) {
    .hero { padding: 1.3rem; }
    .hero h1 { font-size: 1.8rem; }
    .condition { font-size: 1.45rem; }
}
@media (prefers-reduced-motion: no-preference) { .rank-fill { transition: width .5s ease; } }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Model
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Model tidak bisa dimuat. Pastikan file `{MODEL_PATH}` ada.")
    st.caption(f"Detail error: {e}")
    st.stop()


# =========================
# Fungsi bantu
# =========================

def parse_label(raw):
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight', sakit/sehat)"""
    plant, condition = raw.split("___")
    plant = plant.replace("_", " ").strip()
    condition = condition.replace("_", " ").strip()
    condition = condition[0].upper() + condition[1:]
    return plant, condition, condition.lower() == "healthy"


def predict(image):
    resized = image.resize(IMG_SIZE)
    arr = np.array(resized)
    batch = np.expand_dims(arr, axis=0)
    return model.predict(batch, verbose=0)[0]


def render_verdict(probs):
    idx = int(np.argmax(probs))
    plant, condition, healthy = parse_label(class_names[idx])
    conf = float(probs[idx]) * 100
    color = T["ok"] if healthy else T["bad"]
    bg = T["ok_bg"] if healthy else T["bad_bg"]
    status = "Healthy" if healthy else "Disease detected"

    st.markdown(
        f"""<div class="verdict">
<span class="pill" style="background:{bg};color:{color}">{status}</span>
<div class="plant">{plant}</div>
<div class="condition">{condition}</div>
<div class="gauge-wrap">
<div class="gauge" style="background:conic-gradient({color} {conf:.1f}%, var(--track) 0)">
<div class="gauge-inner">{conf:.0f}%</div>
</div>
<div class="gauge-label">Model confidence<br>for this prediction</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )
    return conf


def render_top5(probs):
    top = np.argsort(probs)[-5:][::-1]
    rows = []
    for rank, i in enumerate(top, start=1):
        plant, condition, healthy = parse_label(class_names[i])
        pct = float(probs[i]) * 100
        color = T["ok"] if healthy else T["bad"]
        first = " first" if rank == 1 else ""
        rows.append(
            f'<div class="rank-row{first}">'
            f'<div class="rank-n">{rank}</div>'
            f'<div><div class="rank-name">{condition}<small>{plant}</small></div>'
            f'<div class="rank-bar"><div class="rank-fill" style="width:{pct:.2f}%;background:{color}"></div></div></div>'
            f'<div class="rank-pct">{pct:.1f}%</div>'
            f"</div>"
        )
    st.markdown(
        '<div class="section-title">Top 5 predictions</div>' + "".join(rows),
        unsafe_allow_html=True,
    )


def analyze(image):
    image = image.convert("RGB")
    probs = predict(image)

    col_img, col_res = st.columns([5, 6], gap="medium")
    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with col_res:
        conf = render_verdict(probs)

    if conf < 60:
        st.warning(
            "Confidence is low. Try a sharper, well-lit photo with a single leaf "
            "filling most of the frame."
        )

    render_top5(probs)


# =========================
# Halaman utama
# =========================

head_l, head_r = st.columns([4, 1])
with head_r:
    st.toggle("🌙 Dark mode", key="dark_mode")

st.markdown(
    """<div class="hero">
<div class="eyebrow">🌿 Leaf health check</div>
<h1>Plant Disease Detection</h1>
<p>Upload a photo of a plant leaf and the model will predict which disease it has, or whether it is healthy.</p>
</div>""",
    unsafe_allow_html=True,
)

tab_upload, tab_camera = st.tabs(["Upload image", "Use camera"])

image_source = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a plant leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file)

with tab_camera:
    camera_file = st.camera_input("Take a photo", label_visibility="collapsed")
    if camera_file is not None:
        image_source = Image.open(camera_file)

if image_source is not None:
    st.divider()
    with st.spinner("Analyzing leaf..."):
        analyze(image_source)
else:
    st.markdown('<div class="note">No image yet. Upload a leaf photo or use your camera to start.</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="note">Predictions come from a model trained on 38 plant and disease classes. '
    "Treat them as a first indication, not a final diagnosis.</div>",
    unsafe_allow_html=True,
)