# app.py
import os, io, math, json
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# ===== Config =====
MODEL_PATHS = [
    "cnn_neumonia_3clases.h5"
]
TARGET_SIZE = (150, 150)
MODEL_CHANNELS = 3

# orden alfabetico tipico de flow_from_directory:
# NORMAL, NOT_CXR, PNEUMONIA
CLASS_NAMES_3 = ["NORMAL", "NOT_CXR", "PNEUMONIA"]

# sensibilidad para gate solo cuando el modelo es binario
THRESH_INVALID_BASE = 0.80

st.set_page_config(page_title="Clasificador de Neumonia por Rayos X", page_icon="🫁")

# ===== Estilos =====
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.prob-card{
  padding: 1rem 1.1rem; border: 1px solid rgba(255,255,255,.10);
  border-radius: 14px; background: rgba(255,255,255,.03);
}
.mini{opacity:.75; font-size:.85rem; margin-bottom:.35rem}
.big{font-size:1.35rem}
hr{border-color: rgba(255,255,255,.08)}
</style>
""", unsafe_allow_html=True)

# ===== Utilidades =====
@st.cache_resource
def load_model():
    last_err = None
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                m = tf.keras.models.load_model(p)
                return m, p
            except Exception as e:
                last_err = e
    if last_err:
        raise RuntimeError(f"No pude cargar el modelo. Ultimo error: {last_err}")
    raise FileNotFoundError(f"No encontre archivos: {MODEL_PATHS}")

def to_array(img, channels=MODEL_CHANNELS):
    if channels == 1:
        img = ImageOps.grayscale(img)
    else:
        img = img.convert("RGB")
    img = img.resize(TARGET_SIZE, Image.BILINEAR)
    arr = np.array(img).astype("float32")/255.0
    if channels == 1:
        arr = np.expand_dims(arr, -1)
    return np.expand_dims(arr, 0)

# heuristicas solo usadas si el modelo es binario
def colorfulness_score(img_rgb):
    rgb = np.array(img_rgb.convert("RGB")).astype(np.float32)
    R,G,B = rgb[...,0], rgb[...,1], rgb[...,2]
    rg = np.abs(R-G); yb = np.abs(0.5*(R+G)-B)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return math.sqrt(std_rg**2+std_yb**2) + 0.3*math.sqrt(mean_rg**2+mean_yb**2)

def grayscale_likeness(img):
    rgb = np.array(img.convert("RGB")).astype(np.float32)/255.0
    std_channels = np.std(rgb, axis=-1)
    return float(1.0 - np.clip(np.mean(std_channels)*4.0, 0.0, 1.0))

def aspect_ratio_score(img):
    w,h = img.size; r = w/h if h>0 else 1.0
    d = 0.0
    if r < 0.7: d = (0.7-r)/0.7
    elif r > 1.6: d = (r-1.6)/1.6
    return float(np.clip(1.0-d, 0.0, 1.0))

def validity_score(img):
    cf = colorfulness_score(img)            # mayor = mas color
    grayscale = grayscale_likeness(img)     # mayor = mas gris
    ar = aspect_ratio_score(img)            # 1 si relacion tipica
    cf_norm = np.clip((cf - 8) / 17.0, 0.0, 1.0)
    valid = (1.0 - cf_norm)*0.55 + grayscale*0.35 + ar*0.10
    return float(np.clip(valid, 0.0, 1.0))

def three_class_probs_with_gate(p_neu_bin: float, p_valid: float, invalid_bias=THRESH_INVALID_BASE):
    HARD_GATE = 0.55
    SOFT_GATE = 0.70
    if p_valid < HARD_GATE:
        return 0.0, 0.0, 1.0
    if p_valid < SOFT_GATE:
        p_invalid = np.clip(0.75 + (SOFT_GATE - p_valid)*0.5, 0.0, 1.0)
    else:
        p_invalid = np.clip((1.0 - p_valid)*0.6 + (invalid_bias - 0.5)*0.2, 0.0, 1.0)
    remaining = max(1.0 - p_invalid, 0.0)
    p_neu = remaining * float(p_neu_bin)
    p_norm = remaining * (1.0 - float(p_neu_bin))
    s = p_neu + p_norm + p_invalid
    if s <= 0: return 0.0, 0.0, 1.0
    return p_neu/s, p_norm/s, p_invalid/s

def softmax_probs(pred):
    pred = np.array(pred)
    if pred.ndim == 2:
        pred = pred[0]
    # asegurar softmax numericamente estable
    e = np.exp(pred - np.max(pred))
    return e / np.sum(e)

# ===== UI =====
st.title("Clasificador de Neumonia en Rayos X")
st.caption("Sube una imagen. La app devuelve probabilidades para: Neumonia, Normal e Imagen no valida.")

with st.expander("Ajustes del modelo", expanded=False):
    ts = st.text_input("Tamano de entrada (ancho,alto)", f"{TARGET_SIZE[0]},{TARGET_SIZE[1]}")
    ch = st.selectbox("Canales esperados por el modelo", [1,3], index=1 if MODEL_CHANNELS==3 else 0)
    inv_base = st.slider("Sensibilidad 'Imagen no valida' (solo modelos binarios)", 0.0, 1.0, THRESH_INVALID_BASE, 0.01)
    try:
        w,h = [int(x.strip()) for x in ts.split(",")]
        TARGET_SIZE = (w,h); MODEL_CHANNELS = int(ch); THRESH_INVALID_BASE = float(inv_base)
    except:
        st.warning("Usa el formato ancho,alto")

uploaded = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded:
    raw = uploaded.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    col_img, col_pred = st.columns([1, 1.2])

    with col_img:
        st.image(img, caption="Vista previa", use_column_width=False, width=320)

    with col_pred:
        with st.spinner("Procesando..."):
            model, model_path = load_model()
            x = to_array(img, channels=MODEL_CHANNELS)
            pred = model.predict(x, verbose=0)

            # deteccion de tipo de modelo por dimension de salida
            out_dim = pred.shape[-1]
            use_three_head = (out_dim == 3)

            if use_three_head:
                # modelo 3 clases: NORMAL, NOT_CXR, PNEUMONIA (orden alfabetico tipico)
                p = softmax_probs(pred)
                # mapear a etiquetas de salida amigables
                # indice 0 -> NORMAL, 1 -> NOT_CXR, 2 -> PNEUMONIA
                p_norm, p_invalid, p_neu = float(p[0]), float(p[1]), float(p[2])

            else:
                # modelo binario: mantenemos gate para la tercera clase
                # si 1 nodo: prob(neumonia); si 2 nodos: usamos softmax y tomamos canal neumonia
                if out_dim == 1:
                    p_neu_bin = float(pred[0,0])
                else:
                    p2 = softmax_probs(pred)
                    # suponemos indice -1/neumonia es el mayor canal positivo
                    p_neu_bin = float(p2[-1])
                p_valid = validity_score(img)
                p_neu, p_norm, p_invalid = three_class_probs_with_gate(
                    p_neu_bin, p_valid, THRESH_INVALID_BASE
                )

        # ---- tarjeta de resultados ----
        st.markdown("<div class='prob-card'>", unsafe_allow_html=True)
        st.markdown("<div class='mini'>Probabilidades</div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Neumonia", f"{p_neu*100:.1f}%")
        m2.metric("Normal", f"{p_norm*100:.1f}%")
        m3.metric("Imagen no valida", f"{p_invalid*100:.1f}%")
        st.markdown("<hr/>", unsafe_allow_html=True)

        st.progress(min(1.0, p_neu), text=f"Neumonia: {p_neu:.4f}")
        st.progress(min(1.0, p_norm), text=f"Normal: {p_norm:.4f}")
        st.progress(min(1.0, p_invalid), text=f"No valida: {p_invalid:.4f}")

        etiqueta = ["Neumonia","Normal","Imagen no valida"][int(np.argmax([p_neu,p_norm,p_invalid]))]
        st.markdown("<br/>", unsafe_allow_html=True)
        st.info(f"Prediccion: {etiqueta}")
        st.markdown("</div>", unsafe_allow_html=True)

        if not use_three_head:
            st.caption("Nota: usando compuerta heuristica para 'Imagen no valida' porque el modelo cargado no es de 3 clases.")
        else:
            st.caption(f"Modelo detectado 3 clases: {model_path}")
else:
    st.write("Sube una imagen para comenzar")
