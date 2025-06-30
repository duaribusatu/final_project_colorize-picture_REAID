import streamlit as st
import numpy as np
from PIL import Image
import io
import torch
from colorizers.siggraph17 import siggraph17
from colorizers.utils import preprocess_img, postprocess_tens

st.set_page_config(page_title="Photo Colorizer (SIGGRAPH17)", layout="centered")
st.title("🎨 Black & White Photo Colorization")
st.caption("Model: SIGGRAPH 2017 by Richard Zhang et al.")
st.caption("by Dandi Septiandi")

@st.cache_resource
def load_model():
    model = siggraph17(pretrained=True)
    model.eval()
    return model

model = load_model()

uploaded = st.file_uploader("📤 Upload a black and white photo", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    with st.spinner("🎨 Coloring in progress..."):
        tens_orig_l, tens_rs_l = preprocess_img(img_np, HW=(256, 256))
        with torch.no_grad():
            out_ab = model.inference(tens_rs_l)
            out_img = postprocess_tens(tens_orig_l, out_ab)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original (B&W)", use_container_width=True)
    with col2:
        st.image(out_img, caption="Colorized", use_container_width=True)

    buf = io.BytesIO()
    Image.fromarray((out_img * 255).astype(np.uint8)).save(buf, format="PNG")
    st.download_button("📥 Download Colorized Image", data=buf.getvalue(), file_name="colorized.png", mime="image/png")
else:
    st.info("Please upload a black and white photo to begin.")
