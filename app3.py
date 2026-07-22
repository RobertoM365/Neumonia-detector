st.set_page_config(page_title="Pneumonia X-Ray Classifier", page_icon="🫁")

# ===== UI =====
st.title("Pneumonia X-Ray Classifier")
st.caption("Upload an image. The app will return the probabilities for: Pneumonia, Normal, and Invalid Image.")

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])

...

    with col_img:
        st.image(img, caption="Preview", use_column_width=False, width=320)

    with col_pred:
        with st.spinner("Processing..."):

...

        # ---- result card ----
        st.markdown("<div class='prob-card'>", unsafe_allow_html=True)
        st.markdown("<div class='mini'>Probabilities</div>", unsafe_allow_html=True)

        m1.metric("Pneumonia", f"{p_neu*100:.1f}%")
        m2.metric("Normal", f"{p_norm*100:.1f}%")
        m3.metric("Invalid Image", f"{p_invalid*100:.1f}%")

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.progress(min(1.0, p_neu), text=f"Pneumonia: {p_neu:.4f}")
        st.progress(min(1.0, p_norm), text=f"Normal: {p_norm:.4f}")
        st.progress(min(1.0, p_invalid), text=f"Invalid Image: {p_invalid:.4f}")

        label = ["Pneumonia", "Normal", "Invalid Image"][
            int(np.argmax([p_neu, p_norm, p_invalid]))
        ]

        st.markdown("<br/>", unsafe_allow_html=True)
        st.info(f"Prediction: {label}")
        st.markdown("</div>", unsafe_allow_html=True)

        if not use_three_head:
            st.caption(
                "Note: Using a heuristic gate for 'Invalid Image' because the loaded model is not a three-class model."
            )
        else:
            st.caption(f"Detected 3-class model: {model_path}")

else:
    st.write("Upload an image to get started")
