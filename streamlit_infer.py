import streamlit as st
from infer import ImageCaptioning
from PIL import Image

@st.cache_resource
def load_model():
    img_captioning = ImageCaptioning()
    return img_captioning

st.set_page_config(page_title="Image Captioning App",
                   page_icon="🏞️",
                   layout="wide",
                   initial_sidebar_state="expanded")
img_captioning = load_model()
st.title("Image Captioning")
st.markdown("This is an image captioning app that uses model trained on filker8k dataset. it uses VGG16 encode and LSTM decoder.")

with st.container(height=400):
    img_path = st.file_uploader("Upload an image", type=['jpg', 'png'])
    if img_path:
        img = Image.open(img_path)
        img = img.resize((224,224))
        col1, col2, _ = st.columns(3, vertical_alignment='center', gap='small')
        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=False)
        with col2:
            # st.write("")
            with st.spinner("Processing image..."):
                caption = img_captioning.predict(img_path)
            st.toast("Caption generated successfully", icon="🥂")
            st.markdown(f"<h4>{caption}</h4>", unsafe_allow_html=True)
            # st.experimental_rerun()