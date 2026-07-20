import streamlit as st
from PIL import Image
from inference import predict_image

# 1. Page Configuration
st.set_page_config(
    page_title="Bengali Handwritten OCR Dashboard",
    page_icon="🇧🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURABLE LINKS ---
GITHUB_REPO_URL = "https://github.com/RahatAhmed171/Bengali_Handwritten_Upazila-District_Pair_Name_Recognition"  # Replace with your actual GitHub repo link
PAPER_LINK_URL = "https://drive.google.com/file/d/1vbYmXdCWBwOpDmxM5UmtgABf1wewBmmc/view?usp=sharing"  # Replace with your actual Paper / Research link (or arXiv/Drive URL)

# 2. Sidebar Layout
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/artificial-intelligence.png", width=80)
    st.title("System Info")
    st.markdown("""
    ### 🧠 Architecture
    * **Task:** Handwritten District-Upazila OCR
    * **Framework:** PyTorch & Vision Transformers / ConvNeXt
    * **Environment:** CPU Optimized Production Tier
    """)
    
    st.divider()
    
    # Paper & Source Code Section
    st.markdown("### 🔗 Project Links")
    st.link_button("💻 GitHub Repository", GITHUB_REPO_URL, use_container_width=True)
    st.link_button("📄 Read Research Paper", PAPER_LINK_URL, use_container_width=True)
    
    st.divider()
    
    st.markdown("""
    ### 💡 How to test:
    1. Upload a clear handwritten snippet image.
    2. Click **Run Inference**.
    3. The cached ensemble model yields results instantly.
    """)

# 3. Main Header
st.markdown("# 🇧🇩 Bengali Handwritten Geo-Name Recognition")
st.markdown(
    "*Production Showcase: A deep learning pipeline built to classify and recognize "
    "handwritten administrative regions of Bangladesh.*"
)

# Header quick links row
st.markdown(f"[💻 GitHub Repository]({GITHUB_REPO_URL}) &nbsp;|&nbsp; [📄 Research Paper / Paper Link]({PAPER_LINK_URL})")

st.divider()

# 4. Two-Column Dashboard Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📥 Input Panel")
    
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Drag and drop your handwritten image snippet here", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Preview of Uploaded Text", use_container_width=True)
            submit_btn = st.button("🚀 Run Inference", type="primary", use_container_width=True)
        else:
            submit_btn = False
            if "last_prediction" in st.session_state:
                del st.session_state["last_prediction"]

with col2:
    st.markdown("### 🎯 Analysis & Prediction")
    
    # Run prediction and store in session_state
    if uploaded_file is not None and submit_btn:
        with st.status("🔮 Running neural network inference...", expanded=True) as status:
            try:
                result = predict_image(image_path=image, version="v1")
                status.update(label="✅ Inference Completed!", state="complete", expanded=False)
                st.session_state["last_prediction"] = result
                st.balloons()
            except Exception as e:
                status.update(label="❌ Inference Failed", state="error")
                st.error(f"Error executing model pipeline: {str(e)}")

    # Render prediction output OUTSIDE the status block so it stays permanently visible
    if "last_prediction" in st.session_state and uploaded_file is not None:
        result = st.session_state["last_prediction"]
        
        st.markdown("#### ✨ Predicted Location")
        st.success(f"### **{result['class_name']}**")
        
        confidence_pct = result['confidence'] * 100
        st.metric(label="Model Confidence Score", value=f"{confidence_pct:.2f}%")
        st.progress(int(confidence_pct))
        
        with st.expander("🛠️ View Technical Class Metadata"):
            st.json({
                "Internal Class ID": result['class_id'],
                "Pipeline Version": "v1.0.0",
                "Device Allocated": "CPU (Cached)"
            })
            
    elif uploaded_file is not None and "last_prediction" not in st.session_state:
        st.info("💡 Image loaded successfully. Click the **Run Inference** button to process.")
    else:
        st.info("📥 Please upload a handwritten snippet image on the left to start the analysis.")