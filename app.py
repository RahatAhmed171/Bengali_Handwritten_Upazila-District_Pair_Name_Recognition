import streamlit as st
from PIL import Image
from inference import predict_image  # Your model prediction code

# 1. Professional Page Configuration
st.set_page_config(
    page_title="Bengali Handwritten OCR Dashboard",
    page_icon="🇧🇩",
    layout="wide",  # Wide layout gives it a dashboard feel
    initial_sidebar_state="expanded"
)

# 2. Sidebar for Metadata & Instructions
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/artificial-intelligence.png", width=80)
    st.title("System Info")
    st.markdown("""
    ### 🧠 Architecture
    * **Task:** Handwritten District-Upazila OCR
    * **Framework:** PyTorch & Transformers
    * **Environment:** CPU Optimized Production Tier
    
    ---
    ### 💡 How to test:
    1. Upload a clear snippet image.
    2. Click **Run Inference**.
    3. The cached neural network yields results instantly.
    """)

# 3. Main Dashboard Header
st.markdown("# 🇧🇩 Bengali Handwritten Geo-Name Recognition")
st.markdown(
    "*Production Showcase: A deep learning pipeline built to classify and recognize "
    "handwritten administrative regions of Bangladesh.*"
)

st.divider()

# 4. Two-Column Dashboard Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📥 Input Panel")
    
    # Wrapped file uploader inside a clean card container
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Drag and drop your handwritten image snippet here", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Preview of Uploaded Text", use_container_width=True)
            
            # Action button
            submit_btn = st.button("🚀 Run Inference", type="primary", use_container_width=True)
        else:
            submit_btn = False

with col2:
    st.markdown("### 🎯 Analysis & Prediction")
    
    if uploaded_file is not None and submit_btn:
        # Polished progress container while processing
        with st.status("🔮 Running neural network inference...", expanded=True) as status:
            try:
                # Run your optimized cached inference script
                result = predict_image(image_path=image, version="v1")
                status.update(label="✅ Inference Completed!", state="complete", expanded=False)
                
                # Show results in beautiful UI elements
                st.balloons() # Fun professional touch for a successful demo
                
                st.markdown("#### ✨ Predicted Location")
                st.success(f"### **{result['class_name']}**")
                
                # Metric display with progress bar for visual flair
                confidence_pct = result['confidence'] * 100
                st.metric(label="Model Confidence Score", value=f"{confidence_pct:.2f}%")
                st.progress(int(confidence_pct))
                
                # Technical metadata for technical recruiters
                with st.expander("🛠️ View Technical Class Metadata"):
                    st.json({
                        "Internal Class ID": result['class_id'],
                        "Pipeline Version": "v1.0.0",
                        "Device Allocated": "CPU (Cached)"
                    })
                    
            except Exception as e:
                status.update(label="❌ Inference Failed", state="error")
                st.error(f"Error executing model pipeline: {str(e)}")
                
    elif uploaded_file is not None and not submit_btn:
        st.info("💡 Image loaded successfully. Click the **Run Inference** button to process.")
    else:
        st.info("📥 Please upload a handwritten snippet image on the left to start the analysis.")