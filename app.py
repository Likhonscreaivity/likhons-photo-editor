import streamlit as st
from rembg import remove, new_session
from PIL import Image, ImageEnhance, ImageFilter
import io
import cv2
import numpy as np

# --- Page Config (Must be first) ---
st.set_page_config(page_title="Pro Studio AI Editor", layout="wide", initial_sidebar_state="expanded")

# --- Session State for Speed ---
if 'processed_fg' not in st.session_state:
    st.session_state['processed_fg'] = None
if 'original_upload' not in st.session_state:
    st.session_state['original_upload'] = None

# --- Custom CSS for Modern UI ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #FF4B4B, #FF914D);
        color: white;
        border: none;
        padding: 10px;
        font-weight: bold;
        border-radius: 10px;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- Logic Functions ---

@st.cache_resource
def get_rembg_session():
    return new_session("u2net")

def resize_image(image, max_width=2000):
    width, height = image.size
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        return image.resize((max_width, new_height), Image.LANCZOS)
    return image

def remove_background_smart(input_image):
    """Heavy AI Task - Only runs once"""
    input_image = resize_image(input_image)
    session = get_rembg_session()
    with st.spinner('🤖 AI analyzing & extracting subject...'):
        return remove(input_image, session=session)

def crop_face_smart(image):
    """Smart Crop for Headshots"""
    cv_img = np.array(image)
    cv_img = cv_img[:, :, ::-1].copy()
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        center_x = x + w // 2
        center_y = y + h // 2
        # Tighter crop for passport/portrait style
        crop_size = int(max(w, h) * 3.5) 
        
        left = max(center_x - crop_size // 2, 0)
        top = max(center_y - crop_size // 2, 0)
        right = min(center_x + crop_size // 2, image.width)
        bottom = min(center_y + crop_size // 2, image.height)
        
        return image.crop((left, top, right, bottom))
    return image

def apply_pro_edits(foreground, bg_color, brightness, contrast, saturation, sharpness, is_transparent):
    """Real-time processing"""
    
    # 1. Background Logic
    if is_transparent:
        final_image = foreground
    else:
        new_bg = Image.new("RGBA", foreground.size, bg_color)
        new_bg.paste(foreground, (0, 0), foreground)
        final_image = new_bg.convert("RGB")
    
    # 2. Strong Enhancements
    # Brightness
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(brightness)
    
    # Contrast (New)
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(final_image)
        final_image = enhancer.enhance(contrast)
        
    # Saturation/Color (New)
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(final_image)
        final_image = enhancer.enhance(saturation)

    # Sharpness
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(final_image)
        final_image = enhancer.enhance(sharpness)
        
    return final_image

# --- UI Layout ---

# Sidebar Controls
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    uploaded_file = st.file_uploader("📂 Upload Image", type=['jpg', 'png', 'jpeg'])
    
    # State Management
    if uploaded_file:
        if st.session_state['original_upload'] != uploaded_file.name:
            st.session_state['processed_fg'] = None
            st.session_state['original_upload'] = uploaded_file.name
    
    if uploaded_file and st.button("✨ START EDITING", use_container_width=True):
        original = Image.open(uploaded_file)
        # Heavy processing
        fg = remove_background_smart(original)
        # Auto crop immediately
        st.session_state['processed_fg'] = crop_face_smart(fg)

    st.markdown("---")
    
    # Tools Group 1: Background
    with st.expander("🎨 Background Studio", expanded=True):
        transparent_mode = st.checkbox("Transparent Background (PNG)")
        bg_color = st.color_picker("Pick Any Color", "#F2F2F2")
        
    # Tools Group 2: Lighting & Color
    with st.expander("💡 Light & Color Correction", expanded=True):
        brightness = st.slider("Brightness (Exposure)", 0.5, 1.5, 1.05)
        contrast = st.slider("Contrast (Pop)", 0.5, 1.8, 1.1)
        saturation = st.slider("Saturation (Vibrance)", 0.0, 2.0, 1.1)
    
    # Tools Group 3: Details
    with st.expander("🔍 Details & Sharpening"):
        sharpness = st.slider("Sharpness", 0.0, 3.0, 1.4)

# Main Preview Area
st.title("📸 AI Professional Studio")

if uploaded_file:
    if st.session_state['processed_fg']:
        
        # Real-time editing
        final_result = apply_pro_edits(
            st.session_state['processed_fg'],
            bg_color,
            brightness,
            contrast,
            saturation,
            sharpness,
            transparent_mode
        )
        
        # Display
        col_prev, col_stats = st.columns([3, 1])
        with col_prev:
            st.image(final_result, caption="Final Professional Output", use_container_width=True)
        
        with col_stats:
            st.success("✅ AI Processing Complete")
            st.info(f"Resolution: {final_result.size}")
            
            # Download Logic
            buf = io.BytesIO()
            file_format = "PNG" # Always PNG for quality & transparency
            final_result.save(buf, format=file_format, quality=100)
            
            st.download_button(
                label="📥 Download HD Image",
                data=buf.getvalue(),
                file_name="pro_edited_photo.png",
                mime="image/png",
                use_container_width=True
            )
            
    else:
        st.info("👈 বাম পাশের সাইডবারে 'START EDITING' বাটনে ক্লিক করুন।")
        st.image(uploaded_file, caption="Original Upload", width=400)
else:
    st.write("👈 শুরু করতে বাম পাশের সাইডবার থেকে ছবি আপলোড করুন।")