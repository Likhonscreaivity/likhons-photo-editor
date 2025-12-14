import streamlit as st
from rembg import remove, new_session
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import cv2
import numpy as np
import google.generativeai as genai
from streamlit_drawable_canvas import st_canvas

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Pro AI Studio (High-Res)", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    .stButton>button {
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        color: black; font-weight: bold; border: none; padding: 0.6rem 1.2rem; border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'processed_image' not in st.session_state: st.session_state['processed_image'] = None
if 'original_image' not in st.session_state: st.session_state['original_image'] = None

# --- 3. ADVANCED AI LOGIC ---

@st.cache_resource
def get_rembg_model():
    # Switching to ISNET-GENERAL-USE (Best for detailed edges/hair)
    return new_session("isnet-general-use")

def smart_resize(image, max_w=2000):
    """Resize maintaining high quality"""
    if not image: return None
    if image.width > max_w:
        ratio = max_w / image.width
        new_h = int(image.height * ratio)
        return image.resize((max_w, new_h), Image.LANCZOS)
    return image

def apply_feathering(image, blur_radius=2):
    """Smooths the jagged edges of the cutout"""
    if image.mode != 'RGBA':
        return image
    
    # Split channels
    r, g, b, a = image.split()
    
    # Blur the alpha channel (the mask)
    a_blurred = a.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Merge back
    return Image.merge('RGBA', (r, g, b, a_blurred))

def remove_background_pro(image):
    """The High-Quality Removal Logic"""
    session = get_rembg_model()
    
    # Advanced Alpha Matting Parameters
    # This keeps transparency in hair while making solid parts solid
    no_bg = remove(
        image, 
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10
    )
    
    # Apply Feathering to fix "Sticker Look"
    final_cutout = apply_feathering(no_bg, blur_radius=1)
    return final_cutout

def advanced_processing(image, settings):
    """Photoshop-style Enhancements"""
    if image is None: return None
    
    # Convert to RGB for OpenCV
    if image.mode == 'RGBA':
        base = image.convert('RGB')
    else:
        base = image
    
    cv_img = np.array(base)[:, :, ::-1].copy()
    
    # 1. Clarity (Local Contrast)
    clarity = settings.get('clarity', 0)
    if clarity > 0:
        blurred = cv2.GaussianBlur(cv_img, (0, 0), 3)
        cv_img = cv2.addWeighted(cv_img, 1.0 + clarity, blurred, -clarity, 0)
    
    # 2. Convert back to PIL
    enhanced_rgb = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    
    # 3. Apply standard filters
    enhanced_rgb = ImageEnhance.Color(enhanced_rgb).enhance(settings['sat'])
    enhanced_rgb = ImageEnhance.Brightness(enhanced_rgb).enhance(settings['bright'])
    enhanced_rgb = ImageEnhance.Contrast(enhanced_rgb).enhance(settings['contrast'])
    enhanced_rgb = ImageEnhance.Sharpness(enhanced_rgb).enhance(settings['sharp'])
    
    # 4. Handle Background
    if settings['transparent']:
        # Apply alpha from original cutout
        r, g, b = enhanced_rgb.split()
        a = image.split()[3]
        return Image.merge('RGBA', (r, g, b, a))
    else:
        # Create Solid Background
        bg_img = Image.new("RGBA", enhanced_rgb.size, settings['bg_color'])
        if image.mode == 'RGBA':
            mask = image.split()[3]
            bg_img.paste(enhanced_rgb, (0, 0), mask)
        return bg_img.convert("RGB")

# --- 4. UI LAYOUT ---

st.sidebar.title("💎 Pro AI Studio")
api_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password")
uploaded_file = st.sidebar.file_uploader("Upload High-Res Photo", type=['jpg', 'png', 'jpeg'])

# --- PROCESS LOGIC ---
if uploaded_file:
    # Load new file
    if st.session_state['original_image'] is None or uploaded_file.name != st.session_state.get('filename', ''):
        img = Image.open(uploaded_file)
        st.session_state['original_image'] = smart_resize(img)
        st.session_state['filename'] = uploaded_file.name
        st.session_state['processed_image'] = None # Reset
        
    # Process Button
    if st.sidebar.button("✨ Remove Background (High Quality)"):
        with st.spinner("AI analyzing details (Model: ISNET)..."):
            try:
                # Call the new PRO function
                cutout = remove_background_pro(st.session_state['original_image'])
                st.session_state['processed_image'] = cutout
                
                # Gemini Advice (Optional)
                if api_key:
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        res = model.generate_content(["Give 1 short tip to improve this headshot lighting.", st.session_state['original_image'].resize((512,512))])
                        st.toast(f"AI Tip: {res.text}")
                    except: pass
                    
            except Exception as e:
                st.error(f"Error: {e}")

# --- EDITOR INTERFACE ---
if st.session_state['processed_image']:
    
    # Controls
    col_tools, col_view = st.columns([1, 2.5])
    
    with col_tools:
        st.markdown("### 🎛️ Adjustments")
        
        with st.expander("Background", expanded=True):
            is_trans = st.checkbox("Transparent PNG", value=False)
            bg_col = st.color_picker("Pick Color", "#F0F0F0")
            
        with st.expander("Enhancements", expanded=True):
            clarity = st.slider("Clarity (Detail)", 0.0, 1.0, 0.0)
            bright = st.slider("Brightness", 0.8, 1.3, 1.0)
            contrast = st.slider("Contrast", 0.8, 1.5, 1.05)
            sat = st.slider("Saturation", 0.0, 1.5, 1.05)
            sharp = st.slider("Sharpness", 0.0, 3.0, 1.3)
            
        # Compile Settings
        settings = {
            'transparent': is_trans,
            'bg_color': bg_col,
            'clarity': clarity,
            'bright': bright,
            'contrast': contrast,
            'sat': sat,
            'sharp': sharp
        }
        
    with col_view:
        # Real-time render
        final_img = advanced_processing(st.session_state['processed_image'], settings)
        
        if final_img:
            st.image(final_img, caption="Professional Result", use_container_width=True)
            
            # Download Logic
            buf = io.BytesIO()
            ext = "PNG" if is_trans else "JPEG"
            final_img.save(buf, format=ext, quality=100, dpi=(300, 300))
            
            st.download_button(
                label=f"📥 Download {ext} (HD)",
                data=buf.getvalue(),
                file_name=f"pro_edit.{ext.lower()}",
                mime=f"image/{ext.lower()}",
                use_container_width=True
            )

else:
    st.info("Upload a photo and click the button to start.")
    st.caption("Using 'isnet-general-use' model for superior edge detection.")