import streamlit as st
from rembg import remove, new_session
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import cv2
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Pro Passport & Headshot Studio", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stButton>button {
        background: linear-gradient(90deg, #1CB5E0 0%, #000851 100%);
        color: white; font-weight: bold; border: none; padding: 10px; border-radius: 8px;
    }
    .metric-card {
        background-color: #262730; padding: 10px; border-radius: 8px; text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'processed_fg' not in st.session_state: st.session_state['processed_fg'] = None
if 'final_result' not in st.session_state: st.session_state['final_result'] = None

# --- Core Logic Functions ---

@st.cache_resource
def get_rembg_session():
    return new_session("u2net")

def process_initial_image(image):
    # 1. Resize for speed optimization
    max_w = 2000
    if image.width > max_w:
        ratio = max_w / image.width
        image = image.resize((max_w, int(image.height * ratio)), Image.LANCZOS)
    
    # 2. Remove BG
    session = get_rembg_session()
    no_bg = remove(image, session=session)
    
    # 3. Smart Crop (Face Detection)
    cv_img = np.array(no_bg)
    cv_img = cv_img[:, :, ::-1].copy()
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        # Passport Ratio Logic (Usually vertical rectangle)
        center_x, center_y = x + w//2, y + h//2
        crop_h = int(h * 3.5) # Headroom
        crop_w = int(crop_h * 0.8) # 4:5 Aspect Ratio
        
        left = max(center_x - crop_w // 2, 0)
        top = max(center_y - crop_h // 2, 0)
        right = min(center_x + crop_w // 2, no_bg.width)
        bottom = min(center_y + crop_h // 2, no_bg.height)
        return no_bg.crop((left, top, right, bottom))
        
    return no_bg

def apply_edits(image, bg_color, settings):
    # Background
    bg = Image.new("RGBA", image.size, bg_color)
    bg.paste(image, (0, 0), image)
    final = bg.convert("RGB")
    
    # Adjustments
    if settings['temp'] != 0:
        r, g, b = final.split()
        r = r.point(lambda i: i * (1 + settings['temp'] * 0.1))
        b = b.point(lambda i: i * (1 - settings['temp'] * 0.1))
        final = Image.merge('RGB', (r, g, b))
        
    final = ImageEnhance.Brightness(final).enhance(settings['brightness'])
    final = ImageEnhance.Contrast(final).enhance(settings['contrast'])
    final = ImageEnhance.Color(final).enhance(settings['saturation'])
    final = ImageEnhance.Sharpness(final).enhance(settings['sharpness'])
    
    return final

def create_print_sheet(photo, paper_type, photo_size_mm):
    # 300 DPI Standard
    DPI = 300
    mm_to_px = lambda mm: int(mm * DPI / 25.4)
    
    # Paper Settings
    if paper_type == "A4":
        sheet_w, sheet_h = 2480, 3508 # A4 @ 300 DPI
    elif paper_type == "4x6 Inch":
        sheet_w, sheet_h = 1200, 1800 # 4x6 @ 300 DPI
    
    # Photo Target Size
    p_w_mm, p_h_mm = photo_size_mm
    p_w, p_h = mm_to_px(p_w_mm), mm_to_px(p_h_mm)
    
    # Create Sheet
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    
    # Resize Photo
    photo_resized = photo.resize((p_w, p_h), Image.LANCZOS)
    
    # Add thin border to photos for cutting
    border_w = 2
    photo_bordered = ImageOps.expand(photo_resized, border=border_w, fill='#DDDDDD')
    pb_w, pb_h = photo_bordered.size
    
    # Grid Logic (Simple Margin Calculation)
    margin = 50
    cols = (sheet_w - 2*margin) // (pb_w + 10)
    rows = (sheet_h - 2*margin) // (pb_h + 10)
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            x = margin + c * (pb_w + 20)
            y = margin + r * (pb_h + 20)
            sheet.paste(photo_bordered, (x, y))
            count += 1
            
    return sheet, count

# --- UI Layout ---

st.sidebar.title("🛠️ Studio Tools")
uploaded_file = st.sidebar.file_uploader("Upload Photo", type=['jpg', 'png'])

if uploaded_file and st.sidebar.button("🚀 Start Processing"):
    orig = Image.open(uploaded_file)
    with st.spinner("AI Detecting Face & Cleaning..."):
        st.session_state['processed_fg'] = process_initial_image(orig)

# --- MAIN WORKSPACE ---
if st.session_state['processed_fg']:
    
    # TABS
    tab_edit, tab_resize, tab_print = st.tabs(["🎨 Edit Look", "📏 Resize/Export", "🖨️ Print Lab"])
    
    # 1. EDIT TAB
    with tab_edit:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("##### Adjustments")
            bg_color = st.color_picker("Background", "#F2F2F2")
            b = st.slider("Brightness", 0.8, 1.3, 1.05)
            c = st.slider("Contrast", 0.8, 1.5, 1.1)
            s = st.slider("Saturation", 0.0, 1.5, 1.05)
            sh = st.slider("Sharpness", 1.0, 3.0, 1.4)
            t = st.slider("Warmth", -1.0, 1.0, 0.0)
            
            settings = {'brightness':b, 'contrast':c, 'saturation':s, 'sharpness':sh, 'temp':t}
            
            # Apply Edits
            edited_img = apply_edits(st.session_state['processed_fg'], bg_color, settings)
            st.session_state['final_result'] = edited_img
            
        with c2:
            st.image(edited_img, caption="Live Preview", width=350)

    # 2. RESIZE/EXPORT TAB
    with tab_resize:
        st.subheader("Custom Resolution Export")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            unit = st.selectbox("Unit", ["Pixel (px)", "Millimeter (mm)", "Centimeter (cm)"])
            
            # Helper to maintain aspect ratio
            orig_w, orig_h = st.session_state['final_result'].size
            aspect = orig_h / orig_w
            
            if unit == "Pixel (px)":
                target_w = st.number_input("Width", value=orig_w, min_value=100)
                target_h = st.number_input("Height", value=int(target_w * aspect))
                final_resize_w, final_resize_h = int(target_w), int(target_h)
            
            elif unit == "Millimeter (mm)":
                mm_w = st.number_input("Width (mm)", value=35.0)
                mm_h = st.number_input("Height (mm)", value=45.0)
                # Convert to px at 300 DPI
                final_resize_w = int(mm_w * 300 / 25.4)
                final_resize_h = int(mm_h * 300 / 25.4)
                
            else: # cm
                cm_w = st.number_input("Width (cm)", value=3.5)
                cm_h = st.number_input("Height (cm)", value=4.5)
                final_resize_w = int(cm_w * 10 * 300 / 25.4)
                final_resize_h = int(cm_h * 10 * 300 / 25.4)
                
        with col_r2:
            st.info(f"Output Resolution: **{final_resize_w} x {final_resize_h} px**")
            st.info("Quality: **300 DPI (High Res)**")
            
            if st.button("Download Custom Size"):
                out_img = st.session_state['final_result'].resize((final_resize_w, final_resize_h), Image.LANCZOS)
                buf = io.BytesIO()
                out_img.save(buf, format="JPEG", dpi=(300, 300), quality=100)
                st.download_button("📥 Download Single Photo", buf.getvalue(), "custom_photo.jpg", "image/jpeg")

    # 3. PRINT LAB TAB
    with tab_print:
        st.subheader("🖨️ Create Printable Sheet")
        
        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            paper = st.selectbox("Paper Size", ["4x6 Inch", "A4"])
            p_size = st.selectbox("Photo Size", ["Passport (35x45 mm)", "Stamp (20x25 mm)", "Visa (50x50 mm)"])
            
            # Map selection to dimensions (mm)
            if "35x45" in p_size: dim = (35, 45)
            elif "20x25" in p_size: dim = (20, 25)
            else: dim = (50, 50)
            
            if st.button("Generate Sheet"):
                sheet_img, count = create_print_sheet(st.session_state['final_result'], paper, dim)
                st.session_state['print_sheet'] = sheet_img
                st.session_state['print_count'] = count
                
        with col_p2:
            if 'print_sheet' in st.session_state:
                st.image(st.session_state['print_sheet'], caption=f"Preview: {st.session_state['print_count']} copies", use_container_width=True)
                
                buf_p = io.BytesIO()
                st.session_state['print_sheet'].save(buf_p, format="JPEG", dpi=(300, 300), quality=100)
                st.download_button(
                    label=f"📥 Download Printable Sheet ({paper})",
                    data=buf_p.getvalue(),
                    file_name="printable_sheet.jpg",
                    mime="image/jpeg"
                )

else:
    st.info("Upload a photo to access the Studio.")