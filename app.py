import os
import json
import warnings
import numpy as np
import streamlit as st
import easyocr
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# Silence harmless CPU pin_memory warning
warnings.filterwarnings("ignore", message=".*pin_memory.*")

# Page setup
st.set_page_config(page_title="Screenshot Search", layout="wide")
st.title("🔍 Local Screenshot Search")

# Configuration
FOLDER = r"C:\Users\Admin\Desktop\SS"
INDEX_FILE = "screenshot_index.json"

# Load AI models once and cache them in memory
@st.cache_resource
def load_models():
    reader = easyocr.Reader(['en'])
    clip_model = SentenceTransformer('clip-ViT-B-32')
    return reader, clip_model

reader, clip_model = load_models()

# Sidebar for indexing controls
with st.sidebar:
    st.header("Index Settings")
    if st.button("🔄 Scan & Index New Screenshots"):
        if not os.path.exists(FOLDER):
            st.error("Folder path not found. Please check the path in app.py.")
        else:
            existing_index = []
            indexed_filenames = set()
            if os.path.exists(INDEX_FILE):
                with open(INDEX_FILE, "r") as f:
                    existing_index = json.load(f)
                indexed_filenames = {item["filename"] for item in existing_index}

            all_files = [f for f in os.listdir(FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            new_files = [f for f in all_files if f not in indexed_filenames]

            if not new_files:
                st.info(f"All {len(existing_index)} screenshots are already indexed!")
            else:
                status_text = st.empty()
                progress = st.progress(0)
                total_new = len(new_files)

                for i, filename in enumerate(new_files):
                    status_text.text(f"Scanning {i + 1} of {total_new}: {filename}")
                    path = os.path.join(FOLDER, filename)
                    try:
                        img = Image.open(path).convert("RGB")
                        
                        # Resize tall/scrolling screenshots to prevent RAM spikes
                        ocr_img = img.copy()
                        ocr_img.thumbnail((1600, 1600))
                        text_list = reader.readtext(np.array(ocr_img), detail=0)
                        full_text = " ".join(text_list).lower()

                        embedding = clip_model.encode(img).tolist()

                        existing_index.append({
                            "filename": filename,
                            "path": path,
                            "text": full_text,
                            "embedding": embedding
                        })
                    except Exception as e:
                        st.warning(f"Skipped {filename}: {e}")

                    # Auto-save every 10 images
                    if (i + 1) % 10 == 0:
                        with open(INDEX_FILE, "w") as f:
                            json.dump(existing_index, f)

                    progress.progress((i + 1) / total_new)

                with open(INDEX_FILE, "w") as f:
                    json.dump(existing_index, f)
                status_text.empty()
                st.success(f"Added {total_new} new screenshots! (Total: {len(existing_index)})")

# Main Search Interface
query = st.text_input("Search by remembered text or visual description:", placeholder="e.g., 'receipt', 'football match', or 'order confirmed'")
top_k = st.slider("Number of results to show", min_value=3, max_value=24, value=6, step=3)

if query:
    if not os.path.exists(INDEX_FILE):
        st.warning("No index found. Click 'Scan & Index New Screenshots' in the sidebar first.")
    else:
        with open(INDEX_FILE, "r") as f:
            index = json.load(f)
            
        query_lower = query.lower()
        query_emb = clip_model.encode(query)
        
        results = []
        for item in index:
            text_match = 1.0 if query_lower in item["text"] else 0.0
            visual_score = util.cos_sim(query_emb, item["embedding"]).item()
            total_score = text_match + visual_score
            results.append((total_score, text_match > 0, item))
            
        results.sort(reverse=True, key=lambda x: x)
        
        cols = st.columns(3)
        for idx, (score, is_text_match, item) in enumerate(results[:top_k]):
            with cols[idx % 3]:
                st.image(item["path"], use_container_width=True)
                match_type = "📝 Text Match" if is_text_match else "🖼️ Visual Match"
                st.caption(f"**{item['filename']}** | {match_type} (Score: {score:.2f})")
                with st.expander("View extracted text"):
                    st.write(item["text"] if item["text"] else "*No text detected*")
