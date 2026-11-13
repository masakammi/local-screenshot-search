import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import gc
import json
import warnings
import numpy as np
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore", message=".*pin_memory.*")

st.set_page_config(page_title="Screenshot Search", layout="wide")
st.title("🔍 Local Screenshot Search")

FOLDER = r"C:\Users\Admin\Desktop\SS"
INDEX_FILE = "screenshot_index.json"

@st.cache_resource
def load_clip():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('clip-ViT-B-32')

@st.cache_data
def load_index(file_path, file_mtime):
    with open(file_path, "r") as f:
        return json.load(f)

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
                with st.spinner("Loading EasyOCR model..."):
                    import easyocr
                    reader = easyocr.Reader(['en'])
                    clip_model = load_clip()

                status_text = st.empty()
                progress = st.progress(0)
                total_new = len(new_files)

                for i, filename in enumerate(new_files):
                    status_text.text(f"Scanning {i + 1} of {total_new}: {filename}")
                    path = os.path.join(FOLDER, filename)
                    try:
                        img = Image.open(path).convert("RGB")
                        
                        ocr_img = img.copy()
                        ocr_img.thumbnail((1024, 1024))
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

                    if (i + 1) % 10 == 0:
                        with open(INDEX_FILE, "w") as f:
                            json.dump(existing_index, f)
                        gc.collect()

                    progress.progress((i + 1) / total_new)

                with open(INDEX_FILE, "w") as f:
                    json.dump(existing_index, f)
                
                del reader
                gc.collect()
                status_text.empty()
                st.success(f"Added {total_new} new screenshots! (Total: {len(existing_index)})")
query = st.text_input("Search by remembered text or phrase:", placeholder="e.g., 'twitter', 'receipt', or a phrase from the tweet")
col_a, col_b = st.columns([2, 1])
with col_a:
    top_k = st.slider("Number of results to show", min_value=3, max_value=24, value=9, step=3)
with col_b:
    use_visual_ai = st.checkbox("Enable Visual AI (CLIP)", value=False, help="Uses ~1GB extra RAM. Leave unchecked for instant text search.")

if query:
    if not os.path.exists(INDEX_FILE):
        st.warning("No index found. Click 'Scan & Index New Screenshots' in the sidebar first.")
    else:
        index = load_index(INDEX_FILE, os.path.getmtime(INDEX_FILE))
        query_lower = query.lower()
        query_words = query_lower.split()

        if use_visual_ai:
            clip_model = load_clip()
            query_emb = np.array(clip_model.encode(query), dtype=np.float32)
            query_norm = np.linalg.norm(query_emb)

        results = []
        for item in index:
            text = item["text"]
            # Score exact phrase match highest, then partial word matches
            exact_match = 2.0 if query_lower in text else 0.0
            word_hits = sum(1 for w in query_words if w in text) / max(len(query_words), 1)
            text_score = exact_match + word_hits

            visual_score = 0.0
            if use_visual_ai:
                item_emb = np.array(item["embedding"], dtype=np.float32)
                visual_score = float(np.dot(query_emb, item_emb) / (query_norm * np.linalg.norm(item_emb) + 1e-8))

            total_score = text_score + visual_score
            if total_score > 0:
                results.append((total_score, text_score > 0, item))

        results.sort(reverse=True, key=lambda x: x[0])

        if not results:
            st.info("No matching screenshots found. Try a shorter keyword or different spelling.")
        else:
            st.write(f"Found **{len(results)}** matching screenshots (showing top {min(top_k, len(results))}):")
            cols = st.columns(3)
            for idx, (score, is_text_match, item) in enumerate(results[:top_k]):
                with cols[idx % 3]:
                    st.image(item["path"], use_container_width=True)
                    match_type = "📝 Text Match" if is_text_match else "🖼️ Visual Match"
                    st.caption(f"**{item['filename']}** | {match_type} (Score: {score:.2f})")
                    with st.expander("View extracted text"):
                        st.write(item["text"] if item["text"] else "*No text detected*")
