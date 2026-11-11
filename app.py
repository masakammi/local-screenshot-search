import os
import json
import streamlit as st
import easyocr
from PIL import Image
from sentence_transformers import SentenceTransformer, util

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
    if st.button("🔄 Scan & Index Folder"):
        if not os.path.exists(FOLDER):
            st.error("Folder path not found. Please check the path in app.py.")
        else:
            files = [f for f in os.listdir(FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            progress = st.progress(0)
            index = []
            
            for i, filename in enumerate(files):
                path = os.path.join(FOLDER, filename)
                try:
                    # Extract text
                    text_list = reader.readtext(path, detail=0)
                    full_text = " ".join(text_list).lower()
                    
                    # Extract visual embedding
                    img = Image.open(path).convert("RGB")
                    embedding = clip_model.encode(img).tolist()
                    
                    index.append({
                        "filename": filename,
                        "path": path,
                        "text": full_text,
                        "embedding": embedding
                    })
                except Exception as e:
                    st.warning(f"Skipped {filename}: {e}")
                
                progress.progress((i + 1) / len(files))
                
            with open(INDEX_FILE, "w") as f:
                json.dump(index, f)
            st.success(f"Indexed {len(index)} screenshots!")

# Main Search Interface
query = st.text_input("Search by remembered text or visual description:", placeholder="e.g., 'receipt', 'football match', or 'order confirmed'")
top_k = st.slider("Number of results to show", min_value=3, max_value=24, value=6, step=3)

if query:
    if not os.path.exists(INDEX_FILE):
        st.warning("No index found. Click 'Scan & Index Folder' in the sidebar first.")
    else:
        with open(INDEX_FILE, "r") as f:
            index = json.load(f)
            
        query_lower = query.lower()
        query_emb = clip_model.encode(query)
        
        results = []
        for item in index:
            # Exact text match gets a strong boost
            text_match = 1.0 if query_lower in item["text"] else 0.0
            # Visual similarity score
            visual_score = util.cos_sim(query_emb, item["embedding"]).item()
            total_score = text_match + visual_score
            
            results.append((total_score, text_match > 0, item))
            
        results.sort(reverse=True, key=lambda x: x[0])
        
        # Display results in a 3-column grid
        cols = st.columns(3)
        for idx, (score, is_text_match, item) in enumerate(results[:top_k]):
            with cols[idx % 3]:
                st.image(item["path"], use_container_width=True)
                match_type = "📝 Text Match" if is_text_match else "🖼️ Visual Match"
                st.caption(f"**{item['filename']}** | {match_type} (Score: {score:.2f})")
                with st.expander("View extracted text"):
                    st.write(item["text"] if item["text"] else "*No text detected*")
