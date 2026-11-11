# Local Screenshot Search

A local, privacy-first web app built with Python and Streamlit to search thousands of screenshots using **remembered text (OCR)** or **visual descriptions (AI embeddings)**—no cloud uploads required.

## Features
- **Text Extraction (EasyOCR):** Reads text inside screenshots (tweets, receipts, chats, notes) and indexes it locally.
- **Visual Search (OpenAI CLIP):** Converts images into semantic embeddings (`clip-ViT-B-32`) so you can search by visual concepts like *"dark mode tweet"* or *"football match"*.
- **Incremental Indexing & Auto-Save:** Skips already-indexed files on future scans and saves progress every 10 images.
- **Local UI (Streamlit):** Displays top matches in a clean 3-column gallery with expandable extracted text.

## Setup & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd local-screenshot-search
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your screenshot folder:**
   Open `app.py` and update `FOLDER = r"path\to\your\Screenshots"`.

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   Click **Scan & Index New Screenshots** in the sidebar to build or update your local `screenshot_index.json` database.
