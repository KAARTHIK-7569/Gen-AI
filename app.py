import streamlit as st
import os
import shutil
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. PAGE SETUP
st.set_page_config(page_title="Setu AI - Funding Analyst", layout="wide")
st.title("PolyVest : Your AI Funding Companion !!")
st.markdown("Ask about **Funding Schemes**, or **Upload your own Document** for analysis.")

# --- NEW: DEFINE FOLDERS ---
PERMANENT_DATA_PATH = "./data"
TEMP_UPLOAD_PATH = "./data/uploads"

# Create folders if they don't exist
os.makedirs(PERMANENT_DATA_PATH, exist_ok=True)
os.makedirs(TEMP_UPLOAD_PATH, exist_ok=True)

# --- NEW: CLEANUP FUNCTION (The Magic Trick) ---
# This runs once when the user opens/refreshes the page to ensure a clean slate.
if "cleanup_done" not in st.session_state:
    # Delete all files in the uploads folder
    for filename in os.listdir(TEMP_UPLOAD_PATH):
        file_path = os.path.join(TEMP_UPLOAD_PATH, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path) # Delete file
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    st.session_state.cleanup_done = True
    # Note: We do NOT delete the main 'data' folder, so your CSV/TXT database is safe.

# 2. CONFIGURATION & FILE UPLOAD
st.sidebar.title("⚙️ Settings")

if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
    st.sidebar.success("✅ API Key loaded")
else:
    api_key = st.sidebar.text_input("Enter Groq API Key (gsk_...)", type="password")

if not api_key:
    st.warning("⚠️ Enter Groq API Key to start.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Analyze Your Doc")
st.sidebar.info("ℹ️ Uploaded files are temporary and deleted on refresh.")
uploaded_file = st.sidebar.file_uploader("Upload Pitch Deck / Report", type=["pdf", "txt", "csv"])

if uploaded_file is not None:
    # SAVE TO THE TEMPORARY UPLOADS FOLDER
    save_path = os.path.join(TEMP_UPLOAD_PATH, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.sidebar.success(f"✅ Saved: {uploaded_file.name}")
    
    if st.sidebar.button("🔄 Process & Analyze Now"):
        st.cache_resource.clear()
        st.rerun()

# 3. SETUP THE BRAIN
try:
    Settings.llm = Groq(
        model="llama-3.3-70b-versatile", 
        api_key=api_key,
        context_window=8192,
        request_timeout=120.0
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    Settings.chunk_size = 512       
    Settings.chunk_overlap = 50     

except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# 4. LOAD DATA (Read BOTH Permanent AND Temporary folders)
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner("🧠 Reading Knowledge Base (Static DB + Temp Uploads)..."):
        documents = []
        
        # A. Load Permanent Database (CSV, Investor Text)
        if os.path.exists(PERMANENT_DATA_PATH):
            # Load only files directly in 'data', not subfolders
            # (We use a specific list comprehension to avoid reading 'uploads' twice if using recursive)
            # Actually, SimpleDirectoryReader(recursive=True) is easiest:
            documents += SimpleDirectoryReader(PERMANENT_DATA_PATH, recursive=True).load_data()
        
        if not documents:
            return None
            
        index = VectorStoreIndex.from_documents(documents)
        return index

index = load_data()

if not index:
    st.info("👋 Database empty. Add files to 'data' folder or upload one.")
    st.stop()

# 5. CHAT ENGINE
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context",
        system_prompt=(
            "You are 'Setu', an expert Startup Consultant."
            "DATA SOURCES: You have access to official funding data AND user-uploaded documents."
            "INSTRUCTIONS:"
            "SPECIAL INSTRUCTION: If the user asks to 'Analyze' their uploaded pitch deck, DO NOT just summarize it."
            "Instead, generate a structured 'FUNDING READINESS REPORT' with:"
            "1. ✅ **Eligibility Check:** (Pass/Fail for SISFS based on age/sector)."
            "2. 💰 **Valuation sanity check:** (Is their ask reasonable? Compare with your CSV data)."
            "3. 🎯 **Investor Match:** (Name the top 3 specific investors from the database)."
            "4. 🛑 **Red Flags:** (What is missing? e.g., 'No revenue mentioned')."
            "5. 🏆 **Setu Score:** Give a score out of 10 for funding probability."
            "Always answer in the user's language (Hindi/Tamil/English)."
        )
    )

# 6. CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask: 'Analyze my uploaded pitch deck'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat_engine.chat(prompt)
                st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
            except Exception as e:
                st.error(f"Error: {e}")