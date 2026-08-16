import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.orchestration.pipeline import VoiceRAGPipeline
from app.orchestration.models import VoiceRAGRequest

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import logging

logger = logging.getLogger(__name__)

pipeline = VoiceRAGPipeline()

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing and pre-warming the RAG pipeline models...")
    
    # 1. Warm up the SentenceTransformer PyTorch forward pass using a simple query
    try:
        pipeline.en_retriever.embedding_service.encode_query("warmup")
        logger.info("PyTorch forward pass warmed up.")
    except Exception as e:
        logger.warning(f"Failed to warm up PyTorch forward pass: {e}")

    # 2. Load precomputed sentence embeddings from disk into the generator
    languages = {"en": "english", "hi": "hindi", "ta": "tamil", "te": "telugu", "ml": "malayalam"}
    # backend/app/main.py -> backend/app -> backend -> root
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "indexes")
    
    for lang, folder in languages.items():
        pkl_path = os.path.join(base_dir, folder, "sentence_embeddings.pkl")
        pipeline.extractive_generator.load_precomputed_embeddings(pkl_path)
            
    logger.info("RAG pipeline pre-warming complete. Server is ready.")

@app.post("/api/voice/query")
async def voice_query(audio: UploadFile = File(...), language: str = Form("en")):
    audio_data = await audio.read()
    req = VoiceRAGRequest(audio_data=audio_data, language_hint=language, generation_mode="extractive")
    resp = pipeline.execute(req)
    return resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": settings.app_name
    }

# Serve frontend if it exists
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
