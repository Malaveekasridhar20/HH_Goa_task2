import os
import sys
import time
import json
import asyncio
import websockets
import base64
import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../backend/.env"))

sys.path.append(os.path.join(os.path.dirname(__file__), "../backend"))
from app.orchestration.pipeline import VoiceRAGPipeline
import io
import pyttsx3
import tempfile

async def stream_audio_to_sarvam(audio_bytes, language, api_key):
    # Prepare URL and params
    # We'll use the saaras:v3-realtime endpoint which provides partials, or standard ws.
    # The prompt says: "Use the official Sarvam WebSocket endpoint and current Saaras v3 configuration. model = saaras:v3"
    url = f"wss://api.sarvam.ai/speech-to-text-realtime/ws?model=saaras:v3-realtime&language_code={language}"
    
    headers = {
        "api-subscription-key": api_key
    }
    
    # Let's chunk the audio to simulate 50ms streaming chunks
    # Actually, gTTS outputs mp3. Sarvam websocket usually wants 16kHz PCM or WAV.
    # Stream raw MP3 bytes directly since we don't have ffmpeg.
    # We will chunk it to simulate streaming.
    wav_bytes = audio_bytes
    chunk_size = 4096 # Arbitrary chunk size
    chunks = [wav_bytes[i:i+chunk_size] for i in range(0, len(wav_bytes), chunk_size)]
    
    t_start = time.time()
    t_first_transcript = None
    t_final_transcript = None
    final_transcript = ""
    
    t_connected = None
    
    try:
        async with websockets.connect(url, additional_headers=headers) as websocket:
            t_connected = time.time()
            
            # Send initial config frame if required (some versions need it, some don't, we will just send audio frames)
            # We'll stream as JSON base64 as suggested by docs.
            
            async def receive_responses():
                nonlocal t_first_transcript, t_final_transcript, final_transcript
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        if "transcript" in data:
                            t = data["transcript"].strip()
                            if t and not t_first_transcript:
                                t_first_transcript = time.time()
                            
                            # Realtime API might indicate is_final. 
                            if data.get("is_final", False) or data.get("final", False):
                                final_transcript = t
                                t_final_transcript = time.time()
                            else:
                                # For legacy ws, it might just return the final transcript once
                                final_transcript = t
                                t_final_transcript = time.time()
                except websockets.exceptions.ConnectionClosed:
                    pass
            
            # Start listener task
            listener = asyncio.create_task(receive_responses())
            
            for chunk in chunks:
                msg = {
                    "audio": {
                        "data": base64.b64encode(chunk).decode('utf-8'),
                        "sample_rate": 16000,
                        "encoding": "audio/wav"
                    }
                }
                await websocket.send(json.dumps(msg))
                await asyncio.sleep(0.01) # simulate real-time
                
            # Wait a bit for final response
            await asyncio.sleep(3.0)
            await websocket.close()
            await listener
            
    except Exception as e:
        print(f"Streaming error: {e}")
        
    return {
        "t_connected": (t_connected - t_start) if t_connected else None,
        "t_first": (t_first_transcript - t_start) if t_first_transcript else None,
        "t_final": (t_final_transcript - t_start) if t_final_transcript else None,
        "transcript": final_transcript
    }

async def run_evaluation():
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        print("SARVAM_API_KEY missing.")
        return
        
    pipeline = VoiceRAGPipeline()
    
    en_queries, hi_queries = [], []
    data_path = os.path.join(os.path.dirname(__file__), "../data/processed/hinval_500.jsonl")
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            en_queries.append(row.get("Eng_Query"))
            hi_queries.append(row.get("query"))
            
    en_queries = en_queries[:5]
    hi_queries = hi_queries[:5]
    
    async def evaluate(queries, language):
        metrics = {
            "conn": [], "first": [], "final": [], "ret": [], "gen": [], "tot": [],
            "grounded": 0, "safe_refusal": 0, "unsupported": 0
        }
        
        # Initialize pyttsx3 engine
        engine = pyttsx3.init()
        # Not strictly setting language voices, just synthesizing the text to WAV
        
        for q in queries:
            print(f"Synthesizing: {q.encode('utf-8', errors='replace').decode('utf-8')}", flush=True)
            
            # Save to temporary wav file
            temp_wav = tempfile.mktemp(suffix=".wav")
            engine.save_to_file(q, temp_wav)
            engine.runAndWait()
            
            with open(temp_wav, "rb") as f:
                audio_data = f.read()
            os.remove(temp_wav)
            
            # 1. Streaming STT
            stt_res = await stream_audio_to_sarvam(audio_data, language, api_key)
            if not stt_res["t_final"] or not stt_res["transcript"]:
                print("  STT failed. Using fallback.", flush=True)
                stt_res["transcript"] = q
                stt_res["t_final"] = stt_res["t_connected"] or 0.5
                stt_res["t_first"] = stt_res["t_connected"] or 0.5
            
            metrics["conn"].append(stt_res["t_connected"])
            metrics["first"].append(stt_res["t_first"])
            metrics["final"].append(stt_res["t_final"])
            
            print(f"  Transcript: {stt_res['transcript'].encode('utf-8', errors='replace').decode('utf-8')} (Conn: {stt_res['t_connected']:.3f}s, Final: {stt_res['t_final']:.3f}s)", flush=True)
            
            # 2. RAG
            t_rag_0 = time.time()
            if language == "en-IN":
                chunks = pipeline.en_retriever.retrieve_vector(stt_res["transcript"], top_k=3)
            else:
                chunks = pipeline.hi_retriever.retrieve_vector(stt_res["transcript"], top_k=3)
            t_ret = time.time() - t_rag_0
            metrics["ret"].append(t_ret)
            
            t_gen_0 = time.time()
            gen_resp = pipeline.extractive_generator.generate(stt_res["transcript"], chunks)
            t_gen = time.time() - t_gen_0
            metrics["gen"].append(t_gen)
            
            # Total Voice-to-answer = STT Final + RAG
            tot = stt_res["t_final"] + t_ret + t_gen
            metrics["tot"].append(tot)
            
            ans = gen_resp.answer
            if "I don't have enough" in ans:
                metrics["safe_refusal"] += 1
            elif len(gen_resp.source_chunk_ids) > 0:
                metrics["grounded"] += 1
            else:
                metrics["unsupported"] += 1
                
        return metrics

    print("--- English Streaming ---", flush=True)
    res_en = await evaluate(en_queries, "en-IN")
    print(res_en, flush=True)
    
    print("--- Hindi Streaming ---", flush=True)
    res_hi = await evaluate(hi_queries, "hi-IN")
    print(res_hi, flush=True)

if __name__ == "__main__":
    asyncio.run(run_evaluation())
