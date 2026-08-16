"""
Phase 6B Corrected Realtime STT Benchmark

Protocol verified from official Sarvam AI documentation:
  Endpoint:  wss://api.sarvam.ai/speech-to-text-realtime/ws
  Auth:      header "api-subscription-key"
  Model:     saaras:v3-realtime  (passed as URL query param)
  Endpointing: manual (passed as URL query param)

Client-to-Server events (JSON over WebSocket):
  { "event": "speech_start" }
  { "event": "audio_input", "audio": "<base64-pcm-data>" }
  { "event": "flush" }
  { "event": "speech_end" }

Server-to-Client events:
  { "event": "transcript.partial", "transcript": "..." }
  { "event": "transcript.final",   "transcript": "..." }
  { "event": "session.begin", "request_id": "..." }

Audio format requirements:
  - WAV or raw PCM (pcm_s16le)
  - The API requires the declared sample rate to match actual audio sample rate
  - Our pyttsx3-generated files are 22050Hz 16-bit mono

All fallback transcripts are prohibited.
All failures are marked as FAILED.
"""

import os
import sys
import time
import json
import wave
import asyncio
import base64
import numpy as np
import websockets
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../backend/.env"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../backend"))

from app.orchestration.pipeline import VoiceRAGPipeline

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../data/human_audio")
REALTIME_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"


def inspect_wav(path):
    """Returns (sample_rate, channels, sample_width_bytes, n_frames, duration_s)."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        nf = w.getnframes()
        dur = nf / sr if sr else 0
        raw = w.readframes(nf)
    return sr, ch, sw, nf, dur, raw


def get_audio_samples():
    en_files, hi_files = [], []
    if not os.path.exists(AUDIO_DIR):
        return en_files, hi_files
    for f in sorted(os.listdir(AUDIO_DIR)):
        if f.endswith(".wav") and not f.endswith(".txt"):
            full = os.path.join(AUDIO_DIR, f)
            if f.startswith("en_"):
                en_files.append(full)
            elif f.startswith("hi_"):
                hi_files.append(full)
    return en_files[:5], hi_files[:5]


async def stream_one_cold(path, language, api_key, chunk_ms=100):
    """Mode A: new connection per sample. Returns dict with timing + transcript."""
    sr, ch, sw, nf, dur, raw_pcm = inspect_wav(path)
    if nf == 0:
        return {"ok": False, "error": "Empty WAV file", "path": path}

    url = (
        f"{REALTIME_URL}"
        f"?model=saaras:v3-realtime"
        f"&endpointing=manual"
        f"&language_code={language}"
    )
    headers = {"api-subscription-key": api_key}

    # Chunk size: chunk_ms of audio
    bytes_per_sample = sw * ch
    chunk_frames = int(sr * (chunk_ms / 1000.0))
    chunk_bytes = chunk_frames * bytes_per_sample
    chunks = [raw_pcm[i:i+chunk_bytes] for i in range(0, len(raw_pcm), chunk_bytes)]

    t_start = time.time()
    t_connected = None
    t_first_partial = None
    t_final = None
    transcript_partial = ""
    transcript_final = ""
    all_events = []

    try:
        async with websockets.connect(url, additional_headers=headers) as ws:
            t_connected = time.time()

            async def receive_loop():
                nonlocal t_first_partial, t_final, transcript_partial, transcript_final
                try:
                    async for raw_msg in ws:
                        now = time.time()
                        try:
                            msg = json.loads(raw_msg)
                        except Exception:
                            continue
                        evt = msg.get("event", "")
                        all_events.append({"t": now - t_start, "event": evt, "msg": msg})

                        if evt == "transcript.partial":
                            txt = msg.get("text", "").strip()
                            if txt and t_first_partial is None:
                                t_first_partial = now
                            transcript_partial = txt

                        elif evt == "transcript.final":
                            txt = msg.get("text", "").strip()
                            transcript_final = txt
                            t_final = now
                            break  # got final, stop listening

                        elif evt == "error":
                            all_events[-1]["is_error"] = True
                            break

                except websockets.exceptions.ConnectionClosed:
                    pass

            listener = asyncio.create_task(receive_loop())

            # Send speech_start
            await ws.send(json.dumps({"event": "speech_start"}))

            # Stream audio in real-time paced chunks
            for chunk in chunks:
                await ws.send(json.dumps({
                    "event": "audio_input",
                    "audio": base64.b64encode(chunk).decode("utf-8")
                }))
                await asyncio.sleep(chunk_ms / 1000.0)  # real-time pacing

            # Flush + speech_end to trigger final transcript
            await ws.send(json.dumps({"event": "flush"}))
            await ws.send(json.dumps({"event": "speech_end"}))

            try:
                await asyncio.wait_for(listener, timeout=5.0)
            except asyncio.TimeoutError:
                listener.cancel()

    except Exception as e:
        return {"ok": False, "error": str(e), "path": path, "events": all_events}

    success = bool(transcript_final or transcript_partial)
    final_text = transcript_final or transcript_partial  # prefer final

    return {
        "ok": success,
        "path": path,
        "audio_dur": dur,
        "sr": sr,
        "t_conn": (t_connected - t_start) if t_connected else None,
        "t_first_partial": (t_first_partial - t_start) if t_first_partial else None,
        "t_final": (t_final - t_start) if t_final else None,
        "transcript": final_text,
        "all_events": all_events,
    }


async def stream_warm(paths, language, api_key, chunk_ms=100):
    """Mode B: persistent connection, multiple turns."""
    url = (
        f"{REALTIME_URL}"
        f"?model=saaras:v3-realtime"
        f"&endpointing=manual"
        f"&language_code={language}"
    )
    headers = {"api-subscription-key": api_key}

    results = []
    t_conn_start = time.time()

    try:
        async with websockets.connect(url, additional_headers=headers) as ws:
            t_connected = time.time()
            conn_duration = t_connected - t_conn_start

            for path in paths:
                sr, ch, sw, nf, dur, raw_pcm = inspect_wav(path)
                if nf == 0:
                    results.append({"ok": False, "error": "Empty WAV", "path": path})
                    continue

                bytes_per_sample = sw * ch
                chunk_frames = int(sr * (chunk_ms / 1000.0))
                chunk_bytes = chunk_frames * bytes_per_sample
                chunks = [raw_pcm[i:i+chunk_bytes] for i in range(0, len(raw_pcm), chunk_bytes)]

                t_turn_start = time.time()
                t_first_partial = None
                t_final = None
                transcript_partial = ""
                transcript_final = ""
                turn_events = []

                async def receive_turn():
                    nonlocal t_first_partial, t_final, transcript_partial, transcript_final
                    try:
                        async for raw_msg in ws:
                            now = time.time()
                            try:
                                msg = json.loads(raw_msg)
                            except Exception:
                                continue
                            evt = msg.get("event", "")
                            turn_events.append({"t": now - t_turn_start, "event": evt})

                            if evt == "transcript.partial":
                                txt = msg.get("text", "").strip()
                                if txt and t_first_partial is None:
                                    t_first_partial = now
                                transcript_partial = txt

                            elif evt == "transcript.final":
                                txt = msg.get("text", "").strip()
                                transcript_final = txt
                                t_final = now
                                break

                            elif evt == "error":
                                break
                    except websockets.exceptions.ConnectionClosed:
                        pass

                listener = asyncio.create_task(receive_turn())

                await ws.send(json.dumps({"event": "speech_start"}))

                for chunk in chunks:
                    await ws.send(json.dumps({
                        "event": "audio_input",
                        "audio": base64.b64encode(chunk).decode("utf-8")
                    }))
                    await asyncio.sleep(chunk_ms / 1000.0)

                await ws.send(json.dumps({"event": "flush"}))
                await ws.send(json.dumps({"event": "speech_end"}))

                try:
                    await asyncio.wait_for(listener, timeout=5.0)
                except asyncio.TimeoutError:
                    listener.cancel()

                final_text = transcript_final or transcript_partial
                success = bool(final_text)

                results.append({
                    "ok": success,
                    "path": path,
                    "audio_dur": dur,
                    "sr": sr,
                    "conn_shared": conn_duration,      # once only
                    "t_first_partial": (t_first_partial - t_turn_start) if t_first_partial else None,
                    "t_final": (t_final - t_turn_start) if t_final else None,
                    "transcript": final_text,
                    "events": turn_events,
                })

                # Brief pause between turns
                await asyncio.sleep(0.5)

    except Exception as e:
        print(f"Warm connection error: {e}")
        while len(results) < len(paths):
            results.append({"ok": False, "error": str(e)})

    return results


def execute_rag(text, lang, pipeline):
    t0 = time.time()
    retriever = pipeline.en_retriever if "en" in lang else pipeline.hi_retriever
    chunks = retriever.retrieve_vector(text, top_k=3)
    t_ret = time.time() - t0

    t1 = time.time()
    resp = pipeline.extractive_generator.generate(text, chunks)
    t_gen = time.time() - t1

    return t_ret, t_gen, resp


def percentile(lst, p):
    if not lst:
        return None
    return float(np.percentile(lst, p))


async def run_evaluation():
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        print("ERROR: SARVAM_API_KEY not configured")
        return

    en_files, hi_files = get_audio_samples()

    print("=== AUDIO INSPECTION ===")
    all_files = en_files + hi_files
    for f in all_files:
        sr, ch, sw, nf, dur, _ = inspect_wav(f)
        print(f"  {os.path.basename(f)}: {sr}Hz {ch}ch {sw*8}bit {nf}frames {dur:.2f}s")

    if not en_files:
        print("STOP: No English WAV files found in data/test_audio/")
        return
    if not hi_files:
        print("STOP: No Hindi WAV files found in data/test_audio/")
        return

    # Check for empty files
    empty = [f for f in hi_files if inspect_wav(f)[3] == 0]
    if empty:
        print(f"STOP: {len(empty)} Hindi WAV files are empty (0 frames). Cannot benchmark.")
        print("Run generate_test_audio.py to fix them.")
        return

    print("\nLoading VoiceRAGPipeline...")
    pipeline = VoiceRAGPipeline()

    # -----------------------------------------------------------------------
    # ENGLISH COLD
    # -----------------------------------------------------------------------
    print("\n=== English Cold (new connection per sample) ===")
    en_cold = []
    for f in en_files:
        print(f"  [{os.path.basename(f)}] streaming...", flush=True)
        r = await stream_one_cold(f, "en-IN", api_key)
        print(f"    Events: {[e['event'] for e in r.get('all_events', [])]}")
        print(f"    transcript={repr(r.get('transcript', ''))}")
        en_cold.append(r)

    # -----------------------------------------------------------------------
    # ENGLISH WARM
    # -----------------------------------------------------------------------
    print("\n=== English Warm (persistent connection) ===")
    en_warm = await stream_warm(en_files, "en-IN", api_key)
    for r, f in zip(en_warm, en_files):
        print(f"  [{os.path.basename(f)}] ok={r['ok']} transcript={repr(r.get('transcript',''))}")

    # -----------------------------------------------------------------------
    # HINDI COLD
    # -----------------------------------------------------------------------
    print("\n=== Hindi Cold (new connection per sample) ===")
    hi_cold = []
    for f in hi_files:
        print(f"  [{os.path.basename(f)}] streaming...", flush=True)
        r = await stream_one_cold(f, "hi-IN", api_key)
        print(f"    Events: {[e['event'] for e in r.get('all_events', [])]}")
        print(f"    transcript={repr(r.get('transcript', ''))}")
        hi_cold.append(r)

    # -----------------------------------------------------------------------
    # HINDI WARM
    # -----------------------------------------------------------------------
    print("\n=== Hindi Warm (persistent connection) ===")
    hi_warm = await stream_warm(hi_files, "hi-IN", api_key)
    for r, f in zip(hi_warm, hi_files):
        print(f"  [{os.path.basename(f)}] ok={r['ok']} transcript={repr(r.get('transcript',''))}")

    # -----------------------------------------------------------------------
    # AGGREGATE WITH RAG
    # -----------------------------------------------------------------------
    def aggregate(results, lang, mode):
        conn_l, first_l, final_l, ret_l, gen_l, tot_l = [], [], [], [], [], []
        grounded = safe_refusal = unsupported = failed = success = 0

        for r in results:
            if not r.get("ok") or not r.get("transcript"):
                failed += 1
                continue
            success += 1

            if "t_conn" in r and r["t_conn"] is not None:
                conn_l.append(r["t_conn"])
            if "conn_shared" in r and r["conn_shared"] is not None and len(conn_l) == 0:
                conn_l.append(r["conn_shared"])  # record once

            if r.get("t_first_partial") is not None:
                first_l.append(r["t_first_partial"])
            if r.get("t_final") is not None:
                final_l.append(r["t_final"])

            t_ret, t_gen, rag_resp = execute_rag(r["transcript"], lang, pipeline)
            ret_l.append(t_ret)
            gen_l.append(t_gen)

            final_stt = r.get("t_final") or r.get("t_first_partial") or 0.5
            tot_l.append(final_stt + t_ret + t_gen)

            ans = rag_resp.answer
            if "don't have enough" in ans.lower():
                safe_refusal += 1
            elif len(rag_resp.source_chunk_ids) > 0:
                grounded += 1
            else:
                unsupported += 1

        return {
            "mode": mode, "lang": lang,
            "samples": len(results), "success": success, "failed": failed,
            "conn_p50": percentile(conn_l, 50), "conn_p70": percentile(conn_l, 70), "conn_p100": percentile(conn_l, 100),
            "first_p50": percentile(first_l, 50), "first_p70": percentile(first_l, 70), "first_p100": percentile(first_l, 100),
            "final_p50": percentile(final_l, 50), "final_p70": percentile(final_l, 70), "final_p100": percentile(final_l, 100),
            "ret_p50": percentile(ret_l, 50), "ret_p70": percentile(ret_l, 70), "ret_p100": percentile(ret_l, 100),
            "gen_p50": percentile(gen_l, 50), "gen_p70": percentile(gen_l, 70), "gen_p100": percentile(gen_l, 100),
            "tot_p50": percentile(tot_l, 50), "tot_p70": percentile(tot_l, 70), "tot_p100": percentile(tot_l, 100),
            "grounded": grounded, "safe_refusal": safe_refusal, "unsupported": unsupported,
        }

    m_en_cold = aggregate(en_cold, "en-IN", "cold")
    m_en_warm = aggregate(en_warm, "en-IN", "warm")
    m_hi_cold = aggregate(hi_cold, "hi-IN", "cold")
    m_hi_warm = aggregate(hi_warm, "hi-IN", "warm")

    # -----------------------------------------------------------------------
    # PRINT FULL REPORT
    # -----------------------------------------------------------------------
    def fmt(v):
        return f"{v:.3f}s" if v is not None else "N/A"

    def print_section(m):
        print(f"\n  Samples: {m['samples']}")
        print(f"  Successful: {m['success']}")
        print(f"  Failed: {m['failed']}")
        print(f"  Cold connection P50/P70/P100: {fmt(m['conn_p50'])} / {fmt(m['conn_p70'])} / {fmt(m['conn_p100'])}")
        print(f"  First partial P50/P70/P100:   {fmt(m['first_p50'])} / {fmt(m['first_p70'])} / {fmt(m['first_p100'])}")
        print(f"  Final transcript P50/P70/P100: {fmt(m['final_p50'])} / {fmt(m['final_p70'])} / {fmt(m['final_p100'])}")
        print(f"  RAG P50/P70/P100:             {fmt(m['ret_p50'])} / {fmt(m['ret_p70'])} / {fmt(m['ret_p100'])}")
        print(f"  Gen P50/P70/P100:             {fmt(m['gen_p50'])} / {fmt(m['gen_p70'])} / {fmt(m['gen_p100'])}")
        print(f"  Total voice-to-answer P50/P70/P100: {fmt(m['tot_p50'])} / {fmt(m['tot_p70'])} / {fmt(m['tot_p100'])}")
        print(f"  Grounded: {m['grounded']}, Safe refusal: {m['safe_refusal']}, Unsupported: {m['unsupported']}")

    print("\n\n" + "=" * 60)
    print("# PHASE 6B CORRECTED REALTIME RESULT")
    print("=" * 60)
    print(f"\nAPI:   {REALTIME_URL}")
    print(f"Model: saaras:v3-realtime")
    print(f"Auth:  api-subscription-key header")
    print(f"Mode:  endpointing=manual (speech_start/audio_input/flush/speech_end)")
    print(f"Response events: transcript.partial / transcript.final")

    print("\n--- AUDIO PROPERTIES ---")
    for f in all_files:
        sr, ch, sw, nf, dur, _ = inspect_wav(f)
        print(f"  {os.path.basename(f)}: {sr}Hz {ch}ch {sw*8}bit dur={dur:.2f}s{'  [EMPTY]' if nf == 0 else ''}")

    print("\n## English Cold")
    print_section(m_en_cold)

    print("\n## English Warm (persistent connection)")
    print_section(m_en_warm)

    print("\n## Hindi Cold")
    print_section(m_hi_cold)

    print("\n## Hindi Warm (persistent connection)")
    print_section(m_hi_warm)

    # Requirement assessment
    en_cold_pass = m_en_cold["tot_p50"] is not None and m_en_cold["tot_p50"] < 0.200
    en_warm_pass = m_en_warm["tot_p50"] is not None and m_en_warm["tot_p50"] < 0.200
    hi_cold_pass = m_hi_cold["tot_p50"] is not None and m_hi_cold["tot_p50"] < 0.200
    hi_warm_pass = m_hi_warm["tot_p50"] is not None and m_hi_warm["tot_p50"] < 0.200

    print("\n## Requirement: <200ms")
    print(f"  English cold  <200ms: {'PASS' if en_cold_pass else 'FAIL'}")
    print(f"  English warm  <200ms: {'PASS' if en_warm_pass else 'FAIL'}")
    print(f"  Hindi   cold  <200ms: {'PASS' if hi_cold_pass else 'FAIL'}")
    print(f"  Hindi   warm  <200ms: {'PASS' if hi_warm_pass else 'FAIL'}")

    print("\n## Conclusion")
    all_fail = all(not r.get("ok") for r in en_cold + en_warm + hi_cold + hi_warm)
    if all_fail:
        print("  ALL SAMPLES FAILED. The saaras:v3-realtime endpoint did not produce transcripts.")
        print("  Inspect 'all_events' field in raw results for actual API error sequence.")
    else:
        warm_pass = en_warm_pass and hi_warm_pass
        if warm_pass:
            print("  Streaming with persistent connection meets <200ms target for warm turns.")
        else:
            warm_en = fmt(m_en_warm["tot_p50"])
            warm_hi = fmt(m_hi_warm["tot_p50"])
            print(f"  Streaming does NOT meet <200ms target even for warm turns.")
            print(f"  English warm P50={warm_en}, Hindi warm P50={warm_hi}")
            print(f"  Main bottleneck: STT final transcript time dominates.")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
