import json, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('data/processed/rag_latency_benchmark.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

for r in results:
    lang = r['language']
    n = r['n_queries']
    ret = r['retrieval_ms']
    gen = r['generation_ms']
    grd = r['guardrail_ms']
    ser = r['serialization_ms']
    tot = r['total_rag_ms']
    q = r['quality']

    p50_ok = tot['P50'] < 200
    p70_ok = tot['P70'] < 200
    p100_ok = tot['P100'] < 200

    print(lang + " n=" + str(n))
    print("  Retrieval    P50=" + str(round(ret['P50'],1)) + "ms P70=" + str(round(ret['P70'],1)) + "ms P100=" + str(round(ret['P100'],1)) + "ms")
    print("  Generation   P50=" + str(round(gen['P50'],1)) + "ms P70=" + str(round(gen['P70'],1)) + "ms P100=" + str(round(gen['P100'],1)) + "ms")
    print("  Guardrails   P50=" + str(round(grd['P50'],3)) + "ms P70=" + str(round(grd['P70'],3)) + "ms P100=" + str(round(grd['P100'],3)) + "ms")
    print("  Serialization P50=" + str(round(ser['P50'],3)) + "ms P70=" + str(round(ser['P70'],3)) + "ms P100=" + str(round(ser['P100'],3)) + "ms")
    print("  TOTAL RAG    P50=" + str(round(tot['P50'],1)) + "ms P70=" + str(round(tot['P70'],1)) + "ms P100=" + str(round(tot['P100'],1)) + "ms")
    print("  <200ms       P50=" + ("PASS" if p50_ok else "FAIL") + " P70=" + ("PASS" if p70_ok else "FAIL") + " P100=" + ("PASS" if p100_ok else "FAIL"))
    grounded = q['grounded']
    safe_ref = q['safe_refusal']
    unsup = q['unsupported']
    incomp = q['incomplete']
    print("  Quality      Grounded=" + str(grounded) + " SafeRefusal=" + str(safe_ref) + " Unsupported=" + str(unsup) + " Incomplete=" + str(incomp))
    print()
