#!/usr/bin/env bash
# =============================================================================
# ELMM Ablation Experiment: 4-Configuration Comparison
# =============================================================================
# C1: No-SD baseline (vLLM native UVA offloading)
# C2: EAGLE-3 + UVA (vanilla speculative decoding, no ELMM)
# C3: EAGLE-3 + ELMM (cache only, no prefetch)
# C4: EAGLE-3 + ELMM (cache + draft-guided prefetch)
#
# Each config runs the vLLM server, sends a benchmark workload, measures
# throughput (tok/s), latency, and acceptance rate, then shuts down.
# =============================================================================
set -euo pipefail

MODEL_DIR="/root/models/Qwen3-30B-A3B-Instruct-2507"
EAGLE_DIR="/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
CPU_OFFLOAD_GB=30
PORT=8000
BASE_URL="http://127.0.0.1:${PORT}"
NUM_PROMPTS=20
MAX_TOKENS=200
RESULTS_DIR="/root/MoE-SD/results/elmm_ablation"
LOG_DIR="/root/MoE-SD/logs/elmm_ablation"
WARMUP_PROMPTS=3

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ----- helper functions -----
wait_for_server() {
    local url="${BASE_URL}/v1/models"
    local max_wait=600
    local waited=0
    echo "  Waiting for server at ${url} ..."
    while ! curl -sf "$url" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ "$waited" -ge "$max_wait" ]; then
            echo "  ERROR: server did not start within ${max_wait}s"
            return 1
        fi
    done
    echo "  Server ready after ${waited}s"
}

kill_server() {
    echo "  Shutting down server..."
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 5
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 2
}

run_benchmark() {
    local label="$1"
    local output_file="${RESULTS_DIR}/${label}.jsonl"
    local summary_file="${RESULTS_DIR}/${label}_summary.json"

    echo "  Running benchmark: ${label} (${NUM_PROMPTS} prompts, max_tokens=${MAX_TOKENS})"

    # Warmup
    for i in $(seq 1 $WARMUP_PROMPTS); do
        curl -sf "${BASE_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$(basename $MODEL_DIR)\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Hello, how are you?\"}],
                \"max_tokens\": 50,
                \"chat_template_kwargs\": {\"enable_thinking\": false}
            }" > /dev/null 2>&1 || true
    done
    echo "  Warmup done"

    # Actual benchmark: send prompts sequentially, measure per-request metrics
    python3 - "$output_file" "$summary_file" <<'PYEOF'
import json, sys, time, requests

output_file = sys.argv[1]
summary_file = sys.argv[2]
base_url = "BASE_URL_PLACEHOLDER"
model_name = "MODEL_NAME_PLACEHOLDER"
num_prompts = NUM_PROMPTS_PLACEHOLDER
max_tokens = MAX_TOKENS_PLACEHOLDER

prompts = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute Fibonacci numbers efficiently.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis step by step.",
    "How does a neural network learn from training data?",
    "Compare and contrast democracy and authoritarianism.",
    "Explain how a CPU cache hierarchy works and why it matters.",
    "What is the significance of the Turing test in AI?",
    "Describe the major causes and effects of climate change.",
    "How do hash tables work and what is their time complexity?",
    "Explain the CAP theorem in distributed systems.",
    "What are the key principles of object-oriented programming?",
    "Describe how HTTPS ensures secure communication.",
    "What is quantum entanglement and why is it important?",
    "How does garbage collection work in modern programming languages?",
    "Explain the concept of recursion with a practical example.",
    "What are the advantages of microservices over monolithic architecture?",
    "How does CRISPR gene editing technology work?",
    "Describe the working principle of a transformer neural network.",
    "What are the ethical considerations of artificial intelligence?",
]

results = []
total_tokens = 0
total_time = 0.0

for i in range(min(num_prompts, len(prompts))):
    prompt = prompts[i % len(prompts)]
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        tps = completion_tokens / elapsed if elapsed > 0 else 0
        results.append({
            "prompt_idx": i,
            "completion_tokens": completion_tokens,
            "elapsed_s": round(elapsed, 3),
            "tps": round(tps, 3),
        })
        total_tokens += completion_tokens
        total_time += elapsed
        print(f"    [{i+1}/{num_prompts}] {completion_tokens} tok, {elapsed:.2f}s, {tps:.1f} tok/s")
    except Exception as e:
        print(f"    [{i+1}/{num_prompts}] ERROR: {e}")
        results.append({"prompt_idx": i, "error": str(e)})

# Write per-request results
with open(output_file, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

# Write summary
avg_tps = total_tokens / total_time if total_time > 0 else 0
summary = {
    "total_tokens": total_tokens,
    "total_time_s": round(total_time, 3),
    "avg_tps": round(avg_tps, 3),
    "num_requests": len([r for r in results if "error" not in r]),
    "errors": len([r for r in results if "error" in r]),
}
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Summary: {summary['avg_tps']} tok/s, {summary['total_tokens']} tokens in {summary['total_time_s']}s")
PYEOF

    # Patch placeholders
    sed -i "s|BASE_URL_PLACEHOLDER|${BASE_URL}|g" /dev/null 2>/dev/null || true
}

run_benchmark_python() {
    local label="$1"
    local output_file="${RESULTS_DIR}/${label}.jsonl"
    local summary_file="${RESULTS_DIR}/${label}_summary.json"

    echo "  Running benchmark: ${label} (${NUM_PROMPTS} prompts, max_tokens=${MAX_TOKENS})"

    python3 /root/MoE-SD/scripts/elmm_bench_worker.py \
        --base-url "${BASE_URL}" \
        --model "$(basename $MODEL_DIR)" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-tokens "${MAX_TOKENS}" \
        --warmup "${WARMUP_PROMPTS}" \
        --output "${output_file}" \
        --summary "${summary_file}"
}

# =============================================================================
# Configuration C1: No-SD baseline
# =============================================================================
run_config() {
    local label="$1"
    shift
    local server_args=("$@")

    echo ""
    echo "============================================================"
    echo "  Configuration: ${label}"
    echo "  Server args: ${server_args[*]}"
    echo "============================================================"

    kill_server

    echo "  Starting server..."
    VLLM_PLUGINS="${VLLM_PLUGINS:-}" \
    python3 -m vllm.entrypoints.openai.api_server \
        "${server_args[@]}" \
        > "${LOG_DIR}/${label}_server.log" 2>&1 &

    if ! wait_for_server; then
        echo "  FAILED: server didn't start. Check ${LOG_DIR}/${label}_server.log"
        kill_server
        return 1
    fi

    run_benchmark_python "${label}"
    kill_server
    echo "  Config ${label} complete."
}

echo "Starting ELMM ablation experiment at $(date)"
echo "Model: ${MODEL_DIR}"
echo "Eagle: ${EAGLE_DIR}"
echo "CPU offload: ${CPU_OFFLOAD_GB} GB"
echo ""

# --- C1: No-SD baseline ---
VLLM_PLUGINS="" \
run_config "C1_no_sd" \
    --model "${MODEL_DIR}" \
    --cpu-offload-gb "${CPU_OFFLOAD_GB}" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --port "${PORT}" \
    --disable-log-requests

# --- C2: EAGLE-3 + UVA (vanilla, no ELMM) ---
VLLM_PLUGINS="" \
run_config "C2_eagle3_uva" \
    --model "${MODEL_DIR}" \
    --speculative-model "${EAGLE_DIR}" \
    --num-speculative-tokens 3 \
    --speculative-draft-tensor-parallel-size 1 \
    --cpu-offload-gb "${CPU_OFFLOAD_GB}" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --port "${PORT}" \
    --disable-log-requests

# --- C3: EAGLE-3 + ELMM (cache only, no prefetch) ---
VLLM_PLUGINS="elmm" \
ELMM_CACHE_GB=8 \
ELMM_PREFETCH=0 \
ELMM_LOG_INTERVAL=50 \
run_config "C3_eagle3_elmm_cache" \
    --model "${MODEL_DIR}" \
    --speculative-model "${EAGLE_DIR}" \
    --num-speculative-tokens 3 \
    --speculative-draft-tensor-parallel-size 1 \
    --cpu-offload-gb "${CPU_OFFLOAD_GB}" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --port "${PORT}" \
    --disable-log-requests

# --- C4: EAGLE-3 + ELMM (cache + prefetch) ---
VLLM_PLUGINS="elmm" \
ELMM_CACHE_GB=8 \
ELMM_PREFETCH=1 \
ELMM_LOG_INTERVAL=50 \
run_config "C4_eagle3_elmm_full" \
    --model "${MODEL_DIR}" \
    --speculative-model "${EAGLE_DIR}" \
    --num-speculative-tokens 3 \
    --speculative-draft-tensor-parallel-size 1 \
    --cpu-offload-gb "${CPU_OFFLOAD_GB}" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --port "${PORT}" \
    --disable-log-requests

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "  ELMM Ablation Results Summary"
echo "============================================================"
for f in "${RESULTS_DIR}"/*_summary.json; do
    label=$(basename "$f" _summary.json)
    if [ -f "$f" ]; then
        tps=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('avg_tps', 'N/A'))")
        echo "  ${label}: ${tps} tok/s"
    fi
done
echo ""
echo "Done at $(date)"
