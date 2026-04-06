# =============================================================
# CELL 7a: Start LitServe server in background
# =============================================================
import subprocess, requests, time, base64, json
from PIL import Image
from io import BytesIO

# Kill any existing server
!kill $(lsof -t -i:8000) 2>/dev/null || true
time.sleep(2)

# Start with "speed" preset (fastest to load + demonstrate)
proc = subprocess.Popen(
    ["python", "server/serve.py", "--preset", "speed", "--port", "8000"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT  # merge stderr into stdout
)

print("Starting LitServe server with 'speed' preset...")
print("This loads SDXL + LCM-LoRA — takes 3-8 min on T4.\n")

# Poll the server by trying the /predict endpoint with a tiny request
# LitServe doesn't have /health — we just wait and try
server_ready = False
for i in range(480):  # 8 min timeout
    if proc.poll() is not None:
        print(f"\n❌ Server crashed (exit code {proc.poll()})")
        print(proc.stdout.read().decode()[-3000:])
        break
    try:
        r = requests.post(
            "http://localhost:8000/predict",
            json={"prompt": "test", "height": 64, "width": 64},
            timeout=3,
        )
        if r.status_code == 200:
            print(f"\n✅ Server ready after {i+1}s!")
            server_ready = True
            break
    except (requests.ConnectionError, requests.Timeout):
        pass
    time.sleep(1)
    if i % 60 == 59:
        print(f"  Still loading... ({i+1}s)")

if not server_ready and proc.poll() is None:
    print("\n⚠️ Server still loading — try the next cell in a minute")


# =============================================================
# CELL 7b: Test API — Single inference call (like curl)
# =============================================================
print("=" * 60)
print("API TEST: POST /predict")
print("=" * 60)

test_prompt = "a photo of an astronaut riding a horse on mars, cinematic lighting, detailed"

payload = {
    "prompt": test_prompt,
    "seed": 42,
    "height": 1024,
    "width": 1024,
}

print(f"\nRequest:")
print(f"  curl -X POST http://localhost:8000/predict \\")
print(f"    -H 'Content-Type: application/json' \\")
print(f"    -d '{json.dumps(payload)}'")
print()

response = requests.post(
    "http://localhost:8000/predict",
    json=payload,
    timeout=120,
)

data = response.json()
print(f"Response:")
print(f"  Status:  {response.status_code}")
print(f"  Latency: {data['latency_s']}s")
print(f"  Preset:  {data['preset']} — {data['preset_info']['description']}")
print(f"  Speedup: {data['preset_info']['expected_speedup']} vs baseline")
print(f"  Format:  {data['format']}")
print(f"  Image:   {len(data['image_base64'])} bytes (base64)")

img = Image.open(BytesIO(base64.b64decode(data["image_base64"])))
img.save("results/api_demo_speed.png")
print(f"\n🚀 Generated in {data['latency_s']}s:")
display(img)


# =============================================================
# CELL 7c: Multi-prompt batch test
# =============================================================
print("=" * 60)
print("BATCH TEST: 4 diverse prompts via API")
print("=" * 60)

test_prompts = [
    "a photo of an astronaut riding a horse on mars, cinematic lighting",
    "a beautiful sunset over a calm ocean with sailboats, oil painting style",
    "a corgi wearing a top hat and monocle, portrait photography",
    "a futuristic cityscape at night with neon lights, cyberpunk",
]

api_images = []
api_latencies = []

for i, prompt in enumerate(test_prompts):
    t0 = time.perf_counter()
    r = requests.post(
        "http://localhost:8000/predict",
        json={"prompt": prompt, "seed": 42},
        timeout=120,
    )
    total_time = time.perf_counter() - t0
    d = r.json()
    api_images.append(Image.open(BytesIO(base64.b64decode(d["image_base64"]))))
    api_latencies.append(d["latency_s"])
    print(f"  [{i+1}/4] {d['latency_s']}s — {prompt[:50]}...")

avg_latency = sum(api_latencies) / len(api_latencies)
throughput = 60 / avg_latency
print(f"\n📊 Average latency: {avg_latency:.2f}s/image")
print(f"📊 Throughput: {throughput:.0f} images/min")
print(f"📊 vs Baseline (~5.7s): {5.66/avg_latency:.1f}× faster")

# Display grid
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, img, prompt, lat in zip(axes, api_images, test_prompts, api_latencies):
    ax.imshow(img)
    ax.set_title(f"{lat}s\n{prompt[:35]}...", fontsize=9)
    ax.axis("off")
plt.suptitle(f"LitServe API — 'speed' preset | Avg: {avg_latency:.2f}s/img | {throughput:.0f} img/min", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/api_batch_demo.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 Saved to results/api_batch_demo.png")


# =============================================================
# CELL 7d: Cleanup
# =============================================================
proc.terminate()
proc.wait()
print("✅ Server stopped.")
