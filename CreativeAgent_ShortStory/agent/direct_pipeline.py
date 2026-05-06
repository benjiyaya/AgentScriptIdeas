"""
Creative Story Agent - Direct Pipeline Runner
Runs the pipeline without LLM reasoning - direct tool execution.

Usage:
    python direct_pipeline.py --scene 1           # Full pipeline for scene 1
    python direct_pipeline.py --scene 1-3         # Scenes 1-3
    python direct_pipeline.py --all               # All 10 scenes
    python direct_pipeline.py --status            # Check status
    python direct_pipeline.py --gen-bg 3          # Only generate background
    python direct_pipeline.py --submit 3          # Only submit video job
    python direct_pipeline.py --poll 3            # Only poll video status
"""

import json
import time
import base64
import random
import argparse
import os
import sys
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
WAN_MODEL = "wan2.7-i2v"

BASE_DIR = Path(__file__).parent.parent
GENERATED_DIR = BASE_DIR / "generated"
CHARS_DIR = GENERATED_DIR / "characters"
BGS_DIR = GENERATED_DIR / "backgrounds"
VIDEOS_DIR = GENERATED_DIR / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR = BASE_DIR / "agent" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# ── Scene Data (same as agent/main.py) ─────────────────────────────────
# Import from the agent module
sys.path.insert(0, str(BASE_DIR / "agent"))
from story_agent import SCENES


def generate_background(scene_num: int) -> str:
    """Generate background via ComfyUI."""
    if scene_num not in SCENES:
        return f"Invalid scene: {scene_num}"

    scene = SCENES[scene_num]
    bg_path = BGS_DIR / scene["start_bg"]

    if bg_path.exists():
        return f"SKIP: {bg_path} already exists"

    seed = random.randint(0, 2**63)
    workflow = {
        "9": {"inputs": {"filename_prefix": f"moonlight/{scene['start_bg'].replace('.png','')}",
                          "images": ["65", 0]}, "class_type": "SaveImage"},
        "62": {"inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2",
                          "device": "default"}, "class_type": "CLIPLoader"},
        "63": {"inputs": {"vae_name": "ae.safetensors"}, "class_type": "VAELoader"},
        "64": {"inputs": {"conditioning": ["67", 0]}, "class_type": "ConditioningZeroOut"},
        "65": {"inputs": {"samples": ["70", 0], "vae": ["63", 0]}, "class_type": "VAEDecode"},
        "66": {"inputs": {"unet_name": "Z-Image\\z-anime-distill-8step-bf16.safetensors",
                          "weight_dtype": "default"}, "class_type": "UNETLoader"},
        "67": {"inputs": {"text": scene["global_prompt"], "clip": ["62", 0]},
               "class_type": "CLIPTextEncode"},
        "68": {"inputs": {"width": 1216, "height": 832, "batch_size": 1},
               "class_type": "EmptySD3LatentImage"},
        "69": {"inputs": {"shift": 3, "model": ["66", 0]},
               "class_type": "ModelSamplingAuraFlow"},
        "70": {"inputs": {"seed": seed, "steps": 8, "cfg": 1.0,
                          "sampler_name": "euler_ancestral", "scheduler": "beta",
                          "denoise": 1, "model": ["69", 0], "positive": ["67", 0],
                          "negative": ["64", 0], "latent_image": ["68", 0]},
               "class_type": "KSampler"},
    }

    data = json.dumps({"prompt": workflow, "client_id": "moonlight-direct"}).encode()
    req = urllib.request.Request(f"{COMFYUI_URL}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req).read())
        prompt_id = resp["prompt_id"]
        print(f"  ComfyUI queued: {prompt_id[:12]}... (seed={seed})")

        for i in range(150):
            time.sleep(2)
            hist = json.loads(urllib.request.urlopen(
                f"{COMFYUI_URL}/history/{prompt_id}").read())
            if prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                if status.get("completed", False):
                    for _, node_out in hist[prompt_id].get("outputs", {}).items():
                        for img in node_out.get("images", []):
                            params = urllib.parse.urlencode({
                                "filename": img["filename"],
                                "subfolder": img.get("subfolder", ""),
                                "type": img.get("type", "output"),
                            })
                            urllib.request.urlretrieve(
                                f"{COMFYUI_URL}/view?{params}", bg_path)
                    size_kb = bg_path.stat().st_size / 1024
                    return f"OK: {bg_path} ({size_kb:.0f} KB)"
                if status.get("status_str") == "error":
                    return f"ERROR: {status.get('messages', [])}"
            if i % 25 == 0:
                print(f"  ... waiting ({i*2}s)")
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"


def submit_video(scene_num: int) -> str:
    """Submit video job to Wan 2.7 API."""
    if scene_num not in SCENES:
        return f"Invalid scene: {scene_num}"
    if not DASHSCOPE_API_KEY:
        return "ERROR: DASHSCOPE_API_KEY not set"

    scene = SCENES[scene_num]
    bg_path = BGS_DIR / scene["start_bg"]
    if not bg_path.exists():
        return f"ERROR: Background not found: {bg_path}"

    with open(bg_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    body = {
        "model": WAN_MODEL,
        "input": {
            "prompt": scene["global_prompt"],
            "media": [{"type": "first_frame", "data": f"data:image/png;base64,{img_b64}"}],
            "parameters": {"size": "720P", "duration": 5},
        },
    }

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }

    req = urllib.request.Request(
        f"{DASHSCOPE_BASE_URL}/services/aigc/video-generation/generation",
        data=json.dumps(body).encode(), headers=headers)

    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        task_id = resp.get("output", {}).get("task_id", "")
        if not task_id:
            return f"ERROR: {json.dumps(resp)[:300]}"

        scene_id = f"scene{scene_num:02d}_{scene['name'].lower().replace(' ', '_')}"
        state_file = STATE_DIR / f"{scene_id}_task.json"
        state_file.write_text(json.dumps({
            "scene_num": scene_num, "task_id": task_id,
            "request_id": resp.get("request_id", ""),
            "status": "submitted", "created_at": datetime.now().isoformat(),
        }, indent=2))
        return f"OK: task_id={task_id}"
    except urllib.error.HTTPError as e:
        return f"ERROR: HTTP {e.code} - {e.read().decode()[:300]}"
    except Exception as e:
        return f"ERROR: {e}"


def poll_video(scene_num: int) -> str:
    """Poll video job and download if complete."""
    if scene_num not in SCENES:
        return f"Invalid scene: {scene_num}"
    if not DASHSCOPE_API_KEY:
        return "ERROR: DASHSCOPE_API_KEY not set"

    scene = SCENES[scene_num]
    scene_id = f"scene{scene_num:02d}_{scene['name'].lower().replace(' ', '_')}"
    state_file = STATE_DIR / f"{scene_id}_task.json"
    if not state_file.exists():
        return "ERROR: No task found. Submit first."

    task_data = json.loads(state_file.read_text())
    task_id = task_data["task_id"]

    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
    req = urllib.request.Request(
        f"{DASHSCOPE_BASE_URL}/tasks/{task_id}", headers=headers)

    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        output = resp.get("output", {})
        status = output.get("task_status", "UNKNOWN")

        task_data["status"] = status
        task_data["last_checked"] = datetime.now().isoformat()
        state_file.write_text(json.dumps(task_data, indent=2))

        if status == "SUCCEEDED":
            video_url = output.get("video_url", "")
            if not video_url:
                return f"ERROR: Succeeded but no video_url"
            video_path = VIDEOS_DIR / f"{scene_id}.mp4"
            urllib.request.urlretrieve(video_url, video_path)
            size_mb = video_path.stat().st_size / (1024 * 1024)
            return f"DOWNLOADED: {video_path} ({size_mb:.1f} MB)"
        elif status == "FAILED":
            return f"FAILED: {output.get('message', 'unknown')}"
        else:
            return f"RUNNING: {status} - {output.get('progress', '')}"
    except Exception as e:
        return f"ERROR: {e}"


def run_full_pipeline(scene_num: int) -> str:
    """Run complete pipeline: bg gen -> submit -> poll -> download."""
    scene = SCENES[scene_num]
    scene_id = f"scene{scene_num:02d}_{scene['name'].lower().replace(' ', '_')}"
    video_path = VIDEOS_DIR / f"{scene_id}.mp4"

    if video_path.exists():
        return f"SKIP: Video already exists: {video_path}"

    print(f"\n{'='*50}")
    print(f"  Scene {scene_num}: {scene['name']}")
    print(f"{'='*50}")

    # Step 1: Background
    print(f"\n[1/3] Generating background...")
    result = generate_background(scene_num)
    print(f"  Result: {result}")
    if "ERROR" in result:
        return f"Pipeline stopped at background generation: {result}"

    # Step 2: Submit
    print(f"\n[2/3] Submitting to Wan 2.7 API...")
    result = submit_video(scene_num)
    print(f"  Result: {result}")
    if "ERROR" in result:
        return f"Pipeline stopped at video submission: {result}"

    # Step 3: Poll
    print(f"\n[3/3] Polling for video (every 30s, max 30 min)...")
    for i in range(60):
        time.sleep(30)
        result = poll_video(scene_num)
        print(f"  [{i*30}s] {result}")
        if "DOWNLOADED" in result or "FAILED" in result:
            return f"Pipeline {'complete' if 'DOWNLOADED' in result else 'failed'}: {result}"

    return "Pipeline TIMEOUT after 30 minutes"


def show_status():
    """Show status of all scenes."""
    print(f"\n{'='*70}")
    print(f"  Moonlight Exchange - Scene Status")
    print(f"{'='*70}")
    for num, scene in SCENES.items():
        bg_path = BGS_DIR / scene["start_bg"]
        scene_id = f"scene{num:02d}_{scene['name'].lower().replace(' ', '_')}"
        video_path = VIDEOS_DIR / f"{scene_id}.mp4"
        task_file = STATE_DIR / f"{scene_id}_task.json"

        bg = "OK" if bg_path.exists() else "MISSING"
        vid = "DONE" if video_path.exists() else "pending"
        task_status = ""
        if task_file.exists():
            td = json.loads(task_file.read_text())
            task_status = f" | Task: {td.get('status', '?')}"

        segs = len(scene["local_prompts"].split("|"))
        print(f"  Scene {num:2d}: {scene['name']:<25s} | BG:{bg:<7s} | "
              f"Video:{vid:<7s} | Segs:{segs}{task_status}")
    print(f"{'='*70}\n")


def parse_scenes(range_str: str) -> list:
    scenes = []
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            s, e = part.split("-", 1)
            scenes.extend(range(int(s), int(e) + 1))
        else:
            scenes.append(int(part))
    return [s for s in scenes if 1 <= s <= 10]


def main():
    parser = argparse.ArgumentParser(
        description="Moonlight Exchange - Direct Pipeline Runner")
    parser.add_argument("--scene", type=str, help="Scene(s): '1', '1-3', '1,3,5'")
    parser.add_argument("--all", action="store_true", help="All 10 scenes")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--gen-bg", type=int, help="Generate background only")
    parser.add_argument("--submit", type=int, help="Submit video job only")
    parser.add_argument("--poll", type=int, help="Poll video status only")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.gen_bg:
        result = generate_background(args.gen_bg)
        print(result)
    elif args.submit:
        result = submit_video(args.submit)
        print(result)
    elif args.poll:
        result = poll_video(args.poll)
        print(result)
    elif args.all:
        for num in range(1, 11):
            result = run_full_pipeline(num)
            print(f"\n  -> {result}")
    elif args.scene:
        for num in parse_scenes(args.scene):
            result = run_full_pipeline(num)
            print(f"\n  -> {result}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
