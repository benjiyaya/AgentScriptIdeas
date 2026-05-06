"""
Moonlight Exchange — LangGraph AI Agent
Orchestrates: Story → Image Generation (ComfyUI) → Video Generation (Wan 2.7 API)

The agent manages the full pipeline:
1. Parse scene definitions from story config
2. Generate reference images via ComfyUI API (Z-Anime)
3. Upload images and submit video jobs to Wan 2.7 (DashScope)
4. Poll async jobs until completion
5. Download and organize final videos

Usage:
    python agent/main.py                          # Interactive mode
    python agent/main.py --scene 1                 # Generate single scene
    python agent/main.py --scene 1-3               # Generate scenes 1-3
    python agent/main.py --all                     # Generate all scenes
    python agent/main.py --status                  # Check job status
"""

import asyncio
import json
import os
import sys
import time
import base64
import uuid
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Annotated
from dataclasses import dataclass, field

# LangGraph
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# LangChain
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# ── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
AGENT_DIR = Path(__file__).parent
GENERATED_DIR = BASE_DIR / "generated"
CHARS_DIR = GENERATED_DIR / "characters"
BGS_DIR = GENERATED_DIR / "backgrounds"
VIDEOS_DIR = GENERATED_DIR / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR = AGENT_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")

# DashScope / Wan 2.7 API
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
WAN_MODEL = "wan2.7-i2v"

# LLM for agent reasoning
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6:27b")

# ── Scene Data (loaded from story config) ──────────────────────────────

SCENES = {
    1: {
        "name": "The Discovered Shop",
        "start_bg": "scene01_shop_exterior.png",
        "width": 1280, "height": 720,
        "global_prompt": (
            "A 19-year-old anime girl Mei with messy chestnut hair and a faded blue ribbon "
            "wanders into a narrow old Japanese street at dusk, discovering a mysterious "
            "antique shop called Moonlight Exchange. Warm amber street lamps, evening mist."
        ),
        "local_prompts": (
            "Wide establishing shot of a narrow old Japanese street at dusk, warm amber "
            "street lamps casting soft glow through evening mist, Mei in her cream cardigan "
            "walks into frame from the distance, head slightly down holding a sketchbook"
            "|"
            "Close-up of Mei's face as she looks up, warm amber eyes wide with curiosity, "
            "her expression soft and surprised, misty street background blurred"
            "|"
            "Medium shot of Mei walking toward the antique shop entrance, the hand-painted "
            "Moonlight Exchange sign glowing faintly above, hanging lanterns sway gently"
        ),
    },
    2: {
        "name": "First Meeting",
        "start_bg": "scene02_shop_interior.png",
        "width": 1280, "height": 720,
        "global_prompt": (
            "Mei steps inside the cozy cluttered antique shop filled with golden lamp light "
            "and vintage items. Ren, a tall dark-haired boy with cool grey eyes, looks up "
            "from repairing a music box. Pudding the orange cat sleeps on the counter."
        ),
        "local_prompts": (
            "Interior wide shot of the cozy antique shop, tall wooden shelves filled with "
            "vintage items, warm golden light from a brass lamp, Mei steps through the front "
            "door, the orange cat Pudding visible sleeping on the counter"
            "|"
            "Medium shot of Ren at the counter, dark messy hair, charcoal shirt with rolled "
            "sleeves, fingerless gloves, looking up from a small brass music box, cool grey "
            "eyes meeting Mei's gaze"
            "|"
            "Close-up of Mei's face, slightly nervous but curious expression, looking around "
            "at the shop's warm shelves and trinkets, soft golden light on her chestnut hair"
        ),
    },
    3: {
        "name": "The Colored Key",
        "start_bg": "scene03_shop_counter.png",
        "width": 1280, "height": 720,
        "global_prompt": (
            "Mei stands at the antique shop counter and picks up an old brass key. She sees "
            "a soft pink glowing light radiating from it that only she can perceive. Her "
            "expression becomes emotional. Ren watches her carefully."
        ),
        "local_prompts": (
            "Medium shot of Mei standing at the wooden counter, surrounded by antique "
            "trinkets, warm brass lamp light"
            "|"
            "Close-up of Mei's hand picking up an old brass key, her fingers wrap around "
            "it gently, her expression shifts from curiosity to something deeper"
            "|"
            "Extreme close-up of Mei's amber eyes, slightly teary, soft pink light subtly "
            "reflecting in her irises"
            "|"
            "Ren watching from across the counter, one hand resting on the music box, grey "
            "eyes narrowing slightly with interest and suspicion"
        ),
    },
    4: {
        "name": "Mei's Apartment",
        "start_bg": "scene04_mei_apartment.png",
        "width": 1280, "height": 720,
        "global_prompt": (
            "Mei sits alone at her small wooden desk in her cozy apartment at sunset, "
            "surrounded by colorful illustrations pinned on the walls. She is sketching "
            "Ren from memory in her worn sketchbook."
        ),
        "local_prompts": (
            "Wide shot of Mei's cozy apartment at sunset, walls covered in colorful "
            "illustrations, a small plant on the windowsill, warm orange light streaming "
            "through the window, Mei sits at her desk"
            "|"
            "Medium shot of Mei sketching in her worn sketchbook, pencil moving quickly, "
            "focused expression, a half-empty boba tea cup on the desk"
            "|"
            "Close-up of the sketch showing a portrait of Ren with grey eyes and dark "
            "messy hair, still in progress, charcoal lines"
            "|"
            "Close-up of Mei's hand touching the old brass key on the desk, she pauses, "
            "looks up toward the window, soft smile on her lips"
        ),
    },
    5: {
        "name": "Rooftop at Night",
        "start_bg": "scene05_rooftop_night.png",
        "width": 1280, "height": 720,
        "global_prompt": (
            "Mei and Ren sit on the edge of a rooftop overlooking the old Japanese district "
            "at night. The city stretches below in a sea of tiny warm lights under a deep "
            "blue starlit sky with a crescent moon. They talk quietly."
        ),
        "local_prompts": (
            "Wide establishing shot of the rooftop at night, deep blue starlit sky, crescent "
            "moon, the city stretches below in a sea of warm lights, two silhouettes sit on "
            "the edge with legs dangling"
            "|"
            "Medium two-shot of Mei and Ren sitting side by side on the rooftop edge, backs "
            "to the camera, looking out at the city, quiet atmosphere, wind moves Mei's hair"
            "|"
            "Close-up of Mei's face in profile, moonlight on her features, she looks at Ren "
            "from the corner of her eye, slight blush, nervous smile"
            "|"
            "Close-up of Ren's face, looking straight at the city, calm expression, fingers "
            "tap restlessly on his knee"
        ),
    },
    6: {
        "name": "The Music Box",
        "start_bg": "scene06_workbench.png",
        "width": 1280, "height": 720,
        "global_prompt": (
            "Ren shows Mei the music box he was repairing. He opens it, the tiny ballerina "
            "rotates. Mei freezes because it plays the same melody her mother used to hum."
        ),
        "local_prompts": (
            "Medium shot of Ren at the workbench, holding the opened brass music box, the "
            "tiny ballerina inside slowly rotating, warm lamp light"
            "|"
            "Close-up of the music box, intricate gears visible, the small ballerina "
            "spinning, golden light reflecting off the brass"
            "|"
            "Close-up of Mei's face, eyes widen then fill with tears, hand covers mouth, "
            "deeply emotional, as if remembering something precious"
            "|"
            "Medium shot of Ren watching Mei's reaction, grey eyes softening with concern"
        ),
    },
    7: {
        "name": "Rainy Street",
        "start_bg": "scene07_rainy_street.png",
        "width": 1280, "height": 720,
        "global_prompt": (
            "Mei gets caught in heavy rain on the way to the antique shop. Ren suddenly "
            "appears with a big black umbrella, holding it over both of them. They walk "
            "slowly down the wet street together."
        ),
        "local_prompts": (
            "Wide shot of the old street in heavy rain, cobblestone slick and reflective, "
            "rain streaking diagonally, Mei stands alone under eaves hugging her sketchbook"
            "|"
            "Medium shot of Mei looking down the empty street, frustrated, rain dripping "
            "from her chestnut hair and blue ribbon"
            "|"
            "Medium shot of Ren stepping into frame with a big black umbrella, tilting it "
            "over Mei, his dark hair wet, faint smile"
            "|"
            "Two-shot from behind, Mei and Ren walking slowly under the umbrella, shoulders "
            "almost touching, puddles reflecting amber shop lights"
        ),
    },
    8: {
        "name": "The Attic Discovery",
        "start_bg": "scene08_attic.png",
        "width": 1280, "height": 720,
        "global_prompt": (
            "Mei and Ren explore the dusty attic of the antique shop. Golden light shafts "
            "through a skylight. They find an old locked wooden chest. Mei uses the brass "
            "key to open it, revealing a photograph of younger Ren."
        ),
        "local_prompts": (
            "Wide shot of the dusty attic, golden light shafts through a skylight, dust "
            "motes floating, Mei and Ren walk through stacked trunks"
            "|"
            "Medium shot of Ren pushing aside a dust sheet, revealing an old locked wooden "
            "chest with iron bands"
            "|"
            "Close-up of Mei reaching into her cardigan pocket, pulling out the small brass "
            "key, determined expression"
            "|"
            "Extreme close-up of the key sliding into the chest's lock, a soft metallic click"
            "|"
            "Medium shot of the opened chest revealing an old letter and a faded photograph"
            "|"
            "Close-up of Mei and Ren's faces side by side, both staring at the photo in "
            "stunned silence"
        ),
    },
    9: {
        "name": "The Confession",
        "start_bg": "scene09_rooftop_sunrise.png",
        "width": 1280, "height": 720,
        "global_prompt": (
            "Mei and Ren stand on the rooftop at sunrise, pink and peach sky. They finally "
            "confess their secrets and hold hands. Emotional, vulnerable, warm lighting."
        ),
        "local_prompts": (
            "Wide establishing shot of the rooftop at sunrise, pink and peach sky with "
            "wispy clouds, the city below in warm golden morning light, two figures face "
            "each other near the railing"
            "|"
            "Medium two-shot of Mei and Ren facing each other, wind blowing through their "
            "hair, sunrise light wrapping around them"
            "|"
            "Close-up of Mei's face, eyes determined, tears in her eyes, speaking honestly"
            "|"
            "Close-up of Ren's face, guard dropped, grey eyes vulnerable and open"
            "|"
            "Close-up of their hands, Mei reaches out, Ren hesitates then wraps his fingers "
            "around hers"
        ),
    },
    10: {
        "name": "New Morning",
        "start_bg": "scene10_shop_morning.png",
        "width": 1280, "height": 720,
        "global_prompt": (
            "The Moonlight Exchange shop in bright morning sunlight. The window display is "
            "filled with colorful illustrations. Ren stands in the doorway with Pudding. "
            "Mei hangs a new brass sign. Cheerful, hopeful ending."
        ),
        "local_prompts": (
            "Wide establishing shot of the shop exterior in bright morning sunlight, the "
            "front window filled with colorful illustrations, warm light spilling from "
            "the open door"
            "|"
            "Medium shot of Ren standing in the shop doorway, charcoal shirt sleeves "
            "rolled, arms crossed with a small smile, Pudding the orange cat sits on "
            "the step beside him"
            "|"
            "Medium shot of Mei walking up to the shop holding a new polished brass sign, "
            "bright morning light on her face, confident smile"
            "|"
            "Close-up of Mei's hands hanging the new brass sign above the door, her blue "
            "ribbon catches the morning breeze"
            "|"
            "Wide final shot of the complete shop front, new sign gleaming, window full "
            "of color, Ren and Pudding in doorway, Mei beside him, golden morning light"
        ),
    },
}


# ── State Definition ────────────────────────────────────────────────────

class PipelineState(TypedDict):
    """State for the Moonlight Exchange pipeline."""
    messages: Annotated[list, add_messages]
    scene_num: int
    scene_name: str
    scene_config: dict
    start_frame_path: str
    start_frame_filename: str
    video_output_path: str
    comfyui_prompt_id: str
    comfyui_status: str
    image_generated: bool
    dashscope_task_id: str
    dashscope_status: str
    video_url: str
    video_downloaded: bool
    seed: int
    error: str
    created_at: str


# ── Tools ───────────────────────────────────────────────────────────────

@tool
def check_comfyui_connection() -> str:
    """Check if ComfyUI is running and accessible."""
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5)
        data = json.loads(resp.read())
        devices = data.get("devices", [])
        vram = devices[0].get("vram_total", 0) / (1024**3) if devices else 0
        return f"ComfyUI is running. VRAM: {vram:.1f} GB. Devices: {len(devices)}"
    except Exception as e:
        return f"ComfyUI not reachable: {e}"


@tool
def check_dashscope_api_key() -> str:
    """Check if the DashScope API key is configured."""
    if DASHSCOPE_API_KEY:
        masked = DASHSCOPE_API_KEY[:8] + "***" + DASHSCOPE_API_KEY[-4:]
        return f"DashScope API key configured: {masked}"
    return "DashScope API key NOT set. Set DASHSCOPE_API_KEY environment variable."


@tool
def list_scenes() -> str:
    """List all Moonlight Exchange scenes with their generation status."""
    lines = []
    for num, scene in SCENES.items():
        bg_path = BGS_DIR / scene["start_bg"]
        scene_id = f"scene{num:02d}_{scene['name'].lower().replace(' ', '_')}"
        video_path = VIDEOS_DIR / f"{scene_id}.mp4"
        bg_status = "OK" if bg_path.exists() else "MISSING"
        vid_status = "EXISTS" if video_path.exists() else "pending"
        segments = len(scene["local_prompts"].split("|"))
        lines.append(
            f"Scene {num:2d}: {scene['name']:<25s} | "
            f"BG: {bg_status:<7s} | Video: {vid_status:<7s} | "
            f"Segments: {segments}"
        )
    return "\n".join(lines)


@tool
def generate_background_image(scene_num: int) -> str:
    """Generate a background image for a scene using Z-Anime via ComfyUI.
    
    Args:
        scene_num: Scene number (1-10)
    """
    import urllib.request
    import urllib.parse
    
    if scene_num not in SCENES:
        return f"Invalid scene number: {scene_num}. Must be 1-10."
    
    scene = SCENES[scene_num]
    bg_path = BGS_DIR / scene["start_bg"]
    
    if bg_path.exists():
        return f"Background already exists: {bg_path}"
    
    # Build ComfyUI workflow
    seed = random.randint(0, 2**63)
    workflow = {
        "9": {
            "inputs": {"filename_prefix": f"moonlight/{scene['start_bg'].replace('.png','')}",
                       "images": ["65", 0]},
            "class_type": "SaveImage", "_meta": {"title": "Save Image"},
        },
        "62": {
            "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"},
            "class_type": "CLIPLoader", "_meta": {"title": "Load CLIP"},
        },
        "63": {
            "inputs": {"vae_name": "ae.safetensors"},
            "class_type": "VAELoader", "_meta": {"title": "Load VAE"},
        },
        "64": {
            "inputs": {"conditioning": ["67", 0]},
            "class_type": "ConditioningZeroOut", "_meta": {"title": "ConditioningZeroOut"},
        },
        "65": {
            "inputs": {"samples": ["70", 0], "vae": ["63", 0]},
            "class_type": "VAEDecode", "_meta": {"title": "VAE Decode"},
        },
        "66": {
            "inputs": {"unet_name": "Z-Image\\z-anime-distill-8step-bf16.safetensors",
                       "weight_dtype": "default"},
            "class_type": "UNETLoader", "_meta": {"title": "Load Z-Anime"},
        },
        "67": {
            "inputs": {"text": scene["global_prompt"], "clip": ["62", 0]},
            "class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode"},
        },
        "68": {
            "inputs": {"width": 1216, "height": 832, "batch_size": 1},
            "class_type": "EmptySD3LatentImage", "_meta": {"title": "Empty Latent"},
        },
        "69": {
            "inputs": {"shift": 3, "model": ["66", 0]},
            "class_type": "ModelSamplingAuraFlow", "_meta": {"title": "ModelSamplingAuraFlow"},
        },
        "70": {
            "inputs": {"seed": seed, "steps": 8, "cfg": 1.0,
                       "sampler_name": "euler_ancestral", "scheduler": "beta",
                       "denoise": 1, "model": ["69", 0],
                       "positive": ["67", 0], "negative": ["64", 0],
                       "latent_image": ["68", 0]},
            "class_type": "KSampler", "_meta": {"title": "KSampler"},
        },
    }
    
    # Submit to ComfyUI
    data = json.dumps({"prompt": workflow, "client_id": "moonlight-agent"}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt", data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req).read())
        prompt_id = resp["prompt_id"]
        
        # Wait for completion
        for _ in range(150):  # 5 min timeout
            time.sleep(2)
            hist = json.loads(
                urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}").read()
            )
            if prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                if status.get("completed", False):
                    # Download image
                    outputs = hist[prompt_id].get("outputs", {})
                    for node_id, node_out in outputs.items():
                        for img in node_out.get("images", []):
                            params = urllib.parse.urlencode({
                                "filename": img["filename"],
                                "subfolder": img.get("subfolder", ""),
                                "type": img.get("type", "output"),
                            })
                            urllib.request.urlretrieve(
                                f"{COMFYUI_URL}/view?{params}", bg_path
                            )
                    return f"Background generated: {bg_path} (seed={seed})"
                if status.get("status_str") == "error":
                    return f"ComfyUI error: {status.get('messages', [])}"
        
        return f"Timeout waiting for ComfyUI generation"
    except Exception as e:
        return f"ComfyUI error: {e}"


@tool
def submit_video_job(scene_num: int) -> str:
    """Submit a video generation job to Wan 2.7 API using a background image as start frame.
    
    Args:
        scene_num: Scene number (1-10)
    """
    import urllib.request
    
    if scene_num not in SCENES:
        return f"Invalid scene number: {scene_num}. Must be 1-10."
    
    if not DASHSCOPE_API_KEY:
        return "ERROR: DASHSCOPE_API_KEY not set. Cannot submit video job."
    
    scene = SCENES[scene_num]
    bg_path = BGS_DIR / scene["start_bg"]
    
    if not bg_path.exists():
        return f"Background image not found: {bg_path}. Generate it first."
    
    # Read and base64 encode the image
    with open(bg_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    # Build Wan 2.7 API request
    request_body = {
        "model": WAN_MODEL,
        "input": {
            "prompt": scene["global_prompt"],
            "media": [{
                "type": "first_frame",
                "data": f"data:image/png;base64,{img_b64}",
            }],
            "parameters": {
                "size": "720P",
                "duration": 5,
            },
        },
    }
    
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    
    data = json.dumps(request_body).encode()
    req = urllib.request.Request(
        f"{DASHSCOPE_BASE_URL}/services/aigc/video-generation/generation",
        data=data, headers=headers,
    )
    
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        request_id = resp.get("request_id", "")
        output = resp.get("output", {})
        task_id = output.get("task_id", "")
        
        if not task_id:
            return f"API error: no task_id returned. Response: {json.dumps(resp)[:500]}"
        
        # Save task state
        scene_id = f"scene{scene_num:02d}_{scene['name'].lower().replace(' ', '_')}"
        state_file = STATE_DIR / f"{scene_id}_task.json"
        state_file.write_text(json.dumps({
            "scene_num": scene_num,
            "scene_id": scene_id,
            "task_id": task_id,
            "request_id": request_id,
            "status": "submitted",
            "created_at": datetime.now().isoformat(),
            "prompt": scene["global_prompt"],
        }, indent=2))
        
        return (f"Video job submitted for Scene {scene_num} ({scene['name']}).\n"
                f"  Task ID: {task_id}\n"
                f"  Request ID: {request_id}\n"
                f"  Model: {WAN_MODEL}\n"
                f"  Resolution: 720P, Duration: 5s")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500]
        return f"API HTTP {e.code}: {body}"
    except Exception as e:
        return f"API error: {e}"


@tool
def poll_video_job(scene_num: int) -> str:
    """Check the status of a video generation job and download if complete.
    
    Args:
        scene_num: Scene number (1-10)
    """
    import urllib.request
    
    if scene_num not in SCENES:
        return f"Invalid scene number: {scene_num}. Must be 1-10."
    
    if not DASHSCOPE_API_KEY:
        return "ERROR: DASHSCOPE_API_KEY not set."
    
    scene = SCENES[scene_num]
    scene_id = f"scene{scene_num:02d}_{scene['name'].lower().replace(' ', '_')}"
    state_file = STATE_DIR / f"{scene_id}_task.json"
    
    if not state_file.exists():
        return f"No task found for Scene {scene_num}. Submit a job first."
    
    task_data = json.loads(state_file.read_text())
    task_id = task_data["task_id"]
    
    # Query task status
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
    }
    req = urllib.request.Request(
        f"{DASHSCOPE_BASE_URL}/tasks/{task_id}",
        headers=headers,
    )
    
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        output = resp.get("output", {})
        task_status = output.get("task_status", "UNKNOWN")
        
        # Update state file
        task_data["status"] = task_status
        task_data["last_checked"] = datetime.now().isoformat()
        state_file.write_text(json.dumps(task_data, indent=2))
        
        if task_status == "SUCCEEDED":
            video_url = output.get("video_url", "")
            if not video_url:
                return f"Task succeeded but no video_url found. Output: {json.dumps(output)[:500]}"
            
            # Download video
            video_path = VIDEOS_DIR / f"{scene_id}.mp4"
            urllib.request.urlretrieve(video_url, video_path)
            task_data["video_path"] = str(video_path)
            task_data["video_url"] = video_url
            state_file.write_text(json.dumps(task_data, indent=2))
            
            size_mb = video_path.stat().st_size / (1024 * 1024)
            return (f"Scene {scene_num} video COMPLETED and downloaded!\n"
                    f"  File: {video_path}\n"
                    f"  Size: {size_mb:.1f} MB\n"
                    f"  URL: {video_url}")
        
        elif task_status == "FAILED":
            error_msg = output.get("message", "Unknown error")
            return f"Scene {scene_num} video FAILED: {error_msg}"
        
        elif task_status == "RUNNING":
            progress = output.get("progress", "")
            return f"Scene {scene_num} video still RUNNING. {progress}"
        
        else:
            return f"Scene {scene_num} status: {task_status}. Output: {json.dumps(output)[:300]}"
    
    except Exception as e:
        return f"Poll error: {e}"


@tool
def run_full_pipeline(scene_num: int) -> str:
    """Run the complete pipeline for a scene: generate background, submit video, poll until done.
    
    Args:
        scene_num: Scene number (1-10)
    """
    if scene_num not in SCENES:
        return f"Invalid scene number: {scene_num}. Must be 1-10."
    
    scene = SCENES[scene_num]
    scene_id = f"scene{scene_num:02d}_{scene['name'].lower().replace(' ', '_')}"
    video_path = VIDEOS_DIR / f"{scene_id}.mp4"
    
    if video_path.exists():
        return f"Scene {scene_num} video already exists: {video_path}"
    
    results = []
    
    # Step 1: Check prerequisites
    results.append(f"=== Scene {scene_num}: {scene['name']} ===")
    results.append(f"Step 1: Checking prerequisites...")
    
    # Step 2: Generate background if needed
    bg_path = BGS_DIR / scene["start_bg"]
    if not bg_path.exists():
        results.append(f"Step 2: Generating background image...")
        gen_result = generate_background_image.invoke({"scene_num": scene_num})
        results.append(f"  {gen_result}")
        if "error" in gen_result.lower() or "not found" in gen_result.lower():
            return "\n".join(results)
    else:
        results.append(f"Step 2: Background already exists: {bg_path}")
    
    # Step 3: Submit video job
    results.append(f"Step 3: Submitting video job to Wan 2.7...")
    submit_result = submit_video_job.invoke({"scene_num": scene_num})
    results.append(f"  {submit_result}")
    
    if "ERROR" in submit_result or "error" in submit_result.lower():
        return "\n".join(results)
    
    # Step 4: Poll for completion
    results.append(f"Step 4: Waiting for video generation (polling every 30s)...")
    max_attempts = 60  # 30 minutes max
    for attempt in range(max_attempts):
        time.sleep(30)
        poll_result = poll_video_job.invoke({"scene_num": scene_num})
        if "COMPLETED" in poll_result:
            results.append(f"  {poll_result}")
            results.append(f"Pipeline complete for Scene {scene_num}!")
            return "\n".join(results)
        elif "FAILED" in poll_result:
            results.append(f"  {poll_result}")
            return "\n".join(results)
        else:
            if attempt % 4 == 0:  # Log every 2 minutes
                results.append(f"  [{attempt*30}s] {poll_result}")
    
    results.append(f"  Timeout after {max_attempts * 30}s")
    return "\n".join(results)


# ── Graph Definition ───────────────────────────────────────────────────

# All tools available to the agent
ALL_TOOLS = [
    check_comfyui_connection,
    check_dashscope_api_key,
    list_scenes,
    generate_background_image,
    submit_video_job,
    poll_video_job,
    run_full_pipeline,
]

tool_node = ToolNode(ALL_TOOLS)


def create_agent():
    """Create the LangGraph agent."""
    
    # Use Ollama for local LLM
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL,
        temperature=0,
    ).bind_tools(ALL_TOOLS)
    
    def agent_reasoner(state: PipelineState):
        """The agent node that decides what to do next."""
        messages = state["messages"]
        if not messages:
            messages = [HumanMessage(content="What scenes are available?")]
        
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: PipelineState):
        """Determine whether to use tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END
    
    # Build the graph
    graph = StateGraph(PipelineState)
    
    graph.add_node("agent", agent_reasoner)
    graph.add_node("tools", tool_node)
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    
    return graph.compile()


# ── CLI Interface ───────────────────────────────────────────────────────

def run_interactive():
    """Run the agent in interactive chat mode."""
    print("\n" + "=" * 60)
    print("  Moonlight Exchange — LangGraph Agent")
    print("  Type 'quit' to exit, 'status' for scene list")
    print("=" * 60 + "\n")
    
    graph = create_agent()
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        
        state = PipelineState(
            messages=[HumanMessage(content=user_input)],
            created_at=datetime.now().isoformat(),
        )
        
        result = graph.invoke(state)
        
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\nAgent: {msg.content}\n")
            elif isinstance(msg, ToolMessage):
                print(f"  [Tool] {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")


def run_pipeline(scene_nums: list):
    """Run the full pipeline for specified scenes."""
    graph = create_agent()
    
    for num in scene_nums:
        prompt = (
            f"Run the full pipeline for Scene {num}. "
            f"Generate the background image if needed, submit the video job to Wan 2.7, "
            f"and poll until the video is downloaded. Report the final status."
        )
        
        state = PipelineState(
            messages=[HumanMessage(content=prompt)],
            scene_num=num,
            created_at=datetime.now().isoformat(),
        )
        
        result = graph.invoke(state)
        
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\n{msg.content}\n")


def show_status():
    """Show current status of all scenes."""
    graph = create_agent()
    state = PipelineState(
        messages=[HumanMessage(content="List all scenes and their current status.")],
    )
    result = graph.invoke(state)
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(msg.content)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Moonlight Exchange — LangGraph AI Agent"
    )
    parser.add_argument("--scene", type=str, help="Scene(s): '1', '1-3', '1,3,5'")
    parser.add_argument("--all", action="store_true", help="Run all 10 scenes")
    parser.add_argument("--status", action="store_true", help="Show scene status")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive chat mode")
    args = parser.parse_args()
    
    def parse_scene_range(range_str: str) -> list:
        scenes = []
        for part in range_str.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                scenes.extend(range(int(start), int(end) + 1))
            else:
                scenes.append(int(part))
        return [s for s in scenes if 1 <= s <= 10]
    
    if args.status:
        show_status()
    elif args.interactive or (not args.scene and not args.all):
        run_interactive()
    elif args.all:
        run_pipeline(list(range(1, 11)))
    elif args.scene:
        run_pipeline(parse_scene_range(args.scene))


if __name__ == "__main__":
    main()
