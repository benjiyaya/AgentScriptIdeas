"""
Moonlight Exchange — LTX 2.3 + Prompt Relay Video Generator
Uses generated character/background images as start frames for video generation.

Usage:
    python generate_video.py --scene 1                # Generate scene 1 video
    python generate_video.py --scene 1-3              # Generate scenes 1-3
    python generate_video.py --all                    # Generate all 10 scenes
    python generate_video.py --all --project my-story # Generate into generated/my-story/
    python generate_video.py --list                   # List available start frames
"""

import json
import time
import urllib.request
import urllib.parse
import random
import argparse
import os
import sys
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
COMFYUI_URL = "http://127.0.0.1:8188"
BASE_DIR = Path(__file__).parent
GENERATED_BASE = BASE_DIR / "generated"

# These are set dynamically based on --project argument
GENERATED_DIR = None
CHARS_DIR = None
BGS_DIR = None
VIDEOS_DIR = None


def setup_dirs(project: str = "default"):
    """Initialize per-project output directories."""
    global GENERATED_DIR, CHARS_DIR, BGS_DIR, VIDEOS_DIR
    GENERATED_DIR = GENERATED_BASE / project
    CHARS_DIR = GENERATED_DIR / "characters"
    BGS_DIR = GENERATED_DIR / "backgrounds"
    VIDEOS_DIR = GENERATED_DIR / "videos"
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# LTX 2.3 model config
LTX_MODEL = "ltx2\\ltx-2.3-22b-dev.safetensors"
LTX_CLIP = "gemma_3_12B_it_fp8_scaled.safetensors"
LTX_TEXT_PROJ = "ltx2\\ltx-2.3_text_projection_bf16.safetensors"
LTX_VIDEO_VAE = "LTX23_video_vae_bf16.safetensors"

# Video generation defaults
FPS = 25
VIDEO_DURATION = 20  # seconds per scene
TOTAL_FRAMES = FPS * VIDEO_DURATION  # 500 frames for 20s

# ── Scene Definitions ──────────────────────────────────────────────────
# Maps scene number to: which character image, which background image, and prompts

SCENES = {
    1: {
        "name": "The Discovered Shop",
        "start_frame": None,  # Uses background only
        "start_bg": "scene01_shop_exterior.png",
        "width": 768,
        "height": 512,
        "global_prompt": (
            "A 19-year-old anime girl Mei with messy chestnut hair and a faded blue ribbon "
            "wanders into a narrow old Japanese street at dusk, discovering a mysterious "
            "antique shop called Moonlight Exchange. The atmosphere is warm and misty with "
            "amber street lamps. Anime style, high quality."
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
            "Moonlight Exchange sign glowing faintly above, hanging lanterns sway gently "
            "in the warm evening light"
        ),
        "epsilon": 0.001,
    },
    2: {
        "name": "First Meeting",
        "start_frame": None,
        "start_bg": "scene02_shop_interior.png",
        "width": 768,
        "height": 512,
        "global_prompt": (
            "Mei steps inside the cozy cluttered antique shop filled with golden lamp light "
            "and vintage items. Inside, a tall dark-haired boy named Ren with cool grey eyes "
            "looks up from repairing a music box at the wooden counter. A chubby orange cat "
            "named Pudding stretches lazily on the counter. Anime style, high quality."
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
        "epsilon": 0.001,
    },
    3: {
        "name": "The Colored Key",
        "start_frame": None,
        "start_bg": "scene03_shop_counter.png",
        "width": 768,
        "height": 512,
        "global_prompt": (
            "Mei stands at the antique shop counter and picks up an old brass key. She sees "
            "a soft pink glowing light radiating from the key that only she can perceive. "
            "Her expression becomes emotional. Ren watches her carefully. Anime style."
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
        "epsilon": 0.001,
    },
    4: {
        "name": "Mei's Apartment",
        "start_frame": None,
        "start_bg": "scene04_mei_apartment.png",
        "width": 768,
        "height": 512,
        "global_prompt": (
            "Mei sits alone at her small wooden desk in her cozy apartment at sunset, "
            "surrounded by colorful illustrations pinned on the walls. She is sketching Ren "
            "from memory in her worn sketchbook. Anime style, warm atmosphere."
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
        "epsilon": 0.001,
    },
    5: {
        "name": "Rooftop at Night",
        "start_frame": None,
        "start_bg": "scene05_rooftop_night.png",
        "width": 768,
        "height": 512,
        "global_prompt": (
            "Mei and Ren sit on the edge of a rooftop overlooking the old Japanese district "
            "at night. The city stretches below in a sea of tiny warm lights under a deep "
            "blue starlit sky with a crescent moon. They talk quietly. Anime style."
        ),
        "local_prompts": (
            "Wide establishing shot of the rooftop at night, deep blue starlit sky, crescent "
            "moon, the city stretches below in a sea of warm lights, two small silhouettes "
            "sit on the edge with legs dangling"
            "|"
            "Medium two-shot of Mei and Ren sitting side by side on the rooftop edge, backs "
            "to the camera, looking out at the city, quiet atmosphere, wind gently moves "
            "Mei's chestnut hair"
            "|"
            "Close-up of Mei's face in profile, moonlight on her features, she looks at Ren "
            "from the corner of her eye, slight blush, nervous smile"
            "|"
            "Close-up of Ren's face, looking straight at the city, calm expression, but his "
            "fingers tap restlessly on his knee"
        ),
        "epsilon": 0.001,
    },
    6: {
        "name": "The Music Box",
        "start_frame": None,
        "start_bg": "scene06_workbench.png",
        "width": 768,
        "height": 512,
        "global_prompt": (
            "Back at the antique shop, Ren shows Mei the music box he was repairing. He "
            "opens it, the tiny ballerina rotates, and a soft melody plays. Mei freezes "
            "because it is the same melody her mother used to hum. Anime style, emotional."
        ),
        "local_prompts": (
            "Medium shot of Ren at the workbench, holding the opened brass music box in "
            "his hands, the tiny ballerina inside slowly rotating, warm lamp light"
            "|"
            "Close-up of the music box, intricate gears visible, the small ballerina figure "
            "spinning, golden light reflecting off the brass"
            "|"
            "Close-up of Mei's face, her eyes widen then fill with tears, her hand covers "
            "her mouth, deeply emotional, as if remembering something precious"
            "|"
            "Medium shot of Ren watching Mei's reaction, his grey eyes softening with "
            "concern, he gently sets the music box down"
        ),
        "epsilon": 0.001,
    },
    7: {
        "name": "Rainy Street",
        "start_frame": None,
        "start_bg": "scene07_rainy_street.png",
        "width": 768,
        "height": 512,
        "global_prompt": (
            "Mei gets caught in heavy rain on her way to the antique shop on the old narrow "
            "street. Ren suddenly appears with a big black umbrella, holding it over both "
            "of them. They walk slowly down the wet street together. Anime style."
        ),
        "local_prompts": (
            "Wide shot of the old street in heavy rain, cobblestone street slick and "
            "reflective, rain streaking diagonally, Mei stands alone under the eaves of "
            "a building, hugging her sketchbook, rain soaked"
            "|"
            "Medium shot of Mei looking down the empty street, frustrated, rain dripping "
            "from her chestnut hair and blue ribbon, her cream cardigan darkened with water"
            "|"
            "Medium shot of Ren stepping into frame with a big black umbrella, tilting it "
            "over Mei, his dark hair already wet, a faint smile on his face"
            "|"
            "Two-shot from behind, Mei and Ren walking slowly down the rain-slicked street "
            "under the umbrella, shoulders almost touching, puddles reflecting the amber "
            "shop lights behind them"
        ),
        "epsilon": 0.001,
    },
    8: {
        "name": "The Attic Discovery",
        "start_frame": None,
        "start_bg": "scene08_attic.png",
        "width": 768,
        "height": 512,
        "global_prompt": (
            "Mei and Ren explore the dusty attic of the antique shop. Shafts of golden "
            "afternoon light stream through a skylight. They find an old locked wooden "
            "chest with iron bands. Mei uses the brass key to open it. Anime style."
        ),
        "local_prompts": (
            "Wide shot of the dusty attic, dramatic golden light shafts through a skylight, "
            "dust motes floating, Mei and Ren walk through stacked trunks and covered "
            "furniture"
            "|"
            "Medium shot of Ren pushing aside a dust sheet, revealing an old locked wooden "
            "chest with iron bands, he looks puzzled"
            "|"
            "Close-up of Mei reaching into her cardigan pocket, pulling out the small brass "
            "key, her expression determined"
            "|"
            "Extreme close-up of the key sliding into the chest's lock, a soft metallic "
            "click, Mei's hand and Ren's hand both reach for the lid"
            "|"
            "Medium shot of the opened chest revealing an old yellowed letter and a faded "
            "black-and-white photograph of two smiling teenagers"
            "|"
            "Close-up of Mei and Ren's faces side by side, both staring at the photo in "
            "stunned silence, the two teenagers look exactly like younger Ren and a girl"
        ),
        "epsilon": 0.001,
    },
    9: {
        "name": "The Confession",
        "start_frame": None,
        "start_bg": "scene09_rooftop_sunrise.png",
        "width": 768,
        "height": 512,
        "global_prompt": (
            "Mei and Ren stand on the rooftop at sunrise, the sky painted in pink and peach "
            "gradients. They face each other and finally confess their secrets. The wind "
            "blows gently. They hold hands. Emotional, vulnerable, warm lighting. Anime style."
        ),
        "local_prompts": (
            "Wide establishing shot of the rooftop at sunrise, pink and peach sky with "
            "wispy clouds, the city below in warm golden morning light, two figures stand "
            "facing each other near the railing"
            "|"
            "Medium two-shot of Mei and Ren facing each other, wind blowing through Mei's "
            "chestnut hair and Ren's dark hair, sunrise light wrapping around them"
            "|"
            "Close-up of Mei's face, eyes determined, tears in her eyes, speaking honestly "
            "for the first time"
            "|"
            "Close-up of Ren's face, guard completely dropped, grey eyes vulnerable and "
            "open, listening intently"
            "|"
            "Close-up of their hands, Mei reaches out, Ren hesitates for a fraction of a "
            "second, then wraps his fingers around hers"
        ),
        "epsilon": 0.001,
    },
    10: {
        "name": "New Morning",
        "start_frame": None,
        "start_bg": "scene10_shop_morning.png",
        "width": 768,
        "height": 512,
        "global_prompt": (
            "The Moonlight Exchange antique shop in bright fresh morning sunlight. The front "
            "window display is now filled with colorful illustrations. Ren stands in the "
            "doorway with Pudding the orange cat. Mei hangs a new brass sign. Anime style, "
            "hopeful ending."
        ),
        "local_prompts": (
            "Wide establishing shot of the shop exterior in bright morning sunlight, the "
            "front window now filled with colorful illustrations, warm light spilling from "
            "the open door, potted plants in the doorway"
            "|"
            "Medium shot of Ren standing in the shop doorway, charcoal shirt sleeves "
            "rolled, arms crossed with a small satisfied smile, Pudding the orange cat "
            "sits on the step beside him, one white paw raised"
            "|"
            "Medium shot of Mei walking up to the shop from the cobblestone path, holding "
            "a new polished brass sign, bright morning light on her face, she smiles "
            "confidently"
            "|"
            "Close-up of Mei's hands hanging the new brass sign above the door, it reads "
            "Moonlight Exchange Open, her blue ribbon catches the morning breeze"
            "|"
            "Wide final shot of the complete shop front, new sign gleaming, window full "
            "of color, Ren and Pudding in the doorway, Mei standing beside him with her "
            "sketchbook, all three silhouettes in the golden morning light"
        ),
        "epsilon": 0.5,  # Softer transitions for the finale
    },
}


# ── ComfyUI API Helpers ─────────────────────────────────────────────────

def api_post(endpoint: str, data: dict = None) -> dict:
    """POST to ComfyUI API and return parsed JSON."""
    url = f"{COMFYUI_URL}{endpoint}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"} if body else {},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"  [ERROR] HTTP {e.code}: {e.read().decode()[:500]}")
        return None


def upload_image(image_path: Path, filename: str) -> str:
    """Upload image to ComfyUI input folder, return the filename used."""
    with open(image_path, "rb") as f:
        body = f.read()

    boundary = f"----WebKitFormBoundary{random.randint(100000, 999999)}"
    parts = [
        f"--{boundary}",
        f'Content-Disposition: form-data; name="image"; filename="{filename}"',
        "Content-Type: image/png",
        "",
    ]
    req_body = ("\r\n".join(parts) + "\r\n").encode() + body + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{COMFYUI_URL}/upload/image",
        data=req_body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req).read())
        return resp.get("name", filename)
    except Exception as e:
        print(f"  [ERROR] Upload failed: {e}")
        return None


def wait_for_completion(prompt_id: str, timeout: int = 1200) -> dict:
    """Poll ComfyUI until prompt completes or fails."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            hist = api_post(f"/history/{prompt_id}")
            if hist and prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                if status.get("completed", False) or status.get("status_str") == "error":
                    return hist[prompt_id]
        except Exception:
            pass
        time.sleep(5)
    print(f"  [TIMEOUT] {timeout}s exceeded")
    return None


def download_file(filename: str, subfolder: str, file_type: str, save_path: Path):
    """Download output file from ComfyUI."""
    params = urllib.parse.urlencode({
        "filename": filename,
        "subfolder": subfolder,
        "type": file_type,
    })
    url = f"{COMFYUI_URL}/view?{params}"
    urllib.request.urlretrieve(url, save_path)


# ── Workflow Builder ────────────────────────────────────────────────────

def build_video_workflow(
    start_frame_filename: str,
    global_prompt: str,
    local_prompts: str,
    width: int,
    height: int,
    total_frames: int,
    fps: int,
    epsilon: float,
    seed: int,
    prefix: str,
) -> dict:
    """
    Build LTX 2.3 + Prompt Relay I2V workflow.
    
    Pipeline:
    1. LoadImage (start frame)
    2. VAEEncode (image -> latent)
    3. UNETLoader (LTX 2.3)
    4. DualCLIPLoader (Gemma 3 + text projection)
    5. PromptRelayEncode (global + local prompts)
    6. ConditioningZeroOut (negative)
    7. KSampler
    8. VAEDecode
    9. SaveVideo (VHS_SaveVideo or VAE Encode -> Save)
    """

    workflow = {
        # 1. Load start frame image
        "10": {
            "inputs": {"image": start_frame_filename},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Start Frame"},
        },
        # 2. Load LTX 2.3 model
        "11": {
            "inputs": {
                "unet_name": LTX_MODEL,
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
            "_meta": {"title": "Load LTX 2.3 Model"},
        },
        # 3. Load CLIP (Gemma 3) + text projection
        "12": {
            "inputs": {
                "clip_name1": LTX_CLIP,
                "clip_name2": LTX_TEXT_PROJ,
                "type": "ltxv",
            },
            "class_type": "DualCLIPLoader",
            "_meta": {"title": "Load CLIP + Text Projection"},
        },
        # 4. Load Video VAE
        "13": {
            "inputs": {"vae_name": LTX_VIDEO_VAE},
            "class_type": "VAELoader",
            "_meta": {"title": "Load Video VAE"},
        },
        # 5. VAE Encode start frame to latent
        "14": {
            "inputs": {
                "pixels": ["10", 0],
                "vae": ["13", 0],
            },
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode Start Frame"},
        },
        # 6. Prompt Relay Encode (global + local prompts)
        "15": {
            "inputs": {
                "model": ["11", 0],
                "clip": ["12", 0],
                "latent": ["14", 0],
                "global_prompt": global_prompt,
                "local_prompts": local_prompts,
                "segment_lengths": "",  # Auto-distribute evenly
                "epsilon": epsilon,
            },
            "class_type": "PromptRelayEncode",
            "_meta": {"title": "Prompt Relay Encode"},
        },
        # 7. ConditioningZeroOut (negative)
        "16": {
            "inputs": {"conditioning": ["15", 1]},
            "class_type": "ConditioningZeroOut",
            "_meta": {"title": "Negative Conditioning"},
        },
        # 8. KSampler
        "17": {
            "inputs": {
                "seed": seed,
                "steps": 20,
                "cfg": 1.0,
                "sampler_name": "euler_ancestral",
                "scheduler": "beta",
                "denoise": 1.0,
                "model": ["15", 0],
                "positive": ["15", 1],
                "negative": ["16", 0],
                "latent_image": ["14", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
        },
        # 9. VAE Decode (video)
        "18": {
            "inputs": {
                "samples": ["17", 0],
                "vae": ["13", 0],
            },
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode Video"},
        },
        # 10. Save output
        "19": {
            "inputs": {
                "filename_prefix": prefix,
                "images": ["18", 0],
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Output"},
        },
    }
    return workflow


# ── Generator ───────────────────────────────────────────────────────────

def generate_scene_video(scene_num: int, skip_existing: bool = True) -> bool:
    """Generate video for a single scene."""
    scene = SCENES.get(scene_num)
    if not scene:
        print(f"  [ERROR] Scene {scene_num} not defined")
        return False

    scene_id = f"scene{scene_num:02d}_{scene['name'].lower().replace(' ', '_')}"
    output_path = VIDEOS_DIR / f"{scene_id}.mp4"

    if skip_existing and output_path.exists():
        print(f"  [SKIP] Scene {scene_num} ({scene['name']}) — already exists")
        return True

    # Upload start frame
    bg_path = BGS_DIR / scene["start_bg"]
    if not bg_path.exists():
        print(f"  [ERROR] Background not found: {bg_path}")
        return False

    print(f"  [UPLOAD] Start frame: {scene['start_bg']}")
    uploaded_name = upload_image(bg_path, f"moonlight_{scene['start_bg']}")
    if not uploaded_name:
        return False

    seed = random.randint(0, 2**63)
    prefix = f"moonlight/video_{scene_id}"

    print(f"  [GEN] Scene {scene_num}: {scene['name']}")
    print(f"        Frames: {TOTAL_FRAMES} ({VIDEO_DURATION}s @ {FPS}fps)")
    print(f"        Resolution: {scene['width']}x{scene['height']}")
    print(f"        Segments: {len(scene['local_prompts'].split('|'))}")
    print(f"        Seed: {seed}")

    workflow = build_video_workflow(
        start_frame_filename=uploaded_name,
        global_prompt=scene["global_prompt"],
        local_prompts=scene["local_prompts"],
        width=scene["width"],
        height=scene["height"],
        total_frames=TOTAL_FRAMES,
        fps=FPS,
        epsilon=scene["epsilon"],
        seed=seed,
        prefix=prefix,
    )

    result = api_post("/prompt", {"prompt": workflow, "client_id": "moonlight-video"})
    if not result or "prompt_id" not in result:
        return False

    prompt_id = result["prompt_id"]
    print(f"  [QUEUE] Prompt ID: {prompt_id[:12]}...")

    # Wait for completion
    completed = wait_for_completion(prompt_id, timeout=1800)
    if not completed:
        return False

    status = completed.get("status", {})
    if status.get("status_str") == "error":
        msgs = status.get("messages", [])
        print(f"  [FAIL] Scene {scene_num}: {msgs}")
        return False

    # Download outputs
    outputs = completed.get("outputs", {})
    saved_files = []
    for node_id, node_out in outputs.items():
        for item in node_out.get("images", []):
            fname = item["filename"]
            subfolder = item.get("subfolder", "")
            ftype = item.get("type", "output")
            save_name = f"{scene_id}_{fname}"
            save_p = VIDEOS_DIR / save_name
            download_file(fname, subfolder, ftype, save_p)
            saved_files.append(save_p)
            print(f"  [OK] Saved: {save_p}")

    if not saved_files:
        print(f"  [WARN] No output files found")
        return False

    return True


def list_scenes():
    """List all scenes with their status."""
    print("\n" + "=" * 70)
    print("  Moonlight Exchange — Scene List")
    print("=" * 70)
    for num, scene in SCENES.items():
        scene_id = f"scene{num:02d}_{scene['name'].lower().replace(' ', '_')}"
        video_path = VIDEOS_DIR / f"{scene_id}.mp4"
        bg_path = BGS_DIR / scene["start_bg"]
        bg_ok = "OK" if bg_path.exists() else "MISSING"
        vid_ok = "EXISTS" if video_path.exists() else "pending"
        segments = len(scene["local_prompts"].split("|"))
        print(f"  Scene {num:2d}: {scene['name']:<25s} | BG: {bg_ok:<7s} | "
              f"Video: {vid_ok:<7s} | Segments: {segments}")
    print("=" * 70 + "\n")


def parse_scene_range(range_str: str) -> list:
    """Parse '1-3' or '1,3,5' or '1' into list of scene numbers."""
    scenes = []
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            scenes.extend(range(int(start), int(end) + 1))
        else:
            scenes.append(int(part))
    return [s for s in scenes if 1 <= s <= 10]


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Moonlight Exchange — LTX 2.3 + Prompt Relay Video Generator"
    )
    parser.add_argument("--project", type=str, default="default",
                        help="Project name (output goes to generated/<project>/)")
    parser.add_argument("--scene", type=str, help="Scene(s): '1', '1-3', '1,3,5'")
    parser.add_argument("--all", action="store_true", help="Generate all 10 scenes")
    parser.add_argument("--list", action="store_true", help="List scenes and status")
    parser.add_argument("--fps", type=int, default=25, help="FPS (default: 25)")
    parser.add_argument("--duration", type=int, default=20,
                        help="Duration in seconds (default: 20)")
    parser.add_argument("--seed", type=int, default=None, help="Base seed")
    args = parser.parse_args()

    # Setup per-project directories
    setup_dirs(args.project)
    print(f"  Project: {args.project}")
    print(f"  Output:  {VIDEOS_DIR}")

    fps = args.fps
    duration = args.duration
    total_frames = fps * duration

    if args.seed is not None:
        random.seed(args.seed)

    # Verify ComfyUI
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5)
    except Exception:
        print(f"[ERROR] Cannot connect to ComfyUI at {COMFYUI_URL}")
        sys.exit(1)

    if args.list:
        list_scenes()
        return

    if args.all:
        scene_nums = list(range(1, 11))
    elif args.scene:
        scene_nums = parse_scene_range(args.scene)
    else:
        parser.print_help()
        return

    print(f"\n{'='*60}")
    print(f"  Moonlight Exchange — Video Generation")
    print(f"  LTX 2.3 + Prompt Relay")
    print(f"  Scenes: {scene_nums}")
    print(f"  {total_frames} frames ({duration}s @ {fps}fps)")
    print(f"  Output: {VIDEOS_DIR}")
    print(f"{'='*60}")

    success, fail = 0, 0
    for num in scene_nums:
        if generate_scene_video(num):
            success += 1
        else:
            fail += 1

    print(f"\n  Done: {success} OK, {fail} failed\n")


if __name__ == "__main__":
    main()
