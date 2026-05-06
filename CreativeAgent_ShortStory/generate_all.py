"""
Moonlight Exchange — Z-Anime Batch Image Generator
Generates all character sheets and background scenes via ComfyUI API.
Uses Z-Anime Distill-8Step (8 steps, CFG 1.0, euler_ancestral + beta).

Usage:
    python generate_all.py                          # Generate everything (default project)
    python generate_all.py --project my-story       # Generate into generated/my-story/
    python generate_all.py --chars                  # Characters only
    python generate_all.py --bgs                    # Backgrounds only
    python generate_all.py --scene 3                # Specific scene background only
"""

import json
import time
import urllib.request
import urllib.error
import uuid
import random
import argparse
import os
import sys
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
COMFYUI_URL = "http://127.0.0.1:8188"
BASE_DIR = Path(__file__).parent / "generated"

# Model settings (Z-Anime Distill-8Step)
MODEL = "Z-Image\\z-anime-distill-8step-bf16.safetensors"
CLIP = "qwen_3_4b.safetensors"
VAE = "ae.safetensors"
STEPS = 8
CFG = 1.0
SAMPLER = "euler_ancestral"
SCHEDULER = "beta"
SHIFT = 3

# ── Image Definitions ──────────────────────────────────────────────────

CHARACTERS = [
    {
        "id": "mei_fullbody",
        "name": "Mei — Full Body",
        "prompt": (
            "A full-body illustration of a 19-year-old anime girl named Mei, petite build, "
            "messy chestnut hair tied with a faded blue ribbon, warm amber eyes, wearing an "
            "oversized cream knit cardigan over a white cotton shirt, blue denim skirt, navy "
            "knee-high socks, and scuffed brown loafers. She holds a worn sketchbook in one "
            "arm and a plastic boba tea cup in her other hand. Soft natural lighting, gentle "
            "expression, standing in a sunlit street, high quality anime key visual, fine line work."
        ),
        "width": 832,
        "height": 1216,
    },
    {
        "id": "mei_portrait",
        "name": "Mei — Portrait",
        "prompt": (
            "Detailed anime portrait of Mei, a 19-year-old girl with messy chestnut hair and a "
            "faded blue ribbon, warm amber eyes looking thoughtfully to the side, soft natural "
            "skin shading, gentle slight smile, wearing an oversized cream cardigan, shallow "
            "depth of field, warm afternoon lighting, professional anime illustration quality, "
            "fine line work."
        ),
        "width": 832,
        "height": 1216,
    },
    {
        "id": "ren_fullbody",
        "name": "Ren — Full Body",
        "prompt": (
            "A full-body illustration of a 21-year-old anime boy named Ren, tall and lean build, "
            "dark messy hair with a silver chain necklace, cool grey eyes, wearing a charcoal "
            "button-down shirt with sleeves rolled to the elbows, dark jeans, fingerless gloves "
            "on his left hand. He stands casually with one hand in his pocket, calm unreadable "
            "expression, soft side lighting, high quality anime character design, fine line work."
        ),
        "width": 832,
        "height": 1216,
    },
    {
        "id": "ren_portrait",
        "name": "Ren — Portrait",
        "prompt": (
            "Detailed anime portrait of Ren, a 21-year-old boy with dark messy hair, a silver "
            "chain necklace visible at his collar, cool grey eyes with a subtle guarded "
            "expression, wearing a charcoal button-down with sleeves rolled up, soft dramatic "
            "side lighting, shallow depth of field, professional anime illustration quality."
        ),
        "width": 832,
        "height": 1216,
    },
    {
        "id": "pudding_cat",
        "name": "Pudding the Cat",
        "prompt": (
            "An adorable chubby orange tabby cat with one white front paw, sitting in a pool of "
            "warm sunlight, eyes half-closed in sleepy contentment, soft fluffy fur rendering, "
            "anime style, cozy atmosphere, detailed fur texture."
        ),
        "width": 1024,
        "height": 1024,
    },
]

BACKGROUNDS = [
    {
        "id": "scene01_shop_exterior",
        "name": "Scene 1 — The Discovered Shop (Exterior)",
        "prompt": (
            "Anime illustration of a narrow old Japanese street at dusk, a small antique shop "
            "with wooden shutters and a faded hand-painted sign reading \"Moonlight Exchange.\" "
            "Warm amber street lamps cast a soft glow through evening mist, hanging lanterns "
            "sway gently, potted plants and bicycle parked near the entrance. Atmospheric, "
            "Makoto Shinkai style background, highly detailed, wallpaper quality."
        ),
        "width": 1216,
        "height": 832,
    },
    {
        "id": "scene02_shop_interior",
        "name": "Scene 2 — Inside the Antique Shop",
        "prompt": (
            "Anime interior illustration of a cozy cluttered antique shop at golden hour. Tall "
            "wooden shelves filled with vintage items — music boxes, old cameras, porcelain "
            "figurines, stacks of books. A large wooden counter with a brass lamp casting warm "
            "pool of light. Sunlight streams through a large front window. Dust motes visible "
            "in the light. An orange cat sleeps on the counter. Warm gold and brown palette, "
            "detailed environment art."
        ),
        "width": 1216,
        "height": 832,
    },
    {
        "id": "scene03_shop_counter",
        "name": "Scene 3 — Shop Counter Close-up",
        "prompt": (
            "Close-up anime illustration of a worn wooden antique shop counter, warm brass lamp "
            "light illuminating the scene. An old brass key rests on the wood, surrounded by "
            "scattered antique trinkets — a pocket watch, a small glass vial, dried pressed "
            "flowers. Warm intimate atmosphere, shallow depth of field, detailed texture on the "
            "wood grain, high quality anime still image."
        ),
        "width": 1024,
        "height": 1024,
    },
    {
        "id": "scene04_mei_apartment",
        "name": "Scene 4 — Mei's Apartment",
        "prompt": (
            "Anime interior illustration of a small cozy artist's apartment at sunset. A wooden "
            "desk sits by the window with a warm desk lamp, scattered loose sketches, a set of "
            "colorful markers, and a half-empty boba tea cup. The walls are covered with vibrant "
            "anime-style illustrations pinned up in a collage. A small green potted plant sits "
            "on the windowsill, light blue curtains billowing gently in the breeze. Warm orange "
            "and lavender sunlight streams through the window, casting long soft shadows across "
            "the wooden floor. A low bookshelf in the corner is stacked with manga and art books. "
            "Cozy lived-in atmosphere, highly detailed background art, wallpaper quality, "
            "Makoto Shinkai style lighting."
        ),
        "width": 1216,
        "height": 832,
    },
    {
        "id": "scene05_rooftop_night",
        "name": "Scene 5 — Rooftop at Night",
        "prompt": (
            "Wide anime illustration of a rooftop overlooking an old Japanese district at night. "
            "The rooftop is concrete with a rusty metal railing, a small abandoned potted plant "
            "in the corner, and scattered loose bricks. The city stretches far below in a sea of "
            "tiny warm lights against a deep blue starlit sky. A crescent moon hangs softly in "
            "the upper right with scattered stars. The old district below features traditional "
            "tile roofs mixed with low modern buildings. Cinematic wide shot, atmospheric, "
            "emotional, Studio Ghibli inspired night scene, highly detailed background art, "
            "wallpaper quality."
        ),
        "width": 1216,
        "height": 832,
    },
    {
        "id": "scene06_workbench",
        "name": "Scene 6 — Workbench Detail",
        "prompt": (
            "Close-up anime illustration of an antique repair workbench under a focused brass "
            "desk lamp. An opened brass music box with intricate gears and a tiny rotating "
            "ballerina visible inside. A small dark velvet-lined tray holds tiny screws, "
            "tweezers, and a precision screwdriver. A soft cotton cloth, a magnifying glass, "
            "and scattered handwritten repair notes with sketches lie beside them. Warm amber "
            "light from the lamp creates dramatic pool of light and deep shadows on the dark "
            "scratched wood surface. Intimate detailed scene, high quality anime still frame, "
            "shallow depth of field."
        ),
        "width": 1024,
        "height": 1024,
    },
    {
        "id": "scene07_rainy_street",
        "name": "Scene 7 — Rainy Street",
        "prompt": (
            "Anime illustration of the old narrow Japanese street in heavy rain at dusk. The "
            "cobblestone street is slick and highly reflective, rain puddles mirroring the warm "
            "amber glow of hanging street lamps and shop windows. Raindrops streak diagonally "
            "across the frame in sharp focus. In the background, the antique shop's warm golden "
            "window light stands out against the blue-grey downpour. Wet wooden shutters, "
            "dripping eaves, a parked bicycle with rain running off it. Blue-grey cool atmosphere "
            "contrasted with warm golden shop light. Cinematic, emotional, highly detailed, "
            "Makoto Shinkai rain scene quality."
        ),
        "width": 1216,
        "height": 832,
    },
    {
        "id": "scene08_attic",
        "name": "Scene 8 — The Attic",
        "prompt": (
            "Anime interior illustration of a dusty attic in an old wooden building at "
            "afternoon. A single rectangular skylight in the sloped roof casts dramatic golden "
            "shafts of light through the air, dust motes floating thickly in the beams. An old "
            "locked wooden chest with iron bands sits in the center of the frame, surrounded by "
            "stacked fabric-covered trunks, an old rocking chair, and cobwebs in the corners. "
            "Warm mysterious atmosphere, light rays cutting through darkness, high contrast "
            "between bright and shadow areas, highly detailed environment art, wallpaper quality."
        ),
        "width": 1216,
        "height": 832,
    },
    {
        "id": "scene09_rooftop_sunrise",
        "name": "Scene 9 — Rooftop at Sunrise",
        "prompt": (
            "Wide anime illustration of a rooftop at sunrise. The sky is painted in soft pink, "
            "peach, and orange gradients with wispy cirrus clouds stretching across. The city "
            "below is bathed in warm golden morning light, the old district slowly waking up "
            "with faint smoke from chimneys. The concrete rooftop floor is lit with a warm "
            "glow, the rusty railing catches the first light. A small potted plant in the "
            "corner. Peaceful, emotional, hopeful atmosphere, cinematic wide composition. High "
            "quality anime background art, Makoto Shinkai style dawn scene."
        ),
        "width": 1216,
        "height": 832,
    },
    {
        "id": "scene10_shop_morning",
        "name": "Scene 10 — The Shop Morning",
        "prompt": (
            "Anime illustration of an antique shop exterior in bright fresh morning sunlight. "
            "The front display window is now filled with colorful hand-drawn illustrations "
            "arranged beautifully behind the glass. The wooden front door stands open, warm "
            "interior light spills onto the cobblestone step. A new polished brass sign hangs "
            "above the entrance reading \"Moonlight Exchange — Open.\" Potted plants flank the "
            "doorway, morning light creates soft lens flare. Clean fresh white and gold "
            "lighting, cheerful hopeful atmosphere, highly detailed environment art, Makoto "
            "Shinkai style, wallpaper quality."
        ),
        "width": 1216,
        "height": 832,
    },
]

# ── ComfyUI API Helpers ─────────────────────────────────────────────────

def queue_prompt(workflow: dict) -> str:
    """Submit workflow to ComfyUI, return prompt_id."""
    data = json.dumps({"prompt": workflow, "client_id": "moonlight-exchange"}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req).read())
        return resp["prompt_id"]
    except urllib.error.HTTPError as e:
        print(f"  [ERROR] HTTP {e.code}: {e.read().decode()}")
        return None


def wait_for_completion(prompt_id: str, timeout: int = 300) -> dict:
    """Poll ComfyUI until the prompt completes or fails."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = json.loads(
                urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}").read()
            )
            if prompt_id in resp:
                status = resp[prompt_id].get("status", {})
                if status.get("completed", False) or status.get("status_str") == "error":
                    return resp[prompt_id]
        except Exception:
            pass
        time.sleep(2)
    print(f"  [TIMEOUT] {timeout}s exceeded for {prompt_id}")
    return None


def get_images(prompt_id: str) -> list[dict]:
    """Retrieve output images for a completed prompt."""
    try:
        hist = json.loads(
            urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}").read()
        )
        outputs = hist[prompt_id].get("outputs", {})
        images = []
        for node_id, node_out in outputs.items():
            for img in node_out.get("images", []):
                images.append({
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                })
        return images
    except Exception as e:
        print(f"  [ERROR] Failed to get images: {e}")
        return []


def download_image(filename: str, subfolder: str, img_type: str, save_path: Path):
    """Download image from ComfyUI output to local file."""
    params = urllib.parse.urlencode({
        "filename": filename,
        "subfolder": subfolder,
        "type": img_type,
    })
    url = f"{COMFYUI_URL}/view?{params}"
    urllib.request.urlretrieve(url, save_path)


# ── Workflow Builder ────────────────────────────────────────────────────

def build_workflow(prompt: str, width: int, height: int, seed: int, prefix: str) -> dict:
    """Build a ComfyUI API workflow JSON for Z-Anime Distill-8Step."""
    return {
        # Save Image
        "9": {
            "inputs": {
                "filename_prefix": prefix,
                "images": ["65", 0],
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
        # CLIP Loader
        "62": {
            "inputs": {
                "clip_name": CLIP,
                "type": "lumina2",
                "device": "default",
            },
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP"},
        },
        # VAE Loader
        "63": {
            "inputs": {"vae_name": VAE},
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"},
        },
        # Negative (ConditioningZeroOut)
        "64": {
            "inputs": {"conditioning": ["67", 0]},
            "class_type": "ConditioningZeroOut",
            "_meta": {"title": "ConditioningZeroOut"},
        },
        # VAE Decode
        "65": {
            "inputs": {
                "samples": ["70", 0],
                "vae": ["63", 0],
            },
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        # UNET Loader
        "66": {
            "inputs": {
                "unet_name": MODEL,
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
            "_meta": {"title": "Load Diffusion Model"},
        },
        # CLIP Text Encode (Positive)
        "67": {
            "inputs": {
                "text": prompt,
                "clip": ["62", 0],
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
        # Empty Latent
        "68": {
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
            "class_type": "EmptySD3LatentImage",
            "_meta": {"title": "Empty Latent Image"},
        },
        # Model Sampling AuraFlow
        "69": {
            "inputs": {
                "shift": SHIFT,
                "model": ["66", 0],
            },
            "class_type": "ModelSamplingAuraFlow",
            "_meta": {"title": "ModelSamplingAuraFlow"},
        },
        # KSampler
        "70": {
            "inputs": {
                "seed": seed,
                "steps": STEPS,
                "cfg": CFG,
                "sampler_name": SAMPLER,
                "scheduler": SCHEDULER,
                "denoise": 1,
                "model": ["69", 0],
                "positive": ["67", 0],
                "negative": ["64", 0],
                "latent_image": ["68", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
        },
    }


# ── Generator ───────────────────────────────────────────────────────────

def generate_one(item: dict, save_dir: Path, project: str = "default") -> bool:
    """Generate a single image and save to disk."""
    seed = random.randint(0, 2**63)
    prefix = f"{project}/{item['id']}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{item['id']}.png"

    if save_path.exists():
        print(f"  [SKIP] {item['name']} — already exists")
        return True

    print(f"  [GEN]  {item['name']} ({item['width']}x{item['height']}, seed={seed})")

    workflow = build_workflow(item["prompt"], item["width"], item["height"], seed, prefix)
    prompt_id = queue_prompt(workflow)
    if not prompt_id:
        return False

    result = wait_for_completion(prompt_id)
    if not result:
        return False

    status = result.get("status", {})
    if status.get("status_str") == "error":
        print(f"  [FAIL] {item['name']}: {status.get('messages', [])}")
        return False

    images = get_images(prompt_id)
    if not images:
        print(f"  [FAIL] {item['name']}: no images in output")
        return False

    img = images[0]
    download_image(img["filename"], img["subfolder"], img["type"], save_path)
    print(f"  [OK]   {item['name']} -> {save_path}")
    return True


def generate_all(items: list, save_dir: Path, label: str, project: str = "default"):
    """Generate a batch of images."""
    print(f"\n{'='*60}")
    print(f"  Generating {len(items)} {label}")
    print(f"  Output: {save_dir}")
    print(f"  Model:  Z-Anime Distill-8Step ({STEPS} steps, CFG {CFG})")
    print(f"{'='*60}")

    success, fail = 0, 0
    for item in items:
        if generate_one(item, save_dir, project):
            success += 1
        else:
            fail += 1

    print(f"\n  Done: {success} OK, {fail} failed\n")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Moonlight Exchange — Z-Anime Image Generator")
    parser.add_argument("--project", type=str, default="default",
                        help="Project name (output goes to generated/<project>/)")
    parser.add_argument("--chars", action="store_true", help="Generate character sheets only")
    parser.add_argument("--bgs", action="store_true", help="Generate backgrounds only")
    parser.add_argument("--scene", type=int, help="Generate a specific scene background (1-10)")
    parser.add_argument("--seed", type=int, default=None, help="Base seed (for reproducibility)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Compute per-project output directories
    project_dir = BASE_DIR / args.project
    chars_dir = project_dir / "characters"
    bgs_dir = project_dir / "backgrounds"

    print(f"  Project: {args.project}")
    print(f"  Output:  {project_dir}")

    # Verify ComfyUI is running
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5)
    except Exception:
        print(f"[ERROR] Cannot connect to ComfyUI at {COMFYUI_URL}")
        print("  Make sure ComfyUI is running before using this script.")
        sys.exit(1)

    if args.scene:
        idx = args.scene - 1
        if 0 <= idx < len(BACKGROUNDS):
            generate_one(BACKGROUNDS[idx], bgs_dir, args.project)
        else:
            print(f"[ERROR] Scene {args.scene} out of range (1-{len(BACKGROUNDS)})")
    elif args.chars:
        generate_all(CHARACTERS, chars_dir, "character sheets", args.project)
    elif args.bgs:
        generate_all(BACKGROUNDS, bgs_dir, "background scenes", args.project)
    else:
        generate_all(CHARACTERS, chars_dir, "character sheets", args.project)
        generate_all(BACKGROUNDS, bgs_dir, "background scenes", args.project)


if __name__ == "__main__":
    main()
