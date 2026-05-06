# CreativeAgent — Story Studio

LangGraph-powered anime story creation pipeline with Gradio web UI.

## Installation

```bash
# Clone or navigate to the project
cd CreativeAgent_ShortStory

# Install dependencies
pip install -r agent/requirements.txt
```

### Requirements

- **Python 3.10+**
- **Ollama** running locally with a compatible model (default: `qwen3.6:27b`)
- **ComfyUI** (optional, for image/video generation)
- **DASHSCOPE_API_KEY** (optional, for Wan 2.7 video generation)

### Quick Ollama Setup

```bash
# Install Ollama: https://ollama.com
# Pull the default model
ollama pull qwen3.6:27b

# Or use a different model
export LLM_MODEL=llama3.1:70b
```

## Usage

### Web UI (Recommended)

```bash
# Launch on localhost:7860
python agent/gradio_app.py

# Custom port
python agent/gradio_app.py --port 8080

# Public share link
python agent/gradio_app.py --share
```

### CLI — Story Agent

```bash
# Interactive mode
python agent/story_agent.py -i

# Continue a saved project
python agent/story_agent.py -i --resume

# List all projects
python agent/story_agent.py --status

# Export prompts for a project
python agent/story_agent.py --export <project-name>
```

### Image Generation

```bash
# Generate all characters and backgrounds (default project)
python generate_all.py

# Generate for a specific project
python generate_all.py --project my-story

# Characters only
python generate_all.py --project my-story --chars

# Backgrounds only
python generate_all.py --project my-story --bgs

# Single scene
python generate_all.py --project my-story --scene 3
```

### Video Generation

```bash
# Generate all scenes
python generate_video.py --project my-story --all

# Single scene
python generate_video.py --project my-story --scene 1

# Scene range
python generate_video.py --project my-story --scene 1-3

# List scenes and status
python generate_video.py --project my-story --list

# Custom FPS and duration
python generate_video.py --project my-story --all --fps 30 --duration 15
```

## Story Pipeline Phases

The LangGraph agent guides stories through 6 phases:

| Phase | Description |
|-------|-------------|
| **Idea** | Story concept, genre, tone, logline, scene outline |
| **Architect** | Act structure, subplots, narrative arc, pacing map |
| **Characters** | Detailed character designs with arcs and relationships |
| **Scenes** | Scene backgrounds with lighting, mood, color palette |
| **Prompts** | AI-optimized image and video prompts |
| **Review** | Consistency check across all generated content |

### Story Length Options

| Length | Scenes | Structure |
|--------|--------|-----------|
| Short | 8-12 | Single focused arc, 2-3 characters |
| Medium | 15-20 | Full 3-act structure, 3-5 characters, subplots |
| Long | 20-30+ | Multi-act epic, 4-8 characters, multiple subplots |

## Output Structure

Each project gets its own folder under `generated/`:

```
generated/
├── my-story/
│   ├── characters/       # Character sheet images
│   ├── backgrounds/      # Scene background images
│   └── videos/           # Generated video clips
└── another-story/
    ├── characters/
    ├── backgrounds/
    └── videos/
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `LLM_MODEL` | `qwen3.6:27b` | Ollama model name |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI API endpoint |
| `DASHSCOPE_API_KEY` | *(none)* | DashScope API key for Wan 2.7 video |

## Project Files

Saved projects live in `agent/projects/<project-name>/`:

```
agent/projects/
├── _current_project.txt       # Points to active project
└── my-story/
    ├── story_state.json       # Full pipeline state
    └── prompts_export.md      # Exported prompts for execution
```
