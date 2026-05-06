# AGENTS.md — CreativeAgent Short Story

Run the AI story creation pipeline from the command line using the LangGraph agent in `./agent/`.

## Prerequisites

```bash
pip install -r agent/requirements.txt
```

- **Ollama** running locally (default model: `qwen3.6:27b`)
- **ComfyUI** running locally at `http://127.0.0.1:8188` with Z-Anime or Z Image Turbo model loaded

## Story Creation Agent (`agent/story_agent.py`)

The story agent guides you through 6 creative phases: Idea, Architect, Characters, Scenes, Prompts, Review.

### Start a New Story

```bash
python agent/story_agent.py -i
```

Opens an interactive chat where the agent proposes story concepts, designs characters, builds scenes, and generates optimized prompts. Type your ideas and the agent responds with structured output for each phase.

### Continue a Saved Project

```bash
python agent/story_agent.py -i --resume
```

Loads the last saved project and picks up where you left off.

### List All Projects

```bash
python agent/story_agent.py --status
```

### Export Prompts for Execution

```bash
python agent/story_agent.py --export <project-name>
```

Exports all generated prompts to `agent/projects/<project-name>/prompts_export.md` for use with image generation scripts.

## Story Phases

| Phase | What Happens |
|-------|-------------|
| **Idea** | Propose title, genre, tone, logline, scene outline |
| **Architect** | Act structure, subplots, narrative arc, pacing map |
| **Characters** | Detailed character designs with arcs and relationships |
| **Scenes** | Scene backgrounds with lighting, mood, color palette |
| **Prompts** | AI-optimized image prompts for Z-Anime / Z Image Turbo |
| **Review** | Consistency check across all generated content |

### Navigating Phases

During the interactive session, tell the agent what you want:
- "Let's design characters" — moves to Character phase
- "Build the scenes" — moves to Scene phase
- "Generate prompts" — moves to Prompt phase
- "Review everything" — moves to Review phase
- "Fix the character in scene 5" — revises within current phase
- "Export" — exports prompts and marks project done

## Pipeline Agent (`agent/main.py`)

For running the image generation pipeline after story creation:

```bash
# Interactive mode
python agent/main.py -i

# Generate a single scene
python agent/main.py --scene 1

# Generate scenes 1 through 3
python agent/main.py --scene 1-3

# Generate all scenes
python agent/main.py --all

# Check scene status
python agent/main.py --status
```

## Direct Pipeline (`agent/direct_pipeline.py`)

Run the image generation pipeline without LLM reasoning — direct tool execution:

```bash
# Generate background for a single scene
python direct_pipeline.py --scene 1

# Scenes 1 through 3
python direct_pipeline.py --scene 1-3

# All scenes
python direct_pipeline.py --all

# Check status
python direct_pipeline.py --status
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `LLM_MODEL` | `qwen3.6:27b` | Ollama model name |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI API endpoint |

## Project Structure

```
agent/
  story_agent.py        # Story creation agent (LangGraph, interactive)
  main.py               # Pipeline agent (image generation)
  direct_pipeline.py    # Direct pipeline runner (no LLM)
  gradio_app.py         # Web UI (alternative to CLI)
  requirements.txt      # Python dependencies
  projects/
    <project-name>/
      story_state.json   # Full pipeline state
      prompts_export.md  # Exported prompts for execution

generated/
  characters/            # Character sheet images
  backgrounds/           # Scene background images
```

## Typical Workflow

1. `python agent/story_agent.py -i` — create your story interactively
2. Refine characters, scenes, and prompts through the chat
3. Export prompts when satisfied
4. `python agent/main.py --all` or `python direct_pipeline.py --all` — generate character and background images
5. Check `generated/` for output files
