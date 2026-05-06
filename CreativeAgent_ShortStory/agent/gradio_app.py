"""
CreativeAgent — Gradio Web UI
Web interface for the LangGraph story creation pipeline.

Usage:
    python agent/gradio_app.py              # Launch on localhost:7860
    python agent/gradio_app.py --port 8080  # Custom port
    python agent/gradio_app.py --share      # Public share link
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from story_agent import (
    create_story_agent,
    StoryState,
    PROJECTS_DIR,
    CURRENT_PROJECT_FILE,
)

# ── Config ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6:27b")
GENERATED_BASE = Path(__file__).parent.parent / "generated"


def make_initial_state() -> dict:
    """Create a fresh StoryState."""
    return {
        "messages": [],
        "project_name": "", "genre": "", "tone": "", "logline": "", "story_length": "",
        "acts": [], "subplots": [], "narrative_arc": "",
        "characters": [], "scenes": [],
        "character_prompts": [], "background_prompts": [], "video_prompts": [],
        "phase": "idea", "revision_count": 0,
        "created_at": datetime.now().isoformat(), "updated_at": "",
    }


def load_project_state(project_name: str) -> dict:
    """Load a project's state from disk."""
    state_file = PROJECTS_DIR / project_name / "story_state.json"
    if not state_file.exists():
        return None
    data = json.loads(state_file.read_text(encoding="utf-8"))
    CURRENT_PROJECT_FILE.write_text(project_name)
    state = make_initial_state()
    state.update(data)
    return state


def save_current_state(state: dict) -> str:
    """Save the current state to disk."""
    project_name = state.get("project_name", "")
    if not project_name:
        return "No project name set"

    project_dir = PROJECTS_DIR / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    CURRENT_PROJECT_FILE.write_text(project_name)

    state["updated_at"] = datetime.now().isoformat()
    state_file = project_dir / "story_state.json"
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    return f"Saved to {state_file}"


def list_all_projects() -> list:
    """List all saved project names."""
    if not PROJECTS_DIR.exists():
        return []
    return [d.name for d in PROJECTS_DIR.iterdir()
            if d.is_dir() and d.name != "_archive" and (d / "story_state.json").exists()]


def get_phase_display(state: dict) -> str:
    """Get a formatted phase status display."""
    phase = state.get("phase", "idea")
    phases = ["idea", "architect", "characters", "scenes", "prompts", "review", "done"]
    idx = phases.index(phase) if phase in phases else 0

    lines = ["### Pipeline Progress\n"]
    for i, p in enumerate(phases):
        if i < idx:
            lines.append(f"- [x] **{p.title()}**")
        elif i == idx:
            lines.append(f"- [>] **{p.title()}** ← current")
        else:
            lines.append(f"- [ ] {p.title()}")

    lines.append(f"\n**Project:** {state.get('project_name', '(unnamed)')}")
    lines.append(f"**Genre:** {state.get('genre', '-')}")
    lines.append(f"**Length:** {state.get('story_length', '-')}")
    lines.append(f"**Scenes:** {len(state.get('scenes', []))}")
    lines.append(f"**Characters:** {len(state.get('characters', []))}")
    lines.append(f"**Acts:** {len(state.get('acts', []))}")

    return "\n".join(lines)


def get_state_summary(state: dict) -> str:
    """Get a formatted summary of the current story state."""
    parts = []

    if state.get("project_name"):
        parts.append(f"# {state['project_name']}")
    if state.get("logline"):
        parts.append(f"\n**Logline:** {state['logline']}")
    if state.get("narrative_arc"):
        parts.append(f"\n**Narrative Arc:** {state['narrative_arc']}")

    acts = state.get("acts", [])
    if acts:
        parts.append("\n## Acts")
        for act in acts:
            parts.append(f"- **Act {act.get('act_num', '?')}:** {act.get('title', '?')} — {act.get('theme', '')}")

    chars = state.get("characters", [])
    if chars:
        parts.append("\n## Characters")
        for c in chars:
            parts.append(f"- **{c.get('name', '?')}** ({c.get('role', '?')}): {c.get('appearance', '')[:100]}...")

    scenes = state.get("scenes", [])
    if scenes:
        parts.append(f"\n## Scenes ({len(scenes)} total)")
        for s in scenes[:10]:
            parts.append(f"- **Scene {s.get('num', '?')}:** {s.get('title', '?')} — {s.get('location', '')}")
        if len(scenes) > 10:
            parts.append(f"- ... and {len(scenes) - 10} more")

    subplots = state.get("subplots", [])
    if subplots:
        parts.append("\n## Subplots")
        for sp in subplots:
            parts.append(f"- **{sp.get('name', '?')}:** {sp.get('arc', '')}")

    return "\n".join(parts) if parts else "No story data yet. Start chatting to create your story!"


def get_prompts_export(state: dict) -> str:
    """Generate a markdown export of all prompts."""
    lines = [
        f"# {state.get('project_name', 'Untitled')} — Prompt Export",
        f"\n**Genre:** {state.get('genre', 'N/A')}",
        f"**Tone:** {state.get('tone', 'N/A')}",
        f"**Logline:** {state.get('logline', 'N/A')}",
        f"**Story Length:** {state.get('story_length', 'N/A')}",
        f"\n---\n",
    ]

    char_prompts = state.get("character_prompts", [])
    if char_prompts:
        lines.append("## Character Prompts\n")
        for p in char_prompts:
            lines.append(f"### {p.get('name', '?')} ({p.get('type', '?')})")
            lines.append(f"```\n{p.get('prompt', '')}\n```\n")

    bg_prompts = state.get("background_prompts", [])
    if bg_prompts:
        lines.append("## Background Prompts\n")
        for p in bg_prompts:
            lines.append(f"### Scene {p.get('scene_num', '?')}: {p.get('scene_name', '?')}")
            lines.append(f"```\n{p.get('prompt', '')}\n```\n")

    vid_prompts = state.get("video_prompts", [])
    if vid_prompts:
        lines.append("## Video Prompts\n")
        for p in vid_prompts:
            lines.append(f"### Scene {p.get('scene_num', '?')}")
            lines.append(f"**Global:** {p.get('global_prompt', '')}")
            lines.append(f"```\n{p.get('local_prompts', '')}\n```\n")

    return "\n".join(lines)


# ── Gradio App ──────────────────────────────────────────────────────────

def build_app():
    """Build the Gradio application."""

    # Shared state across tabs
    app_state = {"graph": None, "state": make_initial_state(), "thread_id": "gradio-session"}

    def get_graph():
        if app_state["graph"] is None:
            app_state["graph"] = create_story_agent()
        return app_state["graph"]

    def get_config():
        return {"configurable": {"thread_id": app_state["thread_id"]}}

    # ── Chat Function ────────────────────────────────────────────────

    def chat(message: str, history: list):
        """Process a chat message through the LangGraph agent."""
        if not message.strip():
            return history, ""

        state = app_state["state"]
        state["messages"].append(HumanMessage(content=message))

        graph = get_graph()
        try:
            result = graph.invoke(state, get_config())
        except Exception as e:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"Error: {str(e)}"})
            return history, ""

        app_state["state"] = result

        # Collect assistant responses
        ai_responses = []
        tool_msgs = []
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.content:
                ai_responses.append(msg.content)
            elif isinstance(msg, ToolMessage) and msg.content:
                tool_msgs.append(msg.content[:200])

        # Add to chat history
        history.append({"role": "user", "content": message})

        response = "\n\n".join(ai_responses) if ai_responses else "(No response)"
        if tool_msgs:
            response += "\n\n---\n**Tools:**\n" + "\n".join(f"- {t}" for t in tool_msgs)

        history.append({"role": "assistant", "content": response})

        return history, ""

    # ── Project Management ───────────────────────────────────────────

    def create_project(name: str):
        """Create a new project."""
        if not name.strip():
            return "Please enter a project name", gr.update(), get_phase_display(app_state["state"])

        project_dir = PROJECTS_DIR / name.strip()
        project_dir.mkdir(parents=True, exist_ok=True)

        state = make_initial_state()
        state["project_name"] = name.strip()
        state["messages"] = [
            HumanMessage(content=f"I want to create a new story called '{name}'. Help me plan it out.")
        ]

        app_state["state"] = state
        app_state["thread_id"] = f"gradio-{name}-{datetime.now().strftime('%H%M%S')}"

        # Auto-save
        save_current_state(state)

        return (
            f"Project '{name}' created!",
            gr.update(choices=list_all_projects(), value=name),
            get_phase_display(state),
        )

    def load_project(name: str):
        """Load an existing project."""
        if not name:
            return "Select a project", gr.update(), get_phase_display(app_state["state"])

        state = load_project_state(name)
        if not state:
            return f"Project '{name}' not found", gr.update(), get_phase_display(app_state["state"])

        app_state["state"] = state
        app_state["thread_id"] = f"gradio-{name}-{datetime.now().strftime('%H%M%S')}"

        return (
            f"Loaded '{name}' (phase: {state.get('phase', '?')})",
            gr.update(choices=list_all_projects(), value=name),
            get_phase_display(state),
        )

    def refresh_projects():
        """Refresh the project list."""
        projects = list_all_projects()
        return gr.update(choices=projects)

    # ── Review & Export ──────────────────────────────────────────────

    def refresh_review():
        """Refresh the review display."""
        state = app_state["state"]
        return get_state_summary(state), get_phase_display(state)

    def export_prompts():
        """Export prompts to markdown."""
        state = app_state["state"]
        return get_prompts_export(state)

    def save_state_manual():
        """Manually save the current state."""
        state = app_state["state"]
        result = save_current_state(state)
        return result

    # ── Settings ─────────────────────────────────────────────────────

    def update_settings(model: str, ollama_url: str):
        """Update LLM settings (requires restart for model change)."""
        os.environ["LLM_MODEL"] = model
        os.environ["OLLAMA_BASE_URL"] = ollama_url
        # Reset graph so it picks up new settings
        app_state["graph"] = None
        return f"Settings updated: model={model}, url={ollama_url}. Graph will be recreated on next message."

    # ── Build UI ─────────────────────────────────────────────────────

    with gr.Blocks(
        title="CreativeAgent — Story Studio",
    ) as app:

        gr.Markdown("# CreativeAgent — Story Studio")
        gr.Markdown("LangGraph-powered anime story creation pipeline")
        gr.Markdown("[GitHub — benjiyaya](https://github.com/benjiyaya)")

        with gr.Tabs():

            # ── Tab 1: Story Creation Chat ───────────────────────────
            with gr.Tab("Story Creation"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Story Agent",
                            height=500,
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Describe your story idea, or type a command...",
                                label="Message",
                                scale=4,
                                lines=2,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)

                        gr.Examples(
                            examples=[
                                "I want to create a sci-fi romance anime about time travel",
                                "Let's design the characters now",
                                "Build the scene outlines",
                                "Generate the image and video prompts",
                                "Review the story for consistency",
                                "Make it longer — add more acts and scenes",
                            ],
                            inputs=msg_input,
                            label="Quick Commands",
                        )

                    with gr.Column(scale=1):
                        phase_display = gr.Markdown(
                            value=get_phase_display(app_state["state"]),
                            label="Pipeline Status",
                            elem_classes=["phase-indicator"],
                        )
                        refresh_btn = gr.Button("Refresh Status")
                        save_btn = gr.Button("Save Project")
                        save_output = gr.Textbox(label="Save Status", lines=2, interactive=False)

                # Wire up chat
                msg_input.submit(chat, [msg_input, chatbot], [chatbot, msg_input])
                send_btn.click(chat, [msg_input, chatbot], [chatbot, msg_input])
                refresh_btn.click(refresh_review, [], [gr.Markdown(visible=False), phase_display])
                save_btn.click(save_state_manual, [], [save_output])

            # ── Tab 2: Project Management ────────────────────────────
            with gr.Tab("Projects"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Create New Project")
                        new_project_name = gr.Textbox(
                            label="Project Name",
                            placeholder="my-anime-story",
                        )
                        create_btn = gr.Button("Create Project", variant="primary")
                        create_output = gr.Textbox(label="Status", lines=2, interactive=False)

                    with gr.Column():
                        gr.Markdown("### Load Existing Project")
                        project_dropdown = gr.Dropdown(
                            choices=list_all_projects(),
                            label="Select Project",
                            interactive=True,
                        )
                        load_btn = gr.Button("Load Project")
                        load_output = gr.Textbox(label="Status", lines=2, interactive=False)
                        refresh_list_btn = gr.Button("Refresh List")

                # Wire up project management
                create_btn.click(
                    create_project,
                    [new_project_name],
                    [create_output, project_dropdown, phase_display],
                )
                load_btn.click(
                    load_project,
                    [project_dropdown],
                    [load_output, project_dropdown, phase_display],
                )
                refresh_list_btn.click(refresh_projects, [], [project_dropdown])

            # ── Tab 3: Review & Export ───────────────────────────────
            with gr.Tab("Review & Export"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Story Overview")
                        review_display = gr.Markdown(
                            value="Load or create a project to see its overview.",
                            label="Story Summary",
                        )
                        refresh_review_btn = gr.Button("Refresh")

                    with gr.Column():
                        gr.Markdown("### Prompt Export")
                        export_display = gr.Code(
                            language="markdown",
                            label="Exported Prompts",
                            lines=25,
                        )
                        export_btn = gr.Button("Generate Export")

                refresh_review_btn.click(refresh_review, [], [review_display, phase_display])
                export_btn.click(export_prompts, [], [export_display])

            # ── Tab 4: Settings ─────────────────────────────────────
            with gr.Tab("Settings"):
                gr.Markdown("### LLM Configuration")
                gr.Markdown("Changes take effect on the next message sent to the agent.")

                model_input = gr.Textbox(
                    value=LLM_MODEL,
                    label="Ollama Model",
                    placeholder="qwen3.6:27b",
                )
                url_input = gr.Textbox(
                    value=OLLAMA_BASE_URL,
                    label="Ollama Base URL",
                    placeholder="http://127.0.0.1:11434",
                )
                settings_btn = gr.Button("Apply Settings", variant="primary")
                settings_output = gr.Textbox(label="Status", lines=2, interactive=False)

                settings_btn.click(
                    update_settings,
                    [model_input, url_input],
                    [settings_output],
                )

                gr.Markdown("---")
                gr.Markdown("### Generated Output Structure")
                gr.Markdown(
                    "Each project outputs to `generated/<project_name>/` with subfolders:\n"
                    "- `characters/` — Character sheet images\n"
                    "- `backgrounds/` — Scene background images\n"
                    "- `videos/` — Generated video clips\n\n"
                    "Use `python generate_all.py --project <name>` to generate images.\n"
                    "Use `python generate_video.py --project <name>` to generate videos."
                )

    return app


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CreativeAgent — Gradio Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
