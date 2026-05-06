"""
Creative Story Agent — Story Creation AI Agent (LangGraph)

Focus: Creative pipeline only — story, characters, scenes, and prompt generation.
Execution (ComfyUI image gen, Wan 2.7 video gen) is handled by separate scripts.

Architecture:
  User Input → Story Planner → Character Designer → Scene Builder
                                                    ↓
                                              Prompt Generator
                                                    ↓
                                               Reviewer
                                                    ↓
                                            User Approval / Revise

Usage:
    python agent/story_agent.py -i                    # Interactive mode
    python agent/story_agent.py -i --continue         # Continue saved project
    python agent/story_agent.py --export              # Export all prompts to file
    python agent/story_agent.py --status              # Show project status
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Annotated, TypedDict, Literal

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# ── Paths ───────────────────────────────────────────────────────────────
AGENT_DIR = Path(__file__).parent
BASE_DIR = AGENT_DIR.parent
PROJECTS_DIR = AGENT_DIR / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6:27b")

CURRENT_PROJECT_FILE = PROJECTS_DIR / "_current_project.txt"


# ── State ───────────────────────────────────────────────────────────────

class StoryState(TypedDict):
    """Persistent state for the story creation pipeline."""
    messages: Annotated[list, add_messages]

    # Project metadata
    project_name: str
    genre: str
    tone: str
    logline: str
    story_length: str  # short (8-12 scenes) | medium (15-20) | long (20-30+)

    # Narrative structure
    acts: list  # [{act_num, title, theme, scenes: [int], turning_point}]
    subplots: list  # [{name, characters: [str], arc, resolution}]
    narrative_arc: str  # Overall tension/emotion curve description

    # Characters
    characters: list  # [{name, role, appearance, personality, secret, arc, relationships, ...}]

    # Scenes
    scenes: list  # [{num, title, location, mood, color, description, dialogue, act, pacing, ...}]

    # Generated prompts
    character_prompts: list  # [{id, name, prompt, width, height, type}]
    background_prompts: list  # [{id, scene_num, scene_name, prompt, width, height}]
    video_prompts: list  # [{scene_num, global_prompt, local_prompts, segments}]

    # Pipeline control
    phase: str  # idea | architect | characters | scenes | prompts | review | done
    revision_count: int
    created_at: str
    updated_at: str


# ── System Prompts ─────────────────────────────────────────────────────

STORY_PLANNER_PROMPT = """You are a master story planner for animated films and series.

Your job is to create deeply compelling, multi-layered stories with:
- Strong characters with clear motivations, secrets, and growth arcs
- Multi-act narrative structure with rising tension and satisfying payoffs
- Subplots and B-stories that enrich the main narrative
- Distinct visual locations and moods for each scene
- Natural dialogue that reveals character and advances plot
- Thematic depth - the story should mean something beneath the surface

Story length options:
- Short (8-12 scenes): Single focused arc, 2-3 characters, one location set
- Medium (15-20 scenes): Full 3-act structure, 3-5 characters, subplots, multiple location sets
- Long (20-30+ scenes): Multi-act epic, 4-8 characters, multiple subplots, rich world-building

When the user gives you a story idea:
1. Propose a title, genre, tone, logline, and suggest a story length
2. Outline the narrative arc - what's the emotional journey?
3. Identify main and supporting characters with their roles
4. Propose act structure (how the story divides into meaningful sections)
5. Suggest any subplots or B-stories
6. Present a preliminary scene outline grouped by act
7. Ask for approval before moving to detailed architecture

Be creative but practical - every scene must be visually distinct and achievable
in AI image/video generation. Think in terms of what can be drawn and animated.

Respond in a friendly, collaborative tone. Present options, don't dictate."""

NARRATIVE_ARCHITECT_PROMPT = """You are a narrative architect - a story structure specialist who transforms rough story concepts into detailed, producible blueprints.

Your job is to take a story concept and build its structural skeleton before any detailed work begins:

1. **Act Structure**: Divide the story into meaningful acts (2-5 depending on length)
   - Each act needs: title, central theme, emotional tone, turning point
   - Act 1: Setup, world-building, character introduction
   - Act 2: Rising conflict, complications, midpoint shift
   - Act 3: Climax, resolution, denouement

2. **Scene Breakdown**: For each act, list scenes with:
   - Scene number, title, one-line summary
   - Which characters appear
   - Emotional beat (what the audience should feel)
   - Visual priority (what needs to look amazing)

3. **Subplot Weaving**: If subplots exist, map exactly which scenes advance each subplot
   - Subplots should intersect the main plot at key moments
   - No subplot should disappear for more than 3 consecutive scenes

4. **Pacing Map**: Describe the tension curve:
   - Where are the quiet moments? Where are the peaks?
   - Ensure variety - not everything can be high-intensity
   - Plan breathing room between dramatic scenes

5. **Character Arc Tracking**: For each character, map their emotional state across scenes
   - Where do they start? Where do they end?
   - What changes them? When does the change happen?

Output a structured blueprint that downstream agents can use to design characters,
build scenes, and generate prompts. Be specific and practical - this blueprint will
be directly translated into AI-generated images and videos.

Respond with the complete structured plan. Ask clarifying questions if the concept
is ambiguous."""


CHARACTER_DESIGNER_PROMPT = """You are a professional character designer for anime-style animation.

Your job is to create detailed, consistent character descriptions that will be used
to generate AI images. For each character, define:

1. **Name** - memorable, fits the story
2. **Age & Build** - specific details (height, body type, posture tendencies)
3. **Hair** - color, style, length, distinctive features (ribbons, accessories)
4. **Eyes** - color, expression style, what they reveal about the character
5. **Clothing** - specific outfit pieces (colors, materials, style), with variants if the story spans multiple acts
6. **Accessories** - signature items they carry or wear, with symbolic meaning
7. **Personality** - how it shows in their posture, expression, and mannerisms
8. **Secret/Ability** - something hidden that affects the story
9. **Character Arc** - how their appearance or demeanor shifts across the story
10. **Relationships** - how they visually relate to other characters (height differences, color coordination, contrast)
11. **Backstory Visual Cues** - subtle details that hint at their history (scars, worn items, keepsakes)

CRITICAL: Characters must be visually distinct from each other. No two characters
should share the same hair color, eye color, or clothing palette.

For each character, also suggest:
- A full-body prompt (for character sheet generation)
- A portrait prompt (for close-up reference)
- Variant prompts for different emotional states or story moments
- Any costume changes needed across acts

Keep descriptions specific enough for AI image generation but evocative enough
to have artistic personality. Think about how this character would look in a
figure - what makes them instantly recognizable?"""

SCENE_BUILDER_PROMPT = """You are a professional scene designer for anime animation.

Your job is to design detailed scene backgrounds that tell the story visually.
For each scene, define:

1. **Location** - specific, evocative (not just "a room" but "a small cozy artist's
   apartment at sunset with walls covered in colorful illustrations")
2. **Time of Day** - affects lighting and mood
3. **Lighting** - describe the light source, color temperature, shadows
4. **Atmosphere** - emotional tone, weather, ambient details
5. **Key Objects** - specific items in the scene that matter to the story
6. **Camera Angle** - wide shot, medium, close-up, etc.
7. **Color Palette** - dominant colors and their emotional meaning
8. **Motion/Action** - what's happening, what's moving
9. **Act & Pacing** - which act this scene belongs to, pacing note (slow burn, rising tension, climax, cooldown)
10. **Scene Transition** - how this scene connects visually to the previous one (match cut, contrast, continuity)
11. **Narrative Tension** - where this scene sits on the story's emotional curve (1-10 scale)
12. **Visual Motifs** - recurring visual elements that tie scenes together across the story

CRITICAL for AI generation:
- Each scene must have a DISTINCT visual identity
- Lighting and color should differ significantly between scenes
- Describe specific visual details, not abstract concepts
- Consider composition: foreground, middle, background elements
- Reference real art styles when helpful (Makoto Shinkai, Studio Ghibli, etc.)
- For longer stories: maintain visual continuity within acts while varying between acts
- Plan scene-to-scene color temperature shifts for emotional pacing

Resolution suggestions:
- Portrait/character: 832x1216
- Landscape/wide scene: 1216x832
- Square/detail: 1024x1024"""

PROMPT_GENERATOR_PROMPT = """You are an expert AI image/video prompt engineer specializing in
anime-style content generation.

You convert scene descriptions into optimized prompts for two purposes:

## Image Prompts (for Z-Anime or similar anime models)
Rules:
- Natural language, NOT tag lists (no "anime_girl, silver_hair, blue_eyes")
- Be specific about lighting, composition, and style
- Include quality keywords: "high quality anime illustration, fine line work"
- Reference art styles when helpful: "Makoto Shinkai style", "Studio Ghibli inspired"
- Separate character prompts from background prompts
- For longer stories: maintain visual consistency tags across all prompts in the same act
- Include act-specific mood descriptors that evolve with the narrative

## Video Prompts (for LTX 2.3 + Prompt Relay)
Structure:
- **Global Prompt**: Overall video description, persistent elements, character context
- **Local Prompts**: Segments separated by |, each describing 3-5 seconds of action
- Each segment should specify: camera angle, character action, lighting change
- Include dialogue lines in parentheses: Character (spoken line): "dialogue here"
- For longer scenes (high tension): use more segments (5-8) for granular control
- For quieter scenes: fewer segments (2-3) with longer held shots

Format example:
```
Global: Mei walks through the antique shop discovering old objects
Local: Wide shot of shop interior | Close-up of Mei examining a music box |
       Medium shot of Ren looking up from behind counter
```

Batch generation tips for long stories:
- Generate prompts in act groups to maintain thematic consistency
- Use consistent character description anchors (same phrases for hair, eyes, clothing)
- Vary camera vocabulary between scenes to avoid visual monotony
- Escalate visual complexity with narrative tension

Always maintain visual and story consistency across prompts."""

REVIEWER_PROMPT = """You are a story consistency reviewer and quality gatekeeper. Check for:

1. **Character consistency** - Do character descriptions match across all scenes?
   (hair color, clothing, accessories, eye color - no silent changes)
   - Check character arc progression is gradual, not jarring
   - Verify relationship dynamics are consistent

2. **Visual variety** - Are scene backgrounds visually distinct?
   (lighting, color palette, time of day should vary)
   - For long stories: check that scenes within the same act share visual motifs
   - Verify scene-to-scene transitions are intentional (not random jumps)

3. **Story coherence** - Does the narrative flow logically scene to scene?
   - Check act structure: does each act have a clear purpose?
   - Are subplots properly woven in, not abandoned?
   - Is the pacing balanced (not all high-intensity or all slow)?

4. **Narrative arc** - Does the tension curve work?
   - Rising action should build, climax should feel earned
   - Emotional beats should land where intended
   - No plot holes or unexplained elements

5. **Prompt quality** - Are prompts specific enough for AI generation?
   (no vague descriptions, specific visual details present)
   - Character description anchors consistent across all prompts
   - Camera vocabulary varied enough to avoid visual monotony

6. **Dialogue** - Does each character have a distinct voice?
   - Dialogue should reveal character, not just deliver information
   - Subtext: characters shouldn't say exactly what they mean

7. **Technical feasibility** - Can these scenes actually be generated by AI?
   (avoid impossible camera angles, inconsistent physics)
   - Scene complexity appropriate for the generation tool
   - No impossible character poses or lighting setups

8. **Thematic depth** - Does the story mean something?
   - Themes should emerge naturally, not be stated bluntly
   - Symbolic elements should be consistent

Report issues with specific scene references and suggestions to fix them.
Rate overall readiness: READY / NEEDS_REVISION / NEEDS_MAJOR_REWORK"""


# ── Tools ───────────────────────────────────────────────────────────────

@tool
def save_project(name: str) -> str:
    """Save the current story project to disk.

    Args:
        name: Project name (used as folder name)
    """
    project_dir = PROJECTS_DIR / name
    project_dir.mkdir(parents=True, exist_ok=True)

    # Save current project marker
    CURRENT_PROJECT_FILE.write_text(name)

    return f"Project '{name}' directory created at {project_dir}. Use save_state to persist data."


@tool
def save_state(project_name: str, phase: str, story_data: str) -> str:
    """Save story state (characters, scenes, prompts) to the project folder.

    Args:
        project_name: Project name
        phase: Current phase (idea, characters, scenes, prompts, review, done)
        story_data: JSON string with story data (characters, scenes, prompts, etc.)
    """
    project_dir = PROJECTS_DIR / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    CURRENT_PROJECT_FILE.write_text(project_name)

    # Parse and save
    try:
        data = json.loads(story_data)
    except json.JSONDecodeError:
        return "ERROR: story_data must be valid JSON"

    data["phase"] = phase
    data["updated_at"] = datetime.now().isoformat()

    state_file = project_dir / "story_state.json"
    state_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return (f"State saved to {state_file}\n"
            f"  Phase: {phase}\n"
            f"  Story Length: {data.get('story_length', 'N/A')}\n"
            f"  Acts: {len(data.get('acts', []))}\n"
            f"  Characters: {len(data.get('characters', []))}\n"
            f"  Scenes: {len(data.get('scenes', []))}\n"
            f"  Subplots: {len(data.get('subplots', []))}\n"
            f"  Character Prompts: {len(data.get('character_prompts', []))}\n"
            f"  Background Prompts: {len(data.get('background_prompts', []))}\n"
            f"  Video Prompts: {len(data.get('video_prompts', []))}")


@tool
def load_project(project_name: str) -> str:
    """Load a previously saved story project.

    Args:
        project_name: Project name
    """
    project_dir = PROJECTS_DIR / project_name
    state_file = project_dir / "story_state.json"

    if not state_file.exists():
        return f"ERROR: No saved state found for project '{project_name}'"

    data = json.loads(state_file.read_text(encoding="utf-8"))
    CURRENT_PROJECT_FILE.write_text(project_name)

    return (f"Project loaded: {project_name}\n"
            f"  Phase: {data.get('phase', 'unknown')}\n"
            f"  Title: {data.get('project_name', 'untitled')}\n"
            f"  Genre: {data.get('genre', 'N/A')}\n"
            f"  Logline: {data.get('logline', 'N/A')}\n"
            f"  Characters: {len(data.get('characters', []))}\n"
            f"  Scenes: {len(data.get('scenes', []))}\n"
            f"  Last updated: {data.get('updated_at', 'unknown')}")


@tool
def list_projects() -> str:
    """List all saved story projects."""
    if not PROJECTS_DIR.exists():
        return "No projects directory found."

    projects = [d for d in PROJECTS_DIR.iterdir() if d.is_dir() and d.name != "_archive"]

    if not projects:
        return "No projects found. Create one with save_project."

    lines = []
    for p in sorted(projects):
        state_file = p / "story_state.json"
        if state_file.exists():
            data = json.loads(state_file.read_text(encoding="utf-8"))
            lines.append(
                f"  {p.name:<30s} | Phase: {data.get('phase', '?'):<10s} | "
                f"Scenes: {len(data.get('scenes', []))} | "
                f"Updated: {data.get('updated_at', '?')[:10]}"
            )
        else:
            lines.append(f"  {p.name:<30s} | (empty)")

    current = CURRENT_PROJECT_FILE.read_text().strip() if CURRENT_PROJECT_FILE.exists() else "none"
    return f"Projects (current: {current}):\n" + "\n".join(lines)


@tool
def export_prompts(project_name: str) -> str:
    """Export all generated prompts to a markdown file for the execution pipeline.

    Args:
        project_name: Project name
    """
    project_dir = PROJECTS_DIR / project_name
    state_file = project_dir / "story_state.json"

    if not state_file.exists():
        return f"ERROR: No saved state for '{project_name}'"

    data = json.loads(state_file.read_text(encoding="utf-8"))

    output_file = project_dir / "prompts_export.md"
    lines = [
        f"# {data.get('project_name', 'Untitled')} - Prompt Export",
        f"",
        f"**Genre:** {data.get('genre', 'N/A')}",
        f"**Tone:** {data.get('tone', 'N/A')}",
        f"**Logline:** {data.get('logline', 'N/A')}",
        f"**Story Length:** {data.get('story_length', 'N/A')}",
        f"",
        f"**Exported:** {datetime.now().isoformat()}",
        f"",
        f"---",
        f"",
    ]

    # Narrative Arc
    if data.get("narrative_arc"):
        lines.append("## Narrative Arc")
        lines.append("")
        lines.append(data["narrative_arc"])
        lines.append("")

    # Acts
    acts = data.get("acts", [])
    if acts:
        lines.append("## Act Structure")
        lines.append("")
        for act in acts:
            lines.append(f"### Act {act.get('act_num', '?')}: {act.get('title', 'Untitled')}")
            lines.append(f"- **Theme:** {act.get('theme', 'N/A')}")
            lines.append(f"- **Scenes:** {', '.join(str(s) for s in act.get('scenes', []))}")
            if act.get("turning_point"):
                lines.append(f"- **Turning Point:** {act['turning_point']}")
            lines.append("")

    # Subplots
    subplots = data.get("subplots", [])
    if subplots:
        lines.append("## Subplots")
        lines.append("")
        for sp in subplots:
            lines.append(f"### {sp.get('name', 'Unnamed')}")
            lines.append(f"- **Characters:** {', '.join(sp.get('characters', []))}")
            lines.append(f"- **Arc:** {sp.get('arc', 'N/A')}")
            if sp.get("resolution"):
                lines.append(f"- **Resolution:** {sp['resolution']}")
            lines.append("")

    # Characters
    chars = data.get("characters", [])
    if chars:
        lines.append("## Characters")
        lines.append("")
        for c in chars:
            lines.append(f"### {c.get('name', 'Unnamed')}")
            lines.append(f"- **Role:** {c.get('role', 'N/A')}")
            lines.append(f"- **Appearance:** {c.get('appearance', 'N/A')}")
            lines.append(f"- **Personality:** {c.get('personality', 'N/A')}")
            lines.append(f"- **Secret:** {c.get('secret', 'N/A')}")
            lines.append("")

    # Character Image Prompts
    char_prompts = data.get("character_prompts", [])
    if char_prompts:
        lines.append("## Character Image Prompts")
        lines.append("")
        for p in char_prompts:
            lines.append(f"### {p.get('name', 'Unnamed')} ({p.get('type', 'N/A')})")
            lines.append(f"```")
            lines.append(p.get("prompt", ""))
            lines.append(f"```")
            lines.append(f"- Resolution: {p.get('width', '?')}x{p.get('height', '?')}")
            lines.append("")

    # Background Prompts
    bg_prompts = data.get("background_prompts", [])
    if bg_prompts:
        lines.append("## Background Scene Prompts")
        lines.append("")
        for p in bg_prompts:
            lines.append(f"### Scene {p.get('scene_num', '?')}: {p.get('scene_name', 'N/A')}")
            lines.append(f"```")
            lines.append(p.get("prompt", ""))
            lines.append(f"```")
            lines.append(f"- Resolution: {p.get('width', '?')}x{p.get('height', '?')}")
            lines.append("")

    # Video Prompts
    vid_prompts = data.get("video_prompts", [])
    if vid_prompts:
        lines.append("## Video Prompts (Prompt Relay Format)")
        lines.append("")
        for p in vid_prompts:
            lines.append(f"### Scene {p.get('scene_num', '?')}")
            lines.append(f"**Global Prompt:**")
            lines.append(f"> {p.get('global_prompt', '')}")
            lines.append(f"")
            lines.append(f"**Local Prompts:**")
            lines.append(f"```")
            lines.append(p.get('local_prompts', ''))
            lines.append(f"```")
            lines.append(f"- Segments: {p.get('segments', '?')}")
            lines.append("")

    # Generated output paths
    lines.append("---")
    lines.append("")
    lines.append("## Generated Output Paths")
    lines.append("")
    lines.append(f"Characters: `generated/{project_name}/characters/`")
    lines.append(f"Backgrounds: `generated/{project_name}/backgrounds/`")
    lines.append(f"Videos: `generated/{project_name}/videos/`")
    lines.append("")
    lines.append(f"Run with: `python generate_all.py --project {project_name}`")
    lines.append(f"Video with: `python generate_video.py --project {project_name}`")
    lines.append("")

    output_file.write_text("\n".join(lines), encoding="utf-8")
    return f"Prompts exported to {output_file}"


# ── Graph Nodes ─────────────────────────────────────────────────────────

def create_story_agent():
    """Create the LangGraph story creation agent."""

    tools = [save_project, save_state, load_project, list_projects, export_prompts]
    tool_node = ToolNode(tools)

    # Read from env at call time so Settings tab changes take effect
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model_name = os.getenv("LLM_MODEL", "qwen3.6:27b")

    # LLM instances for each phase
    def get_llm(system_prompt: str):
        return ChatOllama(
            base_url=ollama_url,
            model=model_name,
            temperature=0.7,
        )

    def router(state: StoryState):
        """Route to the appropriate phase handler based on current phase."""
        phase = state.get("phase", "idea")
        messages = state["messages"]

        # Get the last human message to determine intent
        last_human = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human = msg.content
                break

        if not last_human:
            return "agent"

        lower = last_human.lower()

        # Tool-like commands
        if any(kw in lower for kw in ["save", "load", "export", "list project"]):
            return "tools"

        # Phase transitions based on user intent
        if phase == "idea":
            if any(kw in lower for kw in ["architect", "structure", "act", "blueprint"]):
                return "architect"
            if any(kw in lower for kw in ["character", "design character", "create character"]):
                return "characters"
            return "idea"
        elif phase == "architect":
            if any(kw in lower for kw in ["character", "design character", "create character"]):
                return "characters"
            if any(kw in lower for kw in ["revise structure", "change structure", "fix act"]):
                return "architect"
            return "architect"
        elif phase == "characters":
            if any(kw in lower for kw in ["scene", "scene outline", "build scene"]):
                return "scenes"
            if any(kw in lower for kw in ["revise character", "change character", "fix character"]):
                return "characters"
            return "characters"
        elif phase == "scenes":
            if any(kw in lower for kw in ["prompt", "generate prompt", "write prompt"]):
                return "prompts"
            if any(kw in lower for kw in ["revise scene", "change scene", "add scene", "remove scene"]):
                return "scenes"
            return "scenes"
        elif phase == "prompts":
            if any(kw in lower for kw in ["review", "check", "consistency"]):
                return "review"
            if any(kw in lower for kw in ["revise prompt", "fix prompt", "change prompt"]):
                return "prompts"
            return "prompts"
        elif phase == "review":
            if any(kw in lower for kw in ["export", "done", "finish", "approve"]):
                return "done"
            if any(kw in lower for kw in ["fix", "revise", "change"]):
                return "prompts"
            return "review"
        else:
            return "agent"

    def idea_node(state: StoryState):
        """Story planning phase."""
        messages = state["messages"]
        system = SystemMessage(content=STORY_PLANNER_PROMPT)
        llm = get_llm(STORY_PLANNER_PROMPT)
        response = llm.invoke([system] + messages)
        return {"messages": [response]}

    def architect_node(state: StoryState):
        """Narrative architecture phase - structures the story into acts and subplots."""
        messages = state["messages"]
        # Include current state context for the architect
        context_parts = []
        if state.get("project_name"):
            context_parts.append(f"Project: {state['project_name']}")
        if state.get("genre"):
            context_parts.append(f"Genre: {state['genre']}")
        if state.get("logline"):
            context_parts.append(f"Logline: {state['logline']}")
        if state.get("story_length"):
            context_parts.append(f"Target length: {state['story_length']}")
        if state.get("characters"):
            char_names = [c.get("name", "?") for c in state["characters"]]
            context_parts.append(f"Characters: {', '.join(char_names)}")

        context_msg = ""
        if context_parts:
            context_msg = "\n\nCurrent project context:\n" + "\n".join(context_parts)

        system = SystemMessage(content=NARRATIVE_ARCHITECT_PROMPT + context_msg)
        llm = get_llm(NARRATIVE_ARCHITECT_PROMPT)
        response = llm.invoke([system] + messages)
        return {"messages": [response]}

    def characters_node(state: StoryState):
        """Character design phase."""
        messages = state["messages"]
        system = SystemMessage(content=CHARACTER_DESIGNER_PROMPT)
        llm = get_llm(CHARACTER_DESIGNER_PROMPT)
        response = llm.invoke([system] + messages)
        return {"messages": [response]}

    def scenes_node(state: StoryState):
        """Scene building phase."""
        messages = state["messages"]
        system = SystemMessage(content=SCENE_BUILDER_PROMPT)
        llm = get_llm(SCENE_BUILDER_PROMPT)
        response = llm.invoke([system] + messages)
        return {"messages": [response]}

    def prompts_node(state: StoryState):
        """Prompt generation phase."""
        messages = state["messages"]
        system = SystemMessage(content=PROMPT_GENERATOR_PROMPT)
        llm = get_llm(PROMPT_GENERATOR_PROMPT)
        response = llm.invoke([system] + messages)
        return {"messages": [response]}

    def review_node(state: StoryState):
        """Review and consistency check phase."""
        messages = state["messages"]
        system = SystemMessage(content=REVIEWER_PROMPT)
        llm = get_llm(REVIEWER_PROMPT)
        response = llm.invoke([system] + messages)
        return {"messages": [response]}

    def agent_node(state: StoryState):
        """General agent node for routing."""
        messages = state["messages"]
        if not messages:
            messages = [HumanMessage(content="What kind of story would you like to create?")]
        system = SystemMessage(content=(
            "You are a creative story assistant for anime short films. "
            "Help the user plan their story, design characters, build scenes, "
            "and generate prompts for AI image and video generation. "
            "Be collaborative - suggest ideas but always let the user decide. "
            "When the user is happy with a phase, suggest moving to the next step."
        ))
        llm = get_llm("")
        response = llm.invoke([system] + messages)
        return {"messages": [response]}

    def should_use_tools(state: StoryState):
        """Determine if tool calls are needed."""
        messages = state["messages"]
        last_msg = messages[-1] if messages else None

        # Check if current phase's LLM made tool calls
        if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"

        return END

    # Build graph
    graph = StateGraph(StoryState)

    # Nodes
    graph.add_node("agent", agent_node)
    graph.add_node("idea", idea_node)
    graph.add_node("architect", architect_node)
    graph.add_node("characters", characters_node)
    graph.add_node("scenes", scenes_node)
    graph.add_node("prompts", prompts_node)
    graph.add_node("review", review_node)
    graph.add_node("tools", tool_node)

    # Entry
    graph.add_edge(START, "agent")

    # Agent can route to any phase or tools
    graph.add_conditional_edges("agent", router, {
        "idea": "idea",
        "architect": "architect",
        "characters": "characters",
        "scenes": "scenes",
        "prompts": "prompts",
        "review": "review",
        "tools": "tools",
    })

    # Each phase node routes to tools or back to agent
    for node_name in ["idea", "architect", "characters", "scenes", "prompts", "review"]:
        graph.add_conditional_edges(node_name, should_use_tools, {
            "tools": "tools",
            END: END,
        })

    # After tools, always return to agent for next step
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=MemorySaver())


# ── Interactive CLI ─────────────────────────────────────────────────────

def run_interactive(continue_project: bool = False):
    """Run the agent in interactive mode."""
    print("\n" + "=" * 60)
    print("  Creative Story Agent - Story Creation Agent")
    print("  Phases: Idea -> Characters -> Scenes -> Prompts -> Review")
    print("  Type 'quit' to exit")
    print("=" * 60 + "\n")

    graph = create_story_agent()
    config = {"configurable": {"thread_id": "creative-story-agent"}}

    # Load existing project if continuing
    if continue_project and CURRENT_PROJECT_FILE.exists():
        project_name = CURRENT_PROJECT_FILE.read_text().strip()
        initial = HumanMessage(content=f"Continue working on the project '{project_name}'. Load it and show me the current status.")
    else:
        initial = HumanMessage(content="Hi! I want to create an anime short film story. Help me plan it out.")

    state = {
        "messages": [initial],
        "project_name": "", "genre": "", "tone": "", "logline": "", "story_length": "",
        "acts": [], "subplots": [], "narrative_arc": "",
        "characters": [], "scenes": [],
        "character_prompts": [], "background_prompts": [], "video_prompts": [],
        "phase": "idea", "revision_count": 0,
        "created_at": datetime.now().isoformat(), "updated_at": "",
    }

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        state["messages"].append(HumanMessage(content=user_input))

        result = graph.invoke(state, config)
        state = result

        # Print assistant responses
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\nAgent: {msg.content}")
            elif isinstance(msg, ToolMessage) and msg.content:
                print(f"  [Tool] {msg.content[:200]}")


def run_export(project_name: str):
    """Export prompts for a project."""
    from main import export_prompts
    result = export_prompts.invoke({"project_name": project_name})
    print(result)


def run_status():
    """Show all projects."""
    from main import list_projects
    result = list_projects.invoke({})
    print(result)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Creative Story Agent - Story Creation Agent"
    )
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Interactive chat mode")
    parser.add_argument("--resume", action="store_true",
                        help="Continue last saved project")
    parser.add_argument("--export", type=str, metavar="PROJECT",
                        help="Export prompts for a project")
    parser.add_argument("--status", action="store_true",
                        help="List all projects")
    args = parser.parse_args()

    if args.status:
        run_status()
    elif args.export:
        run_export(args.export)
    elif args.interactive or args.resume:
        run_interactive(continue_project=args.resume)
    else:
        parser.print_help()
        print("\nExample:")
        print("  python story_agent.py -i              # Start new project")
        print("  python story_agent.py -i --resume    # Continue saved project")
        print("  python story_agent.py --status        # List projects")
        print("  python story_agent.py --export myproj # Export prompts")


if __name__ == "__main__":
    main()
