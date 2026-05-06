# Start creating a new story (interactive, needs Ollama running)
python agent/story_agent.py -i

# Continue where you left off
python agent/story_agent.py -i --resume

# Check your projects
python agent/story_agent.py --status

# Export prompts for the execution pipeline
python agent/story_agent.py --export my_story



What the Story Agent Does (story_agent.py)
5 creative phases, each with a specialized LLM persona:

Phase	Node	What It Does
💡 Idea	idea	Story concept, genre, tone, logline, scene outline
👤 Characters	characters	Detailed character design (appearance, personality, secrets)
🎬 Scenes	scenes	Scene breakdown (location, lighting, mood, color, action)
📝 Prompts	prompts	Convert scenes → AI-optimized image + video prompts
🔍 Review	review	Consistency check across all phases
5 tools for project management:

save_project / save_state — persist work to disk
load_project — resume previous session
list_projects — browse all projects
export_prompts — export markdown for execution pipeline




Flow Example

You: I want to create a sci-fi anime about a girl who can talk to robots

Agent (Story Planner): Great! Here's a concept...
        [suggests title, genre, 10 scenes, 3 characters]

You: Looks good, let's design the characters

Agent (Character Designer): Here are 3 detailed characters...
        [appearance, personality, visual prompts for each]

You: Change the main character's hair to silver

Agent (Character Designer): Updated! Here's the revised version...

You: Now build the scenes

Agent (Scene Builder): Here are 10 scenes with locations, lighting...
        [detailed visual descriptions for each]

You: Generate the prompts

Agent (Prompt Generator): Here are all image + video prompts...
        [Z-Anime image prompts + LTX 2.3 video prompts]

You: Review everything for consistency

Agent (Reviewer): Found 2 issues...
        1. Scene 5 mentions Mei's blue ribbon but it's actually faded...
        2. Scene 8 lighting contradicts the sunset from scene 7...

You: Fix those and export

Agent: Fixed! Exported to projects/my_story/prompts_export.md