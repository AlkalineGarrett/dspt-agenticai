# Activity Agent

An agent that suggests concrete things to do, based on what a person cares about, tomorrow's weather, and real places from Google Maps. It runs on a local model through [Ollama](https://ollama.com), built with [Google ADK](https://google.github.io/adk-docs/).

For each request it:
1. Checks the forecast (morning / afternoon / evening) with [Open-Meteo](https://open-meteo.com) (free, no key).
2. Picks activity types that fit the person's interests and the weather.
3. Searches Google Maps (Places API (New)) for real places.
4. Recommends 3 places, each with why it fits, the best time to go, and a Maps link.

## Setup

Requires Python 3.10+ and Ollama.

```sh
ollama pull qwen3.5:9b

python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export GOOGLE_MAPS_API_KEY=...
```

The API key's Google Cloud project needs **Places API (New)** enabled. The older "Places API" won't work. If the key has API restrictions, Places API (New) must be on the allowed list.

## Usage

Describe yourself in a text file (see `example_profile.txt`), then start either:

```sh
python server.py me.txt              # web chat at http://127.0.0.1:8000  (--port to change)
python activity_agent.py me.txt      # console chat
```

Both check Google Maps at startup and exit with Google's error message if the key is missing or rejected.

## Files

| File | What it is |
|---|---|
| `activity_agent.py` | Agent, tools (weather, place search), console chat |
| `server.py` | Web server for the chat UI |
| `chat.html` | The chat page |
| `prompt.md` | The agent's instructions. Edit freely; reloaded on every message. `{today}` and `{profile}` are filled in. |
| `example_profile.txt` | Sample profile |

## Notes

- **Fails loudly:** if a tool's API fails, the turn stops with an error rather than letting the model answer without the data. The same applies if the model's answer gets cut off.
- **Guards against made-up places:** if the model checks the weather but skips place search, it's sent back to search. Maps links in the answer that didn't come from a search are flagged.
- **Context window:** set by `num_ctx` in `llm()` (16k by default). Ollama's 4k default is too small once place results are in the conversation.
