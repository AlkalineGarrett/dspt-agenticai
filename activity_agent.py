"""Activity recommender agent: user profile + weather (Open-Meteo) + places (Google Places API New).

Setup:
    export GOOGLE_MAPS_API_KEY=...   # Google Cloud project with "Places API (New)" enabled
    ollama pull qwen3.5:9b

Usage:
    python activity_agent.py about_me.txt   # console chat with that user profile

Or from a notebook:
    from activity_agent import activity_agent, ask
    ask(activity_agent, "I'm in Coimbra tomorrow afternoon, what should I do?")
"""
import json, os, re, sys, urllib.request, urllib.parse, warnings, logging
from datetime import date
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

warnings.filterwarnings("ignore")
logging.getLogger("google_adk").setLevel(logging.CRITICAL)  # chat() prints errors itself

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(HERE, "prompt.md")
PROFILE_FILE = os.path.join(HERE, "example_profile.txt")  # set by configure() from the CLI
PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_FIELDS = ",".join([
    "places.displayName", "places.formattedAddress", "places.primaryTypeDisplayName",
    "places.rating", "places.userRatingCount", "places.priceLevel",
    "places.currentOpeningHours.openNow", "places.editorialSummary", "places.googleMapsUri",
])
WMO_CODES = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog",
             48: "fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
             61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
             75: "heavy snow", 80: "rain showers", 81: "rain showers", 82: "violent showers",
             95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail"}


def llm(model="qwen3.5:9b", num_ctx=16384):
    # Native Ollama API: unlike its OpenAI-compatible /v1 endpoint, it honors num_ctx.
    # Ollama's default 4096-token window is too small once place results are in the
    # conversation, and the answer gets cut off.
    return LiteLlm(model=f"ollama_chat/{model}", api_base="http://localhost:11434",
                   num_ctx=num_ctx, reasoning_effort="none", temperature=0)


class ToolError(RuntimeError):
    """A tool's backing API is unreachable or misconfigured. Aborts the turn instead of
    letting the model quietly answer without the data."""


def _get_json(url, data=None, headers=None, api="API"):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:  # Google: {"error": {"message": ...}}, Open-Meteo: {"reason": ...}
            j = json.loads(body)
            msg = j.get("error", {}).get("message") if isinstance(j.get("error"), dict) else j.get("reason")
        except ValueError:
            msg = None
        raise ToolError(f"{api} HTTP {e.code}: {msg or body[:300]}") from None
    except OSError as e:  # DNS, timeouts, connection resets
        raise ToolError(f"{api} unreachable: {getattr(e, 'reason', e)}") from None


def _maps_key():
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if key:
        return key
    similar = [k for k in os.environ if ("MAPS" in k or "GOOGLE" in k) and "KEY" in k]
    hint = f" Found {', '.join(similar)} instead; rename it." if similar else ""
    raise ToolError("GOOGLE_MAPS_API_KEY is not set in this shell's environment. "
                    f"Run `export GOOGLE_MAPS_API_KEY=...` in the same terminal.{hint}")


def check_google_maps():
    """Preflight: proves the key is present and Places API (New) accepts it. Uses an
    IDs-only field mask, the cheapest Text Search request."""
    key = _maps_key()
    _get_json(PLACES_URL, json.dumps({"textQuery": "cafe in Lisbon", "maxResultCount": 1}).encode(),
              {"Content-Type": "application/json", "X-Goog-Api-Key": key,
               "X-Goog-FieldMask": "places.id"}, api="Google Places")
    return f"...{key[-4:]}"


def _geocode(city):
    q = urllib.parse.quote_plus(city.strip())
    g = _get_json(f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1",
                  api="Open-Meteo geocoding")
    return g["results"][0] if g.get("results") else None


def get_weather(city: str, day: str = "") -> dict:
    """Returns the weather forecast for a city on a given day (YYYY-MM-DD, default today,
    up to 15 days ahead), split into morning, afternoon and evening."""
    loc = _geocode(city)
    if not loc:
        return {"error": f"City '{city}' not found."}
    day = day or date.today().isoformat()
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={loc['latitude']}"
           f"&longitude={loc['longitude']}&timezone=auto&start_date={day}&end_date={day}"
           "&hourly=temperature_2m,precipitation_probability,weather_code,wind_speed_10m")
    h = _get_json(url, api="Open-Meteo forecast")["hourly"]

    periods = {}
    for name, hours in {"morning": range(8, 12), "afternoon": range(12, 18),
                        "evening": range(18, 22)}.items():
        periods[name] = {
            "temp_c": round(sum(h["temperature_2m"][i] for i in hours) / len(hours), 1),
            "rain_chance_pct": max(h["precipitation_probability"][i] for i in hours),
            "max_wind_kmh": max(h["wind_speed_10m"][i] for i in hours),
            "sky": WMO_CODES.get(max(h["weather_code"][i] for i in hours), "unknown"),
        }
    return {"city": loc["name"], "date": day, **periods}


def search_places(query: str, city: str, open_now: bool = False) -> dict:
    """Searches Google Maps for places/businesses matching a query in a city,
    e.g. query='indoor climbing gym' or 'quiet specialty coffee'. Set open_now=True
    only if the user wants to go right now."""
    key = _maps_key()
    body = {"textQuery": f"{query} in {city}", "maxResultCount": 5, "openNow": open_now}
    loc = _geocode(city)
    if loc:  # bias results toward the city centre
        body["locationBias"] = {"circle": {"radius": 10000.0, "center": {
            "latitude": loc["latitude"], "longitude": loc["longitude"]}}}
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": key,
               "X-Goog-FieldMask": PLACES_FIELDS}
    res = _get_json(PLACES_URL, json.dumps(body).encode(), headers, api="Google Places")

    return {"places": [{
        "name": p.get("displayName", {}).get("text"),
        "type": p.get("primaryTypeDisplayName", {}).get("text"),
        "address": p.get("formattedAddress"),
        "rating": p.get("rating"),
        "reviews": p.get("userRatingCount"),
        "price": p.get("priceLevel", "").replace("PRICE_LEVEL_", "").lower() or None,
        "open_now": p.get("currentOpeningHours", {}).get("openNow"),
        "summary": p.get("editorialSummary", {}).get("text"),
        "maps_url": p.get("googleMapsUri"),
    } for p in res.get("places", [])]}


def load_prompt(ctx):
    """Re-reads PROMPT_FILE on every model call, so edits apply without restarting.
    {today} and {profile} in the file are filled in here."""
    profile = open(PROFILE_FILE).read() if os.path.exists(PROFILE_FILE) else "(no profile)"
    return (open(PROMPT_FILE).read()
            .replace("{today}", f"{date.today():%A, %Y-%m-%d}")
            .replace("{profile}", profile))


activity_agent = Agent(name="activity_agent", model=llm(),
    tools=[get_weather, search_places], instruction=load_prompt)


def new_runner(agent):
    return Runner(agent=agent, app_name="activity_agent",
                  session_service=InMemorySessionService(), auto_create_session=True)


NUDGE = ("You answered without calling search_places. Discard that answer: call "
         "search_places for each activity, then answer using only places it returned.")


def _run_turn(runner, session_id, message, called, urls):
    """Runs the agent once, yielding ("tool", text) per call; records tool names called and
    maps URLs returned. The final answer is the generator's return value."""
    answer = ""
    for event in runner.run(user_id="user", session_id=session_id,
                            new_message=types.Content(role="user", parts=[types.Part(text=message)])):
        for call in event.get_function_calls():
            called.add(call.name)
            yield "tool", f"{call.name}({', '.join(f'{k}={v!r}' for k, v in call.args.items())})"
        for resp in event.get_function_responses():
            urls.update(p["maps_url"] for p in (resp.response or {}).get("places", []))
        if event.finish_reason == types.FinishReason.MAX_TOKENS:
            raise ToolError("The model ran out of context window and its answer was cut off. "
                            "Raise num_ctx in llm() or start a new chat.")
        text = "".join(p.text or "" for p in (event.content.parts if event.content else []))
        if event.is_final_response() and text.strip():
            answer = text.strip()
    return answer


def turn(runner, session_id, query):
    """One user turn as a stream of (kind, text): "tool", "note", then one "answer" and an
    optional "warning". Guards against the model recommending places it never looked up.
    Raises (e.g. ToolError) instead of producing an answer from missing data."""
    called, urls = set(), set()
    answer = yield from _run_turn(runner, session_id, query, called, urls)
    if "get_weather" in called and "search_places" not in called:
        yield "note", "skipped place search, retrying"
        answer = yield from _run_turn(runner, session_id, NUDGE, called, urls)
    yield "answer", answer
    unverified = [u for u in re.findall(r"https?://\S*maps\S*", answer)
                  if u.rstrip(").,*") not in urls]
    if unverified:
        yield "warning", (f"{len(unverified)} link(s) above did not come from a Google Maps "
                          "search and may be made up.")


def ask(agent, query, runner=None, session_id="session"):
    """Runs one turn and prints it. Pass the same runner and session_id to keep the conversation."""
    runner = runner or new_runner(agent)
    style = {"tool": "\033[2m  → {}\033[0m", "note": "\033[2m  ({})\033[0m",
             "answer": "\n{}\n", "warning": "\033[33m⚠ {}\033[0m\n"}
    for kind, text in turn(runner, session_id, query):
        print(style[kind].format(text), flush=True)


def chat(agent):
    """Console loop: the agent keeps context across turns until /reset or exit."""
    runner, session = new_runner(agent), 1
    print("Activity agent. Ask for things to do (e.g. 'I'm in Porto on Saturday').\n"
          "Commands: /reset (new conversation), exit\n")
    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in ("exit", "quit", "/exit", "/quit"):
            break
        if query == "/reset":
            session += 1
            print("(new conversation)\n")
            continue
        if not query:
            continue
        print("\033[2m  thinking...\033[0m", flush=True)
        try:
            ask(agent, query, runner, f"session-{session}")
        except KeyboardInterrupt:
            print("\n(interrupted)\n")
        except Exception as e:
            print(f"\n\033[31m[error] {type(e).__name__}: {e}\033[0m\n"
                  "\033[2m(turn aborted; no answer was generated from missing data)\033[0m\n")


def configure(description, argv=None, extra_args=None):
    """Shared CLI setup for the console and web server: parses the required profile file
    (plus any extra_args(parser) options), then fails loudly unless Google Maps works."""
    global PROFILE_FILE
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("profile", help="text file describing the user")
    if extra_args:
        extra_args(parser)
    args = parser.parse_args(argv)
    if not os.path.isfile(args.profile):
        parser.error(f"profile file not found: {args.profile}")
    PROFILE_FILE = args.profile  # load_prompt reads this global on every call
    print(f"Profile: {args.profile}")
    try:
        print(f"Google Maps: OK (key {check_google_maps()})")
    except ToolError as e:
        sys.exit(f"\033[31mGoogle Maps check failed: {e}\033[0m")
    return args


if __name__ == "__main__":
    configure("Chat with the activity recommender.")
    chat(activity_agent)
