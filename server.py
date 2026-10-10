"""Web chat UI for the activity agent.

Usage:
    python server.py about_me.txt [--port 8000]
then open http://127.0.0.1:8000
"""
import json
import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import activity_agent as aa

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI()
runner = aa.new_runner(aa.activity_agent)  # one runner; each browser tab gets its own session


class ChatRequest(BaseModel):
    session: str
    message: str


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "chat.html"))


@app.post("/chat")
def chat(req: ChatRequest):
    """Streams the turn as newline-delimited JSON: {"kind": tool|note|answer|warning|error, "text"}."""
    def stream():
        try:
            for kind, text in aa.turn(runner, req.session, req.message):
                yield json.dumps({"kind": kind, "text": text}) + "\n"
        except Exception as e:  # fail loudly in the UI rather than answering without data
            yield json.dumps({"kind": "error", "text": f"{type(e).__name__}: {e}"}) + "\n"
    return StreamingResponse(stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    args = aa.configure("Web chat for the activity recommender.", extra_args=lambda p: (
        p.add_argument("--port", type=int, default=8000),
        p.add_argument("--host", default="127.0.0.1")))
    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
