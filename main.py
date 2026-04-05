import uuid
import os
import time
import threading
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from services.llm_composer import generate_music_json, _missing_llm_vars
from services.midi_exporter import save_midi_file

# In-memory store for background generation jobs
# Maps job_id -> {"status": "pending"|"running"|"completed"|"error", "result": ..., "error": ..., "created_at": float}
jobs: dict = {}
jobs_lock = threading.Lock()
JOB_TTL_SECONDS = 3600  # Retain completed jobs for up to 1 hour

def _cleanup_old_jobs() -> None:
    """Remove jobs older than JOB_TTL_SECONDS. Must be called with jobs_lock held."""
    cutoff = time.time() - JOB_TTL_SECONDS
    expired = [jid for jid, j in jobs.items() if j["created_at"] < cutoff]
    for jid in expired:
        del jobs[jid]

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)

if os.path.exists("static/assets"):
    app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

static_files = [f for f in os.listdir("static") if f != ".gitkeep"] if os.path.exists("static") else []
if static_files:
    app.mount("/static", StaticFiles(directory="static"), name="static")

class MusicRequest(BaseModel):
    prompt: str

@app.get("/api/health")
async def health_check():
    llm_configured = len(_missing_llm_vars) == 0
    return {
        "status": "ok",
        "llm_configured": llm_configured,
        "missing_env_vars": _missing_llm_vars,
    }

@app.post("/api/generate")
async def generate(request: MusicRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    with jobs_lock:
        _cleanup_old_jobs()
        jobs[job_id] = {"status": "pending", "result": None, "error": None, "created_at": time.time()}

    def run_generation():
        with jobs_lock:
            if job_id not in jobs:
                return
            jobs[job_id]["status"] = "running"
        try:
            print(f"Generating music for prompt: {request.prompt}")
            data = generate_music_json(request.prompt)
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "completed"
                    jobs[job_id]["result"] = data
        except Exception as e:
            print(f"Error generating music: {e}")
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "error"
                    jobs[job_id]["error"] = str(e)

    background_tasks.add_task(run_generation)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "pending"})

@app.get("/api/generate/{job_id}")
async def get_generation_status(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            return JSONResponse(status_code=404, content={"error": "Job not found"})
        job = dict(jobs[job_id])
    if job["status"] == "completed":
        return JSONResponse(content={"status": "completed", "result": job["result"]})
    if job["status"] == "error":
        return JSONResponse(status_code=500, content={"status": "error", "error": job["error"]})
    return JSONResponse(content={"status": job["status"]})

@app.post("/api/export")
async def export_midi(request: Request, background_tasks: BackgroundTasks):
    try:
        music_data = await request.json()
        
        file_id = str(uuid.uuid4())
        filename = f"static/{file_id}.mid"
        
        print(f"Exporting MIDI to {filename}...")
        save_midi_file(music_data, filename)
        background_tasks.add_task(os.remove, filename)
        
        return FileResponse(
            path=filename,
            media_type="audio/midi",
            filename="generated-music.mid"
        )
    except Exception as e:
        print(f"Error exporting MIDI: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def read_root():
    return {"status": "ok", "message": "BeatFlow API is running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)