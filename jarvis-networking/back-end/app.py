from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tempfile, base64
from resume_parser import extract_resume_text, extract_json

app = FastAPI()

# CORS Middleware (if you access localhost separately in dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResumeRequest(BaseModel):
    base64: str

@app.post("/parse_resume")
async def parse_resume(req: ResumeRequest):
    print("parsing")
    content = req.base64.split(",")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(base64.b64decode(content))
        f.flush()
        text = extract_resume_text(f.name)
    structured = extract_json(text)
    return {"parsed": structured}

app.mount("/", StaticFiles(directory="../front-end/build", html=True), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("../front-end/build/index.html")
