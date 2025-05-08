from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from starlette.config import Config
import os, tempfile, base64
from resume_parser import extract_resume_text, extract_json
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()



app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "secret"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = Config('.env')
oauth = OAuth(config)

oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

@app.get("/auth/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)

    # Try id_token first
    try:
        user = await oauth.google.parse_id_token(request, token)
    except Exception:
        # Fallback: manually fetch from full Google userinfo endpoint
        resp = await oauth.google.get(
            'https://www.googleapis.com/oauth2/v3/userinfo',
            token=token
        )
        user = resp.json()

    request.session["user"] = dict(user)
    print(user)
    return RedirectResponse("/")

@app.get("/auth/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse("/")

@app.get("/auth/user")
async def get_user(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    return user


def require_user(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user



class ResumeRequest(BaseModel):
    base64: str

@app.post("/parse_resume")
async def parse_resume(req: ResumeRequest, user=Depends(require_user)):
    print("Parsing resume for:", user["email"])
    content = req.base64.split(",")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(base64.b64decode(content))
        f.flush()
        text = extract_resume_text(f.name)
    structured = extract_json(text)
    return {"parsed": structured}


app.mount("/", StaticFiles(directory="../front-end/build", html=True), name="static")

@app.get("/")
async def serve_index(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/auth/login")
    return FileResponse("../front-end/build/index.html")
