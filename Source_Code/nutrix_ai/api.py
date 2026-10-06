"""
Nutrix AI – FastAPI Application

Persistence layer:
  • MongoDB (via motor)  – Chat messages, Ingredient Scans, Diet Planner inputs
  • ChromaDB             – AI Diet Planner LLM responses (only)
  • Supabase             – Authentication (JWT verification)
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from fastapi import UploadFile, File
from pydantic import BaseModel
import sys
import os

from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
load_dotenv(env_path)

import json
import asyncio
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session

# Ensure the parent directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nutrix_ai.core.nutrix_intelligence import NutrixIntelligence
from nutrix_ai.cli_test import seed_test_data
from fastapi.middleware.cors import CORSMiddleware
from nutrix_ai.database import (
    get_db, SessionLocal, User, UserPreference,
    HealthProfile, DietPlan, Conversation, Message, SavedItem,
)
from nutrix_ai.auth import (
    verify_password, get_password_hash, create_access_token,
    decode_access_token, ACCESS_TOKEN_EXPIRE_MINUTES,
)
from nutrix_ai.security import get_current_user as supabase_get_current_user

# --- MongoDB & ChromaDB imports ---
from nutrix_ai.mongodb import (
    ensure_indexes, close_mongo,
    save_chat_message, get_chat_history, delete_chat_history,
    save_ingredient_scan, get_scan_history, delete_scan_history,
    save_diet_planner_input, get_diet_planner_history,
)
from nutrix_ai.chroma_diet_store import (
    save_diet_plan_to_chroma, get_diet_plans_from_chroma,
    delete_all_diet_plans_for_user,
)

from nutrix_ai.unified_analyzer import run_label_analysis
from nutrix_ai.diet_planner import generate_diet_plan
from project.ocr_new import ocr

from datetime import timedelta
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

intelligence: NutrixIntelligence | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    global intelligence

    # --- STARTUP ---
    print("Seeding test databases...")
    for db_file in ["nutrix_rag.db", "nutrix_rag_vectors.db", "nutrix_web.db", "nutrix_web_vectors.db"]:
        if os.path.exists(db_file):
            os.remove(db_file)
    intelligence = NutrixIntelligence()
    seed_test_data(intelligence)

    # Seed developer bypass account (SQLite – kept for backward compat)
    db = SessionLocal()
    dev_email = "dev@cca.com"
    if not db.query(User).filter(User.email == dev_email).first():
        dev_user = User(email=dev_email, password_hash=get_password_hash("CCA_WELCOMES"))
        db.add(dev_user)
        db.commit()
        db.refresh(dev_user)
        db.add(UserPreference(user_id=dev_user.id))
        db.commit()
    db.close()

    # MongoDB indexes
    try:
        await ensure_indexes()
        print("MongoDB connected and indexes ensured.")
    except Exception as e:
        logger.error("MongoDB setup failed (app will still start): %s", e)
        print(f"WARNING: MongoDB setup failed: {e}")

    print("Nutrix API is ready!")

    yield  # --- app is running ---

    # --- SHUTDOWN ---
    await close_mongo()
    print("MongoDB connection closed.")


app = FastAPI(title="Nutrix AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# ---------------------------------------------------------------------------
# Authentication helper
# ---------------------------------------------------------------------------

def get_current_user(supabase_user=Depends(supabase_get_current_user), db: Session = Depends(get_db)):
    """
    Verify the Supabase JWT and return the local SQLAlchemy User row.
    Auto-creates a local profile on first login.
    """
    email = supabase_user.email
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token: no email")
    user = db.query(User).filter(User.email == email).first()
    if not user:
        # Auto-create local profile for first-time Supabase logins
        user = User(email=email, password_hash="supabase_managed")
        db.add(user)
        db.commit()
        db.refresh(user)
        db.add(UserPreference(user_id=user.id))
        db.commit()
    return user


def _get_supabase_uid(supabase_user=Depends(supabase_get_current_user)) -> str:
    """
    Return the raw Supabase UUID string.
    This is used as the key for MongoDB / ChromaDB – never trust a client-sent ID.
    """
    uid = getattr(supabase_user, "id", None)
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid token: no user id")
    return str(uid)


# ---------------------------------------------------------------------------
# Auth Endpoints (kept for backward compatibility)
# ---------------------------------------------------------------------------

from typing import Optional

class UserCreate(BaseModel):
    email: str
    password: str
    name: Optional[str] = None
    age: Optional[int] = None
    goal: Optional[str] = None

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    new_user = User(email=user.email, password_hash=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Init empty preferences
    pref = UserPreference(user_id=new_user.id)
    db.add(pref)
    db.commit()

    return {"msg": "User created successfully"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


class GoogleAuthRequest(BaseModel):
    id_token: str

@app.post("/auth/google")
def google_auth(request: GoogleAuthRequest, db: Session = Depends(get_db)):
    try:
        GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "YOUR_GOOGLE_CLIENT_ID_HERE.apps.googleusercontent.com")

        idinfo = id_token.verify_oauth2_token(
            request.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        email = idinfo.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Google token does not contain an email")

        user = db.query(User).filter(User.email == email).first()
        if not user:
            hashed_password = get_password_hash(os.urandom(24).hex())
            user = User(email=email, password_hash=hashed_password)
            db.add(user)
            db.commit()
            db.refresh(user)
            db.add(UserPreference(user_id=user.id))
            db.commit()

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}

    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Google token")


# ---------------------------------------------------------------------------
# User Preferences
# ---------------------------------------------------------------------------

class PreferencesUpdate(BaseModel):
    theme: str | None = None
    language: str | None = None
    ui_state: dict | None = None

@app.get("/user/preferences")
def get_preferences(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    pref = db.query(UserPreference).filter(UserPreference.user_id == current_user.id).first()
    return {
        "theme": pref.theme if pref else "light",
        "language": pref.language if pref else "en",
        "ui_state": json.loads(pref.ui_state) if pref and pref.ui_state else {}
    }

@app.post("/user/preferences")
def update_preferences(prefs: PreferencesUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    pref = db.query(UserPreference).filter(UserPreference.user_id == current_user.id).first()
    if not pref:
        pref = UserPreference(user_id=current_user.id)
        db.add(pref)

    if prefs.theme is not None:
        pref.theme = prefs.theme
    if prefs.language is not None:
        pref.language = prefs.language
    if prefs.ui_state is not None:
        pref.ui_state = json.dumps(prefs.ui_state)

    db.commit()
    return {"msg": "Preferences updated"}


# =====================================================================
# CORE APP ENDPOINTS (with MongoDB / ChromaDB persistence)
# =====================================================================

class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None  # optional conversation grouping

class ChatResponse(BaseModel):
    answer: str
    ui_indicators: list[str]
    sources: list[str]


# ---------------------------------------------------------------------------
# /chat  – non-streaming, saves to MongoDB
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    user_id: str = Depends(_get_supabase_uid),
):
    try:
        user_id_str = f"user_{current_user.id}"

        # Load recent chat history from MongoDB to pass as context
        db_history = await get_chat_history(user_id, session_id=request.session_id, limit=20)
        history_for_llm = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in db_history
        ] if db_history else None

        generator = intelligence.handle_query(request.query, user_id=user_id_str, db_chat_history=history_for_llm)

        full_text = ""
        ui_indicators = []
        sources = []

        for chunk in generator:
            if chunk.startswith("[UI:"):
                ui_indicators.append(chunk.strip())
            elif chunk.startswith("Sources:\n") or chunk.startswith("\n\nSources:\n"):
                sources.append(chunk.strip())
            else:
                full_text += chunk

        # --- Persist to MongoDB ---
        try:
            await save_chat_message(user_id, "user", request.query, session_id=request.session_id)
            await save_chat_message(user_id, "assistant", full_text.strip(), session_id=request.session_id)
        except Exception as db_err:
            logger.error("Failed to save chat to MongoDB: %s", db_err)

        return ChatResponse(
            answer=full_text.strip(),
            ui_indicators=ui_indicators,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /chat/stream  – streaming + background save to MongoDB
# ---------------------------------------------------------------------------

@app.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    user_id: str = Depends(_get_supabase_uid),
):
    user_id_str = f"user_{current_user.id}"

    # Load recent chat history from MongoDB to pass as context
    db_history = await get_chat_history(user_id, session_id=request.session_id, limit=20)
    history_for_llm = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in db_history
    ] if db_history else None

    # We accumulate the full response for persistence
    collected_text: list[str] = []

    def generate():
        try:
            generator = intelligence.handle_query(request.query, user_id=user_id_str, db_chat_history=history_for_llm)
            for chunk in generator:
                if chunk.startswith("[UI:"):
                    yield json.dumps({"type": "ui", "content": chunk.strip()}) + "\n"
                elif chunk.startswith("Sources:\n") or chunk.startswith("\n\nSources:\n"):
                    yield json.dumps({"type": "source", "content": chunk.strip()}) + "\n"
                else:
                    collected_text.append(chunk)
                    yield json.dumps({"type": "text", "content": chunk}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    async def _save_after_stream():
        """Background task: persist the user query + full AI response to MongoDB."""
        full_response = "".join(collected_text).strip()
        try:
            await save_chat_message(user_id, "user", request.query, session_id=request.session_id)
            if full_response:
                await save_chat_message(user_id, "assistant", full_response, session_id=request.session_id)
        except Exception as e:
            logger.error("Background save to MongoDB failed: %s", e)

    background_tasks.add_task(_save_after_stream)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# /analyze-label  – Ingredient Scanner + save scan to MongoDB
# ---------------------------------------------------------------------------

@app.post("/analyze-label")
async def analyze_label_endpoint(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    user_id: str = Depends(_get_supabase_uid),
):
    try:
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        print(f"File saved to {tmp_path} for analysis by User {user_id}")
        result_dict = run_label_analysis(tmp_path)
        os.remove(tmp_path)

        # --- Persist to MongoDB ---
        try:
            await save_ingredient_scan(
                user_id=user_id,
                input_type="image",
                input_reference=file.filename or "uploaded_image",
                ai_result=result_dict,
            )
        except Exception as db_err:
            logger.error("Failed to save ingredient scan to MongoDB: %s", db_err)

        return result_dict
    except Exception as e:
        print(f"Analysis Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /diet-plan  – inputs → MongoDB, LLM response → ChromaDB
# ---------------------------------------------------------------------------

class DietPlanRequest(BaseModel):
    age: int
    height: float
    weight: float
    goal: str
    diet_type: str


@app.post("/diet-plan")
async def diet_plan_endpoint(
    request: DietPlanRequest,
    current_user: User = Depends(get_current_user),
    user_id: str = Depends(_get_supabase_uid),
    db: Session = Depends(get_db),
):
    try:
        json_response = generate_diet_plan(
            age=request.age,
            height=request.height,
            weight=request.weight,
            goal=request.goal,
            diet_type=request.diet_type
        )
        if "error" in json_response:
            raise HTTPException(status_code=500, detail=json_response["error"])

        # Save diet plan to legacy SQLite database (backward compat)
        new_plan = DietPlan(user_id=current_user.id, plan_json=json.dumps(json_response))
        db.add(new_plan)
        db.commit()

        # --- Persist LLM response to ChromaDB (tagged with Supabase user_id) ---
        chroma_doc_id = None
        try:
            chroma_doc_id = save_diet_plan_to_chroma(user_id, json_response)
        except Exception as chroma_err:
            logger.error("Failed to save diet plan to ChromaDB: %s", chroma_err)

        # --- Persist user inputs to MongoDB ---
        try:
            await save_diet_planner_input(
                user_id=user_id,
                age=request.age,
                height=request.height,
                weight=request.weight,
                goal=request.goal,
                diet_type=request.diet_type,
                chroma_doc_id=chroma_doc_id,
            )
        except Exception as db_err:
            logger.error("Failed to save diet planner input to MongoDB: %s", db_err)

        return json_response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /ocr-extract  – OCR endpoint (unchanged)
# ---------------------------------------------------------------------------

@app.post("/ocr-extract")
async def ocr_extract_endpoint(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    try:
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        print(f"Running OCR extraction on {tmp_path}")
        text = ocr(tmp_path)
        os.remove(tmp_path)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# HISTORY / RETRIEVAL ENDPOINTS
# =====================================================================

# ---------------------------------------------------------------------------
# Chat History
# ---------------------------------------------------------------------------

@app.get("/chat/history")
async def get_chat_history_endpoint(
    session_id: str | None = Query(None, description="Filter by session/conversation ID"),
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
    user_id: str = Depends(_get_supabase_uid),
):
    """Return the authenticated user's chat history (chronological order)."""
    try:
        messages = await get_chat_history(user_id, session_id=session_id, limit=limit, skip=skip)
        return {"messages": messages, "count": len(messages)}
    except Exception as e:
        logger.error("Failed to fetch chat history: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch chat history")


@app.delete("/chat/history")
async def clear_chat_history_endpoint(
    session_id: str | None = Query(None, description="Delete only this session"),
    user_id: str = Depends(_get_supabase_uid),
):
    """Delete the authenticated user's chat history."""
    try:
        deleted = await delete_chat_history(user_id, session_id=session_id)
        return {"deleted_count": deleted}
    except Exception as e:
        logger.error("Failed to delete chat history: %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete chat history")


# ---------------------------------------------------------------------------
# Scan History
# ---------------------------------------------------------------------------

@app.get("/scans/history")
async def get_scans_history_endpoint(
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
    user_id: str = Depends(_get_supabase_uid),
):
    """Return the authenticated user's ingredient scan history."""
    try:
        scans = await get_scan_history(user_id, limit=limit, skip=skip)
        return {"scans": scans, "count": len(scans)}
    except Exception as e:
        logger.error("Failed to fetch scan history: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch scan history")


@app.delete("/scans/history")
async def clear_scans_history_endpoint(
    user_id: str = Depends(_get_supabase_uid),
):
    """Delete the authenticated user's scan history."""
    try:
        deleted = await delete_scan_history(user_id)
        return {"deleted_count": deleted}
    except Exception as e:
        logger.error("Failed to delete scan history: %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete scan history")


# ---------------------------------------------------------------------------
# Diet Planner History (inputs from MongoDB + plans from ChromaDB)
# ---------------------------------------------------------------------------

@app.get("/diet-plans/history")
async def get_diet_plans_history_endpoint(
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
    user_id: str = Depends(_get_supabase_uid),
):
    """
    Return the authenticated user's diet planner history.
    Combines the input parameters (from MongoDB) with the generated plans (from ChromaDB).
    """
    try:
        inputs = await get_diet_planner_history(user_id, limit=limit, skip=skip)

        # Also fetch the ChromaDB plans
        plans = []
        try:
            plans = get_diet_plans_from_chroma(user_id, limit=limit)
        except Exception as chroma_err:
            logger.error("Failed to fetch diet plans from ChromaDB: %s", chroma_err)

        # Build a lookup of chroma_doc_id → plan data
        plan_lookup = {p["id"]: p["plan"] for p in plans}

        # Merge: attach the plan to each input if we have the chroma_doc_id
        enriched = []
        for inp in inputs:
            entry = dict(inp)
            chroma_id = entry.get("chroma_doc_id")
            if chroma_id and chroma_id in plan_lookup:
                entry["generated_plan"] = plan_lookup[chroma_id]
            enriched.append(entry)

        return {"diet_plans": enriched, "count": len(enriched)}
    except Exception as e:
        logger.error("Failed to fetch diet plan history: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch diet plan history")


@app.delete("/diet-plans/history")
async def clear_diet_plans_history_endpoint(
    user_id: str = Depends(_get_supabase_uid),
):
    """Delete the authenticated user's diet plans from both MongoDB and ChromaDB."""
    try:
        from nutrix_ai.mongodb import get_mongo_db
        db = get_mongo_db()
        result = await db.diet_planner_inputs.delete_many({"user_id": user_id})
        mongo_deleted = result.deleted_count

        chroma_deleted = 0
        try:
            chroma_deleted = delete_all_diet_plans_for_user(user_id)
        except Exception as chroma_err:
            logger.error("Failed to delete diet plans from ChromaDB: %s", chroma_err)

        return {"mongo_deleted": mongo_deleted, "chroma_deleted": chroma_deleted}
    except Exception as e:
        logger.error("Failed to delete diet plan history: %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete diet plan history")


# ---------------------------------------------------------------------------
# Fitness Backend (Local JSON Storage)
# ---------------------------------------------------------------------------

FITNESS_DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fitness_data.json")

def _load_fitness_db() -> dict:
    if os.path.exists(FITNESS_DB_FILE):
        try:
            with open(FITNESS_DB_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_fitness_db(data: dict):
    with open(FITNESS_DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

@app.get("/fitness/status")
async def get_fitness_status(user_id: str = Depends(_get_supabase_uid)):
    db_data = _load_fitness_db()
    user_data = db_data.get(user_id)
    if user_data:
        return user_data
    return {"is_connected": False}

class FitnessPairRequest(BaseModel):
    device_name: str

@app.post("/fitness/pair")
async def pair_fitness_device(
    request: FitnessPairRequest,
    user_id: str = Depends(_get_supabase_uid)
):
    db_data = _load_fitness_db()
    # Mock data to simulate real fitness reading
    profile = {
        "user_id": user_id,
        "connected_device": request.device_name,
        "is_connected": True,
        "daily_goals": {
            "move_percentage": 0.75,
            "exercise_percentage": 0.60,
            "stand_percentage": 0.90
        },
        "recent_workouts": [
            {
                "title": "Morning Run",
                "duration": "45 mins • Outdoor",
                "calories": "420 kcal",
                "icon": "run"
            },
            {
                "title": "HIIT Session",
                "duration": "30 mins • Indoor",
                "calories": "350 kcal",
                "icon": "bolt"
            },
            {
                "title": "Yoga Flow",
                "duration": "60 mins • Recovery",
                "calories": "180 kcal",
                "icon": "yoga"
            }
        ]
    }
    db_data[user_id] = profile
    _save_fitness_db(db_data)
    return profile

@app.post("/fitness/unpair")
async def unpair_fitness_device(user_id: str = Depends(_get_supabase_uid)):
    db_data = _load_fitness_db()
    if user_id in db_data:
        del db_data[user_id]
        _save_fitness_db(db_data)
    return {"is_connected": False}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
