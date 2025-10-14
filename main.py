# main.py
from fastapi import FastAPI, HTTPException, Depends, Request, Form, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, validator, Field
from sqlalchemy import create_engine, Column, Integer, String, Date, ForeignKey, Float, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from passlib.context import CryptContext
import datetime
import random
from typing import List, Optional
import uuid
import html
import os
import shutil
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
import io
import base64

# =====================================
# APP + STATIC SETUP
# =====================================
app = FastAPI(title="NeuroHue Prototype (Full)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static / uploads directories (keep as before)
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# mount static and uploads
try:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
except Exception:
    pass

try:
    app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
except Exception:
    pass

# puzzle image expected in static/puzzle.jpg (fallback placeholder will be created if missing)
PUZZLE_IMAGE = STATIC_DIR / "puzzle.jpg"

# =====================================
# DATABASE SETUP (unchanged)
# =====================================
DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =====================================
# PASSWORD HASHING (unchanged)
# =====================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str):
    if len(password.encode("utf-8")) > 72:
        password = password[:72]
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# =====================================
# DATABASE MODELS (unchanged)
# =====================================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    email = Column(String, unique=True)
    password = Column(String)
    firstname = Column(String)
    lastname = Column(String)
    birthday = Column(Date)
    gender = Column(String)
    address = Column(String)
    role = Column(String)
    condition = Column(String)

    progress = relationship("ShapeGameProgress", back_populates="user")
    payments = relationship("Payment", back_populates="user")
    subscriptions = relationship("UserSubscription", back_populates="user")
    routines = relationship("Routine", back_populates="user")


class Deck(Base):
    __tablename__ = "memory_decks"
    deck_id = Column(String, primary_key=True, index=True)
    level = Column(String, nullable=False)
    mapping = Column(String, nullable=False)  # JSON string
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class MemoryFlipProgress(Base):
    __tablename__ = "memory_flip_progress"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    level = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    completed = Column(String, default="no")
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    user = relationship("User")


class MemoryLevelProgress(Base):
    __tablename__ = "memory_level_progress"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    level = Column(String, nullable=False)
    score = Column(Float, nullable=False, default=0.0)
    completed = Column(String, nullable=False, default="no")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship("User")


class MemoryAttempt(Base):
    __tablename__ = "memory_attempts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    deck_id = Column(String, nullable=False)
    level = Column(String, nullable=False)
    attempts_count = Column(Integer, default=0)
    matched_pairs = Column(Integer, default=0)
    total_pairs = Column(Integer, default=0)
    time_seconds = Column(Float, default=0.0)
    score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ShapeGameProgress(Base):
    __tablename__ = "shape_game_progress"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    level = Column(Integer, nullable=False)
    score = Column(Integer, nullable=False)
    completed = Column(String, nullable=False, default="no")
    date = Column(Date, default=datetime.date.today)
    user = relationship("User", back_populates="progress")
    
    
class EduShapeProgress(Base):
    __tablename__ = "edu_shape_game_progress"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    stage = Column(Integer, nullable=False)                # renamed field 'stage' instead of 'level'
    score = Column(Integer, nullable=False)
    completed = Column(String, nullable=False, default="no")
    played_on = Column(Date, default=datetime.date.today)  # different column name
    # relationship back_populates should match whatever you set on User (use unique name)
    user = relationship("User")  # keep simple to avoid forcing modifications in your Us    


class BehaviorChecklist(Base):
    __tablename__ = "behavior_checklist"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    section = Column(String)
    question = Column(String)
    answer = Column(String)
    date = Column(Date, default=datetime.date.today)
    user = relationship("User")


class SubscriptionPlan(Base):
    __tablename__ = "subscription_plans"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    price = Column(Float)
    description = Column(String, nullable=True)


class Payment(Base):
    __tablename__ = "payments"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    plan_id = Column(Integer, ForeignKey("subscription_plans.id"))
    amount = Column(Float)
    method = Column(String)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    transaction_id = Column(String, nullable=True)
    user = relationship("User", back_populates="payments")


class UserSubscription(Base):
    __tablename__ = "user_subscriptions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    plan_id = Column(Integer, ForeignKey("subscription_plans.id"))
    start_date = Column(Date)
    end_date = Column(Date)
    active = Column(String)
    user = relationship("User", back_populates="subscriptions")
    plan = relationship("SubscriptionPlan")


class Routine(Base):
    __tablename__ = "routines"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    category = Column(String, nullable=True)
    task = Column(String, nullable=False)
    time = Column(String, nullable=False)  # "HH:MM"
    status = Column(String, default="Inprogress")  # Done/Inprogress/Miss
    image = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    user = relationship("User", back_populates="routines")


# create tables
Base.metadata.create_all(bind=engine)


# =====================================
# UTILS: PIL -> data URL + slicing 3x3 tiles (unchanged)
# =====================================
def pil_image_to_data_url(img: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    b = buffer.read()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/{format.lower()};base64,{b64}"


def build_3x3_tiles(image_path: Path, tile_size: Optional[int] = None):
    """
    Slice the source image into 3x3 tiles and return list of dicts:
      [{ "id": 0..8, "data": "data:image/png;base64,..." }, ...]
    If source image missing, a placeholder is generated (numbers rendered at FONT_SIZE).
    """
    if image_path.exists():
        source_img = Image.open(str(image_path)).convert("RGB")
    else:
        # create placeholder 600x600 with colored squares and numbers
        target_side = 600
        source_img = Image.new("RGB", (target_side, target_side), (245, 247, 250))
        draw = ImageDraw.Draw(source_img)
        colors = [
            (237, 125, 49), (91, 155, 213), (112, 173, 71),
            (158, 72, 146), (68, 114, 196), (192, 80, 77),
            (99, 102, 106), (244, 176, 132), (152, 195, 121)
        ]
        step = target_side // 3

        # ---------- Set FONT_SIZE here ----------
        FONT_SIZE = 12
        # ----------------------------------------

        # try common TTF fonts first (you can add your own path like "static/fonts/MyFont.ttf")
        font_paths = [
            "DejaVuSans-Bold.ttf",
            "DejaVuSans.ttf",
            "LiberationSans-Regular.ttf",
            "Arial.ttf",
        ]
        font = None
        for p in font_paths:
            try:
                font = ImageFont.truetype(p, FONT_SIZE)
                break
            except Exception:
                font = None

        if font is None:
            # fallback: default font (size may differ from FONT_SIZE)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

        idx = 0
        for r in range(3):
            for c in range(3):
                x0 = c * step
                y0 = r * step
                draw.rectangle([x0, y0, x0 + step, y0 + step], fill=colors[idx % len(colors)])
                text = str(idx)

                # measure text size safely
                tw, th = (10, 10)
                if font is not None:
                    try:
                        bbox = draw.textbbox((0, 0), text, font=font)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]
                    except Exception:
                        try:
                            tw, th = font.getsize(text)
                        except Exception:
                            tw, th = (10, 10)

                tx = x0 + (step / 2) - (tw / 2)
                ty = y0 + (step / 2) - (th / 2)

                # draw subtle shadow then text for contrast
                try:
                    draw.text((tx + 2, ty + 2), text, fill=(0, 0, 0), font=font)
                except Exception:
                    pass
                draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

                idx += 1

    # crop to square center and resize to consistent size
    w, h = source_img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    source_img = source_img.crop((left, top, left + side, top + side))

    target_side = 600
    if tile_size is None:
        tile_size = target_side // 3
    source_img = source_img.resize((tile_size * 3, tile_size * 3), Image.LANCZOS)

    tiles = []
    idx = 0
    ts = tile_size
    for row in range(3):
        for col in range(3):
            x = col * ts
            y = row * ts
            tile_img = source_img.crop((x, y, x + ts, y + ts))
            data_url = pil_image_to_data_url(tile_img, format="PNG")
            tiles.append({"id": idx, "data": data_url})
            idx += 1
    return tiles





# =====================================
# Pydantic Schemas used across endpoints
# =====================================
class RegisterSchema(BaseModel):
    username: str
    email: EmailStr
    password: str
    firstname: str
    lastname: str
    birthday: str  # YYYY-MM-DD
    gender: str
    address: str
    role: str
    condition: str


class LoginSchema(BaseModel):
    username: str
    password: str


class SubscribeRequest(BaseModel):
    user_id: int
    plan_id: int
    payment_method: str  # "gcash", "paymaya", "credit_debit"


class PaymentCreateResponse(BaseModel):
    payment_id: int
    payment_url: str
    amount: float
    method: str


class PaymentConfirmRequest(BaseModel):
    payment_id: int
    status: str  # "success" or "failed"
    transaction_id: Optional[str] = None


class ShapeMatchSchema(BaseModel):
    user_id: int
    level: int
    matched_shapes: int
    total_shapes: int


class BehaviorAnswer(BaseModel):
    section: str
    question: str
    answer: str


class BehaviorChecklistSubmission(BaseModel):
    user_id: int
    responses: List[BehaviorAnswer]


# ------------------ Routine schemas ------------------
class RoutineCreate(BaseModel):
    user_id: int
    category: str
    task: str
    time: str  # expects "HH:MM"

    @validator("time")
    def validate_time_format(cls, v):
        try:
            datetime.datetime.strptime(v, "%H:%M")
        except Exception:
            raise ValueError("time must be in HH:MM format, 24-hour (e.g. 07:30 or 15:45)")
        return v


class RoutineUpdate(BaseModel):
    category: Optional[str] = None
    task: Optional[str] = None
    time: Optional[str] = None
    status: Optional[str] = None  # "Done", "Inprogress", "Miss"

    @validator("time")
    def validate_time_format(cls, v):
        if v is None:
            return v
        try:
            datetime.datetime.strptime(v, "%H:%M")
        except Exception:
            raise ValueError("time must be in HH:MM format, 24-hour (e.g. 07:30 or 15:45)")
        return v

    @validator("status")
    def validate_status(cls, v):
        if v is None:
            return v
        if v not in ("Done", "Inprogress", "Miss"):
            raise ValueError("status must be 'Done', 'Inprogress' or 'Miss'")
        return v


class RoutineOut(BaseModel):
    id: int
    user_id: int
    category: Optional[str]
    task: str
    time: str
    status: str
    image: Optional[str] = None
    created_at: datetime.datetime
    updated_at: Optional[datetime.datetime]

    model_config = {"from_attributes": True}


class RoutineMarkSchema(BaseModel):
    status: str

    @validator("status")
    def validate_status(cls, v):
        if v not in ("Done", "Inprogress", "Miss"):
            raise ValueError("status must be 'Done', 'Inprogress' or 'Miss'")
        return v


# =====================================
# MEMORY-FLIP (jigsaw-ish memory) config & helpers
# =====================================
MEMORY_LEVELS = {
    "easy": {"pairs": 4},
    "medium": {"pairs": 6},
    "hard": {"pairs": 8},
}


def _select_images_for_pairs(pairs: int):
    images = []
    for i in range(1, 21):
        candidate = STATIC_DIR / f"flip_icon_{i}.png"
        if candidate.exists():
            images.append(f"flip_icon_{i}.png")
    if not images:
        images = ["flip.png"]
    result = []
    for i in range(pairs):
        result.append(images[i % len(images)])
    return result


def _build_deck_for_level(level: str):
    if level not in MEMORY_LEVELS:
        raise ValueError("invalid level")
    pairs = MEMORY_LEVELS[level]["pairs"]
    images = _select_images_for_pairs(pairs)
    mapping = {}
    cards = []
    for i in range(pairs):
        pair_key = f"p{i+1}"
        img = images[i]
        for _ in (1, 2):
            card_id = str(uuid.uuid4())
            mapping[card_id] = {"pair_key": pair_key, "image": img}
            cards.append({"card_id": card_id, "image": img})
    random.shuffle(cards)
    return mapping, cards


def _save_deck(db: Session, deck_id: str, level: str, mapping: dict):
    d = Deck(deck_id=deck_id, level=level, mapping=json.dumps(mapping), created_at=datetime.datetime.utcnow())
    db.add(d)
    db.commit()
    db.refresh(d)
    return d


def _load_deck(db: Session, deck_id: str):
    return db.query(Deck).filter(Deck.deck_id == deck_id).first()


def _save_attempt(db: Session, user_id, deck_id, level, attempts_count, matched_pairs, total_pairs, time_seconds, score):
    a = MemoryAttempt(
        user_id=user_id,
        deck_id=deck_id,
        level=level,
        attempts_count=attempts_count,
        matched_pairs=matched_pairs,
        total_pairs=total_pairs,
        time_seconds=time_seconds,
        score=score,
        created_at=datetime.datetime.utcnow(),
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a

class EduShapeSubmitSchema(BaseModel):
    user_id: int
    level: int
    matched_shapes: int
    total_shapes: int

# ---------------------------
# Game levels mapping (separate variable)
# ---------------------------
EDU_GAME_LEVELS = {
    1: ["circle", "square", "triangle"],
    2: ["hexagon", "pentagon", "star", "heart"],
    3: ["crescent", "octagon", "diamond", "trapezoid", "parallelogram"],
}


# =====================================
# AUTH endpoints
# =====================================
@app.post("/signup")
def signup(user: RegisterSchema, db: Session = Depends(get_db)):
    if db.query(User).filter((User.username == user.username) | (User.email == user.email)).first():
        raise HTTPException(status_code=400, detail="Username or email already exists")
    try:
        birthday_date = datetime.datetime.strptime(user.birthday, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid birthday format. Use YYYY-MM-DD.")
    new_user = User(
        username=user.username,
        email=user.email,
        password=hash_password(user.password),
        firstname=user.firstname,
        lastname=user.lastname,
        birthday=birthday_date,
        gender=user.gender,
        address=user.address,
        role=user.role,
        condition=user.condition,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully!", "user_id": new_user.id}


@app.post("/login")
def login(payload: LoginSchema, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == payload.username).first()
    if not db_user or not verify_password(payload.password, db_user.password):
        raise HTTPException(status_code=400, detail="Invalid username or password")
    user_id = db_user.id
    shape_progress = db.query(ShapeGameProgress).filter(ShapeGameProgress.user_id == user_id).all()
    shape_status = "Done" if any(p.completed.lower() == "yes" for p in shape_progress) else "Inprogress"
    behavior_progress = db.query(BehaviorChecklist).filter(BehaviorChecklist.user_id == user_id).first()
    behavior_status = "Done" if behavior_progress else "Inprogress"
    overall_status = {"Shape Game": shape_status, "Behavior Checklist": behavior_status}
    return {
        "message": "Login successful",
        "user": {"id": db_user.id, "username": db_user.username, "firstname": db_user.firstname, "lastname": db_user.lastname},
        "progress_status": overall_status,
    }


@app.post("/logout")
def logout(payload: dict):
    username = payload.get("username")
    if not username:
        raise HTTPException(status_code=400, detail="Username is required for logout")
    return {"message": "Logout successful", "user": {"username": username}}


# =====================================
# PLANS & PAYMENTS (mock prototype)
# =====================================
@app.get("/plans")
def get_plans(db: Session = Depends(get_db)):
    plans = db.query(SubscriptionPlan).all()
    return [{"id": p.id, "name": p.name, "price": p.price, "description": p.description} for p in plans]


@app.get("/payment-methods")
def get_payment_methods():
    return {"methods": [{"id": "gcash", "label": "GCash"}, {"id": "paymaya", "label": "PayMaya"}, {"id": "credit_debit", "label": "Credit/Debit Card (Prototype)"}]}


@app.post("/subscribe", response_model=PaymentCreateResponse)
def subscribe(req: SubscribeRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == req.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    plan = db.query(SubscriptionPlan).filter(SubscriptionPlan.id == req.plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    method = req.payment_method.lower()
    if method not in ("gcash", "paymaya", "credit_debit"):
        raise HTTPException(status_code=400, detail="Unsupported payment method")
    payment = Payment(user_id=req.user_id, plan_id=plan.id, amount=plan.price, method=method, status="pending")
    db.add(payment)
    db.commit()
    db.refresh(payment)
    payment_url = f"/pay/{method}/{payment.id}"
    return PaymentCreateResponse(payment_id=payment.id, payment_url=payment_url, amount=payment.amount, method=payment.method)


def render_payment_html(payment: Payment, plan: SubscriptionPlan):
    safe_plan_name = html.escape(plan.name)
    safe_method = html.escape(payment.method.upper())
    amount = f"₱{payment.amount:,.2f}"
    html_content = f"""
    <!doctype html><html><head><meta charset="utf-8"/><title>Mock Payment - {safe_plan_name}</title>
    <style>body{{font-family:Arial;padding:2rem;background:#f7f9fc;color:#222}}.card{{background:white;padding:1.5rem;border-radius:10px;width:480px;margin:auto}}</style>
    </head><body><div class="card">
    <h1>Mock {safe_method} Checkout</h1>
    <p><strong>Plan:</strong> {safe_plan_name}</p>
    <p><strong>Amount:</strong> {amount}</p>
    <p><strong>User ID:</strong> {payment.user_id}</p>
    <form method="post" action="/payments/confirm">
    <input type="hidden" name="payment_id" value="{payment.id}" />
    <input type="hidden" name="transaction_id" value="{uuid.uuid4()}" />
    <button name="status" value="success" type="submit">Confirm Payment (simulate success)</button>
    <button name="status" value="failed" type="submit">Simulate Failure</button>
    </form>
    <p style="margin-top:.8rem;color:#666;font-size:0.9rem">Prototype page. No real money is processed.</p>
    </div></body></html>
    """
    return html_content


@app.get("/pay/{method}/{payment_id}", response_class=HTMLResponse)
def mock_payment_page(method: str, payment_id: int, db: Session = Depends(get_db)):
    payment = db.query(Payment).filter(Payment.id == payment_id, Payment.method == method).first()
    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found")
    plan = db.query(SubscriptionPlan).filter(SubscriptionPlan.id == payment.plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    return HTMLResponse(render_payment_html(payment, plan))


@app.post("/payments/confirm")
def confirm_payment(payment_id: int = Form(...), status: str = Form(...), transaction_id: Optional[str] = Form(None), db: Session = Depends(get_db)):
    payment = db.query(Payment).filter(Payment.id == int(payment_id)).first()
    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found")
    req_status = status.lower()
    if req_status not in ("success", "failed"):
        raise HTTPException(status_code=400, detail="Invalid status")
    payment.status = "success" if req_status == "success" else "failed"
    payment.transaction_id = transaction_id or str(uuid.uuid4())
    db.add(payment)
    db.commit()
    db.refresh(payment)
    if payment.status == "success":
        today = datetime.date.today()
        existing = db.query(UserSubscription).filter(UserSubscription.user_id == payment.user_id, UserSubscription.plan_id == payment.plan_id, UserSubscription.active == "yes").order_by(UserSubscription.id.desc()).first()
        if existing and existing.end_date and existing.end_date >= today:
            start_date = existing.end_date + datetime.timedelta(days=1)
            end_date = existing.end_date + datetime.timedelta(days=30)
            existing.end_date = end_date
            db.add(existing)
            db.commit()
            db.refresh(existing)
            subscription = existing
        else:
            start_date = today
            end_date = today + datetime.timedelta(days=30)
            subscription = UserSubscription(user_id=payment.user_id, plan_id=payment.plan_id, start_date=start_date, end_date=end_date, active="yes")
            db.add(subscription)
            db.commit()
            db.refresh(subscription)
        return HTMLResponse(f"<html><body style='font-family:Arial;padding:2rem'><h2>Payment Successful ✅</h2><p>Transaction ID: {html.escape(payment.transaction_id or '')}</p><p>Subscription ID: {subscription.id}</p><p>Start: {subscription.start_date.isoformat()} — End: {subscription.end_date.isoformat()}</p><p><a href='/user-subscriptions/{payment.user_id}'>View your subscriptions</a></p></body></html>")
    return HTMLResponse(f"<html><body style='font-family:Arial;padding:2rem'><h2>Payment Failed ❌</h2><p>Payment ID: {payment.id}</p><p><a href='/pay/{payment.method}/{payment.id}'>Try again</a></p></body></html>")


@app.get("/user-subscriptions/{user_id}")
def get_user_subscriptions(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    subs = db.query(UserSubscription).filter(UserSubscription.user_id == user_id).all()
    out = []
    for s in subs:
        plan = db.query(SubscriptionPlan).filter(SubscriptionPlan.id == s.plan_id).first()
        out.append({"subscription_id": s.id, "plan_id": s.plan_id, "plan_name": plan.name if plan else None, "start_date": s.start_date.isoformat() if s.start_date else None, "end_date": s.end_date.isoformat() if s.end_date else None, "active": s.active})
    return {"user_id": user_id, "username": user.username, "subscriptions": out}


@app.get("/user-payments/{user_id}")
def get_user_payments(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    payments = db.query(Payment).filter(Payment.user_id == user_id).order_by(Payment.created_at.desc()).all()
    return [{"payment_id": p.id, "plan_id": p.plan_id, "amount": p.amount, "method": p.method, "status": p.status, "created_at": p.created_at.isoformat(), "transaction_id": p.transaction_id} for p in payments]


# =====================================
# MATCH THE SHAPES (unchanged logic)
# =====================================
GAME_LEVELS = {
    1: ["circle", "square", "triangle"],
    2: ["hexagon", "pentagon", "star", "heart"],
    3: ["crescent", "octagon", "diamond", "trapezoid", "parallelogram"],
}


@app.get("/match-shapes/level/{level}")
def get_shapes(level: int):
    if level not in GAME_LEVELS:
        raise HTTPException(status_code=404, detail="Level not found")
    shapes = GAME_LEVELS[level]
    outlines = shapes.copy()
    random.shuffle(outlines)
    return {"level": level, "shapes": shapes, "outlines": outlines, "message": f"Drag and match the shapes to their outlines for level {level}."}


@app.post("/match-shapes/submit")
def submit_match(data: ShapeMatchSchema, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == data.user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if data.total_shapes <= 0:
        raise HTTPException(status_code=400, detail="total_shapes must be > 0")
    score = int((data.matched_shapes / data.total_shapes) * 100)
    completed = "yes" if score >= 80 else "no"
    progress = ShapeGameProgress(user_id=data.user_id, level=data.level, score=score, completed=completed, date=datetime.date.today())
    db.add(progress)
    db.commit()
    db.refresh(progress)
    next_level = data.level + 1 if data.level < len(GAME_LEVELS) else None
    return {"message": "Level completed!" if completed == "yes" else "Try again!", "score": score, "completed": completed, "next_level": next_level, "progress_id": progress.id}


@app.get("/shape-progress/{user_id}")
def get_shape_progress(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    progress_summary = []
    for level_id in sorted(GAME_LEVELS.keys()):
        attempts_q = db.query(ShapeGameProgress).filter(ShapeGameProgress.user_id == user_id, ShapeGameProgress.level == level_id)
        attempts = attempts_q.count()
        if attempts == 0:
            progress_summary.append({"level": level_id, "attempts": 0, "latest_score": None, "best_score": None, "completed": None, "last_played": None})
            continue
        latest = attempts_q.order_by(ShapeGameProgress.id.desc()).first()
        best = attempts_q.order_by(ShapeGameProgress.score.desc()).first()
        progress_summary.append({"level": level_id, "attempts": attempts, "latest_score": latest.score, "best_score": best.score, "completed": latest.completed, "last_played": latest.date.isoformat() if latest.date else None})
    return {"user_id": user_id, "username": user.username, "progress": progress_summary}


# =====================================
# BEHAVIOR CHECKLIST
# =====================================
BEHAVIOR_CHECKLIST = {
    "Section 1: Sensory Processing": [
        "I feel overwhelmed in environments with bright lights or loud noises.",
        "I avoid certain textures (clothing, food, etc.) because they feel uncomfortable.",
        "I am unusually sensitive to smell, touch, or taste.",
        "I need to stim (e.g., fidgeting, rocking, tapping) to stay regulated.",
        "I get fatigued or distracted in busy or noisy places (like malls or schools).",
    ],
    "Section 2: Communication & Social Patterns": [
        "I often take things literally and miss sarcasm or hidden meanings.",
        "I rehearse conversations before or after social situations.",
        "I find small talk or group conversations draining.",
        "I prefer written communication over spoken communication.",
        "I need more time to process what someone says before I respond.",
    ],
    "Section 3: Executive Functioning": [
        "I struggle to start tasks, even if they’re important to me.",
        "I lose track of time when focused on something interesting.",
        "I often forget appointments or misplace things.",
        "I find it difficult to prioritize tasks.",
        "I rely on systems (apps, lists, routines) to manage daily life.",
    ],
    "Section 4: Emotional Regulation": [
        "I experience intense emotional reactions that others may not understand.",
        "I mask or hide my emotions to fit in socially.",
        "I have difficulty shifting gears when something unexpected happens.",
        "I feel mentally exhausted after social interactions.",
        "I feel emotions in a physically intense way (e.g., stomachaches, headaches).",
    ],
    "Section 5: Strengths & Adaptations": [
        "I have deep knowledge or passion in specific areas of interest.",
        "I can focus intensely on things I care about.",
        "I notice details others often miss.",
        "I’m creative and think in unique ways.",
        "I’ve developed personal systems or tools to help me.",
    ],
}


@app.get("/behavior-checklist/questions")
def get_behavior_checklist():
    return {"scale": ["Never", "Rarely", "Sometimes", "Often", "Always"], "checklist": BEHAVIOR_CHECKLIST, "instructions": "Answer each question based on your usual experiences."}


@app.post("/behavior-checklist/submit")
def submit_behavior_checklist(data: BehaviorChecklistSubmission, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    for response in data.responses:
        record = BehaviorChecklist(user_id=data.user_id, section=response.section, question=response.question, answer=response.answer, date=datetime.date.today())
        db.add(record)
    db.commit()
    return {"message": "Behavior checklist submitted successfully!"}



from pathlib import Path
from fastapi.responses import FileResponse, HTMLResponse
import html

# replace your existing root handler with this
@app.get("/", include_in_schema=False)
def serve_index():
    """
    Serve index.html from project root first, then static/index.html.
    Uses FileResponse so linked assets (CSS/JS) are requested normally by the browser.
    """
    root_candidate = Path("index.html")
    static_candidate = STATIC_DIR / "index.html"

    # 1) Serve index.html from project root
    if root_candidate.exists():
        try:
            return FileResponse(path=str(root_candidate), media_type="text/html")
        except Exception as e:
            return HTMLResponse(f"<html><body><h3>Unable to serve index.html from project root</h3><pre>{html.escape(str(e))}</pre></body></html>", status_code=500)

    # 2) Fallback to static/index.html
    if static_candidate.exists():
        try:
            return FileResponse(path=str(static_candidate), media_type="text/html")
        except Exception as e:
            return HTMLResponse(f"<html><body><h3>Unable to serve static/index.html</h3><pre>{html.escape(str(e))}</pre></body></html>", status_code=500)

    # 3) Not found - helpful instructions
    msg = """
    <html><body style="font-family:Arial;padding:2rem">
      <h2>index.html not found</h2>
      <p>Place <code>index.html</code> in the project root (recommended) or in <code>static/index.html</code>.</p>
      <p>After pushing, open <a href="/">/</a> or <a href="/static/index.html">/static/index.html</a>.</p>
    </body></html>
    """
    return HTMLResponse(msg, status_code=404)





# =====================================
# DAILY ROUTINES endpoints
# =====================================
@app.post("/routines", response_model=RoutineOut, status_code=201)
def create_routine(payload: RoutineCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    r = Routine(user_id=payload.user_id, category=payload.category.strip(), task=payload.task.strip(), time=payload.time, status="Inprogress", created_at=datetime.datetime.utcnow(), updated_at=datetime.datetime.utcnow())
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


@app.get("/routines/{user_id}", response_model=List[RoutineOut])
def list_routines(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    routines = db.query(Routine).filter(Routine.user_id == user_id).order_by(Routine.time).all()
    return routines


@app.get("/routines/{user_id}/today", response_model=List[RoutineOut])
def list_routines_today(user_id: int, db: Session = Depends(get_db)):
    return list_routines(user_id, db)


@app.get("/routine/{routine_id}", response_model=RoutineOut)
def get_routine(routine_id: int, db: Session = Depends(get_db)):
    r = db.query(Routine).filter(Routine.id == routine_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Routine not found")
    return r


@app.put("/routine/{routine_id}", response_model=RoutineOut)
def replace_routine(routine_id: int, payload: RoutineCreate, db: Session = Depends(get_db)):
    r = db.query(Routine).filter(Routine.id == routine_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Routine not found")
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    r.user_id = payload.user_id
    r.category = payload.category.strip()
    r.task = payload.task.strip()
    r.time = payload.time
    r.status = "Inprogress"
    r.updated_at = datetime.datetime.utcnow()
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


@app.patch("/routine/{routine_id}", response_model=RoutineOut)
def update_routine(routine_id: int, payload: RoutineUpdate, db: Session = Depends(get_db)):
    r = db.query(Routine).filter(Routine.id == routine_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Routine not found")
    if payload.category is not None:
        r.category = payload.category.strip()
    if payload.task is not None:
        r.task = payload.task.strip()
    if payload.time is not None:
        r.time = payload.time
    if payload.status is not None:
        r.status = payload.status
    r.updated_at = datetime.datetime.utcnow()
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


@app.delete("/routine/{routine_id}")
def delete_routine(routine_id: int, db: Session = Depends(get_db)):
    r = db.query(Routine).filter(Routine.id == routine_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Routine not found")
    db.delete(r)
    db.commit()
    return {"message": "Routine deleted", "routine_id": routine_id}


@app.post("/routine/{routine_id}/mark", response_model=RoutineOut)
def mark_routine(routine_id: int, payload: RoutineMarkSchema, db: Session = Depends(get_db)):
    r = db.query(Routine).filter(Routine.id == routine_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Routine not found")
    r.status = payload.status
    r.updated_at = datetime.datetime.utcnow()
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


@app.post("/routine/{routine_id}/image")
def upload_routine_image(routine_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    r = db.query(Routine).filter(Routine.id == routine_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Routine not found")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    ext = os.path.splitext(file.filename)[1]
    safe_name = f"routine_{routine_id}_{uuid.uuid4().hex}{ext}"
    dest_path = UPLOAD_DIR / safe_name
    try:
        with dest_path.open("wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    finally:
        file.file.close()
    r.image = f"/uploads/{safe_name}"
    r.updated_at = datetime.datetime.utcnow()
    db.add(r)
    db.commit()
    db.refresh(r)
    return {"message": "Image uploaded", "image_url": r.image}


# =====================================
# JIGSAW endpoints (3x3)
# =====================================
class JigsawCheckRequest(BaseModel):
    level: int
    arrangement: List[Optional[int]]
    user_id: Optional[int] = None

class JigsawCheckResponse(BaseModel):
    correct: int
    total: int
    percent: float
    completed: bool


@app.get("/api/jigsaw/tiles")
def api_jigsaw_tiles(request: Request, level: int = Query(1, ge=1, le=3), seed: Optional[str] = Query(None)):
    # Log simple request info to help debugging when frontend 'Failed to fetch'
    client = request.client.host if request.client else "unknown"
    print(f"[jigsaw] tiles requested from {client} - level={level} seed={seed}")
    try:
        tiles = build_3x3_tiles(PUZZLE_IMAGE)
    except Exception as e:
        print(f"[jigsaw] tile build error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare tiles: {e}")
    if seed == "solution":
        return {"tiles": tiles, "level": level}
    shuffled = tiles.copy()
    random.shuffle(shuffled)
    return {"tiles": shuffled, "level": level}

@app.post("/api/jigsaw/check", response_model=JigsawCheckResponse)
def api_jigsaw_check(request: Request, payload: JigsawCheckRequest):
    client = request.client.host if request.client else "unknown"
    print(f"[jigsaw] check from {client} - user_id={payload.user_id} level={payload.level}")
    if not isinstance(payload.arrangement, list) or len(payload.arrangement) != 9:
        raise HTTPException(status_code=400, detail="arrangement must be an array of length 9")
    correct = 0
    total = 9
    for i, v in enumerate(payload.arrangement):
        if v is None:
            continue
        try:
            tid = int(v)
        except Exception:
            continue
        if tid == i:
            correct += 1
    percent = round((correct / total) * 100.0, 2)
    completed = (correct == total)
    return {"correct": correct, "total": total, "percent": percent, "completed": completed}

# =====================================
# Serve jigsaw.html from project root (if present)
# =====================================
@app.get("/jigsaw.html", response_class=HTMLResponse)
def serve_jigsaw_root():
    """
    Serve jigsaw.html from project root if it exists.
    This is useful when the HTML file is not inside /static.
    """
    root_file = Path("jigsaw.html")
    if root_file.exists():
        try:
            content = root_file.read_text(encoding="utf-8")
            return HTMLResponse(content)
        except Exception as e:
            msg = f"<html><body><h3>Unable to read jigsaw.html</h3><pre>{html.escape(str(e))}</pre></body></html>"
            return HTMLResponse(msg, status_code=500)
    # fallback instructions if file not found
    msg = """
    <html><body style="font-family:Arial;padding:2rem">
      <h2>jigsaw.html not found in project root</h2>
      <p>Place your <code>jigsaw.html</code> file in the project root or in the <code>static/</code> folder.</p>
      <p>To open the UI:</p>
      <ul>
        <li><a href="/jigsaw.html">/jigsaw.html</a> (serves file from project root)</li>
        <li><a href="/static/jigsaw.html">/static/jigsaw.html</a> (if you moved it to static/)</li>
      </ul>
      <p>Or pass <code>?api_origin=http://127.0.0.1:8000</code> to the file when opening via <code>file://</code> in the browser.</p>
    </body></html>
    """
    return HTMLResponse(msg, status_code=404)

# convenience redirect
@app.get("/jigsaw")
def jigsaw_redirect():
    return RedirectResponse(url="/jigsaw.html")

# small health endpoint
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "neurohue", "static_dir": str(STATIC_DIR.resolve()), "puzzle_exists": PUZZLE_IMAGE.exists()}

# =====================================
# Root index (updated to point to /jigsaw.html)
# =====================================
@app.get("/", response_class=HTMLResponse)
def root_index():
    html_content = f"""
    <html><body style="font-family:Arial;padding:2rem">
      <h2>NeuroHue API (Full)</h2>
      <ul>
        <li><a href="/jigsaw.html">jigsaw.html (project root)</a></li>
        <li><a href="/static/jigsaw.html">static/jigsaw.html (static folder)</a></li>
        <li>/api/jigsaw/tiles?level=1</li>
        <li>/api/jigsaw/check (POST)</li>
      </ul>
      <p>Health: <a href="/api/health">/api/health</a></p>
    </body></html>
    """
    return HTMLResponse(html_content)


# =====================================
# MEMORY-FLIP endpoints
# =====================================
class MFLevelInfo(BaseModel):
    id: str
    pairs: int


class MFStartRequest(BaseModel):
    level: str
    user_id: Optional[int] = None


class MFCard(BaseModel):
    card_id: str
    image: str


class MFStartResponse(BaseModel):
    deck_id: str
    level: str
    cards: List[MFCard]


class MFAttemptRequest(BaseModel):
    deck_id: str
    user_id: Optional[int] = None
    matched_pairs: List[List[str]]
    attempts_count: int = Field(..., ge=0)
    time_seconds: float = Field(..., ge=0.0)


class MFAttemptResponse(BaseModel):
    ok: bool
    matched_pairs: int
    total_pairs: int
    score: float
    message: Optional[str] = None


@app.get("/memory-flip/levels", response_model=List[MFLevelInfo])
def memory_levels():
    return [{"id": k, "pairs": v["pairs"]} for k, v in MEMORY_LEVELS.items()]


@app.post("/memory-flip/start", response_model=MFStartResponse)
def memory_start(payload: MFStartRequest, db: Session = Depends(get_db)):
    level = payload.level
    if level not in MEMORY_LEVELS:
        raise HTTPException(status_code=400, detail="Invalid level")
    mapping, cards = _build_deck_for_level(level)
    deck_id = str(uuid.uuid4())
    _save_deck(db, deck_id, level, mapping)
    return MFStartResponse(deck_id=deck_id, level=level, cards=[MFCard(**c) for c in cards])


@app.post("/memory-flip/attempt", response_model=MFAttemptResponse)
def memory_attempt(payload: MFAttemptRequest, db: Session = Depends(get_db)):
    deck = _load_deck(db, payload.deck_id)
    if not deck:
        raise HTTPException(status_code=404, detail="Deck not found")
    try:
        mapping = json.loads(deck.mapping)
    except Exception:
        raise HTTPException(status_code=500, detail="Internal deck mapping error")
    total_pairs = MEMORY_LEVELS.get(deck.level, {}).get("pairs", 0)
    correct = 0
    seen = set()
    for pair in payload.matched_pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        a, b = pair
        if a == b:
            continue
        if a not in mapping or b not in mapping:
            continue
        seen.add(a); seen.add(b)
        if mapping[a]["pair_key"] == mapping[b]["pair_key"]:
            correct += 1
    accuracy = correct / total_pairs if total_pairs else 0.0
    attempts_penalty = max(1, payload.attempts_count)
    time_penalty = max(1.0, payload.time_seconds / 10.0)
    raw_score = accuracy * 100.0
    score = raw_score / (attempts_penalty**0.4 * (time_penalty**0.2))
    score = round(float(score), 2)
    _save_attempt(db, payload.user_id, payload.deck_id, deck.level, payload.attempts_count, correct, total_pairs, payload.time_seconds, score)
    return MFAttemptResponse(ok=True, matched_pairs=correct, total_pairs=total_pairs, score=score, message="Recorded")


@app.get("/memory-flip/deck/{deck_id}")
def memory_get_deck(deck_id: str, db: Session = Depends(get_db)):
    deck = _load_deck(db, deck_id)
    if not deck:
        raise HTTPException(status_code=404, detail="deck not found")
    return {"deck_id": deck.deck_id, "level": deck.level, "created_at": deck.created_at.isoformat(), "mapping_count": len(json.loads(deck.mapping))}


@app.get("/memory-flip/attempts/{user_id}")
def memory_user_attempts(user_id: int, db: Session = Depends(get_db)):
    rows = db.query(MemoryAttempt).filter(MemoryAttempt.user_id == user_id).order_by(MemoryAttempt.created_at.desc()).all()
    out = []
    for r in rows:
        out.append({"id": r.id, "deck_id": r.deck_id, "level": r.level, "attempts_count": r.attempts_count, "matched_pairs": r.matched_pairs, "total_pairs": r.total_pairs, "time_seconds": r.time_seconds, "score": r.score, "created_at": r.created_at.isoformat() if r.created_at else None})
    return out


class MemoryFlipProgressCreate(BaseModel):
    user_id: int
    level: str
    score: float
    completed: Optional[bool] = None


class MemoryFlipProgressOut(BaseModel):
    id: int
    user_id: int
    level: str
    score: float
    completed: bool
    updated_at: Optional[datetime.datetime]
    model_config = {"from_attributes": True}


@app.post("/memory-flip/progress", response_model=MemoryFlipProgressOut)
def upsert_memory_flip_progress(payload: MemoryFlipProgressCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    threshold = 20.0
    completed_flag = payload.completed if payload.completed is not None else (payload.score >= threshold)
    row = db.query(MemoryFlipProgress).filter(MemoryFlipProgress.user_id == payload.user_id, MemoryFlipProgress.level == payload.level).first()
    if row:
        row.score = float(payload.score)
        row.completed = "yes" if completed_flag else "no"
        row.updated_at = datetime.datetime.utcnow()
        db.add(row)
        db.commit()
        db.refresh(row)
        return row
    else:
        new = MemoryFlipProgress(user_id=payload.user_id, level=payload.level, score=float(payload.score), completed="yes" if completed_flag else "no", updated_at=datetime.datetime.utcnow())
        db.add(new)
        db.commit()
        db.refresh(new)
        return new


@app.get("/memory-flip/progress/{user_id}")
def get_memory_flip_progress(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    rows = db.query(MemoryFlipProgress).filter(MemoryFlipProgress.user_id == user_id).all()
    out = {}
    for r in rows:
        out[r.level] = {"score": float(r.score), "completed": True if (r.completed or "").lower() == "yes" else False, "updated_at": r.updated_at.isoformat() if r.updated_at else None}
    return {"user_id": user_id, "progress": out}


# =====================================
# ASSESSMENT STATUS helper
# =====================================
@app.get("/assessment-status/{user_id}")
def get_assessment_status(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    shape_progress = db.query(ShapeGameProgress).filter(ShapeGameProgress.user_id == user_id).all()
    behavior_progress = db.query(BehaviorChecklist).filter(BehaviorChecklist.user_id == user_id).first()
    return {
        "user_id": user.id,
        "username": user.username,
        "shape_game": "Done" if any((p.completed or "").lower() == "yes" for p in shape_progress) else "Inprogress",
        "behavior_checklist": "Done" if behavior_progress else "Inprogress"
    }


# =====================================
# STARTUP helpers
# =====================================
def seed_plans(db: Session):
    if db.query(SubscriptionPlan).count() == 0:
        plans = [
            SubscriptionPlan(name="Basic Plan", price=450.0, description="Basic monthly subscription"),
            SubscriptionPlan(name="Premium Plan", price=1050.0, description="Premium monthly subscription"),
            SubscriptionPlan(name="Institutional Plan", price=3500.0, description="Institutional monthly subscription"),
        ]
        db.add_all(plans)
        db.commit()


def ensure_routines_image_column():
    with engine.connect() as conn:
        res = conn.execute(text("PRAGMA table_info(routines)")).fetchall()
        cols = [row[1] for row in res]
        if "image" not in cols:
            try:
                conn.execute(text("ALTER TABLE routines ADD COLUMN image VARCHAR"))
                print("Added 'image' column to routines table.")
            except Exception as e:
                print("Could not add image column:", e)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    # seed plans if missing
    with SessionLocal() as db:
        seed_plans(db)
    ensure_routines_image_column()
    print("Startup complete. Static dir:", STATIC_DIR.resolve())


# =====================================
# SIMPLE root page
# =====================================
@app.get("/", response_class=HTMLResponse)
def root_index():
    html_content = """
    <html><body style="font-family:Arial;padding:2rem">
      <h2>NeuroHue API (Full)</h2>
      <ul>
        <li><a href="/static/jigsaw.html">Static jigsaw UI (if present)</a></li>
        <li>/api/jigsaw/tiles?level=1</li>
        <li>/api/jigsaw/check (POST)</li>
        <li>/routines endpoints (create/list/update/delete/mark)</li>
      </ul>
    </body></html>
    """
    return HTMLResponse(html_content)


@app.get("/shape-game/level/{level}")
def get_shape_stage(level: int):
    if level not in EDU_GAME_LEVELS:
        raise HTTPException(status_code=404, detail="Stage not found")
    shapes = EDU_GAME_LEVELS[level]
    outlines = shapes.copy()
    random.shuffle(outlines)
    return {
        "level": level,
        "shapes": shapes,
        "outlines": outlines,
        "message": f"Drag and match the shapes to their outlines for stage {level}."
    }

# ---------------------------
# POST submit results
# ---------------------------
@app.post("/shape-game/submit")
def submit_shape_stage(data: EduShapeSubmitSchema, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == data.user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if data.total_shapes <= 0:
        raise HTTPException(status_code=400, detail="total_shapes must be > 0")
    score = int((data.matched_shapes / data.total_shapes) * 100)
    completed_flag = "yes" if score >= 80 else "no"
    record = EduShapeProgress(
        user_id=data.user_id,
        stage=data.level,
        score=score,
        completed=completed_flag,
        played_on=datetime.date.today()
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    # determine next stage
    max_stage = max(EDU_GAME_LEVELS.keys())
    next_stage = data.level + 1 if data.level < max_stage else None

    return {
        "message": "Stage completed!" if completed_flag == "yes" else "Try again!",
        "score": score,
        "completed": completed_flag,
        "next_stage": next_stage,
        "progress_id": record.id
    }

# ---------------------------
# GET progress summary for a user
# ---------------------------
@app.get("/shape-game/progress/{user_id}")
def get_edu_shape_progress(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    progress_summary = []
    for stage_id in sorted(EDU_GAME_LEVELS.keys()):
        q = db.query(EduShapeProgress).filter(EduShapeProgress.user_id == user_id, EduShapeProgress.stage == stage_id)
        attempts = q.count()
        if attempts == 0:
            progress_summary.append({
                "level": stage_id,
                "attempts": 0,
                "latest_score": None,
                "best_score": None,
                "completed": None,
                "last_played": None
            })
            continue
        latest = q.order_by(EduShapeProgress.id.desc()).first()
        best = q.order_by(EduShapeProgress.score.desc()).first()
        progress_summary.append({
            "level": stage_id,
            "attempts": attempts,
            "latest_score": latest.score,
            "best_score": best.score,
            "completed": latest.completed,
            "last_played": latest.played_on.isoformat() if latest.played_on else None
        })

    return {"user_id": user_id, "username": user.username, "progress": progress_summary}
