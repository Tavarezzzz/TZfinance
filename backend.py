from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio, os, json, hashlib, bcrypt, jwt
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import httpx
import duckdb
from pathlib import Path
from pydantic import BaseModel
import news_engine


class AssetForm(BaseModel):
    ticker: str
    quantity: float
    average_price: float

# ─── Configurações ────────────────────────────────────────────────────────────
JWT_SECRET    = os.getenv("JWT_SECRET", "investai-secret-key-dev")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"
NEWS_REFRESH_INTERVAL_SECONDS = int(os.getenv("NEWS_REFRESH_INTERVAL_SECONDS", "300"))

# Banco de dados local
DB_PATH = "./data/investai_local.duckdb"

SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS seq_users;
CREATE SEQUENCE IF NOT EXISTS seq_portfolio;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY DEFAULT nextval('seq_users'),
    name VARCHAR, email VARCHAR UNIQUE,
    password_hash VARCHAR, plan VARCHAR DEFAULT 'free'
);
CREATE TABLE IF NOT EXISTS news (
    id VARCHAR PRIMARY KEY,
    title VARCHAR, url VARCHAR, source VARCHAR,
    sentiment VARCHAR, impact_score INTEGER,
    classified_at TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS portfolio (
    id INTEGER PRIMARY KEY DEFAULT nextval('seq_portfolio'),
    user_id INTEGER,
    ticker VARCHAR,
    quantity DOUBLE,
    average_price DOUBLE,
    added_at TIMESTAMP DEFAULT NOW()
);
"""

db_conn = None
db_lock = asyncio.Lock()
_news_task: asyncio.Task | None = None

async def _news_background_loop():
    while True:
        try:
            async with db_lock:
                added = await news_engine.refresh_news(db_conn, GROQ_API_KEY)
            if added:
                print(f"📰 {added} notícia(s) nova(s) coletada(s).")
        except Exception as e:
            print(f"⚠️  Erro ao atualizar notícias: {e}")
        await asyncio.sleep(NEWS_REFRESH_INTERVAL_SECONDS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_conn, _news_task
    Path("./data").mkdir(parents=True, exist_ok=True)
    db_conn = duckdb.connect(DB_PATH)
    db_conn.execute(SCHEMA)
    print("✅ DuckDB iniciado localmente.")
    _news_task = asyncio.create_task(_news_background_loop())
    print(f"📡 Radar de notícias ativo (atualização a cada {NEWS_REFRESH_INTERVAL_SECONDS}s).")
    yield
    _news_task.cancel()
    db_conn.close()

app = FastAPI(title="InvestAI API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ─── Rotas de Autenticação ────────────────────────────────────────────────────
@app.post("/auth/register")
async def register(name: str, email: str, password: str):
    existing = db_conn.execute("SELECT id FROM users WHERE email=?", [email]).fetchone()
    if existing: raise HTTPException(409, "E-mail já cadastrado")
    pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    # O DuckDB agora cuida do ID automaticamente
    db_conn.execute("INSERT INTO users(name, email, password_hash) VALUES(?, ?, ?)", [name, email, pw])
    return {"message": "Sucesso"}

@app.post("/auth/login")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    row = db_conn.execute("SELECT id, name, email, password_hash FROM users WHERE email=?", [form.username]).fetchone()
    if not row or not bcrypt.checkpw(form.password.encode(), row[3].encode()):
        raise HTTPException(401, "Credenciais incorretas")
    
    token = jwt.encode({"sub": str(row[0]), "email": row[2], "exp": datetime.utcnow() + timedelta(hours=24)}, JWT_SECRET, algorithm="HS256")
    return {"access_token": token, "token_type": "bearer"}

def current_user(token: str = Depends(oauth2)):
    try: return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except: raise HTTPException(401, "Token inválido")

# ─── Rotas de Dados (Valuation & IA) ──────────────────────────────────────────
@app.get("/valuation/{ticker}")
async def valuation(ticker: str, user=Depends(current_user)):
    tk = ticker.upper()
    data = await asyncio.to_thread(_fetch_yfinance, tk)
    if "error" in data: raise HTTPException(404, "Ticker não encontrado")
    return data

def _fetch_yfinance(ticker: str):
    try:
        info = yf.Ticker(ticker).info
        price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        eps = info.get("trailingEps", 0)
        bvps = info.get("bookValue", 0)
        gn = round((22.5 * max(eps, 0) * max(bvps, 0)) ** 0.5, 2) if eps > 0 and bvps > 0 else None
        mos = round((gn - price) / price * 100, 1) if gn and price else None

        return {
            "ticker": ticker, "name": info.get("longName", ticker), "sector": info.get("sector", "N/A"),
            "price": price,
            "multiples": {
                "pe": round(info.get("trailingPE", 0), 2), "pb": round(info.get("priceToBook", 0), 2),
                "roe": round(info.get("returnOnEquity", 0) * 100, 2), "dividend_yield": round(info.get("dividendYield", 0) * 100, 2),
            },
            "graham": { "number": gn, "margin_of_safety_pct": mos, "is_undervalued": (mos or -1) > 0 }
        }
    except: return {"error": True}

@app.get("/news/stats")
async def news_stats(user=Depends(current_user)):
    total = db_conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    positive = db_conn.execute("SELECT COUNT(*) FROM news WHERE sentiment='positive'").fetchone()[0]
    negative = db_conn.execute("SELECT COUNT(*) FROM news WHERE sentiment='negative'").fetchone()[0]
    neutral = total - positive - negative
    return {
        "total": total, "positive": positive, "negative": negative, "neutral": neutral,
        "positive_pct": round((positive / total) * 100, 1) if total else 0,
    }

@app.get("/news/feed")
async def get_news(limit: int = 30, user=Depends(current_user)):
    rows = db_conn.execute(
        "SELECT id, title, url, source, sentiment, impact_score, classified_at "
        "FROM news ORDER BY classified_at DESC LIMIT ?", [limit]
    ).fetchall()
    return [
        {
            "id": r[0], "title": r[1], "url": r[2], "source": r[3],
            "sentiment": r[4], "impact_score": r[5],
            "classified_at": r[6].isoformat() if r[6] else None,
        }
        for r in rows
    ]

@app.post("/news/refresh")
async def force_news_refresh(user=Depends(current_user)):
    async with db_lock:
        added = await news_engine.refresh_news(db_conn, GROQ_API_KEY)
    return {"added": added}

@app.post("/chat")
async def chat(question: str, user=Depends(current_user)):
    if not GROQ_API_KEY:
        return {"answer": "Configure a GROQ_API_KEY no .env para ativar o assistente de IA.", "source": "Sistema"}

    system = (
        "Você é o assistente do TzFinance, especializado em educação financeira e análise "
        "fundamentalista conservadora para o mercado brasileiro. Responda em português, de forma "
        "clara e didática. Nunca apresente recomendações de compra ou venda como certeza absoluta; "
        "sempre contextualize riscos e mencione que não substitui um profissional certificado."
    )
    payload = {
        "model": GROQ_CHAT_MODEL,
        "temperature": 0.4,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
        return {"answer": answer, "source": "Llama-3 (Groq)"}
    except Exception:
        return {"answer": "Não foi possível consultar a IA agora. Tente novamente em instantes.", "source": "Sistema"}

@app.post("/portfolio")
async def add_portfolio_asset(asset: AssetForm, user=Depends(current_user)):
    user_id = int(user['sub'])
    tk = asset.ticker.upper()
    db_conn.execute(
        "INSERT INTO portfolio(user_id, ticker, quantity, average_price) VALUES(?, ?, ?, ?)", 
        [user_id, tk, asset.quantity, asset.average_price]
    )
    return {"message": "Ativo adicionado com sucesso!"}

@app.get("/portfolio")
async def get_portfolio(user=Depends(current_user)):
    user_id = int(user['sub'])
    # Agrupa os ativos para somar quantidades e fazer a média do preço
    rows = db_conn.execute("""
        SELECT ticker, SUM(quantity) as qty, SUM(quantity * average_price)/SUM(quantity) as avg_price 
        FROM portfolio WHERE user_id=? GROUP BY ticker
    """, [user_id]).fetchall()
    
    assets = []
    total_invested = 0
    total_current = 0
    
    for row in rows:
        tk, qty, avg_price = row[0], row[1], row[2]
        
        # Busca o preço atual em tempo real no Yahoo Finance
        try:
            info = yf.Ticker(tk).info
            current_price = info.get("currentPrice") or info.get("regularMarketPrice", avg_price)
        except:
            current_price = avg_price
            
        invested = qty * avg_price
        current_val = qty * current_price
        total_invested += invested
        total_current += current_val
        
        assets.append({
            "ticker": tk,
            "quantity": qty,
            "average_price": round(avg_price, 2),
            "current_price": round(current_price, 2),
            "total_value": round(current_val, 2),
            "profit_pct": round(((current_price / avg_price) - 1) * 100, 2) if avg_price else 0
        })
        
    profit_total = total_current - total_invested
    profit_pct_total = ((total_current / total_invested) - 1) * 100 if total_invested else 0
    
    return {
        "summary": {
            "total_invested": round(total_invested, 2),
            "total_current": round(total_current, 2),
            "total_profit": round(profit_total, 2),
            "profit_pct": round(profit_pct_total, 2)
        },
        "assets": assets
    }


