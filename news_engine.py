"""Motor de notícias em tempo real: coleta via RSS + classificação de sentimento via Groq (Llama-3)."""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TypedDict

import feedparser
import httpx

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_NEWS_MODEL = "llama-3.1-8b-instant"

RSS_SOURCES: list[tuple[str, str]] = [
    ("InfoMoney", "https://www.infomoney.com.br/feed/"),
    ("Money Times", "https://www.moneytimes.com.br/mercados/feed/"),
    ("G1 Economia", "https://g1.globo.com/rss/g1/economia/"),
]

MAX_ITEMS_PER_SOURCE = 12

POSITIVE_KEYWORDS = (
    "alta", "lucro", "supera", "dispara", "avança", "recorde",
    "valoriza", "sobe", "otimismo", "crescimento", "aprova",
)
NEGATIVE_KEYWORDS = (
    "queda", "prejuízo", "cai", "despenca", "corte", "recessão",
    "desaba", "crise", "pessimismo", "demissão", "investigação",
)


class RawNewsItem(TypedDict):
    id: str
    title: str
    url: str
    source: str


def _make_id(link: str) -> str:
    return hashlib.sha1(link.encode("utf-8")).hexdigest()[:16]


def _fetch_source(name: str, feed_url: str) -> list[RawNewsItem]:
    parsed = feedparser.parse(feed_url)
    items: list[RawNewsItem] = []
    for entry in parsed.entries[:MAX_ITEMS_PER_SOURCE]:
        link = entry.get("link", "")
        title = entry.get("title", "").strip()
        if not link or not title:
            continue
        items.append({"id": _make_id(link), "title": title, "url": link, "source": name})
    return items


async def fetch_all_raw() -> list[RawNewsItem]:
    results: list[RawNewsItem] = []
    for name, url in RSS_SOURCES:
        try:
            items = await asyncio.to_thread(_fetch_source, name, url)
            results.extend(items)
        except Exception:
            continue
    return results


def _fallback_sentiment(title: str) -> tuple[str, int]:
    t = title.lower()
    pos = sum(1 for w in POSITIVE_KEYWORDS if w in t)
    neg = sum(1 for w in NEGATIVE_KEYWORDS if w in t)
    if pos > neg:
        return "positive", min(5 + pos, 10)
    if neg > pos:
        return "negative", min(5 + neg, 10)
    return "neutral", 3


async def _classify_batch_groq(titles: list[str], api_key: str) -> list[tuple[str, int]]:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    system = (
        "Você é um analista financeiro. Classifique cada manchete quanto ao sentimento para o "
        "mercado brasileiro (positive, negative ou neutral) e dê um impact_score de 1 a 10 "
        "(potencial de mover preços de ativos). Responda APENAS com um JSON array, um objeto por "
        'manchete na mesma ordem, formato [{"sentiment": "positive", "impact_score": 7}], sem texto '
        "adicional, sem markdown."
    )
    payload = {
        "model": GROQ_NEWS_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": numbered},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(GROQ_CHAT_URL, json=payload, headers=headers)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

    content = content.strip().strip("`")
    if content.lower().startswith("json"):
        content = content[4:].strip()
    parsed = json.loads(content)

    out: list[tuple[str, int]] = []
    for item in parsed:
        sentiment = item.get("sentiment", "neutral")
        if sentiment not in ("positive", "negative", "neutral"):
            sentiment = "neutral"
        score = max(1, min(int(item.get("impact_score", 3)), 10))
        out.append((sentiment, score))
    return out


async def classify_titles(titles: list[str], api_key: str) -> list[tuple[str, int]]:
    if not titles:
        return []
    if not api_key:
        return [_fallback_sentiment(t) for t in titles]
    try:
        results = await _classify_batch_groq(titles, api_key)
        if len(results) != len(titles):
            raise ValueError("Groq retornou quantidade de itens divergente")
        return results
    except Exception:
        return [_fallback_sentiment(t) for t in titles]


async def refresh_news(db_conn, api_key: str) -> int:
    """Busca notícias novas via RSS, classifica sentimento e persiste no DuckDB. Retorna qtd. de itens novos."""
    raw_items = await fetch_all_raw()
    if not raw_items:
        return 0

    existing_ids = {row[0] for row in db_conn.execute("SELECT id FROM news").fetchall()}
    new_items = [item for item in raw_items if item["id"] not in existing_ids]
    if not new_items:
        return 0

    classifications = await classify_titles([item["title"] for item in new_items], api_key)

    for item, (sentiment, impact) in zip(new_items, classifications):
        db_conn.execute(
            "INSERT INTO news(id, title, url, source, sentiment, impact_score) VALUES(?, ?, ?, ?, ?, ?)",
            [item["id"], item["title"], item["url"], item["source"], sentiment, impact],
        )
    return len(new_items)
