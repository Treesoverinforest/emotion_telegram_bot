import os
import requests
from fastapi import FastAPI, Request
from typing import List

# --- Конфиг ---
MODEL_ID = "Djacon/rubert-tiny2-russian-emotion-detection"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HF_TOKEN = os.environ.get("HF_TOKEN")  # положи сюда Hugging Face токен
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")  # токен от BotFather
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Если не указаны токены — приложение падает, чтобы не работать в небезопасном режиме
if not HF_TOKEN or not TELEGRAM_TOKEN:
    raise RuntimeError("Нужны переменные окружения HF_TOKEN и TELEGRAM_TOKEN")

app = FastAPI()

@app.get("/")
async def root():
    return {"ok": True, "msg": "Emotion-bot webhook ready."}

@app.post("/webhook")
async def telegram_webhook(request: Request):
    payload = await request.json()
    # Telegram шлёт разные update; нас интересует update.message.text
    message = payload.get("message") or payload.get("edited_message")
    if not message:
        return {"ok": True}
    chat_id = message["chat"]["id"]
    text = message.get("text", "")
    if not text:
        send_telegram_message(chat_id, "Я понимаю только текстовые сообщения.")
        return {"ok": True}

    # Вызов HF Inference API
    emotions = predict_emotions_via_hf(text)
    reply = "Эмоции:\n" + "\n".join(emotions)
    send_telegram_message(chat_id, reply)
    return {"ok": True}

def predict_emotions_via_hf(text: str, threshold: float = 0.3) -> List[str]:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text}
    try:
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
    except Exception as e:
        return [f"Ошибка запроса к Hugging Face: {e}"]
    if resp.status_code != 200:
        return [f"Hugging Face вернул статус {resp.status_code}: {resp.text[:200]}"]

    try:
        data = resp.json()
    except Exception:
        return ["Пустой/неожиданный ответ от Hugging Face"]

    # Разные модели/эндпойнты возвращают разную структуру:
    #  - часто это список dict {'label': 'Joy', 'score': 0.92}
    #  - или словарь с 'labels' и 'scores'
    labels = []
    if isinstance(data, list):
        for item in data:
            label = item.get("label", "unknown")
            score = item.get("score", 0.0)
            if score >= threshold:
                labels.append(f"{label} ({score:.2f})")
    elif isinstance(data, dict):
        # Попробуем обычные форматы
        if "labels" in data and "scores" in data:
            for lbl, sc in zip(data["labels"], data["scores"]):
                if sc >= threshold:
                    labels.append(f"{lbl} ({sc:.2f})")
        else:
            # fallback: покажем весь словарь кратко
            labels.append(str(data)[:800])
    if not labels:
        labels = ["Нейтрально (0.00)"]
    return labels

def send_telegram_message(chat_id: int, text: str):
    url = f"{TELEGRAM_API}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text})
    except Exception as e:
        print("Ошибка отправки сообщения в Telegram:", e)
