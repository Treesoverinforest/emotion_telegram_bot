# bot.py
import os
import logging
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# === Настройка ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загружаем модель один раз при старте
MODEL_NAME = "Djacon/rubert-tiny2-russian-emotion-detection"
logger.info("Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
logger.info("Модель загружена!")

EMOTIONS = [
    "Нейтрально", "Радость", "Грусть", "Гнев", "Интерес",
    "Удивление", "Отвращение", "Страх", "Вина", "Стыд"
]

def predict_emotions(text: str, threshold=0.3):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().numpy()
    if probs.ndim == 0:
        probs = np.array([probs])
    emotions = []
    for i, prob in enumerate(probs):
        if prob >= threshold:
            emotions.append(f"{EMOTIONS[i]} ({prob:.2f})")
    return emotions if emotions else ["Нейтрально (0.00)"]

# === Обработчики ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Напиши мне что-нибудь на русском — я определю эмоции.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    logger.info(f"Сообщение: {text}")
    emotions = predict_emotions(text)
    await update.message.reply_text("Эмоции:\n" + "\n".join(emotions))

# === Flask-приложение для webhook ===
app = Flask(__name__)

# Создаём бота один раз
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "ВАШ_ТОКЕН_ЗДЕСЬ")
application = Application.builder().token(TELEGRAM_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Запускаем webhook при старте
with application:
    application.bot.set_webhook(url=f"https://{os.environ.get('USERNAME', 'yourname')}.pythonanywhere.com/{TELEGRAM_TOKEN}")

@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def telegram_webhook():
    update = Update.de_json(request.get_json(force=True), application.bot)
    application.update_queue.put_nowait(update)
    return "OK"

# Health-check
@app.route("/")
def hello():
    return "Telegram emotion bot is running!"
