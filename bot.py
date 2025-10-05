import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# === Настройка логирования ===
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# === Загрузка модели и токенизатора ===
MODEL_NAME = "Djacon/rubert-tiny2-russian-emotion-detection"
logger.info("Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  # режим инференса
logger.info("Модель загружена!")

# === Эмоции согласно документации ===
EMOTIONS = [
    "Нейтрально",
    "Радость",
    "Грусть",
    "Гнев",
    "Интерес",
    "Удивление",
    "Отвращение",
    "Страх",
    "Вина",
    "Стыд"
]

# === Функция предсказания эмоций ===
def predict_emotions(text: str, threshold=0.3):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().numpy()

    # Берём эмоции с вероятностью выше порога
    emotions = []
    for i, prob in enumerate(probs):
        if prob >= threshold:
            emotions.append(f"{EMOTIONS[i]} ({prob:.2f})")
    return emotions if emotions else ["Нейтрально (0.00)"]

# === Обработчики команд ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправь мне любое сообщение на русском языке, и я определю его эмоциональный тон."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    logger.info(f"Получено сообщение: {text}")
    emotions = predict_emotions(text)
    response = "Эмоции:\n" + "\n".join(emotions)
    await update.message.reply_text(response)

# === Основная функция запуска ===
def main():
    # Вставь сюда свой токен от BotFather
    import os
    TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен!")
    application.run_polling()

if __name__ == "__main__":
    main()