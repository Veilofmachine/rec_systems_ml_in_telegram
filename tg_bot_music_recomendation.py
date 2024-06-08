import telebot
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from auth_t import token
#Инициализация бота
bot = telebot.TeleBot(token)

#Препроцесс данных
df = pd.read_csv("data/data.csv", low_memory=False)[:1000]
df = df.drop_duplicates(subset="Song Name")
df = df.dropna(axis=0)
df = df.drop(df.columns[3:], axis=1)
df["Artist Name"] = df["Artist Name"].str.replace(" ", "")
df["data"] = df.apply(lambda value: " ".join(value.astype("str")), axis=1)

#Создание матрицы схожести
vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df["data"])
similarities = cosine_similarity(vectorized)
df_tmp = pd.DataFrame(similarities, columns=df["Song Name"], index=df["Song Name"]).reset_index()

#Имплементация в телеграм боте
@bot.message_handler(commands=['start'])
def send_instructions(message):
    bot.reply_to(message, "Этот бот предназначен для рекомендации музыки")

@bot.message_handler(func=lambda message: True)
def recommend_song(message):
    input_song = message.text

    if input_song in df_tmp.columns:
        recommendation = df_tmp.nlargest(11, input_song)["Song Name"]
        response = "Вам бы понравились эти песни:\n" + "\n".join(recommendation.values[1:])
    else:
        response = "Не могу опознать музыку и предложить рекомендацию."

    bot.reply_to(message, response)

# Start the bot
bot.polling()