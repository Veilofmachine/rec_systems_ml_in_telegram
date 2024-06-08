import telebot
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from auth_t import token
#
bot = telebot.TeleBot(token)

# Загрузка данных и их подготовка
credits = pd.read_csv("data/5000_credits.csv")
movies = pd.read_csv("data/5000_movies.csv")
credits_column_renamed = credits.rename(index=str, columns={"movie_id": "id"})
movies_merge = movies.merge(credits_column_renamed, on='id')
movies_cleaned = movies_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])

# замена ненужных значений
movies_cleaned['overview'] = movies_cleaned['overview'].fillna('')

# Создание TF-IDF матрицы и sigmoid kernel
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), stop_words='english')
tfv_matrix = tfv.fit_transform(movies_cleaned['overview'])
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
indices = pd.Series(movies_cleaned.index, index=movies_cleaned['original_title']).drop_duplicates()

# Создание самого телеграмм бота
@bot.message_handler(commands=['recommend'])
def handle_recommendation(message):
    input_title = message.text.split(' ', 1)[1].strip()

    if input_title in movies_cleaned['original_title'].values:
        recommendations = give_recomendations(input_title, sig)
        response = "Recommended movies for '{}':\n".format(input_title) + "\n".join(recommendations)
    else:
        response = "Movie '{}' not found in the database.".format(input_title)

    bot.reply_to(message, response)

# Основная рекомендательная функция фильмов
def give_recomendations(title, sig=sig):
    idx = indices[title]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:11]
    movie_indices = [i[0] for i in sig_scores]
    return movies_cleaned['original_title'].iloc[movie_indices].tolist()

# Старт бота
bot.polling()