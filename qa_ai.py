import os
from autocorrect import Speller
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Đọc dữ liệu từ file CSV
# File CSV: question,answer

def load_qa_data(csv_path):
    df = pd.read_csv(csv_path)
    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    return questions, answers

def build_embeddings(questions, model_name='sentence-transformers/all-mpnet-base-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(questions, convert_to_numpy=True)
    return model, embeddings

def save_cache(cache_path, embeddings, questions, answers):
    np.savez_compressed(cache_path, embeddings=embeddings, questions=questions, answers=answers)

def load_cache(cache_path):
    data = np.load(cache_path, allow_pickle=True)
    return data['embeddings'], data['questions'].tolist(), data['answers'].tolist()

def find_best_match(query, model, embeddings, questions, answers):
    query_emb = model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, embeddings)[0]
    idx = np.argmax(sims)
    return questions[idx], answers[idx], sims[idx]

def answer_mc_question(user_question, choices, model, qa_questions, qa_answers, qa_embeddings):
    # Ghép từng đáp án vào câu hỏi, tính similarity với các câu hỏi gốc
    best_score = -1
    best_choice = None
    for label, choice in choices.items():
        full_query = f"{user_question} {choice}"
        _, _, score = find_best_match(full_query, model, qa_embeddings, qa_questions, qa_answers)
        if score > best_score:
            best_score = score
            best_choice = label
    return best_choice, best_score

def main():
    csv_path = 'rok_qa.csv'
    cache_path = 'rok_qa_cache.npz'
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    if os.path.exists(cache_path):
        print('Đang load embeddings từ cache...')
        qa_embeddings, qa_questions, qa_answers = load_cache(cache_path)
        model = SentenceTransformer(model_name)
    else:
        print('Đang build embeddings, lần đầu sẽ hơi lâu...')
        qa_questions, qa_answers = load_qa_data(csv_path)
        model, qa_embeddings = build_embeddings(qa_questions, model_name)
        save_cache(cache_path, qa_embeddings, qa_questions, qa_answers)
    spell = Speller(lang='en')
    while True:
        print('Nhập câu hỏi (để trống để thoát):')
        user_question = input('Câu hỏi: ')
        if not user_question.strip():
            print('Kết thúc.')
            break
        corrected_question = spell(user_question)
        orange = '\033[38;5;208m'
        reset = '\033[0m'
        if corrected_question != user_question:
            print(f'Câu hỏi đã sửa lỗi chính tả: {orange}{corrected_question}{reset}')
        else:
            print('Không phát hiện lỗi chính tả.')
        matched_question, matched_answer, score = find_best_match(corrected_question, model, qa_embeddings, qa_questions, qa_answers)
        print(f'Câu hỏi gần nhất: {matched_question}')
        # In đậm và đổi màu xanh lá cho câu trả lời gợi ý
        bold_green = '\033[1;32m'
        print(f'Câu trả lời gợi ý: {bold_green}{matched_answer}{reset} (score: {score:.2f})')

if __name__ == '__main__':
    main()

# Hướng dẫn build file exe:
# 1. Cài pyinstaller: pip install pyinstaller
# 2. Build: pyinstaller --onefile qa_ai.py
# File exe sẽ nằm trong thư mục dist/
