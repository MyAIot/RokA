import os
from autocorrect import Speller
try:
    from underthesea import correct as vi_correct
except ImportError:
    vi_correct = None
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Thêm import cho dịch tự động
from transformers import MarianMTModel, MarianTokenizer

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
    if vi_correct is None:
        print('Cảnh báo: underthesea chưa được cài, không sửa lỗi chính tả tiếng Việt.')

    # Khởi tạo model dịch Anh
    print('Đang load model dịch sang tiếng Anh...')
    trans_model_name = 'Helsinki-NLP/opus-mt-vi-en'
    trans_tokenizer = MarianTokenizer.from_pretrained(trans_model_name)
    trans_model = MarianMTModel.from_pretrained(trans_model_name)

    def is_vietnamese(text):
        # Kiểm tra nếu có ký tự tiếng Việt phổ biến
        vietnamese_chars = set('ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ')
        return any(c in vietnamese_chars for c in text.lower())

    def translate_vi_to_en(text):
        batch = trans_tokenizer([text], return_tensors="pt", padding=True)
        gen = trans_model.generate(**batch)
        return trans_tokenizer.decode(gen[0], skip_special_tokens=True)
    while True:
        print('Nhập câu hỏi (để trống để thoát):')
        user_question = input('Câu hỏi: ')
        if not user_question.strip():
            print('Kết thúc.')
            break
        # Nếu là tiếng Anh thì mới sửa lỗi chính tả, tiếng Việt giữ nguyên
        if is_vietnamese(user_question):
            corrected_question = user_question
        else:
            corrected_question = spell(user_question)
        orange = '\033[38;5;208m'
        reset = '\033[0m'
        if corrected_question != user_question:
            print(f'Câu hỏi đã sửa lỗi chính tả: {orange}{corrected_question}{reset}')
        else:
            print('Không phát hiện lỗi chính tả.')

        # Nếu là tiếng Việt thì mới dịch sang tiếng Anh, còn tiếng Anh thì giữ nguyên
        if is_vietnamese(corrected_question):
            translated_question = translate_vi_to_en(corrected_question)
            blue = '\033[1;34m'
            print(f'Câu hỏi dịch sang tiếng Anh: {blue}{translated_question}{reset}')
        else:
            translated_question = corrected_question
            print('Câu hỏi đã là tiếng Anh, không cần dịch.')
        # Sử dụng câu hỏi đã dịch hoặc giữ nguyên để tìm câu trả lời
        matched_question, matched_answer, score = find_best_match(translated_question, model, qa_embeddings, qa_questions, qa_answers)
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
