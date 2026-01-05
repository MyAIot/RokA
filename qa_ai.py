import os
from autocorrect import Speller
from transformers import pipeline
try:
    vi_correction = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")
except Exception as e:
    print(f"Lỗi khởi tạo model sửa lỗi chính tả tiếng Việt: {e}")
    vi_correction = None
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# ...existing code...
from gemini_suggest import GeminiClient



# Đọc dữ liệu từ file CSV
# File CSV: question,answer

def load_cache(cache_path):
    data = np.load(cache_path, allow_pickle=True)
    return data['embeddings'], data['questions'].tolist(), data['answers'].tolist()

def find_best_match(query, model, embeddings, questions, answers):
    query_emb = model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, embeddings)[0]
    idx = np.argmax(sims)
    # Trả về top 3 chỉ số lớn nhất
    top_idx = np.argsort(sims)[::-1][:3]
    top_qas = [(questions[i], answers[i], sims[i]) for i in top_idx]
    return top_qas

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
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    cache_path = 'train/rok_qa_cache_en.npz'
    gemini_api_key = "AIzaSyC0Iuwbc0GRe3v0clwxRqMNlEG_HXxXr48"
    print('Sử dụng dữ liệu tiếng Anh.')

    if os.path.exists(cache_path):
        print('Đang load embeddings từ cache...')
        qa_embeddings, qa_questions, qa_answers = load_cache(cache_path)
        model = SentenceTransformer(model_name)
    else:
        print('Không tìm thấy cache. Vui lòng chạy train/train.py để build cache trước.')
        return
    spell = Speller(lang='en')
    if vi_correction is None:
        print('Cảnh báo: Không có model sửa lỗi chính tả tiếng Việt.')



    def is_vietnamese(text):
        # Kiểm tra nếu có ký tự tiếng Việt phổ biến
        vietnamese_chars = set('ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ')
        return any(c in vietnamese_chars for c in text.lower())

    gemini_client = GeminiClient(gemini_api_key)
    # print("Các model Gemini khả dụng:")
    # try:
    #     models = gemini_client.list_models()
    #     for m in models:
    #         print(f"- {m}")
    # except Exception as e:
    #     print(f"Lỗi khi lấy danh sách model Gemini: {e}")
    while True:
        print("\n" + "="*50)
        print('Nhập câu hỏi (để trống để thoát):')
        user_question = input('Câu hỏi: ')
        if not user_question.strip():
            print('Kết thúc.')
            break
        # Sửa lỗi chính tả tiếng Việt bằng transformers pipeline nếu có, còn lại dùng autocorrect cho tiếng Anh
        if is_vietnamese(user_question) and vi_correction is not None:
            corrected_question = vi_correction(user_question)[0]['generated_text']
        elif is_vietnamese(user_question):
            corrected_question = user_question
        else:
            corrected_question = spell(user_question)
        orange = '\033[38;5;208m'
        reset = '\033[0m'
        if corrected_question != user_question:
            print(f'Câu hỏi đã sửa lỗi chính tả: {orange}{corrected_question}{reset}')
        else:
            print('Không phát hiện lỗi chính tả.')


        # Tìm kiếm trên ngôn ngữ đã chọn
        top_qas = find_best_match(corrected_question, model, qa_embeddings, qa_questions, qa_answers)

        # Câu trả lời gợi ý nhất (đậm, xanh lá)
        matched_question, matched_answer, score = top_qas[0]
        bold_green = '\033[1;32m'
        print(f'Câu hỏi gần nhất: {matched_question}')


        print(f'Câu trả lời gợi ý: {bold_green}{matched_answer}{reset} (score: {score:.2f})')

        gemini_response = gemini_client.generate(corrected_question)
        orange = '\033[33m'  # Màu vàng/cam cơ bản
        reset = '\033[0m'
        print(f'Gemini AI trả lời: {orange}{gemini_response}{reset}')
        # Hiển thị thêm 2 câu và đáp án gần giống nhất tiếp theo (màu xanh nhạt)
        # light_green = '\033[0;32m'
        # for i in range(1, 3):
        #     q, a, s = top_qas[i]
        #     print(f'Gợi ý phụ #{i}: {q}')
        #     print(f'{light_green}Đáp án phụ: {a}{reset} (score: {s:.2f})')

if __name__ == '__main__':
    main()

# Hướng dẫn build file exe:
# 1. Cài pyinstaller: pip install pyinstaller
# 2. Build: pyinstaller --onefile qa_ai.py
# File exe sẽ nằm trong thư mục dist/
