import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Đọc dữ liệu từ file CSV
# File CSV: question,answer,question_vi,answer_vi

def load_qa_data(csv_path, use_vietnamese=False):
    df = pd.read_csv(csv_path)
    if use_vietnamese and 'question_vi' in df.columns and 'answer_vi' in df.columns:
        questions = df['question_vi'].fillna('').tolist()
        answers = df['answer_vi'].fillna('').tolist()
    else:
        questions = df['question'].tolist()
        answers = df['answer'].tolist()
    return questions, answers

def build_embeddings(questions, model_name='sentence-transformers/all-mpnet-base-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(questions, convert_to_numpy=True)
    return model, embeddings

def save_cache(cache_path, embeddings, questions, answers):
    np.savez_compressed(cache_path, embeddings=embeddings, questions=questions, answers=answers)

def main():
    csv_path = 'train/rok_qa.csv'  # File CSV trong thư mục root
    model_name = 'sentence-transformers/all-mpnet-base-v2'

    # Build cache cho tiếng Anh
    print('Đang build embeddings cho tiếng Anh...')
    qa_questions_en, qa_answers_en = load_qa_data(csv_path, use_vietnamese=False)
    model_en, qa_embeddings_en = build_embeddings(qa_questions_en, model_name)
    save_cache('train/rok_qa_cache_en.npz', qa_embeddings_en, qa_questions_en, qa_answers_en)
    print('Đã lưu cache cho tiếng Anh: train/rok_qa_cache_en.npz')

    # Build cache cho tiếng Việt
    print('Đang build embeddings cho tiếng Việt...')
    qa_questions_vi, qa_answers_vi = load_qa_data(csv_path, use_vietnamese=True)
    model_vi, qa_embeddings_vi = build_embeddings(qa_questions_vi, model_name)
    save_cache('train/rok_qa_cache_vi.npz', qa_embeddings_vi, qa_questions_vi, qa_answers_vi)
    print('Đã lưu cache cho tiếng Việt: train/rok_qa_cache_vi.npz')

if __name__ == '__main__':
    main()