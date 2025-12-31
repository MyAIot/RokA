
import pandas as pd
from bs4 import BeautifulSoup
from googletrans import Translator

# Đọc file HTML
with open('data.html', 'r', encoding='utf-8') as f:
    html = f.read()

soup = BeautifulSoup(html, 'html.parser')

qa_pairs = []
questions = soup.find_all('td', class_='column-1')
answers = soup.find_all('td', class_='column-2')

for q, a in zip(questions, answers):
    q_text = q.get_text(strip=True)
    a_text = a.get_text(strip=True)
    if len(q_text) > 10 and len(a_text) > 0:
        qa_pairs.append({'question': q_text, 'answer': a_text})

if qa_pairs:
    df = pd.DataFrame(qa_pairs).drop_duplicates()
    # Xuất file CSV chỉ có tiếng Anh trước
    df.to_csv('rok_qa.csv', index=False)
    print(f'Đã xuất {len(df)} cặp Q&A ra file rok_qa.csv (chỉ tiếng Anh)')

    # Sau đó dịch từng batch 20 dòng và cập nhật cột tiếng Việt cho question và answer
    translator = Translator()
    batch_size = 20
    # Dịch song song question và answer theo từng batch
    questions_en = df['question'].tolist()
    answers_en = df['answer'].tolist()
    questions_vi = []
    answers_vi = []
    total_batches = (len(questions_en) + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, len(questions_en), batch_size)):
        print(f"[Batch {batch_idx+1}/{total_batches}] Dịch question và answer...")
        batch_q = questions_en[i:i+batch_size]
        batch_a = answers_en[i:i+batch_size]
        # Dịch batch question
        text_q = '\n'.join(batch_q)
        try:
            t_q = translator.translate(text_q, src='en', dest='vi')
            translated_q = t_q.text
        except Exception as e:
            print(f"Lỗi dịch question batch {batch_idx+1}: {e}")
            translated_q = '\n'.join([''] * len(batch_q))
        lines_q = [line.strip() for line in translated_q.split('\n')]
        while len(lines_q) < len(batch_q):
            lines_q.append('')
        questions_vi.extend(lines_q)
        # Dịch batch answer
        text_a = '\n'.join(batch_a)
        try:
            t_a = translator.translate(text_a, src='en', dest='vi')
            translated_a = t_a.text
        except Exception as e:
            print(f"Lỗi dịch answer batch {batch_idx+1}: {e}")
            translated_a = '\n'.join([''] * len(batch_a))
        lines_a = [line.strip() for line in translated_a.split('\n')]
        while len(lines_a) < len(batch_a):
            lines_a.append('')
        answers_vi.extend(lines_a)
    df['question_vi'] = questions_vi
    df['answer_vi'] = answers_vi
    df.to_csv('rok_qa.csv', index=False)
    print(f'Đã cập nhật file rok_qa.csv với cột tiếng Việt cho cả câu hỏi và trả lời')
else:
    print('Không tìm thấy cặp Q&A nào!')
