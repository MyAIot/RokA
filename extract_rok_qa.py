import pandas as pd
from bs4 import BeautifulSoup

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
    df.to_csv('rok_qa.csv', index=False)
    print(f'Đã xuất {len(df)} cặp Q&A ra file rok_qa.csv')
else:
    print('Không tìm thấy cặp Q&A nào!')
