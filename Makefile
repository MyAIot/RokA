venv310:
	python3.10 -m venv venv310
	venv310/bin/pip install --upgrade pip
	venv310/bin/pip install -r requirements.txt

install: venv310

makeCsv:
	source venv310/bin/activate && python extract_rok_qa.py

run:
	source venv310/bin/activate && python3 qa_ai.py