run:
	venv:
	python3 -m venv venv
	pip3 install --upgrade pip
	venv/bin/pip install -r requirements.txt

install: venv

run:
	source venv/bin/activate && python3 qa_ai.py