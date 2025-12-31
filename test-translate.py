
from googletrans import Translator

def test_google_translate():
	translator = Translator()
	text = "'A crater on the far side of the Moon is named after a Chinese official who was (supposedly) the first person to travel into space. What was his "
	try:
		result = translator.translate(text, src='en', dest='vi')
		print(f"English: {text}")
		print(f"Vietnamese: {result.text}")
	except Exception as e:
		print(f"Google Translate error: {e}")

if __name__ == "__main__":
	test_google_translate()
