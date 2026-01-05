# import thư viện cần thiết
from google import genai


# Class GeminiClient để quản lý kết nối và gọi Gemini AI
class GeminiClient:
	def __init__(self, api_key):
		"""
		Khởi tạo GeminiClient với API key.
		"""
		self.client = genai.Client(api_key=api_key)
		self.api_key = api_key

	def generate(self, prompt, model="gemini-3-flash-preview"):
		"""
		Gửi prompt đến Gemini AI và trả về kết quả sinh ra.
		"""
		try:
			response = self.client.models.generate_content(
				model=model,
				contents="For the following multiple-choice question with four options (A, B, C, D), just select the answer—no explanation required.\n" + prompt
			)
			return response.text if hasattr(response, 'text') else str(response)
		except Exception as e:
			return f"Lỗi khi gọi Gemini AI: {e}"
	def list_models(self):
		"""
		Liệt kê các mô hình có sẵn từ Gemini AI.
		"""
		try:
			return self.client.models.list()
		except Exception as e:
			print(f"Lỗi khi liệt kê mô hình: {e}")
			return ""
