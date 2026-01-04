class BasicException(Exception):
	"""
	Basic custom exception to display clean error messages in terminal.
	"""

	def __init__(self, message: str):
		super().__init__(message)
		self.message = message

	def __str__(self):
		return f"[ERROR] {self.message}"
