try:
	import whisper
except Exception as _import_err:
	# Provide a clear ImportError when the whisper package isn't available
	class _WhisperMissing:
		@staticmethod
		def load(*args, **kwargs):
			raise ImportError(
				"whisper package not found. Install it with: pip install -U openai-whisper"
			)

	whisper = _WhisperMissing()

model = whisper.load_model("turbo")

