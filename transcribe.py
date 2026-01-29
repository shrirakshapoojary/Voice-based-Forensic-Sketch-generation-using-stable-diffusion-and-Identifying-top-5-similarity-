# import whisper

# # Load Whisper model
# model = whisper.load_model("small")

# # Transcribe audio file and force English output
# result = model.transcribe("description.wav", language="en")

# # Print the text in English
# print("Heard (English):", result["text"])
import whisper
model = whisper.load_model("small")
result = model.transcribe("description.wav", language="en")
print(result["text"].strip())
