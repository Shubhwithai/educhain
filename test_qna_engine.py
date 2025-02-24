import time
from educhain.engines.qna_engine import QnAEngine
from educhain.core.config import LLMConfig

# Initialize the QnA Engine
qna_engine = QnAEngine()

start_time = time.time()
questions = qna_engine.generate_questions(
    topic="Fractions",
    num=25,
    batch_size=5
)
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")
questions.show()
# questions.save_to_file("questions.json")
