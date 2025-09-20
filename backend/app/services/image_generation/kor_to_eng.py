import ollama


def load_llm():
    llm = ollama.load_llm("gemma3:4b")
    return llm

def kor_to_eng(korean_text: str) -> str:
    llm = load_llm()
    prompt = f"Translate the following Korean text to English: {korean_text}"
    response = llm(prompt)
    return response['text']

if __name__ == "__main__":
    korean_text = "침대에 누워있는 고양이의 사진을 보여줘"
    english_text = kor_to_eng(korean_text)
    print(english_text)