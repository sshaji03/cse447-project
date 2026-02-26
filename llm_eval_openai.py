import time
from openai import OpenAI

client = OpenAI()

MODEL = "gpt-4o-mini"

def query_llm(prefix):
    prompt = (
        "Given the following text prefix, "
        "return ONLY the three most likely next characters, "
        "with no explanation and no spaces.\n\n"
        f"Prefix: {prefix}\n\n"
        "Answer:"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a language model performing next-character prediction."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    result = response.choices[0].message.content.strip()
    return result[:3]


def evaluate(file_path):
    correct = 0
    total = 0

    start_time = time.time()

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            prefix, true_char = line.strip().split("\t")
            prediction = query_llm(prefix)

            if true_char in prediction:
                correct += 1

            total += 1

    runtime = time.time() - start_time
    accuracy = correct / total

    print(f"File: {file_path}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Runtime: {runtime:.2f} seconds")
    print("-" * 40)


if __name__ == "__main__":
    evaluate("opus_eval_200.txt")
    evaluate("additional_eval.txt")
    evaluate("example_eval.txt")