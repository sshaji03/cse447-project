import random

def create_prefix_dataset(input_file, output_file, n=200):
    lines = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) > 5:
                lines.append(line)

    sample = random.sample(lines, min(n, len(lines)))

    with open(output_file, "w", encoding="utf-8") as f:
        for line in sample:
            prefix = line[:-1]
            true_char = line[-1]
            f.write(prefix + "\t" + true_char + "\n")

create_prefix_dataset(
    "opus-100-corpus/v1.0/supervised/en-fr/opus.en-fr-dev.en",
    "opus_eval_200.txt",
    200
)

create_prefix_dataset("additional_full_text.txt", "additional_eval.txt", 999)
create_prefix_dataset("example_full_text.txt", "example_eval.txt", 999)