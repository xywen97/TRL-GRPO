from datasets import load_dataset

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, and the final answer is enclosed within \\boxed\{\} tags, i.e., "
    "<think> think process here </think><answer> solution here \\boxed\{your final answer here\} </answer>"
)

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

def load_and_process_data(dataset_id):
    train_dataset, test_dataset = load_dataset(dataset_id, split=["train[:5%]", "test[:5%]"])
    train_dataset = train_dataset.map(make_conversation)
    test_dataset = test_dataset.map(make_conversation)
    train_dataset = train_dataset.remove_columns(["messages", "problem"])
    return train_dataset, test_dataset 