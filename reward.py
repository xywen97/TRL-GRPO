# 原始repo
# https://hugging-face.cn/learn/cookbook/fine_tuning_llm_grpo_trl

import re
from math_verify import LatexExtractionConfig, parse, verify

def format_reward(completions, **kwargs):
    """
    <think> 和 <answer> 必须成对出现，否则reward=0。
    若成对出现，再判断<answer>内是否有\boxed{}，有则reward=1.0，否则0.5。
    """
    pattern_think = r"<think>.*?</think>"
    pattern_answer = r"<answer>(.*?)</answer>"
    pattern_boxed = r"\\boxed{.*?}"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        has_think = bool(re.search(pattern_think, content, re.DOTALL))
        answer_match = re.search(pattern_answer, content, re.DOTALL)
        has_answer = bool(answer_match)
        if not (has_think and has_answer):
            rewards.append(0.0)
        else:
            answer_content = answer_match.group(1)
            has_boxed = bool(re.search(pattern_boxed, answer_content, re.DOTALL))
            if has_boxed:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
    return rewards

def extract_boxed(text):
    # 匹配 \boxed{...}，支持多行和空格
    match = re.search(r"\\boxed{(.*?)}", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def accuracy_reward(completions, **kwargs):
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        pred = extract_boxed(content)
        gold = extract_boxed(solution)
        if gold is None:
            rewards.append(1.0)
        elif pred is not None and pred == gold:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

if __name__ == "__main__":
    # 测试extract_boxed
    test_cases = [
        r"<answer>\boxed{42}</answer>",
        r"<answer>答案是\boxed{123}</answer>",
        r"<answer>\boxed{a+b}</answer>",
        r"<answer>没有boxed</answer>",
        r"<answer>\boxed{  3.14  }</answer>",
        r"<answer>\boxed{多行\n内容}</answer>",
        r"<answer>\boxed{}</answer>",
        r"<answer>\boxed{  }</answer>",
        r"<answer>\boxed{特殊符号!@#}</answer>",
        r"<answer>To solve this problem, we need to determine the values of \\(a\\) and \\(b\\) based on the given conditions and then calculate \\(a - b\\). 1. **Determine \\(a\\):** - \\(a\\) is the smallest non-negative number, which is \\(0\\). 2. **Determine \\(b\\):** - The opposite of \\(b\\) is the largest negative integer, which is \\(-1\\). Therefore, \\(b = 1\\). 3. **Calculate \\(a - b\\):** - With \\(a = 0\\) and \\(b = 1\\), we compute \\(a - b\\). Let's write Python code with SymPy to verify these steps. ```python import sympy as sp # Define the variables a = 0 # smallest non-negative number b = 1  # opposite of the largest negative integer # Calculate a - b result = a - b # Print the result print(result) ``` ```output -1 ``` The result of \\(a - b\\) when \\(a\\) is the smallest non-negative number and the opposite of \\(b\\) is the largest negative integer is \\(\\boxed{-1}\\).</answer>",
    ]
    print("==== extract_boxed 测试 ====")
    for i, case in enumerate(test_cases):
        result = extract_boxed(case)
        print(f"Case {i+1}: {case}\n  提取结果: {result}\n")

    # 测试 format_reward
    print("==== format_reward 测试 ====")
    format_cases = [
        # 三者都有
        r"<think>思考过程</think><answer>\boxed{42}</answer>",
        # 只有<think>和<answer>
        r"<think>思考过程</think><answer>答案是42</answer>",
        # 只有<think>和<answer>，且<answer>内有\boxed{}
        r"<think>思考过程</think><answer>答案是42 \boxed{42}</answer>",
        # 只有<answer>和\boxed
        r"<answer>\boxed{42}</answer>",
        # 只有<think>和\boxed
        r"<think>思考过程</think>\boxed{42}",
        # 只有<answer>
        r"<answer>答案</answer>",
        # 只有<think>
        r"<think>思考过程</think>",
        # 都没有
        r"无格式内容",
    ]
    format_completions = [[{"content": c}] for c in format_cases]
    format_rewards = format_reward(format_completions)
    for i, (c, r) in enumerate(zip(format_cases, format_rewards)):
        print(f"Case {i+1}: {c}\n  format_reward: {r}\n")

    # 测试 accuracy_reward
    print("==== accuracy_reward 测试 ====")
    acc_completions = [[{"content": r"<answer>\boxed{42}</answer>"}], [{"content": r"<answer>\boxed{43}</answer>"}]]
    acc_solutions = [r"\boxed{42}", r"\boxed{42}"]
    acc_rewards = accuracy_reward(acc_completions, solution=acc_solutions)
    for i, (c, s, r) in enumerate(zip(acc_completions, acc_solutions, acc_rewards)):
        print(f"Case {i+1}: completion={c[0]['content']} | solution={s}\n  accuracy_reward: {r}\n")



