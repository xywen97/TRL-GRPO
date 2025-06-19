import re
from math_verify import LatexExtractionConfig, parse, verify

def format_reward(completions, **kwargs):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def accuracy_reward(completions, **kwargs):
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]

    # print("*****************", "This is completions", "*****************")
    # print(completions)
    # print("*****************", "This is solutions", "*****************")
    # print(solutions)
    # print()
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        # print("*****************", "This is gold_parsed", "*****************")
        # print(gold_parsed)
        # print()
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards 


# [[{'role': 'assistant', 'content': 'To find the sum \\(m + n\\) where \\(m\\) and \\(n\\) are natural numbers and \\(\\lceil x \\rceil\\) represents t
# he smallest integer greater than or equal to \\(x\\), we aim to find these values based on the relationships between \\(a\\) and \\(b\\), the number'}
# ], [{'role': 'assistant', 'content': 'Given the information provided:\n\n1. The number \\( a \\) is \\( n \\) times greater than the number \\( b \\).
# \n2. The sum of the numbers \\( a \\) and \\( b \\) is \\( m \\) times greater than their difference.\n3. These relationships imply that:\n\n   - If'}
# ], [{'role': 'assistant', 'content': "Given:\n\n1. \\( a \\) is \\( n \\) times greater than \\( b \\).\n2. The sum of \\( a \\) and \\( b \\) is \\(
# m \\) times greater than their difference.\n\nLet's denote \\( a = kn \\) where \\( k \\) is a common multiplier."}], [{'role': 'assistant', 'content'
# : "Given the situation, let's represent:\n\n1. \\( a = kn \\)\n2. \\( b = kn-1 \\)\n3. The sum of \\( a \\) and \\( b \\) is \\( m \\) times greater t
# han their difference:\n\n\\[ a + b = 2m(kn-1)\\"}], [{'role': 'assistant', 'content': "To calculate the total number of servings that can be made with
#  the given ingredients in the restaurant's warehouse, first, let's determine the number of servings they can prepare based on the proportion of cucumb
# ers: tomatoes:bananas.\n\n- Tomatoes: Total # of bunches: cucumbers\n- Bananas"}], [{'role': 'assistant', 'content': 'To determine how many servings o
# f salad can be made, we can follow these steps:\n\n1. Calculate the total amount of food required for one serving (2 cucumbers, 2 tomatoes, 75 grams o
# f brynza, and 1 pepper)\n2. Use that total to determine how many servings can'}], [{'role': 'assistant', 'content': "First, let's determine how many s
# ervings each ingredient constitutes when making one salad containing these components.\n\n1. Cucumbers:\n   - COURSE = 2 ♦ MEANS 2 x 1 cucumbers = 2 c
# ucumbers\n   - REQUIRED NUMERIC_VALUE = 2\n   -"}], [{'role': 'assistant', 'content': 'To calculate the total number of servings that can be made from
#  the ingredients one can gather in just 60 peppers, 4.2 kg of brynza, 116 tomatoes, and 117 cucumbers, we simply need to divide the total weight of th
# ese ingredients by the serving size'}]]


# [[{'role': 'assistant', 'content': 'To draw a sample of 40 students, we need to calculate the total number of students that have blood type O. If ther
# e are 200 students with blood type O, and we need to get a total of 40 students, then the number of students with O blood type must be:\n\n\\['}], [{'
# role': 'assistant', 'content': 'To determine how many students with blood type O should be drawn to study the relationship between blood type and colo
# r blindness, follow these steps:\n\n1. **Total Number of Students**: First, we need to calculate the total number of students for the grade-levels (fi
# rst and second grade).\n\n2. **Total Number of'}], [{'role': 'assistant', 'content': "To calculate the number of students with blood type O needed for
#  the study of the relationship with color blindness, we first need to determine which part of the population there are between blood type O and O and
# ones with blood type O and blood type B or O and blood type A.\n\n1. **Let's start with '"}], [{'role': 'assistant', 'content': "To solve this problem
# , let's break down the information given:\n\n1. Total number of students = 500\n2. Number of students with blood type O = 200\n3. Number of students w
# ith blood type A = 125\n4. Number of students with blood type B"}], [{'role': 'assistant', 'content': 'In a stratified sampling method, the population
#  is divided into strata based on the type of blood an individual belongs to. In this case, the population includes all 500 students, of which 200 are
# students with blood type O, 125 are with blood type A, 1'}], [{'role': 'assistant', 'content': 'To determine the number of students with blood type O
# to be drawn in stratified sampling, assign a percentage to indicate the number of students each group represents. We can start by listing the total nu
# mber of students in different groups:\n\n- **Group A:** **A:** 200\n- **Group B:**'}], [{'role': 'assistant', 'content': 'If a stratified sampling met
# hod is used to draw a sample of 40 students, to determine the number of students with blood type O in the sample, you can follow these steps:\n\n1. Ca
# lculate the size of the total class to be sampled, which is \\(40 \\times \\text{number of O'}], [{'role': 'assistant', 'content': 'To determine how m
# any students with blood type O should be drawn, we first need to calculate the sample size needed for a stratified sampling with a stratified sampling
#  object, which is the design matrix representing a sample.\n\nThe stratified sampling determines the starting population size for sampling of a specif
# ic group. The formula for this is'}]]
