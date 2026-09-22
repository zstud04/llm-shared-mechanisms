def is_correct_match(output: str, correct_ans: str, incorrect_ans: str) -> int:
    out = output.lower()
    c = correct_ans.lower()
    i = incorrect_ans.lower()

    idx_c = out.find(c)
    idx_i = out.find(i)

    if idx_c == -1 and idx_i == -1:
        return -1
    if idx_c == -1:
        return 0
    if idx_i == -1:
        return 1
    return 1 if idx_c < idx_i else 0

