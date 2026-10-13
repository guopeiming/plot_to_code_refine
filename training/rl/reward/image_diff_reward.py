import re
import logging
from typing import List, Union, Dict

from image_diff_verl_training.config.constants import CODE_REEXP, COMMENT_REEXP

logger = logging.getLogger(__name__)


def common_prefix_length(a: str, b: str) -> int:
    match_len = 0
    for x, y in zip(a, b):
        if x == y:
            match_len += 1
        else:
            break
    return match_len


def code_diff_lines(code1: str, code2: str):
    lines1 = code1.splitlines()
    lines2 = code2.splitlines()
    n, m = len(lines1), len(lines2)

    # 初始化 DP 表
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i  # 删除
    for j in range(m + 1):
        dp[0][j] = j  # 插入

    # 填充 DP 表
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if lines1[i - 1] == lines2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 不需要修改
            else:
                delete_dist = dp[i - 1][j] + 1
                insert_dist = dp[i][j - 1] + 1
                replace_dist = dp[i - 1][j - 1] + (1 if common_prefix_length(lines1[i-1], lines2[j-1]) >= 6 else 999)
                dp[i][j] = min(delete_dist, insert_dist, replace_dist)

    # 回溯
    i, j = n, m
    added, deleted, modified = [], [], []
    added_pairs, modified_pairs = [], []

    while i > 0 or j > 0:
        # 相同行
        if i > 0 and j > 0 and lines1[i - 1] == lines2[j - 1]:
            i -= 1
            j -= 1

        # 新增
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            added.append(j)
            added_pairs.append((i, j))
            j -= 1

        # 删除
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deleted.append(i)
            i -= 1

        # 修改
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            modified.append(i)
            modified_pairs.append((i, j))
            i -= 1
            j -= 1

        else:
            raise ValueError("回溯过程中出现未预料的情况")

    return {
        "added": sorted(added),
        "deleted": sorted(deleted),
        "modified": sorted(modified),
        "added_pairs": sorted(added_pairs),
        "modified_pairs": sorted(modified_pairs),
    }


def compute_comment_score(solution_str: str, ground_truth: List[List[str]], extra_info=None) -> float:
    output_code = re.search(CODE_REEXP, solution_str)
    if output_code is None:
        return 0.
    else:
        output_code = output_code.group(1).strip()

    code_diffs = code_diff_lines(extra_info["origin_code"], output_code)
    if len(code_diffs["modified"]) != 0 or len(code_diffs["deleted"]) != 0:
        return 0.

    reward = 0.1
    output_code_line = output_code.split("\n")
    triple_list = []
    for idx, (code_line_num, _) in enumerate(code_diffs["added_pairs"]):
        comment_code = output_code_line[code_line_num+idx]
        comment_code_re_res = re.search(COMMENT_REEXP, comment_code)
        if comment_code_re_res is None:
            triple_list.append([
                str(-len(triple_list)-1),
                str(-len(triple_list)-1),
                str(-len(triple_list)-1)
            ])
        else:
            triple_list.append([
                str(code_line_num),
                comment_code_re_res.group(1),
                comment_code_re_res.group(2) if comment_code_re_res.group(2) is not None else "0"
            ])

    set_a = {item[0] for item in triple_list}
    set_b = {item[0] for item in ground_truth}
    intersec_set = set_a & set_b
    reward += len(intersec_set) / (len(set_a) + len(set_b) - len(intersec_set))

    set_a = {(item[0], item[1]) for item in triple_list}
    set_b = {(item[0], item[1]) for item in ground_truth}
    intersec_set = set_a & set_b
    reward += len(intersec_set) / (len(set_a) + len(set_b) - len(intersec_set))

    set_a = {(item[0], item[1], item[2]) for item in triple_list}
    set_b = {(item[0], item[1], item[2]) for item in ground_truth}
    intersec_set = set_a & set_b
    reward += len(intersec_set) / (len(set_a) + len(set_b) - len(intersec_set))

    return reward


def compute_type_score(solution_str: str, ground_truth: List[str], extra_info=None) -> float:
    output_types = re.search(r"^[^\n]*", solution_str)
    if output_types is None:
        return 0.0
    else:
        output_types = output_types.group(0).strip()
    if "python" in output_types:
        return 0.0

    reward = 0.1
    set_a = set()
    if "<data>" in output_types:
        set_a.add("<data>")
    if "<global_style>" in output_types:
        set_a.add("<global_style>")
    if "<local_style>" in output_types:
        set_a.add("<local_style>")
    set_b = set(ground_truth)

    intersec_set = set_a & set_b
    reward += len(intersec_set) / (len(set_a) + len(set_b) - len(intersec_set))
    return reward


def compute_score(data_source: str, solution_str: str, ground_truth: List[Union[str, List[str]]], extra_info=None) -> Dict[str, float]:
    reward_dict = dict()
    reward_dict["type_reward"] = compute_type_score(solution_str, ground_truth["type_list"], extra_info)
    reward_dict["comment_reward"] = compute_comment_score(solution_str, ground_truth["comment_list"], extra_info)
    reward_dict["score"] = reward_dict["type_reward"] + reward_dict["comment_reward"]
    # reward = 0.0
    # reward += compute_type_score(solution_str, ground_truth["type_list"], extra_info)
    # reward += compute_comment_score(solution_str, ground_truth["comment_list"], extra_info)
    return reward_dict
