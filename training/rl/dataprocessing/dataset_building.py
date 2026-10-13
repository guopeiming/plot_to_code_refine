import os
import re
import json
import random
import argparse
from typing import List, Dict
from PIL import Image
from datasets import Dataset


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


def get_data_list(data: List[Dict]) -> List[Dict]:
    data_list = []
    for example in data:
        origin_code = re.search(r"Code Snippet:\n```python\n([\s\S]*?)```", example["instruction"]).group(1).strip()
        output_code = re.search(r"```python\n([\s\S]*?)```", example["output"]).group(1).strip()
        output_types = re.search(r"^[^\n]*", example["output"]).group(0).strip()

        diff_type_list = []
        if "<data>" in output_types:
            diff_type_list.append("<data>")
        if "<global_style>" in output_types:
            diff_type_list.append("<global_style>")
        if "<local_style>" in output_types:
            diff_type_list.append("<local_style>")

        code_diffs = code_diff_lines(origin_code, output_code)
        assert len(code_diffs["modified"]) == len(code_diffs["deleted"]) == 0, "code diffs error"
        output_code_line = output_code.split("\n")
        triple_list = []
        for idx, (code_line_num, _) in enumerate(code_diffs["added_pairs"]):
            comment_code = output_code_line[code_line_num+idx]
            assert comment_code.strip().startswith("# TODO: "), f"code error: {comment_code}"
            comment_code_re_res = re.search(r"(Modify|Delete|Add)[^\d]*(\d+)?", comment_code)
            assert comment_code_re_res is not None, "regular expresssion match error"
            triple_list.append([
                str(code_line_num),
                comment_code_re_res.group(1),
                comment_code_re_res.group(2) if comment_code_re_res.group(2) is not None else "0"
            ])

        data_list.append({
            "data_source": "image_diff",
            "prompt": [{"role": "user", "content": example["instruction"]}],
            "images": [
                example["images"][0],
                example["images"][1]
            ],
            "reward_model": {"style": "rule", "ground_truth": {"type_list": diff_type_list, "comment_list": triple_list}},
            "extra_info": {"origin_code": origin_code}
        })

    return data_list


def dataset_building(args: argparse.Namespace) -> None:
    with open(args.dataset_path, "r") as file:
        traindata = json.load(file)
    random.shuffle(traindata)

    dataset = get_data_list(traindata[:5000])
    dataset = Dataset.from_list(dataset)
    dataset.to_parquet(os.path.join(args.verl_dataset_dir, "train.parquet"))

    dataset = get_data_list(traindata[5000:])
    dataset = Dataset.from_list(dataset)
    dataset.to_parquet(os.path.join(args.verl_dataset_dir, "test.parquet"))


# def pil_transform(example: Dict) -> Dict:
#     example["images"] = [
#         Image.open(example["images"][0]),
#         Image.open(example["images"][1])
#     ]
#     return example


# def test():
#     from datasets import load_dataset
#     dataset = load_dataset("parquet", data_files="./data/image_diff/test.parquet")["train"]
#     dataset = dataset.with_transform(pil_transform)
#     print(dataset[0])
#     print(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Building")
    parser.add_argument("--dataset_path", type=str, default="/data8/huangjinpeng/Chart2CodeFixer/Fixer/train_data/raw_data.json")
    parser.add_argument("--verl_dataset_dir", type=str, default="./data/image_diff/")
    args = parser.parse_args()
    dataset_building(args)
    # test()