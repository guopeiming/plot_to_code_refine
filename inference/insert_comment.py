import os
import json
from collections import defaultdict

CODE_DIR = "new_train_data"
OUTPUT_FILE = os.path.join(CODE_DIR, "reverse_results.jsonl")

def merge_lines(nums):
    if not nums: return []
    nums = sorted(nums)
    result = []
    start = end = nums[0]
    for n in nums[1:]:
        if n == end + 1:
            end = n
        else:
            result.append((start, end))
            start = end = n
    result.append((start, end))
    return result

def annotate_code_string(code, changes):
    lines = code.splitlines(True)
    del_ranges = merge_lines(changes.get("deletions", []))
    mod_ranges = merge_lines(changes.get("modifications", []))
    add_ranges = merge_lines(changes.get("additions", []))
    inserts = []
    for start, end in del_ranges:
        count = end - start + 1
        comment = f"# TODO: Delete the following {count} lines\n" if count > 1 else "# TODO: Delete the following 1 lines\n"
        inserts.append((start-1, comment))
    for start, end in mod_ranges:
        count = end - start + 1
        comment = f"# TODO: Modify the following {count} lines\n" if count > 1 else "# TODO: Modify the following 1 lines\n"
        inserts.append((start-1, comment))
    for line_num in sorted(changes.get("additions", [])):
        comment = "# TODO: Add extra code in the next line\n"
        inserts.append((line_num, comment))
    pos_dict = defaultdict(list)
    for pos, comment in inserts:
        pos_dict[pos].append(comment.strip())
    merged_inserts = []
    for pos, comments in pos_dict.items():
        if len(comments) == 1:
            merged_comment = comments[0] + "\n"
        else:
            cleaned_comments = [c.replace("# TODO: ", "") for c in comments]
            merged_comment = f"# TODO: {'; '.join(cleaned_comments)}\n"
        merged_inserts.append((pos, merged_comment))
    merged_inserts.sort(key=lambda x: x[0], reverse=True)
    for pos, comment in merged_inserts:
        if 0 <= pos < len(lines):
            lines.insert(pos, comment)
    return "".join(lines)

def main():
    enhanced_records = []
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            prev_path = os.path.join(CODE_DIR, record["prev_code_file"])
            if os.path.exists(prev_path):
                with open(prev_path, 'r', encoding='utf-8') as pf:
                    prev_code = pf.read()
                record["prev_code_with_comments"] = annotate_code_string(prev_code, record["changes"])
            enhanced_records.append(record)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in enhanced_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
if __name__ == "__main__":
    main()