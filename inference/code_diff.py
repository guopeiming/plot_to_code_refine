import os
import json

CODE_DIR = "new_train_data"
OUTPUT_FILE = os.path.join(CODE_DIR, "results.jsonl")

def read_code(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def check_comment_line(line):
    stripped_line = line.strip()
    return stripped_line.startswith('#') or stripped_line.startswith('"""') or stripped_line.startswith("'''")

def common_prefix_length(a: str, b: str) -> int:
    match_len = 0
    for x, y in zip(a, b):
        if x == y:
            match_len += 1
        else:
            break
    return match_len

def count_brackets(line: str) -> tuple:
    return line.count('('), line.count(')')

def find_balanced_brackets(lines, start_index):
    if start_index < 0 or start_index >= len(lines):
        return start_index
    open_brackets = 0
    close_brackets = 0
    for i in range(start_index, len(lines)):
        line = lines[i]
        open_brackets += line.count('(')
        close_brackets += line.count(')')
        if open_brackets == close_brackets:
            return i
    return len(lines) - 1

def remove_comments(line):
    in_string = False
    string_char = None
    escape_next = False
    result = []
    for i, char in enumerate(line):
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            result.append(char)
            continue
        if char in ('"', "'") and not in_string:
            in_string = True
            string_char = char
            result.append(char)
        elif char == string_char and in_string:
            in_string = False
            string_char = None
            result.append(char)
        elif char == '#' and not in_string:
            break
        else:
            result.append(char)
    return ''.join(result).rstrip()

def only_comment_change(line1, line2):
    code1 = remove_comments(line1).strip()
    code2 = remove_comments(line2).strip()
    return code1 == code2

def process_deletions(deleted_list, modified_list, lines_a):
    filtered_deleted = deleted_list.copy()
    to_remove = []
    for del_line in deleted_list:
        prev_line = del_line - 1
        if prev_line > 0 and prev_line in modified_list:
            if prev_line - 1 < len(lines_a):
                prev_line_code = lines_a[prev_line - 1]
                open_brackets, close_brackets = count_brackets(prev_line_code)
                if open_brackets != close_brackets:
                    to_remove.append(del_line)
    for line in to_remove:
        if line in filtered_deleted:
            filtered_deleted.remove(line)
    return filtered_deleted

def code_diff_lines(code1, code2):
    lines1 = code1.splitlines()
    lines2 = code2.splitlines()
    n, m = len(lines1), len(lines2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if lines1[i - 1] == lines2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                delete_dist = dp[i - 1][j] + 1
                insert_dist = dp[i][j - 1] + 1
                replace_dist = dp[i - 1][j - 1] + (1 if common_prefix_length(lines1[i-1], lines2[j-1]) >= 6 else 999)
                dp[i][j] = min(delete_dist, insert_dist, replace_dist)
    i, j = n, m
    added, deleted, modified = [], [], []
    added_pairs, modified_pairs = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and lines1[i - 1] == lines2[j - 1]:
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            added.append(j)
            added_pairs.append((i, j))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deleted.append(i)
            i -= 1
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

def compare_codes(code_a, code_b):
    result = code_diff_lines(code_a, code_b)
    lines_a = code_a.splitlines()
    lines_b = code_b.splitlines()
    filtered_modified = []
    filtered_modified_pairs = []
    for i, (line_num, pair) in enumerate(zip(result["modified"], result["modified_pairs"])):
        if (line_num - 1 < len(lines_a) and pair[1] - 1 < len(lines_b) and not check_comment_line(lines_a[line_num - 1]) and not check_comment_line(lines_b[pair[1] - 1])):
            if not only_comment_change(lines_a[line_num - 1], lines_b[pair[1] - 1]):
                filtered_modified.append(line_num)
                filtered_modified_pairs.append(pair)
    filtered_deleted = []
    for line_num in result["deleted"]:
        if line_num - 1 < len(lines_a) and not check_comment_line(lines_a[line_num - 1]):
            filtered_deleted.append(line_num)
    filtered_deleted = process_deletions(filtered_deleted, filtered_modified, lines_a)
    source_additions = []
    for pair in result["added_pairs"]:
        insert_position = pair[0]
        source_additions.append(insert_position)
    source_additions = sorted(set(source_additions))
    added_pairs_to_keep = result["added_pairs"].copy() 
    additions_to_remove = set()
    for source_line in source_additions:
        if source_line in result["modified"]:
            target_lines = [pair[1] for pair in added_pairs_to_keep if pair[0] == source_line]
            if not target_lines:
                continue
            min_target_line = min(target_lines)
            if source_line - 1 < len(lines_a) and min_target_line - 1 < len(lines_b):
                line_a = lines_a[source_line - 1]
                line_b = lines_b[min_target_line - 2]
                left_paren_idx_a = line_a.find('(')
                left_paren_idx_b = line_b.find('(')
                if left_paren_idx_a != -1 and left_paren_idx_b != -1:
                    prefix_a = line_a[:left_paren_idx_a].strip()
                    prefix_b = line_b[:left_paren_idx_b].strip()
                    if prefix_a == prefix_b:
                        open_a, close_a = count_brackets(line_a)
                        open_b, close_b = count_brackets(line_b)
                        if (open_a != open_b or close_a != close_b) and open_b != close_b:
                            balanced_line = find_balanced_brackets(lines_b, min_target_line - 2)
                            pairs_to_remove = []
                            for pair in added_pairs_to_keep:
                                if min_target_line <= pair[1] <= balanced_line + 1:
                                    pairs_to_remove.append(pair)
                            for pair in pairs_to_remove:
                                added_pairs_to_keep.remove(pair)
                    else:
                        pass
        elif source_line - 1 in result["modified"]:
            target_lines = [pair[1] for pair in added_pairs_to_keep if pair[0] == source_line]
            if not target_lines:
                continue
            min_target_line = min(target_lines)
            if source_line - 2 < len(lines_a) and min_target_line - 1 < len(lines_b):
                line_a = lines_a[source_line - 2]
                line_b = lines_b[min_target_line - 2]
                left_paren_idx_a = line_a.find('(')
                left_paren_idx_b = line_b.find('(')
                if left_paren_idx_a != -1 and left_paren_idx_b != -1:
                    prefix_a = line_a[:left_paren_idx_a].strip()
                    prefix_b = line_b[:left_paren_idx_b].strip()
                    if prefix_a == prefix_b:
                        open_a, close_a = count_brackets(line_a)
                        open_b, close_b = count_brackets(line_b)
                        if open_a != close_a and open_b != close_b:
                            balanced_linea = find_balanced_brackets(lines_a, source_line - 2)
                            balanced_lineb = find_balanced_brackets(lines_b, min_target_line - 2)
                            pairs_to_remove = []
                            if balanced_linea + 1 == source_line:
                                for pair in added_pairs_to_keep:
                                    if min_target_line <= pair[1] <= balanced_lineb + 1:
                                        pairs_to_remove.append(pair)
                                for pair in pairs_to_remove:
                                    added_pairs_to_keep.remove(pair)    
    final_source_additions = []
    for pair in added_pairs_to_keep:
        source_line = pair[0]
        if source_line not in additions_to_remove:
            final_source_additions.append(source_line)
    final_source_additions = sorted(set(final_source_additions))
    print({
        "deletions": filtered_deleted,
        "modifications": filtered_modified,
        "additions": final_source_additions
    })
    return {
        "deletions": filtered_deleted,
        "modifications": filtered_modified,
        "additions": final_source_additions
    }

def get_modify_types(order, a, b):
    if a == 0 and b == 1:
        return [order[0]] if len(order) > 0 else []
    elif a == 1 and b == 2:
        return [order[1]] if len(order) > 1 else []
    elif a == 2 and b == 3:
        return [order[2]] if len(order) > 2 else []
    elif a == 0 and b == 2:
        return order[0:2] if len(order) > 1 else []
    elif a == 1 and b == 3:
        return order[1:3] if len(order) > 2 else []
    elif a == 0 and b == 3:
        return order[0:3] if len(order) > 2 else []
    return []

def main():
    order_file = os.path.join(CODE_DIR, "modification_orders.json")
    with open(order_file, 'r', encoding='utf-8') as f:
        orders_list = json.load(f)
    record_counter = 1
    pairs = [(0,1), (1,2), (2,3), (0,2), (1,3), (0,3)]
    orders = {}
    for order in orders_list:
        if 'id' in order:
            orders[str(order['id'])] = order
    print(f"{len(orders)}个有效记录")
        
    for item_id, order_info in orders.items():
        order = order_info.get('order', [])
        print(f"处理第{record_counter}个样本 ID:{item_id}")
        for a, b in pairs:
            code_a = os.path.join(CODE_DIR, f"code_{item_id}_{a}.py")
            code_b = os.path.join(CODE_DIR, f"code_{item_id}_{b}.py")
            image_a = os.path.join(CODE_DIR, f"image_{item_id}_{a}.png")
            image_b = os.path.join(CODE_DIR, f"image_{item_id}_{b}.png")
            if not all(os.path.exists(f) for f in [code_a, code_b, image_a, image_b]):
                continue
            mod_types = get_modify_types(order, a, b)
            if not mod_types:
                continue
            code_a_text = read_code(code_a)
            code_b_text = read_code(code_b)
            if not code_a_text or not code_b_text:
                continue
            changes = compare_codes(code_a_text, code_b_text)
            record = {
                "id": str(record_counter).zfill(4),
                "cid": str(item_id),
                "prev_code_file": f"code_{item_id}_{a}.py",
                "prev_image_file": f"image_{item_id}_{a}.png",
                "post_code_file": f"code_{item_id}_{b}.py",
                "post_image_file": f"image_{item_id}_{b}.png",
                "modification_types": mod_types,
                "changes": changes
            }
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')  
            record_counter += 1

if __name__ == "__main__":
    main()