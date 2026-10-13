import os
import json
import re
import base64
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import io
from contextlib import redirect_stdout, redirect_stderr
from vllm import LLM, SamplingParams
import shutil
from PIL import Image

PREVIOUS_RESULT_FILE = "chartmaster/generated_code.jsonl"
GENERATION_MODEL_PATH = "qwen3_vl_4b_sft"
CORRECTION_MODEL_PATH = "qwen3_vl_4b_sft"
NEW_DATASET_DIR = "chart_test_results/chartmimic"
DIMENSIONS_FILE = os.path.join(NEW_DATASET_DIR, "filtered_dimentions_info.jsonl")
OUTPUT_DIR = "chart_test_results/qwen3_4b_sft_0415"
MAX_ITERATIONS = 6
GENERATED_CODE_FILE = f"{OUTPUT_DIR}/generated_code.jsonl"
BATCH_SIZE = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["NCCL_P2P_DISABLE"] = "1"
generation_llm = LLM(
    model=GENERATION_MODEL_PATH,
    max_model_len=32768,
    tensor_parallel_size=8,
    gpu_memory_utilization=0.9,
    trust_remote_code=True
)
correction_llm = generation_llm
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["NCCL_P2P_DISABLE"] = "1"
correction_llm = LLM(
    model=CORRECTION_MODEL_PATH,
    max_model_len=32768,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9,
    trust_remote_code=True
)
generation_sampling_params = SamplingParams(
    max_tokens=32768
)
correction_sampling_params = SamplingParams(
    max_tokens=32768
)

def encode_image_to_base64(img_path):
    with open(img_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def process_instruction_with_images(instruction, img_paths, img_dir):
    parts = instruction.split('<image>')
    content = []
    for i, part in enumerate(parts):
        if part.strip():
            content.append({"type": "text", "text": part.strip()})
        if i < len(img_paths):
            img_path = os.path.join(img_dir, img_paths[i])
            if os.path.exists(img_path):
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(img_path)}"}
                })
    return content

def batch_call_vllm_model(instructions, images_list, model_type, img_dir):
    if model_type == "generation":
        llm = generation_llm
        sampling_params = generation_sampling_params
    else:
        llm = correction_llm
        sampling_params = correction_sampling_params
    all_messages = []
    for instruction, images in zip(instructions, images_list):
        content = process_instruction_with_images(instruction, images, img_dir)
        all_messages.append([{"role": "user", "content": content}])
    outputs = llm.chat(all_messages, sampling_params=sampling_params)
    results = []
    for output in outputs:
        if output.outputs and len(output.outputs) > 0:
            results.append(output.outputs[0].text)
    return results

def call_vllm_model(instruction, images, model_type, img_dir):
    if model_type == "generation":
        llm = generation_llm
        sampling_params = generation_sampling_params
    else:
        llm = correction_llm
        sampling_params = correction_sampling_params
    content = process_instruction_with_images(instruction, images, img_dir)
    messages = [{"role": "user", "content": content}]
    outputs = llm.chat(messages, sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    return generated_text

def execute_code(code_str, save_path):
    try:
        plt.close('all')
        matplotlib.rcdefaults()
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code_str)
            fig = plt.gcf()
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path)
            plt.close('all')
        stderr_output = stderr_capture.getvalue()
        if stderr_output:
            return False, stderr_output, None
        return True, "", fig
    except Exception as e:
        error_msg = f"执行错误: {str(e)}\n{traceback.format_exc()}"
        plt.close('all')
        return False, error_msg, None

def extract_results(response):
    if not response:
        return set(), ""
    error_pattern = r'<([^>]+)>'
    error_matches = re.findall(error_pattern, response)
    error_types = set()
    for match in error_matches:
        if match.strip():
            error_types.add(match.strip())
    code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', response, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ""
    return error_types, code

def batch_execute_codes(code_list, save_paths):
    results = []
    for code_str, save_path in zip(code_list, save_paths):
        success, error_msg, _ = execute_code(code_str, save_path)
        results.append((success, error_msg))
    return results

def load_image_files_from_jsonl(jsonl_path, dataset_dir):
    image_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                idx = data.get("idx", "")
                file_name = data.get("file", "")
                code = data.get("code", "")
                width = data.get("width", 0)
                height = data.get("height", 0)
                if file_name.endswith('.py'):
                    base_name = file_name[:-3]
                    image_file = f"{base_name}.png"
                else:
                    image_file = f"{idx}.png"
                image_path = os.path.join(dataset_dir, image_file)
                if os.path.exists(image_path):
                    image_data.append({
                        "idx": idx,
                        "file_name": file_name,
                        "image_file": image_file,
                        "image_path": image_path,
                        "ground_truth_code": code,
                        "width": width,
                        "height": height
                    })
                else:
                    print(f"警告: 图片文件不存在: {image_path}")
    image_data.sort(key=lambda x: x["idx"])
    return image_data

def remove_todo_comments(code_str):
    if not code_str:
        return code_str
    lines = code_str.split('\n')
    cleaned_lines = []
    for line in lines:
        if not re.match(r'^\s*#\s*TODO\b', line):
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def load_previous_results(result_file):
    previous_results = {}
    if not os.path.exists(result_file):
        print(f"警告: 之前的结果文件不存在: {result_file}")
        return previous_results
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    sample_id = data.get("sample_id")
                    iteration = data.get("iteration", 0)
                    code = data.get("code", "")
                    if iteration == 0 and code:
                        previous_results[sample_id] = code
    except Exception as e:
        print(f"加载之前的结果文件失败: {e}")
    return previous_results

def extract_code_modifications(response):
    if not response:
        return []
    modifications = []
    delimiter_pattern = r'---OLD---\s*(.*?)\s*---NEW---\s*(.*?)(?=---OLD---|$)'
    matches = re.findall(delimiter_pattern, response, re.DOTALL)
    for before, after in matches:
        if before.strip() and after.strip():
            modifications.append({
                'before': before.strip(),
                'after': after.strip()
            })
    if modifications:
        return modifications

def apply_modifications_to_code(original_code, modifications):
    if not modifications:
        return original_code
    current_code = original_code
    applied_mods = []
    failed_mods = []
    for i, mod in enumerate(modifications):
        before = mod['before']
        after = mod['after']
        if before in current_code:
            current_code = current_code.replace(before, after, 1)
            applied_mods.append(i)
        else:
            lines_before = [line.strip() for line in before.split('\n') if line.strip()]
            lines_current = [line.strip() for line in current_code.split('\n') if line.strip()]
            found = False
            for j in range(len(lines_current) - len(lines_before) + 1):
                if lines_current[j:j+len(lines_before)] == lines_before:
                    original_lines = current_code.split('\n')
                    failed_mods.append({
                        'index': i,
                        'reason': 'Matched after stripping whitespace, but not exactly'
                    })
                    found = True
                    break
            if not found:
                failed_mods.append({
                    'index': i,
                    'reason': 'Not found in code'
                })
    if applied_mods or failed_mods:
        print(f"  Applied {len(applied_mods)} out of {len(modifications)} modifications.")
        if failed_mods:
            for fm in failed_mods[:3]:
                print(f"    Modification {fm['index']}: {fm['reason']}")
            if len(failed_mods) > 3:
                print(f"    ... and {len(failed_mods)-3} more failures")
    return current_code

def build_modification_prompt(current_code, error_types_section):
    prompt = f"""You are a helpful assistant who can revise Python matplotlib code to reproduce the reference chart image.

I will provide:
1. A Python code snippet that generates a chart, but the output does not match the reference chart image.
2. A list of difference types describing how the generated chart differs from the reference image.

Your task is:
1. Carefully analyze and infer the visual differences between the reference chart image and my code snippet.
2. Identify specific code segments that need to be modified.
3. Return only the code segments that need to be modified, not the complete code.

IMPORTANT: Return the modifications in the following format:
---OLD---
[original code segment 1]
---NEW---
[modified code segment 1]
---OLD---
[original code segment 2]
---NEW---
[modified code segment 2]

You can have multiple OLD/NEW pairs if multiple segments need to be modified.

Difference types:
{error_types_section}

Code snippet:
```python
{current_code}
```

Reference image:
<image>

Generated image:
<image>

Now, analyze the differences and provide ONLY the code segments that need to be modified:"""
    return prompt

def plot_test():
    previous_results = load_previous_results(PREVIOUS_RESULT_FILE)
    image_data = load_image_files_from_jsonl(DIMENSIONS_FILE, NEW_DATASET_DIR)
    all_results = []
    with open(GENERATED_CODE_FILE, 'w', encoding='utf-8') as jsonl_file:
        for batch_start in tqdm(range(0, len(image_data), BATCH_SIZE), desc="处理批次"):
            batch_end = min(batch_start + BATCH_SIZE, len(image_data))
            batch_samples = image_data[batch_start:batch_end]
            batch_data = []
            for idx, sample in enumerate(batch_samples, start=batch_start):
                ground_truth_path = sample["image_path"]
                ground_truth_copy_filename = f"{sample['idx']}.png"
                ground_truth_copy_path = os.path.join(OUTPUT_DIR, ground_truth_copy_filename)
                if not os.path.exists(ground_truth_copy_path):
                    shutil.copy2(ground_truth_path, ground_truth_copy_path)
                batch_data.append({
                    "idx": sample["idx"],
                    "image_file": sample["image_file"],
                    "image_path": sample["image_path"],
                    "ground_truth_path": ground_truth_path,
                    "ground_truth_code": sample["ground_truth_code"],
                    "width": sample["width"],
                    "height": sample["height"],
                    "current_code": None
                })
            for iteration in range(MAX_ITERATIONS):
                generation_prompts = []
                generation_images = []
                generation_indices = []
                for data in batch_data:
                    idx = data["idx"]
                    image_file = data["image_file"]
                    current_code = data["current_code"]
                    if iteration == 0:
                        if idx in previous_results:
                            previous_code = previous_results[idx]
                            data["generated_code"] = previous_code
                            data["current_code"] = previous_code
                            cleaned_code = remove_todo_comments(previous_code)
                            data["cleaned_code"] = cleaned_code
                            continue
                        else:
                            width = data["width"]
                            height = data["height"]
                            gen_prompt = f"You are an expert Python developer who specializes in writing matplotlib code based on a given picture. I found a very nice picture in a STEM paper, but there is no corresponding source code available. I need your help to generate the Python code that can reproduce the picture based on the picture I provide.\nNote that it is necessary to use figsize=({width}, {height}) to set the image size to match the original size.\nNow, please give me the matplotlib code that reproduces the picture below.\nThe generated code should be surrounded by ```python and ```\n" + "<image>"
                            generation_prompts.append(gen_prompt)
                            generation_images.append([image_file])
                            generation_indices.append(idx)
                    elif data.get("success", False):
                        error_type_explanations = []
                        correction_error_types = data.get("correction_error_types", [])
                        if "data" in correction_error_types:
                            error_type_explanations.append("data: number and values of shapes and data points")
                        if "local_style" in correction_error_types:
                            error_type_explanations.append("local_style: styles of specific graphic shapes and elements")
                        if "global_style" in correction_error_types:
                            error_type_explanations.append("global_style: styles of global elements like Axis, Title, Annotations, Grid, Extra text")
                        error_types_section = ""
                        if error_type_explanations:
                            error_types_section = f"{', '.join(error_type_explanations)}\n"
                        last_generated_image = f"generated_image_{idx}_{iteration-1}.png"
                        images_to_send = [image_file, last_generated_image]
                        if iteration > 0 and data.get("success", False):
                            correction_prompt = build_modification_prompt(current_code, error_types_section)
                        else:
                            correction_prompt = f"""You are a helpful assistant who can revise Python matplotlib code to reproduce the reference chart image.

I will provide:
1. A Python code snippet that generates a chart, but the output does not match the reference chart image. The code contains # TODO comments indicating the specific locations that require modification.
2. A list of difference types describing how the generated chart differs from the reference image.

Your task is:
1. Carefully analyze and infer the visual differences between the reference chart image and my code snippet.
2. Refer to the comments and modify the code to reproduce the reference image.
3. Return the modified complete code surrounded by ```python and ```.

Code snippet:
python
{current_code}
Difference types:
{error_types_section}

Reference image:
<image>

Generated image:
<image>
"""
                        generation_prompts.append(correction_prompt)
                        generation_images.append(images_to_send)
                        generation_indices.append(idx)
                    else:
                        correction_prompt = f"""You are a helpful assistant that can generate Python code using matplotlib.
Please modify the following code based on the error message to create a chart that looks like the given chart.
Code:{data["cleaned_code"]}
Error Message:{data.get("error_msg", "")}
The generated code should be surrounded by ```python and ```\n<image>"""
                        generation_prompts.append(correction_prompt)
                        generation_images.append([image_file])
                        generation_indices.append(idx)
                
                if generation_prompts:
                    generation_responses = batch_call_vllm_model(generation_prompts, generation_images, "generation", OUTPUT_DIR)
                    for response_idx, (response, data_idx) in enumerate(zip(generation_responses, generation_indices)):
                        for data in batch_data:
                            if data["idx"] == data_idx:
                                if response:
                                    if iteration > 0 and data.get("success", False):
                                        modifications = extract_code_modifications(response)
                                        if modifications:
                                            print(f"  样本 {data_idx} 第{iteration}次迭代: 提取到 {len(modifications)} 个修改片段")
                                            original_code = data["current_code"]
                                            modified_code = apply_modifications_to_code(original_code, modifications)
                                            data["generated_code"] = modified_code
                                            data["current_code"] = modified_code
                                            cleaned_code = remove_todo_comments(modified_code)
                                            data["cleaned_code"] = cleaned_code
                                            data["modifications"] = modifications
                                        else:
                                            print(f"  样本 {data_idx} 第{iteration}次迭代: 未提取到修改片段，尝试提取完整代码")
                                            error_types, current_code = extract_results(response)
                                            if current_code:
                                                data["generated_code"] = current_code
                                                data["current_code"] = current_code
                                                cleaned_code = remove_todo_comments(current_code)
                                                data["cleaned_code"] = cleaned_code
                                    else:
                                        error_types, current_code = extract_results(response)
                                        data["generated_code"] = current_code
                                        data["current_code"] = current_code
                                        cleaned_code = remove_todo_comments(current_code)
                                        data["cleaned_code"] = cleaned_code
                                break
                
                execution_codes = []
                execution_paths = []
                execution_indices = []
                for data in batch_data:
                    if data.get("cleaned_code"):
                        generated_image_filename = f"generated_image_{data['idx']}_{iteration}.png"
                        current_image_path = os.path.join(OUTPUT_DIR, generated_image_filename)
                        execution_codes.append(data["cleaned_code"])
                        execution_paths.append(current_image_path)
                        execution_indices.append(data["idx"])
                
                if execution_codes:
                    execution_results = batch_execute_codes(execution_codes, execution_paths)
                    for exec_idx, (success, error_msg) in enumerate(execution_results):
                        data_idx = execution_indices[exec_idx]
                        for data in batch_data:
                            if data["idx"] == data_idx:
                                data["success"] = success
                                data["error_msg"] = error_msg
                                data["generated_image_path"] = execution_paths[exec_idx] if success else ""
                                break
                
                correction_prompts = []
                correction_images = []
                correction_indices = []
                for data in batch_data:
                    if data.get("success", False):
                        correction_instruction = """I write a code snippet to reproduce a science chart. I will give you the original chart and the generated chart by my code.
Please help me to modify my code to make the generated chart closer to the original chart.
Output difference types and code snippet with TODO comments.

Difference types include: 
1. data: number and values of shapes and data points
2. local_style: styles of specific graphic shapes and elements
3. global_style: styles of global elements like Axis, Title, Annotations, Grid, Extra text

TODO comments should be inserted into my code snippet and conform the following format:
1. # TODO: Add extra code in the next line 
2. # TODO: Modify the following <number> lines
3. # TODO: Delete the following <number> lines

Return difference types surrounded < and > in one line.
Return the modified complete code surrounded by ```python and ```.

Original Chart:
<image>

Generated Chart:
<image>

Code Snippet:
python
{generated_code}
"""
                        full_correction_prompt = correction_instruction.format(generated_code=data["cleaned_code"])
                        correction_prompts.append(full_correction_prompt)
                        correction_images.append([data["image_file"], f"generated_image_{data['idx']}_{iteration}.png"])
                        correction_indices.append(data["idx"])
                if correction_prompts:
                    correction_responses = batch_call_vllm_model(correction_prompts, correction_images, "correction", OUTPUT_DIR)
                    for resp_idx, (response, data_idx) in enumerate(zip(correction_responses, correction_indices)):
                        for data in batch_data:
                            if data["idx"] == data_idx:
                                if response:
                                    correction_error_types_list, corrected_code_with_todos = extract_results(response)
                                    data["correction_error_types"] = list(correction_error_types_list) if correction_error_types_list else []
                                    data["corrected_code_with_todos"] = corrected_code_with_todos 
                                    if corrected_code_with_todos:
                                        data["current_code"] = corrected_code_with_todos
                                break
                for data in batch_data:
                    record = {
                        "sample_id": data["idx"],
                        "iteration": iteration,
                        "ground_truth_path": data["ground_truth_path"],
                        "generated_image_path": data.get("generated_image_path", ""),
                        "code": data["generated_code"],
                        "cleaned_code": data.get("cleaned_code", ""),
                        "ground_truth_code": data["ground_truth_code"],
                        "correction_error_types": data.get("correction_error_types", []),
                        "corrected_code_with_todos": data.get("corrected_code_with_todos", ""),
                        "modifications": data.get("modifications", []),
                        "execution_error": data.get("error_msg", "") if not data.get("success", True) else "",
                        "success": data.get("success", False)
                    }
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl_file.flush()
    return all_results

if __name__ == "__main__":
   plot_test()

