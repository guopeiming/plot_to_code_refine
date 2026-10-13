import os
import re
import json
import random
import base64
import shutil
import numpy as np
from openai import OpenAI
import matplotlib.pyplot as plt

CODE_PATH = "Chart2Code-160k/json/chart2code.json"
IMAGE_PATH = "Chart2Code-160k/images"
OUTPUT_DIR = "new_train_data"
ORDER_FILE = os.path.join(OUTPUT_DIR, "modification_orders.json")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def load_code(code_path):
    with open(code_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_valid_samples(count, exclude_ids=[]):
    code_data = load_code(CODE_PATH)
    all_ids = [item["id"] for item in code_data if "id" in item and item["id"] not in exclude_ids]
    
    if not all_ids:
        print("没有可用的新样本了")
        return []
    
    sample_count = int(count * 3)
    selected_ids = random.sample(all_ids, min(sample_count, len(all_ids)))
    
    valid_ids = []
    for item_id in selected_ids:
        if len(valid_ids) >= count*2:
            break
        code = extract_code_json(code_data, item_id)
        if not code:
            continue
        image_path = get_image_path(item_id, IMAGE_PATH)
        if not image_path:
            continue
        cleaned_code = clean_code(code)
        temp_path = os.path.join(OUTPUT_DIR, "temp.png")
        if test_code_execution(cleaned_code, temp_path):
            valid_ids.append(item_id)
            with open(os.path.join(OUTPUT_DIR, f"code_{item_id}_0.py"), 'w') as f:
                f.write(cleaned_code)
            shutil.copy2(image_path, os.path.join(OUTPUT_DIR, f"image_{item_id}_0.png"))
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    if len(valid_ids) < count:
        print(f"样本不足: {len(valid_ids)}/{count}")
    return valid_ids

def get_existing_samples():
    if not os.path.exists(OUTPUT_DIR):
        return []
    code_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('code_') and f.endswith('_0.py')]
    ids = []
    for file in code_files:
        match = re.match(r'code_(\d+)_0\.py', file)
        if match:
            item_id = int(match.group(1))
            image_file = f"image_{item_id}_0.png"
            if os.path.exists(os.path.join(OUTPUT_DIR, image_file)):
                ids.append(item_id)
    print(f"找到{len(ids)}个已存在的样本")
    return ids

def extract_code_json(code, item_id):
    for item in code:
        if item.get("id") == item_id:
            conversations = item.get("conversations", [])
            for conv in conversations:
                if conv.get("from") == "gpt":
                    code_text = conv.get("value", "")
                    if '```python' in code_text:
                        return code_text.split('```python')[1].split('```')[0].strip()
                    elif '```' in code_text:
                        code = code_text.split('```')[1].split('```')[0].strip()
                        if code.startswith('python'):
                            return code[6:].strip()
                        return code
                    else:
                        return code_text.strip()
    return None

def get_image_path(item_id, images_base_dir):
    for ext in ['.png', '.jpg', '.jpeg']:
        image_path = os.path.join(images_base_dir, f"{item_id}{ext}")
        if os.path.exists(image_path):
            return image_path
    return None

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_gpt4o_api(image, code, prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt} Code Snippet: {code}"
                        }
                    ]
                }
            ],
        )
        response_text = response.choices[0].message.content
        return response_text
    except Exception as e:
        print(str(e))
        return None

def extract_code_gpt(response_text):
    if '```python' in response_text:
        return response_text.split('```python')[1].split('```')[0].strip()
    elif '```' in response_text:
        code = response_text.split('```')[1].split('```')[0].strip()
        if code.startswith('python'):
            return code[6:].strip()
        return code
    return response_text.strip()

def test_code_execution(code, output_path):
    try:
        initial_figures = plt.get_fignums()
        exec(code)
        current_figures = plt.get_fignums()
        new_figures = [f for f in current_figures if f not in initial_figures] 
        if new_figures:
            fig = plt.figure(new_figures[0])
            fig.savefig(output_path)
            plt.close(fig)
            return True
        else:
            print("代码执行成功但未生成图片")
            return False
    except Exception as e:
        print(str(e))
        return False
    finally:
        plt.close('all')

def get_data_prompt():
    return f"""
You are an expert Python developer who specializes in writing matplotlib code. Please modify the given code snippet according to the requirements.

Requirements:
1. Change the data used in the code snippet to generate a different image.
2. The changed data should have enough variation to ensure the noticeable difference in the generated image.
3. Add or reduce the number of graphic shapes, and change the values of the data.
4. Keep the color and the style of graphic shapes of the image unchanged.
5. Ensure that the overall structure and style of the code remain unchanged.

Response Format:
Do not include any code for saving the image in the code.
Return the modified complete code surrounded by ```python and ```.
"""

def get_global_style_prompt():
    return f"""
You are an expert Python developer who specializes in writing matplotlib code. Please modify the given code snippet according to the requirements.

Requirements:
1. Change the global styles used in the code snippet to generate a different image.
2. Gloabl styles include: Axis ranges, Axis scales, Axis labels, Axis ticks, Title, Annotations, Grid, and Extra text.
3. The change should have enough variation to ensure the noticeable difference in the generated image.
4. Keep the data and the color and other styles of graphic shapes of the image unchanged
5. Ensure that the overall structure and style of the code remain unchanged.

Response Format:
Do not include any code for saving the image in the code.
Return the modified complete code surrounded by ```python and ```.
"""

def get_local_style_prompt():
    return f"""
You are an expert Python developer who specializes in writing matplotlib code. Please modify the given code snippet according to the requirements.

Requirements:
1. Change the style of graphic shapes and elements in the chart to generate a different image.
2. Change, add or reduce various parameters in the graphic shape rendering function.
3. The change should have enough variation to ensure the noticeable difference in the generated image.
4. Only when the original code uses the pie function, add or change or reduce parameters wedgeprops or explode with one-half probability. 
5. Keep the data and the style of axes of the image unchanged.
6. Ensure that the overall structure and style of the code remain unchanged.

Response Format:
Return the modified complete code surrounded by ```python and ```.
Do not include any code for saving the image in the code.
"""

def clean_code(code):
    lines = code.split('\n')
    cleaned_lines = []
    save_found = False
    for i, line in enumerate(lines):
        if not save_found and re.search(r'plt\.savefig|fig\.savefig|plt\.save', line):
            save_found = True
            if i > 0 and lines[i-1].strip().startswith('#'):
                if cleaned_lines and cleaned_lines[-1].strip() == lines[i-1].strip():
                    cleaned_lines.pop()
            continue
        if save_found:
            continue
        cleaned_lines.append(line)
    if not save_found:
        return code
    return '\n'.join(cleaned_lines)

def modify_with_retry(current_code, current_image_path, modification_type, max_attempts=3):
    prompt_functions = {
        'data': get_data_prompt,
        'global_style': get_global_style_prompt,
        'local_style': get_local_style_prompt
    }
    prompt = prompt_functions[modification_type]()
    for attempt in range(max_attempts):
        image_base64 = image_to_base64(current_image_path)
        response = call_gpt4o_api(image_base64, current_code, prompt)
        if not response:
            continue
        modified_code = extract_code_gpt(response)
        if not modified_code:
            continue
        temp_image_path = os.path.join(OUTPUT_DIR, "temp_modify.png")
        if test_code_execution(modified_code, temp_image_path):
            return modified_code, temp_image_path
        else:
            continue 
    return None, None

def process_sample(item_id):
    init_code_path = os.path.join(OUTPUT_DIR, f"code_{item_id}_0.py")
    init_image_path = os.path.join(OUTPUT_DIR, f"image_{item_id}_0.png")
    if not os.path.exists(init_code_path) or not os.path.exists(init_image_path):
        return False
    with open(init_code_path, 'r', encoding='utf-8') as f:
        current_code = f.read()
    current_image_path = init_image_path
    modification_types = ['data', 'global_style', 'local_style']
    random.shuffle(modification_types)
    modification_order = {
        'item_id': item_id,
        'order': modification_types,
        'success': []
    }
    for i, mod_type in enumerate(modification_types, 1):
        modified_code, new_image_path = modify_with_retry(current_code, current_image_path, mod_type)
        if modified_code and new_image_path:
            code_path = os.path.join(OUTPUT_DIR, f"code_{item_id}_{i}.py")
            image_path = os.path.join(OUTPUT_DIR, f"image_{item_id}_{i}.png")
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(modified_code)
            shutil.copy2(new_image_path, image_path)
            current_code = modified_code
            current_image_path = image_path
            modification_order['success'].append(True)
        else:
            modification_order['success'].append(False)
    save_modification_order(modification_order)
    return True

def save_modification_order(order_info):
    if os.path.exists(ORDER_FILE):
        with open(ORDER_FILE, 'r', encoding='utf-8') as f:
            orders = json.load(f)
    else:
        orders = []
    new_record = {
        'id': order_info['item_id'],
        'order': order_info['order'],
        'success': order_info['success']
    }
    orders.append(new_record)
    with open(ORDER_FILE, 'w', encoding='utf-8') as f:
        json.dump(orders, f, indent=2, ensure_ascii=False)

def save_modification_order(order_info):
    if os.path.exists(ORDER_FILE):
        with open(ORDER_FILE, 'r', encoding='utf-8') as f:
            orders = json.load(f)
    else:
        orders = []
    updated = False
    for i, order in enumerate(orders):
        if order.get('id') == order_info['item_id']:
            orders[i] = {
                'id': order_info['item_id'],
                'order': order_info['order'],
                'success': order_info['success']
            }
            updated = True
            break
    if not updated:
        orders.append({
            'id': order_info['item_id'],
            'order': order_info['order'],
            'success': order_info['success']
        })
    with open(ORDER_FILE, 'w', encoding='utf-8') as f:
        json.dump(orders, f, indent=2, ensure_ascii=False)

def generate_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_samples = 500
    valid_ids = get_existing_samples()
    success_count = 0
    for i, item_id in enumerate(valid_ids, 1):
        print(f"处理第{i}/{len(valid_ids)}个样本")
        if process_sample(item_id):
            success_count += 1
    print(f"成功处理{success_count}/{len(valid_ids)}个样本")

def get_failed_samples():
    if not os.path.exists(ORDER_FILE):
        return []
    with open(ORDER_FILE, 'r', encoding='utf-8') as f:
        orders = json.load(f)
    failed_ids = []
    for order in orders:
        if False in order.get('success', []):
            failed_ids.append(order['id'])
    print(f"找到{len(failed_ids)}个需要重新处理的样本")
    return failed_ids

def reprocess_failed_samples():
    failed_ids = get_failed_samples()
    if not failed_ids:
        return
    success_count = 0
    for i, item_id in enumerate(failed_ids, 1):
        print(f"重新处理第{i}/{len(failed_ids)}个样本(ID:{item_id})")
        for i in range(1, 3):
            code_path = os.path.join(OUTPUT_DIR, f"code_{item_id}_{i}.py")
            image_path = os.path.join(OUTPUT_DIR, f"image_{item_id}_{i}.png")
            if os.path.exists(code_path):
                os.remove(code_path)
            if os.path.exists(image_path):
                os.remove(image_path)
        if process_sample(item_id):
            success_count += 1
    print(f"重新处理{success_count}/{len(failed_ids)}个样本成功")

if __name__ == "__main__":
    reprocess_failed_samples()