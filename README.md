Install dependencies with:

```bash
pip install -r requirements.txt
```


### 🔹 Test on **ScreenSpot**

```bash
python screenspot_test.py \
  --qwen_path "${model_path}" \
  --screenspot_imgs "${screenspot_imgs}" \
  --screenspot_test "${screenspot_test}" \
  --task all \
  --lora_path "Qwen-VL-Chat"
```

### 🔹 Test on **ScreenSpot-V2**

```bash
python screenspot_test.py \
  --qwen_path "${model_path}" \
  --screenspot_imgs "${screenspot_imgs_v2}" \
  --screenspot_test "${screenspot_test_v2}" \
  --task all \
  --lora_path "Qwen-VL-Chat"
```

---

### 🔹 Test on **AndroidControl**

Follow the UGround evaluation pipeline:  
📎 [UGround AndroidControl Evaluation Guide](https://github.com/OSU-NLP-Group/UGround/tree/main/offline_evaluation/AndroidControl)

You may replace `uground_qwen2vl.py` with `ground_qwen25vl_local.py` to evaluate local models.

#### AndroidControl-Low

```bash
python ground_qwen25vl_local.py \
  --model-path ${your_model_path} \
  --question-file UGround/offline_evaluation/AndroidControl/data/query_gpt-4o_low.jsonl \
  --image-folder ${Android_images_path} \
  --image-key image \
  --temperature 0.1 \
  --answers-file android_low_answer.jsonl
```

#### AndroidControl-High

```bash
python ground_qwen25vl_local.py \
  --model-path ${your_model_path} \
  --question-file UGround/offline_evaluation/AndroidControl/data/query_gpt-4o_high.jsonl \
  --image-folder ${Android_images_path} \
  --image-key image \
  --temperature 0.1 \
  --answers-file android_high_answer.jsonl
```

