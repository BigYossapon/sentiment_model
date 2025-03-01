from transformers import (
    CamembertTokenizer,
    AutoModelForSequenceClassification,
    TFAutoModelForSequenceClassification,
    pipeline,
    AutoTokenizer,
    Trainer, 
    TrainingArguments
)
import os
import sys
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report



# from transformers import TFAutoModelForSequenceClassification
from pythainlp.tokenize import word_tokenize
# from thai2transformers.preprocess import process_transformers
from datasets import Dataset



def preprocess_text(text):
    """ ใช้ word_tokenize เพื่อแยกคำก่อนนำเข้าโมเดล """
    return " ".join(word_tokenize(text, keep_whitespace=False))

def save_model(model,tokenizer,path):
    # กำหนดเส้นทางที่ต้องการบันทึก
    # save_directory = './my_model'

    # บันทึกโมเดลและ tokenizer
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def load_model(path):
    # โหลดโมเดลจากที่บันทึกไว้
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else 
                      "cpu")
    model.to(device)
    return model,tokenizer

def run_model(path_model):
    # Load pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
                                    'airesearch/wangchanberta-base-att-spm-uncased',
                                    revision='main')
    tokenizer.additional_special_tokens = ['<s>NOTUSED', '</s>NOTUSED', '<_>']

    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(
                                    'airesearch/wangchanberta-base-att-spm-uncased',
                                    revision='finetuned@wisesight_sentiment')

    # model = TFAutoModelForSequenceClassification.from_pretrained(
    #                                 'airesearch/wangchanberta-base-att-spm-uncased',
    #                                 revision='finetuned@wisesight_sentiment')                                
    save_model(model,tokenizer,path_model)

   

def train_model(path_model,path_csv):
    model,tokenizer = load_model(path=path_model)
 
    df = pd.read_csv(path_csv)
    
    # ตรวจสอบว่ามีคอลัมน์ label กับ text ไหม
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV file must contain 'label' and 'text' columns")

    # (ถ้าต้องการแปลง label เป็นตัวเลข)
    first_label = df.loc[0, "label"].strip()  # .strip() ใช้ตัดช่องว่างที่อาจเกินมา
    print("Label แรก:", first_label)

    label2id = {}
    is_number = False
    data = {}
    if first_label == "Positive" or first_label == "Negative" or first_label == "Neutral"  :
        label2id = {"Positive": 2, "Negative": 0, "Neutral": 1}
        df["label_id"] = df["label"].map(label2id)
    elif first_label == "Pos" or first_label == "Neg" or first_label == "Neu"  :
        label2id = {"Pos": 2, "Neg": 0, "Neu": 1}
        df["label_id"] = df["label"].map(label2id)
    elif first_label == "positive" or first_label == "negative" or first_label == "neutral"  :
        label2id = {"positive": 2, "negative": 0, "neutral": 1}
        df["label_id"] = df["label"].map(label2id)
    elif first_label == "pos" or first_label == "neg" or first_label == "neu":
        label2id = {"pos": 2, "neg": 0, "neu": 1}
        df["label_id"] = df["label"].map(label2id)
    else:
        is_number = True
        
    


    # label2id = {"Positive": 2, "Negative": 0, "Neutral": 1}
    # df["label_id"] = df["label"].map(label2id)

    # สร้าง Dataset สำหรับเทรน
    if is_number :
        data = {
            "text": df["text"].tolist(),
            "label": df["label_id"].tolist()
        }
    else :
         data = {
        "text": df["text"].tolist(),
        "label": df["label"].tolist()
    }

    dataset = Dataset.from_dict(data)
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True ,max_length=128)

    dataset = Dataset.from_dict(data)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
    output_dir='./results',           # Path สำหรับเก็บผลลัพธ์
    num_train_epochs=3,              # จำนวน epoch ที่ต้องการ
    per_device_train_batch_size=8,    # ขนาด batch สำหรับการฝึก
    per_device_eval_batch_size=16,    # ขนาด batch สำหรับการทดสอบ
    warmup_steps=500,                 # จำนวนขั้นตอนการ warmup
    weight_decay=0.01,                # ค่า weight decay
    logging_dir='./logs',             # Path สำหรับเก็บ log
    logging_steps=10,                 # จำนวนการ log ต่อ 10 steps
    )

    trainer = Trainer(
    model=model,                        # โมเดลที่ใช้ในการฝึก
    args=training_args,                 # พารามิเตอร์การฝึก
    train_dataset=tokenized_datasets,   # ชุดข้อมูลฝึก
    )

    trainer.train()  # เริ่มการฝึกโมเดล

def convert_to_polarity(probabilities):
    """
    แปลงค่า Probability จากโมเดล Sentiment Classification เป็น Polarity Score
    """
    return (
        probabilities["Positive"] * 1.0 +  # Positive เป็น 1.0
        probabilities["Neutral"] * 0.0 +   # Neutral เป็น 0.0
        probabilities["Negative"] * -1.0   # Negative เป็น -1.0
    )

def polarity_calculate2(model,tokenizer,comment):
   
   
    inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True,max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    # แปลง logits เป็น probability ด้วย softmax
    probabilities = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
    
    # ค่าความน่าจะเป็นของแต่ละคลาส
    labels = ["Negative", "Neutral", "Positive"]
    prob_dict = dict(zip(labels, probabilities))

    # คำนวณค่า Polarity Score
    polarity_score = convert_to_polarity(prob_dict)
    print(f"probabilities: {prob_dict}\n"  
        f"polarity_score: {polarity_score }"  )
    return {
        "probabilities": prob_dict,  # Probability ของแต่ละ class
        "polarity_score": polarity_score  # ค่า Polarity Score
    }


def use_model_for_sentiment(list_comment,path_csv,path_model):

    model , tokenizer = load_model(path_model)
    model.eval()
    classify_sequence = pipeline(task='sentiment-analysis',
            tokenizer=tokenizer,
            model=model)
    # input_text = "บริษัทนี้ดูแล้วดีจริง อยากบอกต่อ"
    # input_text = "กรรมการบริษัทชุดนี้บริหารงานกันแปลกๆ"
    for comment in list_comment :

        processed_input_text = preprocess_text(comment)
        print('\n', processed_input_text, '\n')
        print(classify_sequence(processed_input_text))
        # สมมติว่า logits คือผลลัพธ์ที่ได้จากการทำนายของโมเดล
        # สมมุติว่าคุณมีโมเดลและ tokenizer ที่โหลดไว้แล้ว
        polarity_calculate(model,tokenizer=tokenizer,comment=comment)
        polarity_calculate2(model,tokenizer=tokenizer,comment=comment)
    
        
      
  

def polarity_calculate(model,tokenizer,comment):
   
         # Tokenize ข้อความ
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True)
    
    # รับผลลัพธ์จากโมเดล
    outputs = model(**inputs)
    logits = outputs.logits
    
    # ใช้ softmax กับ logits เพื่อแปลงเป็น probabilities
    probs = F.softmax(logits, dim=-1)
    
    # สมมติว่าเรามี 3 คลาส (negative, neutral, positive)
    negative_prob = probs[0, 0].item()
    neutral_prob = probs[0, 1].item()
    positive_prob = probs[0, 2].item()
    question_prob = probs[0, 3].item()
        
        # # คำนวณ Polarity Score แบบ Weighted Average
        # polarity_score = (negative_prob - positive_prob) / (positive_prob + neutral_prob + negative_prob)
        # print(f"polarity score : {polarity_score}")
        
        # return polarity_score

   
    # วิธีที่ 1: การคำนวณ Polarity Score จากความแตกต่างระหว่าง positive และ negative
    polarity_score_1 = (positive_prob - negative_prob)
    print(f"Polarity Score (positive - negative): {polarity_score_1}")

    # วิธีที่ 2: การคำนวณ Polarity Score โดยการใช้ weighted average ของคลาสทั้งหมด
    polarity_score_2 = (negative_prob - positive_prob) / (positive_prob + neutral_prob + negative_prob + question_prob)
    print(f"Polarity Score (Weighted Average): {polarity_score_2}")

    # วิธีที่ 3: การคำนวณจากคะแนนรวม
    polarity_score_3 = (positive_prob + neutral_prob - negative_prob)
    print(f"Polarity Score (Summed): {polarity_score_3}")

    # วิธีที่ 4: คำนวณ Polarity Score โดยการหาค่าความน่าจะเป็นของบวกและลบแล้วปรับเป็น 0-1
    polarity_score_4 = (positive_prob - negative_prob) / (positive_prob + negative_prob)
    print(f"Polarity Score (Normalized): {polarity_score_4}")

    # ผลลัพธ์ทั้งหมด
    print(f"Polarity Score (positive - negative): {polarity_score_1}, "
          f"Polarity Score (Weighted Average): {polarity_score_2}, "
          f"Polarity Score (Summed): {polarity_score_3}, "
          f"Polarity Score (Normalized): {polarity_score_4}")

def evaluate_sentiment(csv_path):
    # โหลดโมเดล
    model, tokenizer = load_model("./model_sentiment")
    classify_sequence = pipeline(task='sentiment-analysis', tokenizer=tokenizer, model=model)

    # โหลดชุดข้อมูล CSV
    df = pd.read_csv(csv_path)

    # ตรวจสอบว่าไฟล์มีคอลัมน์ที่ถูกต้องหรือไม่
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV ต้องมีคอลัมน์ 'text' และ 'label'")

    # เก็บค่าที่คาดการณ์และ label จริง
    true_labels = []
    predicted_labels = []
    target_names = []

    # แปลง label ที่ได้จากโมเดลเป็นค่าที่ใช้เปรียบเทียบ
    first_label = df.loc[0, "label"].strip()  # .strip() ใช้ตัดช่องว่างที่อาจเกินมา
    print("Label แรก:", first_label)
    label_map = {}
    if first_label == "Positive" or first_label == "Negative" or first_label == "Neutral"  :
        label_map = {"Positive": "pos", "Negative": "neg", "Neutral": "neu"}
        # target_names = 
    elif first_label == "Pos" or first_label == "Neg" or first_label == "Neu"  :
        label_map = {"Pos": "pos", "Neg": "neg", "Neu": "neu"}
       
    elif first_label == "positive" or first_label == "negative" or first_label == "neutral"  :
        label_map = {"positive": "pos", "negative": "neg", "neutral": "neu"}
       
    elif first_label == "pos" or first_label == "neg" or first_label == "neu":
        label_map = {"pos": "pos", "neg": "neg", "neu": "neu"}
        
    else:
        label_map = {2: "pos", 0: "neg", 1: "neu"}
        # is_number = True
    # label_map = {
    #     "pos": "pos",
    #     "neg": "neg",
    #     "neu": "neu",
    #     # "q":"q"
    # }

    # วนลูปทดสอบข้อมูลทั้งหมด
    for index, row in df.iterrows():
        comment = preprocess_text(row['text'])
        true_label = row['label']  # ค่าจริง

        # ทำนายผล sentiment
        result = classify_sequence(comment)
        predicted_label = result[0]['label']

        # แปลง label เป็นค่าที่สามารถเปรียบเทียบได้
        predicted_labels.append(label_map.get(predicted_label, "unknown"))
        true_labels.append(true_label)

        print(f"✅ คอมเมนต์: {comment}")
        print(f"🔹 ค่าจริง: {true_label}, 🔸 โมเดลทำนาย: {predicted_label}\n")

    # คำนวณความแม่นยำ
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\n📊 Accuracy: {accuracy:.2%}\n")
    
    # รายงานผลลัพธ์แบบละเอียด
    print(classification_report(true_labels, predicted_labels, target_names=["pos", "neu", "neg"]))
  

def upload_model_to_hub(path_model,username_hf,model_name):
    # โหลดโมเดลและ tokenizer ที่คุณได้เทรนไว้
    
    model = AutoModelForSequenceClassification.from_pretrained(path_model)
    tokenizer = AutoTokenizer.from_pretrained(path_model)

    # กำหนดชื่อ repository ที่ต้องการ push ไปที่ Hugging Face
    repo_name = f"{username_hf}/{model_name}"  # ปรับให้เป็นชื่อของคุณและชื่อโมเดลที่ต้องการ

    # push โมเดลและ tokenizer
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)


def tokenize_function(tokenizer,examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

if __name__ == "__main__":
    list_comment_for_sentiment = []
    if len(sys.argv) > 1:
        print(sys.argv[1:])
        list_comment_for_sentiment = sys.argv[1:]
    else :
        list_comment_for_sentiment = ["ผู้บริหารใจดี บริษัทดูมีอนาคต","ทรงหุ้นตัวนี้ดูแปลกๆ","ผมละเกลียดคนแบบนี้จริงๆ","ทำไมเขาถึงทำแบบนี้"]
        # python model.py "data" "delta" 
    path_model = "./model_sentiment"
    path_csv = "./datasets/twitter_training.csv"
    username_hf= "BigYossapon"
    model_name = "SENTIMENT_TEST_FROM_WangchanBERTa"

    # ถ้ายังไม่มี model
    # run_model(path_model)

    # ถ้าต้องการ train
    # train_model(path_model=path_model,path_csv=path_csv)

    # ถ้าต้องการใช้   
   
    use_model_for_sentiment(list_comment_for_sentiment,path_csv=path_csv,path_model=path_model)

    # ถ้าต้องการเทส accuracy
    # evaluate_sentiment(path_csv)

    #ถ้าต้องการเก็บโมเดลไง้ที่ hunging face 1.สมัคร 2.huggingface-cli login 3.ไปสร้าง model ไว้ 
    # upload_model_to_hub(path_model=path_model,model_name=model_name,username_hf=username_hf)
    # หมายเหตุถ้าเป็นคำ eng หมดจะถูกมองว่า neutral เนื่องจากไม่มีการเทรน eng เลย

