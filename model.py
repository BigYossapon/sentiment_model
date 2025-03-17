from transformers import (
    AutoModelForSequenceClassification,
    pipeline,
    AutoTokenizer,
    Trainer, 
    TrainingArguments
)
import emoji
import sys
import re
import torch
from pythainlp.spell import spell
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct
from pythainlp.util import emoji_to_thai, normalize
from pythainlp.corpus import thai_stopwords
from pythainlp.spell import NorvigSpellChecker
from datasets import Dataset
from huggingface_hub import get_safetensors_metadata

def preprocess_text(text):
    """ ใช้ word_tokenize เพื่อแยกคำก่อนนำเข้าโมเดล """
    return " ".join(word_tokenize(text, keep_whitespace=False))

def save_model(model,tokenizer,path):
    # บันทึกโมเดลและ tokenizer
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def load_model(path):
    # โหลดโมเดลจากที่บันทึกไว้
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    # device = torch.device("mps" if torch.backends.mps.is_available() else 
    #                   "cuda" if torch.cuda.is_available() else 
    #                   "cpu")
    # model.to(device)
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
                                 
    save_model(model,tokenizer,path_model)
    
    
def run_model_from_my_hf(path_model):
        # Load pre-trained tokenizer
        model_name = "BigYossapon/SENTIMENT_TEST_FROM_WangchanBERTa"
        tokenizer = AutoTokenizer.from_pretrained(
                                        model_name,
                                        )

        # Load pre-trained model
        model = AutoModelForSequenceClassification.from_pretrained(
                                        model_name,
                                        )
                           
        save_model(model,tokenizer,path_model)

   

def train_model(path_model,path_csv):
    model,tokenizer = load_model(path=path_model)
 
    df = pd.read_csv(path_csv)
    
    # ตรวจสอบว่ามีคอลัมน์ label กับ text ไหม
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("Csv ไม่มีหัว column ถ้าเป็นคำให้ หัวคอลัมเป็น text ถ้าเป็น polarity ให้เป็น label")

    first_label = df.loc[0, "label"].strip() 
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
        return tokenizer(examples['text'], padding="max_length", truncation=True ,max_length=512)
    # max length ความยาวข้อความได้มากสุด 512 character
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

    trainer.train()  

def convert_to_polarity(probabilities):
    return (
        probabilities["Neutral"] * 0.0 +   
        probabilities["Negative"] * 1.0 -  
        probabilities["Positive"] * 1.0  
    )

def split_string(text:str, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def polarity_calculate_for_list(model,tokenizer,comment,result):
    list_string = split_string(comment)
    magnitude_avg = 0
    magnitude = 0
    for res in result :
        magnitude += res[0]["score"] 
    magnitude_avg = magnitude / len(result)
    list_neg_score = []
    list_pos_score = []
    list_neu_score = []
    list_polarity_score = []
    for string in list_string :
        inputs = tokenizer(string, return_tensors="pt", padding=True, truncation=True,max_length=512)
    # max length ความยาวข้อความได้มากสุด 512 character
    #     BERT-based models (เช่น Camembert, BERT, etc.): ปกติจะมีขีดจำกัดที่ 512 tokens. หากคุณตั้งค่า max_length มากกว่า 512, โมเดลจะไม่สามารถรองรับได้ และจะเกิดข้อผิดพลาด.

    # GPT-based models (เช่น GPT-2, GPT-3, etc.): โมเดลบางตัวเช่น GPT-2 มีขีดจำกัดที่ 1024 tokens หรือ 2048 tokens, ขึ้นอยู่กับขนาดของเวอร์ชันโมเดล (เช่น GPT-2 small, medium, large).

    # Longformer, BigBird (โมเดลสำหรับเอกสารยาว): โมเดลที่ออกแบบมาเพื่อจัดการกับเอกสารที่มีความยาวมาก ๆ (เช่น Longformer หรือ BigBird) สามารถรองรับความยาวได้มากกว่า 512 token (เช่น 4096 tokens หรือมากกว่านั้น).
        with torch.no_grad():
            outputs = model(**inputs)

        # แปลง logits เป็น probability ด้วย softmax
        probabilities = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
        
        labels = ["Negative", "Neutral", "Positive"]
        prob_dict = dict(zip(labels, probabilities))

        # คำนวณค่า Polarity Score
        polarity_score = convert_to_polarity(prob_dict)

        print(f"probabilities: {prob_dict}\n"  
            f"polarity_score: {polarity_score }"  )
        list_neg_score.append(prob_dict["Negative"])
        list_neu_score.append(prob_dict["Neutral"])
        list_pos_score.append(prob_dict["Positive"])

        list_polarity_score.append(polarity_score)

    score = {'Negative': sum(list_neg_score) / len(list_neg_score), 'Neutral': sum(list_neu_score) / len(list_neu_score), 'Positive': sum(list_pos_score) / len(list_pos_score)}
    average = sum(list_polarity_score) / len(list_polarity_score)
    polarity = ""
    if len(list_string)>1:
        if average > 0.5:
            polarity = "Positive"
            average = abs(average)
        elif average < -0.5:
            polarity = "Negative"
            average = -abs(average)
        else:
            average = abs(average)
            polarity = "Neutral"
    else:
        if result[0][0]["label"] == "pos":
            polarity = "Positive"
            average = abs(average)
        elif result[0][0]["label"] == "neg":
            polarity = "Negative"
            average = -abs(average)
        else: 
            polarity = "Neutral"
            average = abs(average)
    print({
            "text" : list_string,
            "magnitude" : magnitude_avg,
            "probabilities": score,  # Probability ของแต่ละ class
            "polarity_score": average,  # ค่า Polarity Score
            "polarity" : polarity
        })
    return {
            "text" : list_string,
            "magnitude" : magnitude_avg,
            "probabilities": score,  # Probability ของแต่ละ class
            "polarity_score": average,  # ค่า Polarity Score
            "polarity" : polarity
        }





def polarity_calculate2(model,tokenizer,comment,polarity):


    inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True,max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)

    # แปลง logits เป็น probability ด้วย softmax
    probabilities = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
    
    # ค่าความน่าจะเป็นของแต่ละคลาส
    labels = ["Negative", "Neutral", "Positive"]
    prob_dict = dict(zip(labels, probabilities))

    # คำนวณค่า Polarity Score
    polarity_score = convert_to_polarity(prob_dict)
    # print(f"probabilities: {prob_dict}\n"  
    #     f"polarity_score: {polarity_score }"  )
    return {
        "probabilities": prob_dict,  # Probability ของแต่ละ class
        "polarity_score": polarity_score  # ค่า Polarity Score
    }
    
def polarity_calculate_new(model,tokenizer,comment,polarity):


    inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True,max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    # แปลง logits เป็น probability ด้วย softmax
    probabilities = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
    
    # ค่าความน่าจะเป็นของแต่ละคลาส
    labels = ["Negative", "Neutral", "Positive"]
    prob_dict = dict(zip(labels, probabilities))

    # คำนวณค่า Polarity Score
    polarity_score = convert_to_polarity(prob_dict)
    if polarity == "neg":
        polarity_score = -abs(polarity_score)  # Always return the negative value
    elif polarity == "pos":
        polarity_score =  abs(polarity_score)  # Always return the positive value
    

    print(
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

    for comment in list_comment :

        processed_input_text = preprocess_text(comment)
        print('\n', processed_input_text, '\n')
        result =  classify_sequence(processed_input_text)
        
        print(result[0])
        # สมมติว่า logits คือผลลัพธ์ที่ได้จากการทำนายของโมเดล
        # สมมุติว่าคุณมีโมเดลและ tokenizer ที่โหลดไว้แล้ว
        # polarity_calculate(model,tokenizer=tokenizer,comment=comment)
        # polarity_calculate2(model,tokenizer=tokenizer,comment=comment)
        polarity_calculate_for_list(model,tokenizer=tokenizer,comment=comment)
        polarity_calculate_new(model,tokenizer=tokenizer,comment=comment,polarity=result[0]['label'])
        
def use_my_model_for_sentiment_new(list_comment,path_csv,path_model):

    model , tokenizer = load_model(path_model)
    model.eval()
    classify_sequence = pipeline(task='sentiment-analysis',
            tokenizer=tokenizer,
            model=model)

    for comment in list_comment :
        list_string = split_string(comment)
        print("=====================")
        print(list_string)
        result = []
        for string in list_string:
            
            processed_input_text = preprocess_text(string)
            # print('\n', processed_input_text, '\n')
            
            result.append(classify_sequence(processed_input_text))
            # print(result)
        # print(result[0])
        # สมมติว่า logits คือผลลัพธ์ที่ได้จากการทำนายของโมเดล
        # สมมุติว่าคุณมีโมเดลและ tokenizer ที่โหลดไว้แล้ว
        # polarity_calculate(model,tokenizer=tokenizer,comment=comment)
        # polarity_calculate2(model,tokenizer=tokenizer,comment=comment)
        # polarity_calculate3(model,tokenizer=tokenizer,comment=comment)
        polarity_calculate_for_list(model,tokenizer=tokenizer,comment=comment,result=result)
        # polarity_calculate_new(model,tokenizer=tokenizer,comment=comment,polarity=result[0]['label'])
    
def use_my_model_for_sentiment(list_comment,path_csv,path_model):

    model , tokenizer = load_model(path_model)
    model.eval()
    classify_sequence = pipeline(task='sentiment-analysis',
            tokenizer=tokenizer,
            model=model)
    # input_text = "บริษัทนี้ดูแล้วดีจริง อยากบอกต่อ"
    # input_text = "กรรมการบริษัทชุดนี้บริหารงานกันแปลกๆ"
    for comment in list_comment :
        list_string = split_string(comment)
       
            
        processed_input_text = preprocess_text(comment)
        # print('\n', processed_input_text, '\n')
        result =  classify_sequence(processed_input_text)
    
        # print(result[0])
        # สมมติว่า logits คือผลลัพธ์ที่ได้จากการทำนายของโมเดล
        # สมมุติว่าคุณมีโมเดลและ tokenizer ที่โหลดไว้แล้ว
        # polarity_calculate(model,tokenizer=tokenizer,comment=comment)
        # polarity_calculate2(model,tokenizer=tokenizer,comment=comment)
        # polarity_calculate3(model,tokenizer=tokenizer,comment=comment)
        polarity_calculate_for_list(model,tokenizer=tokenizer,comment=comment,magnitude=result[0]['label'])
        # polarity_calculate_new(model,tokenizer=tokenizer,comment=comment,polarity=result[0]['label'])
      
def evaluate_sentiment(model_path,csv_path):
    # โหลดโมเดล
    model, tokenizer = load_model(model_path)
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

        print(f"comment : {comment}")
        print(f"true value : {true_label}, 🔸 predict : {predicted_label}\n")

    # คำนวณความแม่นยำ
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\n Accuracy: {accuracy:.2%}\n")
    
    # รายงานผลลัพธ์แบบละเอียด
    print(classification_report(true_labels, predicted_labels, target_names=["pos", "neu", "neg"]))
  

def upload_model_to_hub(path_model,username_hf,model_name):
    # โหลดโมเดลและ tokenizer ที่เทรนไว้
    
    model = AutoModelForSequenceClassification.from_pretrained(path_model)
    tokenizer = AutoTokenizer.from_pretrained(path_model)

    # ชื่อ repository ที่ push ไปที่ Hugging Face
    repo_name = f"{username_hf}/{model_name}"  # ปรับให้เป็นชื่อuserและชื่อโมเดลที่ต้องการ

    # push โมเดลและ tokenizer
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)

stopword_list = frozenset(thai_stopwords())

def clean_text(text):
    text = text.lower()  
    text = emoji.replace_emoji(text, replace="") 
    text = re.sub(r"[^\w\s]", "", text)  
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = normalize(text) 
    checker = NorvigSpellChecker()
    words =  spell(text)
    corrected_words = []
    for word in words:
        if word in ["พ่อแม่", "พี่น้อง"]: 
            corrected_words.append(word)
        else:
            corrected = checker.correct(word=word) 
            corrected_words.append(corrected if corrected else word)
    return " ".join(corrected_words)

def clean_text_csv(csv_path_to_clean:str):
    df = pd.read_csv(csv_path_to_clean)
    df["clean_text"] = df["text"].apply(clean_text)
    df.to_csv(csv_path_to_clean, index=False, encoding="utf-8-sig")
    
def check_memory_required(model_id = ""):
    dtype_bytes =  {"F32":4,"F16":2,"BF16":2,"FB":1}
    
    metadata = get_safetensors_metadata(model_id)
    memory = (sum(count * dtype_bytes[key.split("_")[0]] for key ,count in metadata.parameter_count.items())/(1024**3)*1.18)
    
    print(f"{model_id=} requires {memory=}GB")
    
def tokenize_function(tokenizer,examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

if __name__ == "__main__":
    list_comment_for_sentiment = []
    if len(sys.argv) > 1:
        print(sys.argv[1:])
        list_comment_for_sentiment = sys.argv[1:]
    else :
        list_comment_for_sentiment = ["ในโลกปัจจุบันที่เทคโนโลยีและอินเทอร์เน็ตเข้ามามีบทบาทสำคัญในชีวิตประจำวันของเรา การสื่อสารผ่านช่องทางดิจิทัลได้กลายเป็นสิ่งที่ขาดไม่ได้ ไม่ว่าจะเป็นการส่งข้อความ การโทรผ่านวิดีโอคอล หรือแม้แต่การประชุมออนไลน์ที่ช่วยให้เราสามารถติดต่อสื่อสารกันได้อย่างสะดวกสบายโดยไม่ต้องเดินทาง เทคโนโลยีสารสนเทศได้เปลี่ยนแปลงวิธีการดำเนินชีวิตของผู้คนไปอย่างสิ้นเชิง ธุรกิจหลายแห่งหันมาใช้แพลตฟอร์มออนไลน์ในการให้บริการลูกค้า ไม่ว่าจะเป็นการขายสินค้า การให้คำปรึกษาทางไกล หรือแม้กระทั่งการเรียนการสอนออนไลน์ที่ได้รับความนิยมเพิ่มขึ้นเป็นอย่างมาก โดยเฉพาะในช่วงที่เกิดสถานการณ์โรคระบาดที่ทำให้การเดินทางและการพบปะสังสรรค์ในสถานที่สาธารณะถูกจำกัด ความสะดวกสบายที่เทคโนโลยีมอบให้ยังช่วยลดข้อจำกัดด้านระยะทางและเวลา ทำให้ผู้คนสามารถเข้าถึงข้อมูลข่าวสาร ความรู้ และบริการต่าง ๆ ได้จากทุกที่ทั่วโลก นอกจากนี้ การใช้เทคโนโลยีอย่างมีประสิทธิภาพยังสามารถช่วยให้ธุรกิจเติบโตได้อย่างรวดเร็วและเพิ่มโอกาสในการแข่งขันในตลาดโลก อย่างไรก็ตาม การพึ่งพาเทคโนโลยีมากเกินไปก็อาจนำไปสู่ปัญหาต่าง ๆ เช่น ความปลอดภัยของข้อมูล การละเมิดสิทธิส่วนบุคคล และการเสพติดสื่อออนไลน์ ดังนั้น การใช้เทคโนโลยีอย่างมีสติและความรับผิดชอบจึงเป็นสิ่งสำคัญที่ทุกคนควรตระหนัก","ทรงหุ้นตัวนี้ดูแปลกๆ","ผมละเกลียดคนแบบนี้จริงๆ","ทำไมเขาถึงทำแบบนี้","รถไฟฉึกฉึกฉึกฉักฉักฉัก","แย่มากๆ"]
        # python model.py "data" "delta" 
    path_model = "./model_sentiment"
    path_csv = "./datasets/data_test.csv"
    username_hf= "BigYossapon"
    model_name = "SENTIMENT_TEST_FROM_WangchanBERTa"
    # ทดสอบ
    # run_model_from_my_hf("./my_model_hf")
    # use_my_model_for_sentiment_new(list_comment_for_sentiment,path_csv=path_csv,path_model="./my_model_hf")
    # evaluate_sentiment(model_path="./my_model_hf",csv_path=path_csv)
    clean_text_csv("./datasets/data_for_clean_test.csv")
    
    # 1 ถ้ายังไม่มี model
    # path สำหรับ save model
    # run_model(path_model)

    # ถ้าต้องการ train
    # train_model(path_model=path_model,path_csv=path_csv)

    # 2 ถ้าต้องการใช้   
    # use_model_for_sentiment(list_comment_for_sentiment,path_csv=path_csv,path_model=path_model)

    # 3 ถ้าต้องการเทส accuracy
    # evaluate_sentiment(model_path=path_model,csv_path=path_csv)

    #ถ้าต้องการเก็บโมเดลไง้ที่ hunging face 1.สมัคร 2.huggingface-cli login 3.ไปสร้าง model ไว้ 
    # upload_model_to_hub(path_model=path_model,model_name=model_name,username_hf=username_hf)
    # หมายเหตุถ้าเป็นคำ eng หมดจะถูกมองว่า neutral เนื่องจากไม่มีการเทรน eng เลย

    # ทุกไฟล์ที่นำเข้ามาใน datasets ต้องกำหนดหัว column ถ้าเป็นคำให้ column เป็น text และค่า polarity เป็น label 
