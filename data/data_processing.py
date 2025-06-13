#%% LIB
import pandas as pd
import numpy as np

import nltk
from nltk import ngrams
from nltk.corpus import stopwords
nltk.download('punkt_tab')
nltk.download('stopwords')

from pyvi import ViTokenizer

from collections import Counter

from gensim import corpora
from gensim.models import LdaModel

import re
import string

from dotenv import load_dotenv
import os
load_dotenv()

import openai
from openai import OpenAI, Embedding
api_key = os.getenv("SECRETE_KEY")
client = OpenAI(api_key=api_key)

import time

import matplotlib.pyplot as plt
import seaborn as sns

import tiktoken
encoding = tiktoken.encoding_for_model("text-embedding-3-small")

import yt_dlp

class DataRepository:
    def __init__(self):
        pass
    
    #%% CLEAN DATA
    def load_documents(self):
    
        # NGAYMOIONLINE DATA
        df_ngaymoi = pd.read_json('/Users/dungnguyen/Desktop/Data Science off/Python Programming/3. Publication/DA15_silver_economy/data/database/ngaymoionline_article_data.json')
        df_ngaymoi = df_ngaymoi.drop_duplicates(keep=False)
        len(df_ngaymoi)
        df_ngaymoi['text'] = df_ngaymoi['text'].apply(
            lambda x: x.replace(' TẠP CHÍ NGÀY MỚI ONLINE Cơ quan chủ quản: Hội Người cao tuổi Việt Nam Giấy phép hoạt động báo chí số 47/GP-BTTTT do Bộ Thông tin và Truyền thông cấp ngày 5/2/2020. Tổng biên tập: Lê Quang Phó Tổng biên tập: Nguyễn Thị Kim Thoa Trưởng phòng Điện tử: ThS. Lê Đình Vũ Tòa soạn: 12 Lê Hồng Phong, Ba Đình, Hà Nội Liên hệ:  Tạp chí điện tử của Tạp chí Người cao tuổi', '')
            )
        
        # DANTRI DATA
        
        df_dantri = pd.read_json('/Users/dungnguyen/Desktop/Data Science off/Python Programming/3. Publication/DA15_silver_economy/data/database/dantri_article_data.json')
        df_dantri = df_dantri.drop_duplicates(keep=False)
        df_dantri['text'] = df_dantri['text'].apply(
            lambda x: x.replace('Thông tin doanh nghiệp - sản phẩm', '')
            )
        # df_dantri = df_dantri[~df_dantri['text'].str.contains('--')]
        len(df_dantri)
        
        # MANUAL DATA
        df_manual = pd.read_json('/Users/dungnguyen/Desktop/Data Science off/Python Programming/3. Publication/DA15_silver_economy/data/database/manual_article_data_v2.json')
        df_manual = df_manual.melt()
        df_manual.dropna(inplace=True)
        df_manual = df_manual.drop_duplicates(subset='value', keep=False)
        manual_title_list = df_manual['variable'].unique()
        
        df_manual_combine = {'text': []}
        for title in manual_title_list:
            content = df_manual[df_manual['variable']==title]['value'].to_list()
            doc = " ".join(d for d in content)
            df_manual_combine['text'].append(doc)
        
        len(df_manual_combine['text'])
        
        df = pd.concat([df_ngaymoi, df_dantri])
        
        documents = df['text'].to_list() + df_manual_combine['text']
        documents = [doc.replace('\n', '').replace('\t', '').replace('\r', '')
                     for doc in documents]
        
        return documents
    
    def load_doc_sentiment(self):
        
        df_full = pd.DataFrame({
            'doc': self.load_documents()
            })
        
        df_class = pd.read_csv('/Users/dungnguyen/Desktop/Data Science off/Python Programming/3. Publication/DA15_silver_economy/data/database/doc_classification.csv')
        
        df_sentiment = pd.read_csv('/Users/dungnguyen/Desktop/Data Science off/Python Programming/3. Publication/DA15_silver_economy/data/database/doc_sentiment_classification.csv')
        
        df_full = df_full.join(df_class)
        df_full = df_full.join(df_sentiment)
        
        return df_full

    #%% OPENAI CLASSIFICATION
    
    def classify_gpt(self, csv_file_name):
        api_key = os.getenv("SECRETE_KEY")
    
        def classify_text(text):
        
            # labels = [
            #     "Chính sách của nhà nước với người cao tuổi"
            #     "Vai trò và công việc của người cao tuổi trong gia đình và xã hội",
            #     "Các thách thức, khó khăn, và nhu cầu của người cao tuổi",
            #     "Khác"]
            
            labels = [
                "Tích cực",
                "Tiêu cực",
                "Trung lập"]
            
            messages = [
                {"role": "system", 
                 "content": "Bạn là một mô hình phân loại văn bản. NCT là viết tắt của người cao tuổi."},
                {"role": "user", 
                 "content": f"""Cho văn bản sau: \" {text} \"
                Danh sách nhãn: {', '.join(labels)}
                Chỉ trả về một nhãn duy nhất. Không thêm kí tự nào.
                """}
                ]
            
            client = OpenAI(api_key = api_key)
            
            response = client.chat.completions.create( 
                model="gpt-4o-mini",
                messages = messages,
                temperature=0
                )
            
            return response.choices[0].message.content
    
        def classify_documents(documents):
            values = []
            
            for text in documents:
                topic = classify_text(text)
                values.append(topic)
                
            df = pd.Series(values, name='topics')
                
            return df
    
        doc_classified_df = classify_documents(self.load_documents())
        
        doc_classified_df.to_csv(csv_file_name)
        
        return doc_classified_df
    
    

    

    #%% UNSUPERVISED TOPIC DETECT W/T LATENT DIRECHLET ALLOCATION
    def grouping_lda(self):
        # Stop word
        with open("/Users/dungnguyen/Desktop/Data Science off/Python Programming/3. Publication/DA15_silver_economy/data/vietnamese-stopwords.txt", "r") as file:
            vn_stopwords = file.read().splitlines()
        
        vn_stopwords = [word.replace(' ', '_') for word in vn_stopwords]
        
        eng_stopwords = stopwords.words('english')
        
        eng_stopwords = [word.replace(' ', '_') for word in eng_stopwords]
        
        #%%    
        tokens = [  
            [word.lower() for word in ViTokenizer.tokenize(doc).split()
             if word.lower() not in vn_stopwords
             and word.lower() not in eng_stopwords
             and "." not in word
             ] for doc in self.load_documents()
            ]
        
        def keep_meaningful_word(word):
            if word.isdigit():
                return False
            
            if re.fullmatch(r'\W+', word):
                return False
            
            if all(char in string.punctuation for char in word):
                return False
            
            return True
        
        #%%
        tokens_updated = [
            [w for w in token if keep_meaningful_word(w)]
            for token in tokens
            ]
        
        all_words = []
        for token in tokens_updated:
            all_words.extend(token)
        
        counter_word = Counter(all_words)
        top_common_word = [
            word[0] for word in counter_word.most_common(int(0.05*len(counter_word)))
            ]
        top_10k_word = [
            word[0] for word in counter_word.most_common(
                int(0.05*len(counter_word)) + 10000
                )
            ]
        
        (pd.Series(top_common_word[:100])
         .rename('top_100_most_common')
         .to_excel('top_common_words.xlsx')
         )
        
        tokens_updated = [
            [word for word in token 
             if word not in top_common_word and word in top_10k_word] 
            for token in tokens_updated
            ]
        
        tokens_updated = [token for token in tokens_updated if len(token)>=1]
        
        
        dictionary = corpora.Dictionary(tokens_updated)
        
        corpus = [dictionary.doc2bow(text) for text in tokens_updated]
        
        num_topics = 15
        
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=30,
            random_state=42
            )
        
        topics = lda_model.print_topics(num_topics=num_topics, num_words=20)
        
        topics_dict = {"top_weighted_words": []}
        
        for topic in topics:
            topics_dict["top_weighted_words"].append(
                ", ".join(topic[1].split('"')[1::2])
                )
            
        df_topics = pd.DataFrame(topics_dict)
        
        
        #%%
        doc_topics = [lda_model.get_document_topics(bow) for bow in corpus]
        
        topic_interest = dict(zip(range(num_topics), [0]*num_topics))
        
        for doct in doc_topics:
            for i, prob in doct:
                topic_interest[i] += prob
        
        topic_interest = pd.DataFrame(
            {"prob": topic_interest.values()},
            index=topic_interest.keys()
            )
        
        topic_interest = topic_interest/ len(self.load_documents())
        
        result = topic_interest.join(df_topics)
        
        result.to_excel('topic_allocation.xlsx')
        
        result['topic_words'] = result['top_weighted_words'].apply(
            lambda x: ", ".join(x.split(',')[:4])
            )
        
        result = result.sort_values('prob', ascending=False)
        
        result = result.reset_index()
        
        result['topic_words'] = result.index.astype(str) + '. ' + result['topic_words']
        
        
        plt.figure(figsize=(6, 8), dpi=200)
        sns.barplot(
            x=result['prob']*100,
            y=result['topic_words'],
            orient='y'
            )
        plt.xlabel('Average proportion of topics [%]')
        plt.ylabel('Four most weighted words of the group')
        plt.show()
        
    #%% N-GRAM ANALYSIS
    def export_ngram_list(self, num_gram=2, num_top=50):
        text=" ".join(self.load_documents())
        
        result = {"word": [],
                  "frequency": []}
        
        tokens = text.lower().split()
        n_grams = list(ngrams(tokens, num_gram))
        counter_ngrams = Counter(n_grams)
        counter_ngrams_top = counter_ngrams.most_common(num_top)
        
        for i in range(num_top):
            word = " ".join(doc for doc in counter_ngrams_top[i][0])
            count = counter_ngrams_top[i][1]
            
            # Save in result
            result["word"].append(word)
            result["frequency"].append(count)
        
        df = pd.DataFrame(result)
        df.to_excel(f'{num_top} most common {num_gram}-gram words.xlsx')
        return df
    
    #%% EMBEDDING
  
    def count_tokens(self, text):
        return len(encoding.encode(text))
    
    def create_token_safe_batches(self, documents, max_tokens=8192): 
        batches = []
        current_batch = []
        current_tokens = 0
    
        for idx, doc in enumerate(documents):
            doc_tokens = self.count_tokens(doc)
    
            if doc_tokens > max_tokens:
                print(f"⚠️ Skipping document {idx} with {doc_tokens} tokens (too long)")
                continue
    
            if current_tokens + doc_tokens > max_tokens:
                batches.append(current_batch)
                current_batch = [doc]
                current_tokens = doc_tokens
            else:
                current_batch.append(doc)
                current_tokens += doc_tokens
    
        if current_batch:
            batches.append(current_batch)
    
        return batches

    def get_batch_openai_embedding(self, documents, file_name='embedding.npy'):
        safe_batches = self.create_token_safe_batches(documents)
        all_embeddings = []

        for i, batch in enumerate(safe_batches):
            try:
                print(f"Embedding batch {i+1} with {len(batch)} docs, approx {sum(self.count_tokens(d) for d in batch)} tokens")
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                embeddings = [e.embedding for e in response.data]
                
                # store or append your embeddings here
                all_embeddings.extend(embeddings)
                
            except Exception as e:
                print(f"❌ Error on batch {i+1}: {e}")
                
        print(f"Embedded {len(all_embeddings)} documents.")
        
        np.save(file_name, np.array(all_embeddings))
        
        print("Save embedding successfully!")
        
    def remove_long_docs(self, docs):
    
        MAX_TOKENS = 8192  

        # Keep only short enough documents
        filtered_docs = [doc for doc in docs if self.count_tokens(doc) <= MAX_TOKENS]
        
        print(f"✅ Filtered {len(docs) - len(filtered_docs)} overly long documents!")
        
        return filtered_docs
        
    def load_embedding(self, file_name):
        return np.load(file_name)
    
    def download_audio_yt(self, url, file_name='audio.m4a'):
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': file_name,
            'ffmpeg_location': '/opt/homebrew/bin/ffmpeg',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
                'preferredquality': '192',
            }],
            'quiet': False
        }
    
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    
    def transcribe_audio(self, file_path):
        audio_file = open(file_path, "rb")
        
        transcription = client.audio.transcriptions.create(
            model='gpt-4o-transcribe',
            file=audio_file
            )
        
        return transcription.text
