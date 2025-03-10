# Information Retrieval Project - Personalized News Headline Generation using LLM-Based RAG Pipeline

## **Project Overview**
This project explores the role of linguistic style in personalizing Large Language Models (LLMs) and aims to bridge the gap between the universal capabilities of LLMs and the rising demand for individualized interactions. The core idea is to generate personalized news headlines by combining content-dependent and content-independent style representations.

---

## **Dataset: LAMP-4**
The LAMP-4 dataset was used for training and evaluation. 

---

## **Methodology**
### **Approach**
Our approach consists of the following key components:  

### **Pipeline 1: Dense Retrieval Using FAISS**  
1. **Query Variants Generation**  
   - The original query is used to generate k variations using the LLM - Gemini Flash 1.5.
     
2. **Dense Retrieval & FAISS Indexing**  
   - The expanded queries are used to retrieve the top m semantically similar documents using FAISS indexing.  

### **Pipeline 2: Style Embedding-Based Retrieval**  
1. **Style Embeddings for Personalization**  
   - Wegmann et al. (2022)’s model is used to extract style embeddings from the user’s past authored documents.  
   - The average embedding is computed to capture the user’s overall stylistic tendencies.
   - Ranks retrieved documents based on cosine similarity to the author’s style embedding.
  
### **Hybrid Model for Personalization**  
1. **Augmenting Query Context**  
   - A union of dense retrieval and style-based ranking is used to generate the final personalized headline.
   - The top k retrieved documents (from both pipelines) are appended to the original query, enriching the input with additional context. 
   - This union-based ranking helps personalize news headlines without losing contextual accuracy.

2. **Headline Generation**  
  - The Flan-T5 Base Model generates the final personalized headline, ensuring it aligns with both the retrieved semantic content and the user’s unique writing style.

---

## Key Findings
- **✅ Hybrid approach** (content + style embeddings) achieves the best performance in both ROUGE-1 score of 0.0985 and ROUGE-L score of 0.0894.  
- **✅ BM25** remains a strong baseline for text retrieval and headline generation.  
- **✅ Dense Search & Style Embeddings alone** perform worse than hybrid models.  
 

---

## **Conclusions**
✔ **Hybrid Model Effectiveness** – Combining content-based and style-based methods enhances personalization. Encoding linguistic style improves the quality of generated headlines.  
✔ **Traditional IR Methods Still Work** – BM25 remains a competitive retrieval technique for LLM-based text generation.  

---

## **Future Work**
🚀 **Refining Sentence Embeddings** – Improve how style embeddings are extracted from document texts.  
🚀 **Exploring Alternative LLMs** – Evaluate different models for improved style adaptation.  

---

## **Installation & Usage**
### **Requirements**
- Python 3.8+
- Hugging Face Transformers
- FAISS
- NLTK
- PyTorch
- Scikit-learn
