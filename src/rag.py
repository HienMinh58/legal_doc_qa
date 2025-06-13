from sentence_transformers import SentenceTransformer
from pymilvus import Collection
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import logging

logger = logging.getLogger(__name__)

# Khởi tạo tokenizer và mô hình BERT
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def retrieve_relevant_chunks(query, collection, top_k=3, max_tokens=700):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedding_model.encode([query])
    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["data"]
    )
    chunks = [hit.entity.get("data") for hit in results[0]]
    
    context = ""
    current_tokens = 0
    for chunk in chunks:
        tokens = tokenizer.encode(chunk, add_special_tokens=False)
        if current_tokens + len(tokens) <= max_tokens:
            context += chunk + "\n"
            current_tokens += len(tokens)
        else:
            break
    
    logger.debug(f"Retrieved context: {context}")
    return context.strip() if context.strip() else None

def answer_question(question, answer_text):
    if not answer_text or not question:
        return "Không có đủ thông tin để trả lời."
    
    logger.debug(f"Processing question: {question}, context: {answer_text}")
    # Chuẩn hóa câu hỏi
    if not any(char in question.lower() for char in ['what', 'where', 'when', 'how', 'why', 'is', 'are']):
        question = f"What is {question}?"
    
    # Tạo encoding với attention_mask
    encoding = tokenizer.encode_plus(
        question, answer_text,
        truncation=True, max_length=512,
        padding='max_length', return_tensors="pt"
    )
    input_ids = encoding['input_ids']
    token_type_ids = encoding['token_type_ids']
    attention_mask = encoding['attention_mask']
    
    logger.debug(f"Input IDs shape: {input_ids.shape}")
    logger.debug(f"Token Type IDs shape: {token_type_ids.shape}")
    logger.debug(f"Attention Mask shape: {attention_mask.shape}")

    # Dự đoán với attention_mask
    outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    logger.debug(f"Start scores shape: {start_scores.shape}, End scores shape: {end_scores.shape}")
    
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = tokens[answer_start]
    for i in range(answer_start + 1, answer_end + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]
    
    logger.debug(f"Extracted answer: {answer}")
    return answer if answer.strip() else "Không tìm thấy câu trả lời."

def rag_query(query, collection):
    relevant_context = retrieve_relevant_chunks(query, collection)
    if not relevant_context:
        return "Không tìm thấy thông tin phù hợp."
    answer = answer_question(query, relevant_context)
    return answer
