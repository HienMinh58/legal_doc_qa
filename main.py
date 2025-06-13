from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import logging
import sys
import tempfile

# Thêm đường dẫn src vào sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_processing import pdf_to_images, process_and_chunk
from embedding import generate_embeddings, read_chunk_files, save_to_milvus
from rag import rag_query

from pymilvus import connections, Collection
import shutil

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Legal Doc QA API", description="API for uploading PDF and querying legal documents with BERT")

# Khởi tạo kết nối Milvus
connections.connect(host='localhost', port='19530')

# Biến toàn cục để lưu collection
collection = None

def initialize_collection():
    global collection
    try:
        collection = Collection("chunked_legal_vectors")
        collection.load()
    except Exception as e:
        logger.error(f"Lỗi tải collection: {e}")

initialize_collection()

# Phục vụ file tĩnh
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    pdf_path = None
    temp_pdf_path = None
    try:
        base_dir = os.path.dirname(__file__)
        logger.debug(f"Base directory: {base_dir}")
        pdf_dir = os.path.join(base_dir, "data/input_pdfs")
        logger.debug(f"PDF directory: {pdf_dir}")
        os.makedirs(pdf_dir, exist_ok=True)
        
        safe_filename = file.filename.replace(" ", "_")
        pdf_path = os.path.join(pdf_dir, safe_filename)
        logger.debug(f"PDF path: {pdf_path}")
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_pdf_path = temp_file.name
            logger.debug(f"Temporary PDF path: {temp_pdf_path}")
            file.file.seek(0)
            file_content = file.file.read()
            logger.debug(f"File content length: {len(file_content)} bytes")
            temp_file.write(file_content)
        
        shutil.copy2(temp_pdf_path, pdf_path)
        logger.debug(f"Copied to {pdf_path}, size: {os.path.getsize(pdf_path)} bytes")
        
        if not os.path.isfile(pdf_path):
            raise Exception(f"File {pdf_path} không được tạo sau khi sao chép.")
        
        image_dir = os.path.join(base_dir, "data/pdf_images")
        logger.debug(f"Image directory: {image_dir}")
        image_files = pdf_to_images(pdf_path, image_dir)
        
        text_dir = os.path.join(base_dir, "data/extracted_text")
        chunk_dir = os.path.join(base_dir, "data/chunked_text")
        logger.debug(f"Text directory: {text_dir}, Chunk directory: {chunk_dir}")
        chunked_texts = process_and_chunk(image_dir, text_dir, chunk_dir, max_words=100, overlap_sentences=1)
        
        chunks = read_chunk_files(chunk_dir)
        embeddings = generate_embeddings(chunks)
        save_to_milvus(chunks, embeddings)
        
        initialize_collection()
        
        return JSONResponse(content={"message": f"File {safe_filename} đã được xử lý và lưu thành công."})
    except Exception as e:
        logger.error(f"Error in upload_file: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        for path in [temp_pdf_path, pdf_path]:
            if path and os.path.exists(path):
                logger.debug(f"Removing file: {path}")
                os.remove(path)

@app.get("/query/")
async def query_data(keyword: str):
    if not collection:
        return JSONResponse(content={"error": "Collection chưa được khởi tạo."}, status_code=500)
    
    try:
        answer = rag_query(keyword, collection)
        return JSONResponse(content={"query": keyword, "answer": answer})
    except Exception as e:
        logger.error(f"Error in query_data: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
