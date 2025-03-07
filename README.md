# RAG in Harmonized System Codes - Based Goods Classification

## 📌 Giới thiệu

Dự án này áp dụng **Retrieval-Augmented Generation (RAG)** để tự động phân loại hàng hóa dựa trên **Harmonized System (HS) Codes**. Đây là một vấn đề quan trọng trong thương mại quốc tế, giúp tự động hóa quy trình phân loại hàng hóa, giảm thiểu sai sót và tăng hiệu suất làm việc.

## 📑 Nội dung chính
- **Hệ thống Mã HS** và các quy tắc diễn giải.
- **Retrieval-Augmented Generation (RAG)** và ứng dụng của nó.
- **Các phương pháp xử lý dữ liệu**, bao gồm chunking, embedding, và truy xuất.
- **Cài đặt hệ thống và đánh giá kết quả**.

---

## ⚙️ Cài đặt

### Yêu cầu hệ thống:
- Python 3.8+
- Các thư viện:
  ```bash
  pip install langchain textgrad bm25s[full] faiss-cpu
  ```
- API mô hình ngôn ngữ lớn (LLM) từ OpenAI hoặc Huggingface.

### Cấu trúc thư mục:
```
├── data/                  # Dữ liệu đầu vào
├── models/                # Mô hình đã huấn luyện
├── src/                   # Mã nguồn chính
│   ├── preprocess.py      # Xử lý dữ liệu
│   ├── retrieval.py       # Truy xuất thông tin
│   ├── generation.py      # Tạo phản hồi
│   ├── evaluation.py      # Đánh giá mô hình
├── results/               # Kết quả thực nghiệm
├── README.md              # Tài liệu dự án
```

---

## 🚀 Hướng dẫn sử dụng

### 1️⃣ Xử lý dữ liệu
```bash
python src/preprocess.py --input data/raw_data.csv --output data/processed_data.csv
```

### 2️⃣ Truy xuất thông tin
```bash
python src/retrieval.py --query "Mô tả sản phẩm cần phân loại"
```

### 3️⃣ Sinh mã HS bằng RAG
```bash
python src/generation.py --input "Tên sản phẩm cần phân loại"
```

### 4️⃣ Đánh giá mô hình
```bash
python src/evaluation.py --model "gemma2-9b"
```

---

## 📊 Kết quả
| Model           | Accuracy | F1   | Recall | Precision |
|----------------|----------|------|--------|-----------|
| gemma2-9b     | 0.89     | 0.822 | 0.825  | 0.820     |
| llama3.1-70b  | 0.77     | 0.624 | 0.628  | 0.624     |
| llama3.3-70b  | 0.89     | 0.803 | 0.810  | 0.800     |

---

## 🔥 Hướng phát triển
- Kết hợp **BM25 và embedding models** để cải thiện truy xuất thông tin.
- **Fine-tune mô hình RAG** trên tập dữ liệu chuyên biệt.
- **Tối ưu hóa hiệu suất**, giảm độ trễ khi truy vấn dữ liệu.
- Mở rộng bộ dữ liệu **để phù hợp với nhiều loại sản phẩm hơn**.

📌 **Contributors:** Đặng Ngọc Hưng, Lê Thị Mỹ Tiên  
📌 **Advisor:** TS. Ngô Minh Mẫn
