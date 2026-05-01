# 📄 RAG-Based PDF Question Answering System
### Week 3 Assignment | Mini Project

---

## 1. Problem Statement

Traditional AI chatbots answer from their general training data, which means they **cannot answer questions about your specific documents**. This project solves that problem by building a **RAG (Retrieval-Augmented Generation) system** that:

- Takes any PDF document as input
- Understands and indexes its content
- Answers user questions **strictly based on that PDF**
- Responds with *"I could not find this in the document."* if the answer isn't there

This makes the system accurate, document-specific, and trustworthy — unlike a general chatbot.

---

## 2. Tools & Technologies Used

| Tool / Library | Purpose |
|---|---|
| **Python** | Core programming language |
| **Streamlit** | Web interface / frontend |
| **Gemini API** | Embeddings + Answer generation (LLM) |
| **FAISS** | Vector database for similarity search |
| **LangChain** | RAG pipeline orchestration |
| **pypdf** | PDF text extraction |
| **python-dotenv** | Load API key from `.env` file |

---

## 3. Folder Structure

```
rag-pdf-qa/
│
├── app.py               ← Main Streamlit application (all RAG logic here)
├── requirements.txt     ← All Python packages needed
├── .env.example         ← Template for your API key (copy → .env)
├── .env                 ← Your actual API key (DO NOT share or commit!)
├── .gitignore           ← Files Git should ignore
└── README.md            ← This file
```

---

## 4. How the RAG System Works

RAG stands for **Retrieval-Augmented Generation**. It has two key phases:

```
┌─────────────────────────────────────────────────────────────────┐
│                     INDEXING PHASE (on upload)                  │
│                                                                 │
│  PDF Upload → Extract Text → Split into Chunks                  │
│       → Gemini Embeddings → Store in FAISS                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    QUERYING PHASE (on question)                 │
│                                                                 │
│  User Question → Convert to Embedding → Search FAISS           │
│       → Retrieve Top Chunks → Send to Gemini LLM               │
│       → Generate Answer (from context only)                     │
└─────────────────────────────────────────────────────────────────┘
```

**Step-by-step:**
1. User uploads a PDF
2. `pypdf` extracts all text from the PDF
3. `LangChain` splits the text into chunks of ~1000 characters (with 200-char overlap)
4. `Gemini Embeddings` converts each chunk into a vector (list of numbers)
5. `FAISS` stores all these vectors in an in-memory vector database
6. User types a question
7. The question is also converted into a vector using Gemini Embeddings
8. `FAISS` finds the 4 most similar chunks (by cosine similarity)
9. The retrieved chunks + question are sent to `Gemini LLM`
10. Gemini generates an answer **only from the retrieved context**

---

## 5. Setup Steps for Windows PowerShell

### Step 1 — Open PowerShell and navigate to your project folder

```powershell
cd "C:\Users\Hp\OneDrive\Desktop\Projects\Rag project\rag-pdf-qa"
```

### Step 2 — Create a virtual environment

```powershell
python -m venv venv
```

### Step 3 — Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

> ⚠️ If you get an error about execution policy, first run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then re-run the Activate command above.

### Step 4 — Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 5 — Install all required packages

```powershell
python -m pip install -r requirements.txt
```

---

## 6. How to Add Your Gemini API Key

### Get your free Gemini API Key:
1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key

### Add it to your project:
1. In your project folder, copy `.env.example` and rename the copy to `.env`
2. Open `.env` in VS Code
3. Replace `your_gemini_api_key_here` with your actual key:

```
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXX
```

> ⚠️ **Important:** Never share your `.env` file or upload it to GitHub. It's already in `.gitignore` for protection.

---

## 7. How to Run the App

Make sure your virtual environment is activated, then run:

```powershell
python -m streamlit run app.py
```

The app will open automatically in your browser at:
```
http://localhost:8501
```

---

## 8. Sample Inputs and Outputs

### Sample Input 1
- **PDF Uploaded:** A document about climate change
- **Question:** *"What are the main causes of climate change?"*
- **Output:** *"According to the document, the main causes of climate change are greenhouse gas emissions from burning fossil fuels, deforestation, and industrial processes..."*

### Sample Input 2
- **PDF Uploaded:** Same climate change document
- **Question:** *"What is the capital of France?"*
- **Output:** *"I could not find this in the document."*
(✅ Correctly refuses to answer from outside knowledge)

### Sample Input 3
- **PDF Uploaded:** A college syllabus PDF
- **Question:** *"How many credits does this course have?"*
- **Output:** Answers based on the exact credit information found in the syllabus

---

## 9. Viva Explanation

### ❓ Where is RAG used?
RAG is the **entire architecture** of this project. Instead of asking Gemini a question directly (which would use general AI knowledge), we first **retrieve relevant sections from the PDF** (using FAISS), and then **generate an answer based only on those sections** (using Gemini). The combination of retrieval + generation = RAG.

### ❓ Where is the Gemini API used?
Gemini API is used in **two places**:
1. **Embeddings** — `GoogleGenerativeAIEmbeddings` converts text chunks and questions into vectors using the `embedding-001` model
2. **Answer Generation** — `ChatGoogleGenerativeAI` uses the `gemini-1.5-flash` model to read the retrieved context and generate a natural language answer

### ❓ Where is FAISS used?
FAISS (Facebook AI Similarity Search) is used as the **vector database**:
- After embedding all PDF chunks, they are stored in FAISS using `FAISS.from_texts()`
- When the user asks a question, FAISS performs a **similarity search** to find the 4 most relevant chunks using `as_retriever(search_kwargs={"k": 4})`

### ❓ How does the system answer only from the uploaded PDF?
This is enforced by the **Prompt Template** in `generate_answer()`. The prompt explicitly instructs Gemini:
> *"Answer the question using ONLY the context provided below. Do not use any outside knowledge. If the answer is not present in the context, respond with exactly: 'I could not find this in the document.'"*

Since the only context given to Gemini is the retrieved PDF chunks (not the entire internet), the model can only answer from what's in the document.

---

## 10. How to Create the Project Folder in VS Code

### Manual Setup in VS Code:

1. **Open VS Code**

2. **Open the Terminal** → Menu: `Terminal` → `New Terminal`

3. **Navigate to your Projects folder:**
   ```powershell
   cd "C:\Users\Hp\OneDrive\Desktop\Projects\Rag project"
   ```

4. **Create the project folder:**
   ```powershell
   mkdir rag-pdf-qa
   cd rag-pdf-qa
   ```

5. **Open the folder in VS Code:**
   ```powershell
   code .
   ```

6. **Create each file** by right-clicking in the VS Code Explorer (left panel) → `New File`:

   | File to Create | Content to Paste |
   |---|---|
   | `app.py` | The entire Python code from `app.py` |
   | `requirements.txt` | The package list |
   | `.env.example` | The API key template |
   | `.env` | Your actual API key (copy from `.env.example` and fill in) |
   | `.gitignore` | The gitignore content |
   | `README.md` | This README content |

7. **Paste the content** into each file and save with `Ctrl + S`

---

*Built for Week 3 Assignment | RAG Mini Project*
