# 🧠 ARIS – AI Research Intelligence System

🚀 Revolutionizing how researchers explore, analyze, and synthesize knowledge.  
Built for the **CodeMate Hackathon** using Python, Streamlit, and AI-powered NLP.

---

## 🌟 Unique Selling Points (USP)

- **Advanced Document Intelligence**  
  Entity extraction, methodology detection, semantic structure parsing, and automated quality scoring.

- **Knowledge Graph Construction**  
  Dynamic, weighted concept mapping with research gap detection and author collaboration networks.

- **Systematic Literature Analysis**  
  AI-powered thematic clustering, temporal trend mapping, and methodological distribution tracking.

- **Research Insights Engine**  
  Actionable intelligence for future research directions and methodology recommendations.

- **Interactive Dashboards**  
  Visual analytics of trends, impact scores, networks, and concept evolution.

- **Production-Grade Design**  
  Modular architecture, model switching, system configuration tools, database optimization, and backup support.

---

## 🏗️ Project Structure

```

root/
│
├── app.py                 # Streamlit UI
├── main.py                # Entrypoint wrapper (for platforms using `python main.py`)
├── research\_agent.py       # Orchestrator: embeddings, retrieval, reasoning
├── requirements.txt        # Pinned dependencies
├── analysis/               # Specialized analysis modules
│   ├── document\_processing.py
│   ├── impact\_scoring.py
│   ├── knowledge\_graph.py
│   ├── literature\_review\.py
│   └── trends.py
└── utils/                  # Support code
├── file\_readers.py
├── visualization.py
└── config.py

````

---

## ⚙️ Installation

Clone and set up the environment:

```bash
git clone https://github.com/your-username/aris.git
cd aris

# Create environment
conda create -n aris python=3.10 -y
conda activate aris

# Install dependencies
pip install -r requirements.txt

# If using spaCy NLP
python -m spacy download en_core_web_sm
````

### Core Dependencies

* `streamlit` – web UI
* `sentence-transformers` – embeddings
* `faiss-cpu` (or `faiss-gpu`) – vector search
* `transformers` – local LLM reasoning
* `pypdf`, `python-docx` – document parsing
* `spacy`, `textstat` – NLP & readability metrics
* `networkx`, `pyvis` – knowledge graphs
* `matplotlib`, `plotly`, `pandas` – visualizations
* `reportlab` – PDF export

---

## 🚀 Usage

### Run the app

```bash
# Local development
streamlit run app.py

# On CodeMate platform
python main.py
```

### Workflow

1.  Upload research documents (`.pdf`, `.docx`, `.txt`, `.md`, `.csv`).
2. The system automatically:

   * Extracts semantic sections (abstract, methodology, results, limitations).
   * Generates embeddings and indexes locally with FAISS.
   * Builds knowledge graphs, impact scores, and trend maps.
   * Performs retrieval + reasoning using local NLP models.
3. Explore results in interactive dashboards.
4. Export findings as PDF/Markdown research reports.

---


## 📂 Document Ingestion Supported Formats

* `.txt`, `.md`, `.csv`, `.json`
* `.pdf` (via PyPDF2 / pypdf)
* `.docx` (via python-docx)

Large PDFs are chunked automatically to prevent spaCy memory errors.

---

## 🧪 Extensibility

* **Swap embedding model**

  ```python
  from sentence_transformers import SentenceTransformer
  embedder = SentenceTransformer("all-mpnet-base-v2")
  ```

* **Swap reasoning model**

  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
  model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct")
  ```

* **Add new analysis**: create a module in `analysis/` and hook it into `research_agent.py`.

---

## 🌐 Hosting

### Streamlit Community Cloud

- Push repo to GitHub.
- Deploy via [streamlit.io/cloud](https://streamlit.io/cloud).
- Set entrypoint to `app.py`.
- Every GitHub push redeploys automatically.

### Self-host (EC2 / VPS / Docker)

Run with:

```bash
streamlit run app.py
```

Use `pm2`, `systemd`, or Docker to keep the service alive.

---
## 🛠️ Troubleshooting

* **spaCy error (text too long)** → increase `nlp.max_length` or chunk input.
* **Large model download slow** → switch to smaller embeddings (`all-mpnet-base-v2`).
* **Scanned PDFs** → require OCR (`pytesseract + pdf2image`).
* **CodeMate IDE issues** → remove heavy folders before upload.

---

## 📜 License

MIT License – free to use and extend with credit.

---

## 👨‍💻 Authors

Team ARIS – CodeMate Hackathon 2025