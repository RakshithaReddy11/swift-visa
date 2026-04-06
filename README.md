# SwiftVisa - AI-Powered Visa Eligibility Checker

An intelligent visa eligibility assessment platform using RAG (Retrieval-Augmented Generation) and LLM technology to help users understand their visa eligibility across USA, UK, and Australia.

## Features

✅ **Multi-Country Support**: USA, UK, Australia visa assessments  
✅ **AI-Powered**: Uses Groq LLM (llama-3.1-8b-instant) with fallback models  
✅ **RAG Pipeline**: Retrieves relevant policy documents from ChromaDB vector database  
✅ **Structured Assessment**: Eligibility scoring, checklists, and recommendations  
✅ **Interactive UI**: Streamlit-based user interface  
✅ **Multiple Visa Types**: Work, Student, Tourist, Business, Dependent visas  
✅ **Localized Currency**: INR with USD conversion support

---

## Setup Instructions

### 1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/swift-visa.git
cd swift-visa
```

### 2. **Create Virtual Environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Configure Environment Variables**
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your Groq API key
# Get your API key from: https://console.groq.com
```
Edit `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.1-8b-instant
```

### 5. **Vector Database (Pre-built & Included)**
The project includes a pre-built ChromaDB vector database (`backend/chroma_db/`) with all visa policy embeddings.

**For production deployment** → No additional setup needed!

**For local development** (if you want to add new documents):
- Raw visa policy PDFs are not included in the git repo (to keep deployment size small)
- To rebuild the database locally, add your PDF files to `backend/data/{USA,uk_visa,AUSTRALIA}/` and run:
```bash
python backend/store_dataset.py      # USA & Australia
python backend/store_uk_pdf.py       # UK
```

### 5. **Run the Application**
```bash
streamlit run backend/app.py
```

The app will start on `http://localhost:8501`

---

## Project Structure

```
swift-visa/
├── backend/
│   ├── app.py                    # Streamlit UI (entry point)
│   ├── rag_chatbot.py            # RAG pipeline & LLM logic
│   ├── eligibility_llm.py        # Eligibility checklist logic
│   ├── visa_graph.py             # Visa pathway graph logic
│   ├── graph_app.py              # Graph visualization
│   │
│   ├── chroma_db/                # ✅ Vector database (INCLUDED in deployment)
│   │                             # Contains prebuilt embeddings of all visa policies
│   │
│   └── [Local Development Only - not committed]
│       ├── data/                 # Raw policy PDFs (for building database)
│       ├── store_dataset.py      # Script to ingest USA/Australia PDFs
│       └── store_uk_pdf.py       # Script to ingest UK PDFs
│
├── .env.example                  # Environment template (configure with Groq API key)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Version control excludes
└── README.md                     # This file
```

### Why This Structure?
- **chroma_db/** is included because it contains the precomputed vector database needed at runtime
- **data/** folder is excluded because it contains raw PDFs (~100-500MB) only needed for initial setup
- Scripts like `store_dataset.py` are for local development, not needed in production

---

## How It Works

1. **User Input**: User provides visa details (country, visa type, qualifications)
2. **Question Expansion**: Query is expanded into multiple semantic variations
3. **Vector Search**: ChromaDB retrieves relevant policy documents (~top 5)
4. **Keyword Search**: Additional matching via keyword patterns
5. **LLM Assessment**: Groq LLM analyzes user profile against policy context
6. **Structured Output**: Returns:
   - Eligibility Score (0-100%)
   - Decision (Likely/Maybe/Unlikely)
   - What you have vs. what's needed
   - Next steps & visa suggestions

---

## Configuration

### LLM provider
Configured to use **Groq** with fallback models:
- Primary: `llama-3.1-8b-instant`
- Fallback: `llama-3.3-70b-versatile`

To switch models, edit `GROQ_MODEL` in `.env`.

### Currency
- **Default**: INR (Indian Rupees)
- **Conversion**: USD to INR at 83:1 rate
- Change in `backend/app.py` → `DEFAULT_CURRENCY` & `USD_TO_INR_RATE`

### Database
- **Type**: ChromaDB (vector DB)
- **Path**: `backend/chroma_db/`
- **Collections**: `us_visa_collection`, `uk_visa_collection`, `australia_collection`

---

## Troubleshooting

### "Groq SDK not installed"
```bash
pip install groq==0.9.0
```

### Streamlit port 8501 already in use
```bash
streamlit run backend/app.py --server.port=8502
```

### ChromaDB collections not found
Run the data ingestion scripts:
```bash
python backend/store_dataset.py
python backend/store_uk_pdf.py
```

### API quota exceeded
The app falls back to an offline heuristic answer. Wait and retry after quota resets.

---

## API Keys & Security

⚠️ **Never commit `.env` file** — it contains secrets!

1. Get your **Groq API key** from [console.groq.com](https://console.groq.com)
2. Create `.env` locally (using `.env.example` as template)
3. Add to `.gitignore` (already done)

---

## Future Enhancements

- [ ] Database backup/recovery procedures
- [ ] Multi-language support
- [ ] Advanced visa comparison tool
- [ ] Interview preparation module
- [ ] Document checklist generator
- [ ] Real-time policy updates

---

## License

This project is licensed under the MIT License.

---

## Contact & Support

For issues, questions, or contributions, please open a GitHub issue or contact the team.

---

## Development Notes

**Python Version**: 3.11+  
**Primary Framework**: Streamlit  
**Backend**: LangChain + Groq LLM  
**Vector DB**: ChromaDB  
**Policy Source**: Official government visa documents

---

**Last Updated**: April 2026  
**Status**: Production Ready ✅
