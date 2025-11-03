# ai_learning_farheen

## Streamlit RAG Demo

This project now includes a minimal retrieval-augmented generation (RAG) app that answers questions about `src/sales_data.csv`.

### Quick start

1. Create/activate your virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Add your OpenAI credentials to a `.env` file (the template in the repo already expects `OPENAI_API_KEY` and `LLM_MODEL`).
3. Launch the Streamlit UI:
   ```bash
   streamlit run streamlit_app.py
   ```

The interface allows you to tweak the underlying model names and explore the retrieved context used to answer each question.
