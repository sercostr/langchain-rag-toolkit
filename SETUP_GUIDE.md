# Setup Guide

## Prerequisites

- Python 3.9 or higher
- pip or conda
- OpenAI API account (or Anthropic)

## Step 1: Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or on Windows: venv\Scripts\activate
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Set Up API Keys

### Get OpenAI API Key

1. Go to https://platform.openai.com/
2. Sign up or log in
3. Go to API Keys section
4. Create a new key
5. Copy the key (you won't see it again!)

### Create .env File

```bash
cp .env.example .env
```

Edit `.env` and add your key:

```
OPENAI_API_KEY=sk-your-actual-key-here
```

## Step 4: Test Installation

```bash
python 01_hello_langchain.py
```

If you see output, you're ready to go! 🎉

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution**: Make sure virtual environment is activated

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: OpenAI API Error

**Solution**: Check your API key in .env file

```bash
# Test your key
python -c "from openai import OpenAI; client=OpenAI(); print('API key works!')"
```

### Issue: Rate Limit Error

**Solution**: You might need to add credits to your OpenAI account

- Go to https://platform.openai.com/account/billing
- Add $5-10 to start (you won't use much for learning)

## Cost Estimates

For learning with these scripts:

- **GPT-4o-mini**: ~$0.10 per million tokens
- **Embeddings**: ~$0.02 per million tokens
- **Expected cost**: $1-5 for completing all tutorials

## Optional: Set Up LangSmith (Recommended)

LangSmith provides tracing and debugging (like observability tools for AI):

1. Sign up at https://smith.langchain.com/
2. Create an API key
3. Add to .env:

```
LANGCHAIN_API_KEY=your-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=langchain-rag-toolkit
```

## Next Steps

Once setup is complete:

1. Read `RECOMMENDATIONS.md` for learning path
2. Start with `01_hello_langchain.py`
3. Progress through the numbered files
4. Document your learning in `LEARNING_LOG.md`

Happy learning! 🚀
