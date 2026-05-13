FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    # Tell HuggingFace to store downloaded models here
    HF_HOME=/home/user/app/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/home/user/app/hf_cache

WORKDIR $HOME/app

# Copy files and set correct permissions
COPY --chown=user . $HOME/app

# Create cache directories explicitly to ensure correct permissions
RUN mkdir -p $HOME/app/cache && mkdir -p $HOME/app/hf_cache

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Expose Hugging Face Space default port
EXPOSE 7860

# Run FastAPI using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
