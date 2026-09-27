FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libatspi2.0-0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libcups2 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Playwright on newer Debian variants may require either libasound2 or libasound2t64.
RUN apt-get update && (apt-get install -y --no-install-recommends libasound2 || apt-get install -y --no-install-recommends libasound2t64) \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY README.md LICENSE ./
COPY docseek/ docseek/

RUN pip install --no-cache-dir -e . \
    && playwright install chromium

EXPOSE 8010

CMD ["uvicorn", "docseek.server:app", "--host", "0.0.0.0", "--port", "8010"]
