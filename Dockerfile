FROM python:3.10-slim

WORKDIR /IE7374-Group6

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy source code and other directories
COPY src/ ./src/
COPY data/ ./data/
COPY utils/ ./utils/
COPY configs/ ./configs/
COPY outputs/ ./outputs/
COPY app.py .
COPY templates/ ./templates/
COPY static/ ./static/

ENV PYTHONPATH=/IE7374-Group6

CMD ["python", "src/model_runner.py"]

# Expose port for Flask app
EXPOSE 5001
# Use ENTRYPOINT and CMD for flexibility
ENTRYPOINT ["python"]
CMD ["app.py"]
