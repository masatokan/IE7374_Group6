FROM python:3.10-slim

WORKDIR /IE7374-Group6

COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
libgl1-mesa-glx \
libglib2.0-0 \
&& rm -rf /var/lib/apt/lists/*

COPY src/ src/ COPY utils/ utils/ COPY configs/ configs/

ENV PYTHONPATH=/IE7374-Group6

CMD ["python", "src/model_runner.py"]