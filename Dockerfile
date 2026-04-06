FROM python:3.11-slim

WORKDIR /app

COPY docker-requirement.txt requirements.txt 

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app.py .

EXPOSE 8003

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003" ]