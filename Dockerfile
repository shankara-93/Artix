# Dockerfile
FROM python:3.9-slim-buster

WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./app ./app
COPY ./backend ./backend
COPY ./config.py .
COPY ./model ./model  # Copy the entire 'model' directory

# Set the model path as an environment variable (optional but good practice)
ENV MODEL_PATH=/app/model/best.pt

# Command to run the application.  Important:  Use 'python -m uvicorn'
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]