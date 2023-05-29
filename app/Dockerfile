FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install -r requirements.txt

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
CMD ["python", "main.py"]