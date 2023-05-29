FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip3 install --upgrade pip
RUN SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip3 install -r requirements.txt

CMD ["python", "main.py"]
EXPOSE 8080