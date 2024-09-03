FROM python:3.12.3
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 8000
ENV NAME .venv
CMD ["python3", "main.py"]
