FROM python:3.7.5-stretch
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["uwsgi", "app.ini"]
