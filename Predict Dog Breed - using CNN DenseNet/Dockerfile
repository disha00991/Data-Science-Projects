FROM python:3.6-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./densenet_model.pkl /deploy/
COPY ./imagenet_classes.txt /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]