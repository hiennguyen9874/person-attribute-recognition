FROM pytorch/torchserve:0.2.0-cpu

COPY eager_model.pt eager_model.pt

COPY handler.py handler.py

COPY config.properties config.properties

RUN mkdir model_store && \
    torch-model-archiver --model-name eager_model \
    --version 1.0 \
    --serialized-file eager_model.pt \
    --handler handler.py \
    --export-path model_store \
    -f

EXPOSE 8080 8081

CMD ["torchserve", \
    "--start",\
    "--ncs", \
    "--model-store", "model_store",\
    "--ts-config", "config.properties"]
