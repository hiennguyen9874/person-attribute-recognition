# Deploy person attribute recognition with [torchserve](https://github.com/pytorch/serve)

- copy and set path of pre-trained model in [model.py](model.py)
- set attribute path
- Create model as torchscript: `python3 model.py`
- Create a directory to store your models: `mkdir model_store`
- Archive the model by using the model archiver: `torch-model-archiver --model-name eager_model --version 1.0 --serialized-file eager_model.pt --handler handler.py --export-path model_store -f`
- Set python path (which python3) in file config.properties.
- Start TorchServe to serve the model: `torchserve --start --ncs --model-store model_store --ts-config config.properties`
