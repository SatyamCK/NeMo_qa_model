import nemo.collections.nlp as nemo_nlp

# List all available pre-trained models in NeMo
available_models = nemo_nlp.models.QAModel.list_available_models()

# Print the available models
for model in available_models:
    print(f"Model Name: {model.pretrained_model_name}")
