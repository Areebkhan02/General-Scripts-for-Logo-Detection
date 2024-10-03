import torch

def load_model(file_path):
    # Load the model
    try:
        model = torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)
    except Exception as e:
        print(f"Failed to load model with weights_only=True due to: {e}")
        print("Attempting to load the model without weights_only...")
        model = torch.load(file_path, map_location=torch.device('cpu'))

    # Check if it's a state_dict
    model_architecture = None
    if isinstance(model, dict) and 'state_dict' in model:
        print("Model is loaded as a state_dict.")
        model_architecture = model.get('model', None)
        if model_architecture:
            #print(f"Model architecture: {model_architecture}")
            print("Loading model from state_dict...")
            model_instance = model_architecture()
            model_instance.load_state_dict(model['state_dict'])
            model = model_instance
        else:
            print("Loaded state_dict only. Model architecture not available in the file.")
            return model, model_architecture

    return model, model_architecture

def print_model_info(model):
    # Print the model architecture
    print("Model Architecture:")
    print(model)
    
    # Attempt to extract class names (if they exist)
    class_names = None
    if hasattr(model, 'classes'):
        class_names = model.names
    elif isinstance(model, dict):
        class_names = model.get('classes', None)
    
    if class_names:
        print("\nClass Names:")
        for idx, class_name in enumerate(class_names):
            print(f"{idx}: {class_name}")
    else:
        print("\nClass names are not available in the model.")

def main(file_path):
    model, model_architecture = load_model(file_path)
    if model:
        print_model_info(model)

if __name__ == "__main__":
    # Provide the path to your .pt file here
    file_path = "/home/areebadnan/Areeb_code/work/Atheritia/All_models/Large/14layers_freeze_3heads_merged_retrain.pt"
    main(file_path)


