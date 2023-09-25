import torch.onnx
import torchvision.models as models

def export_resnet50_to_onnx():
    # Load a pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)

    # Set the model to evaluation mode
    model.eval()

    # Define dummy input to the model. For ResNet-50, it's a tensor with shape [1, 3, 224, 224]
    x = torch.randn(1, 3, 224, 224)

    # Export the model to ONNX format
    torch.onnx.export(model, x, "resnet50.onnx")
    print("Model has been successfully exported to resnet50.onnx")

if __name__ == "__main__":
    export_resnet50_to_onnx()
