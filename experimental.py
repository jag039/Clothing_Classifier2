import cv2
import torch
import time
from PIL import Image
from models.resnet_clothing_model import ClothingClassifier
from data.transformation import CustomResNetTransform

def main():
    # Load model
    NUM_CLASSES = 50
    model = ClothingClassifier(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load('clothing_classifier.pth', map_location=torch.device('cpu')))
    model.eval()
    transform = CustomResNetTransform()
    
    # Open the default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    last_inference_time = 0
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Camera Feed', frame)
        
        current_time = time.time()
        if current_time - last_inference_time >= 60:
            last_inference_time = current_time
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            input_tensor = transform(pil_img)
            input_tensor = input_tensor.unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            print(f"Inference at {time.ctime(current_time)}: Predicted class = {predicted_class}")
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
