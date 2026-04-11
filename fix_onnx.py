import os
import onnx

model_path = r"c:\Users\samai\Desktop\codes backup\aiml\ani-flow-optimizer\models\screen_classifier.onnx"
print(f"Loading {model_path}...")

# Load the model and its external data
model = onnx.load(model_path, load_external_data=True)

# Delete existing files
os.remove(model_path)
data_path = model_path + ".data"
if os.path.exists(data_path):
    os.remove(data_path)

# Save as a single combined file
onnx.save(model, model_path, save_as_external_data=False)

print(f"✅ Model saved to single file. New size: {os.path.getsize(model_path)} bytes")
