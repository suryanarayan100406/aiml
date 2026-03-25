import os
import joblib
import numpy as np
from onnxmltools.convert import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType

def export():
    model_path = "models/speech_classifier.pkl"
    onnx_path = "models/speech_classifier.onnx"
    
    print(f"Loading {model_path}...")
    model = joblib.load(model_path)
    
    print("Converting to ONNX using onnxmltools...")
    initial_type = [('input', FloatTensorType([None, 52]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)
    
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print(f"✅ Exported successfully to {onnx_path}")

if __name__ == "__main__":
    export()
