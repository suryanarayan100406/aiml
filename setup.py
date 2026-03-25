from setuptools import setup, find_packages

setup(
    name="ani-flow-optimizer",
    version="1.0.0",
    description="ANI Creative Flow Optimizer — Multimodal AI Cognitive State Classifier",
    author="ANI Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "xgboost>=2.0.0",
        "transformers>=4.38.0",
        "torch>=2.2.0",
        "datasets>=2.18.0",
        "scikit-learn>=1.4.0",
        "librosa>=0.10.1",
        "skl2onnx>=1.16.0",
        "onnxruntime>=1.17.0",
        "shap>=0.44.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "joblib>=1.3.2",
        "Pillow>=10.2.0",
        "tqdm>=4.66.0",
    ],
)
