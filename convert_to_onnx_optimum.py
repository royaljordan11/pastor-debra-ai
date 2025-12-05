from optimum.exporters.onnx import main_export

main_export(
    
model_name_or_path="/Users/mj/Desktop/numerology_bot_training/my_model",
    output="/Users/mj/Desktop/numerology_bot_training/onnx",
    task="text-generation"
)

