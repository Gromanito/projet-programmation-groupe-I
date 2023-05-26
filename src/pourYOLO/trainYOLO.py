from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
#les classes sont : 
model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

# Train the model (obligé de mettre le path complet, c'est chiant mais c'est comme ça)
model.train(data='/home/romain/Perso/Cours/L3_deuxieme_semestre/projet-Programmation/TestYolo/upperSansBord/', epochs=3, imgsz=128)




