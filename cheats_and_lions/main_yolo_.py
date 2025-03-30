"""

-requeriments.txt:
torch
torchvision
torchaudio
matplotlib
opencv-python
pycocotools
git+https://github.com/ultralytics/yolov5.git


-commands:
 pip install -r requirements.txt



"""



import os
import shutil
import torch
from pathlib import Path

# way
yolo_dir = Path("yolov5")
dataset_zip = "YOLO_dataset.zip"
dataset_path = yolo_dir / "YOLOv5/obj/dataset.yaml"

# Clonar o repositório YOLOv5 se ele ainda não existir
if not yolo_dir.exists():
    os.system("git clone https://github.com/ultralytics/yolov5.git")

# Instalar as dependências do YOLOv5
os.system(f"pip install -qr {yolo_dir}/requirements.txt")

# Descompactar o conjunto de dados (se necessário)
if os.path.exists(dataset_zip):
    shutil.unpack_archive(dataset_zip, extract_dir=yolo_dir / "YOLOv5")

# Criar o arquivo de configuração do dataset
yaml_content = """
train: yolov5/YOLOv5/obj/images/train
val: yolov5/YOLOv5/obj/images/val
test: yolov5/YOLOv5/obj/images/test

nc: 2
names: ['chita', 'leao']
"""

with open(dataset_path, 'w') as file:
    file.write(yaml_content)

# Treinar o modelo YOLOv5
os.system(f"python {yolo_dir}/train.py --img 640 --batch 16 --epochs 50 "
          f"--data {dataset_path} --weights yolov5s.pt --device 0")

# Carregar modelo treinado
model = torch.hub.load('ultralytics/yolov5', 'custom', 
                       path=str(yolo_dir / "runs/train/exp/weights/best.pt"),
                       force_reload=True)
model.eval()

# Descompactar arquivos de predição, se necessário
predict_zip = "predict.zip"
if os.path.exists(predict_zip):
    shutil.unpack_archive(predict_zip, extract_dir="predict")

# Rodar o modelo de detecção em imagens de predição
os.system(f"python {yolo_dir}/detect.py --source predict --weights "
          f"runs/train/exp2/weights/best.pt --img-size 640 --conf-thres 0.15 "
          f"--iou-thres 0.45 --device 0")

# Exibir imagens de saída
from IPython.display import Image, display

output_folder = Path("runs/detect/exp13")
if output_folder.exists():
    image_files = [f for f in output_folder.iterdir() if f.suffix in ['.jpg', '.jpeg']]
    
    for image_file in image_files:
        display(Image(filename=str(image_file)))
