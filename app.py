import io
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import traceback

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

app = Flask(__name__)
CORS(app) 
labels = ('plane', 'car', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

modelo_vulneravel = SimpleCNN().to(device)
modelo_vulneravel.load_state_dict(torch.load('./models/modelo_vulneravel_cifar10.pth', map_location=device))
modelo_vulneravel.eval()

modelo_robusto = SimpleCNN().to(device)
modelo_robusto.load_state_dict(torch.load('./models/modelo_robusto_cifar10.pth', map_location=device))
modelo_robusto.eval()

print("Modelos (vulnerável e robusto) do CIFAR-10 carregados.")

preprocess_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def fgsm_attack(model, loss_fn, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_images = images + epsilon * sign_data_grad
    perturbed_images = torch.clamp(perturbed_images, -1, 1) 
    return perturbed_images

def get_prediction(model, input_batch, top_k=3):
    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_probs, top_catids = torch.topk(probabilities, top_k)

    predictions_list = []
    for i in range(top_k):
        predictions_list.append({
            "label": labels[top_catids[i].item()],
            "confidence": f"{top_probs[i].item()*100:.2f}%"
        })
        
    return predictions_list

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400

    if file:
        try:
            # Processamento
            img_bytes = file.read()
            image_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            input_tensor = preprocess_transform(image_pil) 
            input_batch = input_tensor.unsqueeze(0).to(device)

            # Predição Original
            preds_orig_vuln = get_prediction(modelo_vulneravel, input_batch)
            preds_orig_rob = get_prediction(modelo_robusto, input_batch)

            # Ataque
            input_batch.requires_grad = True 
            output = modelo_vulneravel(input_batch)
            _, original_pred_idx = torch.max(output, 1)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, original_pred_idx)
            modelo_vulneravel.zero_grad()
            loss.backward()
            data_grad = input_batch.grad.data
            epsilon_attack = 0.1
            perturbed_image_tensor = fgsm_attack(modelo_vulneravel, loss_fn, input_batch, original_pred_idx, epsilon_attack)

            preds_atk_vuln = get_prediction(modelo_vulneravel, perturbed_image_tensor)
            preds_atk_rob = get_prediction(modelo_robusto, perturbed_image_tensor)

            return jsonify({
                "original_image_predictions": {
                    "vulnerable_model": {
                        "top_prediction": preds_orig_vuln[0],
                        "all_predictions": preds_orig_vuln
                    },
                    "robust_model": {
                        "top_prediction": preds_orig_rob[0],
                        "all_predictions": preds_orig_rob
                    }
                },
                "attacked_image_predictions": {
                    "vulnerable_model": {
                        "top_prediction": preds_atk_vuln[0],
                        "all_predictions": preds_atk_vuln
                    },
                    "robust_model": {
                        "top_prediction": preds_atk_rob[0],
                        "all_predictions": preds_atk_rob
                    }
                },
                "attack_strength_epsilon": epsilon_attack
            })

        except Exception as e:
            print(f"Erro detalhado: {e}") 
            traceback.print_exc()
            return jsonify({"error": f"Erro ao processar imagem: {str(e)}"}), 500

if __name__ == '__main__':
    print("Iniciando o servidor...")
    app.run(debug=True, port=5000)