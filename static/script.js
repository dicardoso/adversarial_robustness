document.addEventListener('DOMContentLoaded', () => {

    const fileInput = document.getElementById('file-input');
    const uploadButton = document.getElementById('upload-button');
    const loader = document.getElementById('loader');
    const resultsContainer = document.getElementById('results-container');
    const imagePreview = document.getElementById('image-preview');

    const cardOrigVuln = document.getElementById('card-orig-vuln');
    const cardOrigRob = document.getElementById('card-orig-rob');
    const cardAtkVuln = document.getElementById('card-atk-vuln');
    const cardAtkRob = document.getElementById('card-atk-rob');

    const labelOrigVuln = document.getElementById('label-orig-vuln');
    const confOrigVuln = document.getElementById('conf-orig-vuln');
    const labelOrigRob = document.getElementById('label-orig-rob');
    const confOrigRob = document.getElementById('conf-orig-rob');
    const labelAtkVuln = document.getElementById('label-atk-vuln');
    const confAtkVuln = document.getElementById('conf-atk-vuln');
    const labelAtkRob = document.getElementById('label-atk-rob');
    const confAtkRob = document.getElementById('conf-atk-rob');

    const predsOrigVuln = document.getElementById('preds-orig-vuln');
    const predsOrigRob = document.getElementById('preds-orig-rob');
    const predsAtkVuln = document.getElementById('preds-atk-vuln');
    const predsAtkRob = document.getElementById('preds-atk-rob');

    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => { imagePreview.src = e.target.result; }
            reader.readAsDataURL(file);
            resultsContainer.classList.remove('hidden');
        }
    });

    uploadButton.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) {
            alert('Por favor, selecione uma imagem primeiro.');
            return;
        }

        loader.classList.remove('hidden');
        resultsContainer.classList.add('hidden'); 
        resetCardStyles();

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) throw new Error(`Erro na API: ${response.statusText}`);
            
            const data = await response.json();
            
            updateUI(data);

        } catch (error) {
            console.error('Erro ao chamar a API:', error);
            alert('Falha ao analisar a imagem. Verifique o console.');
        } finally {
            loader.classList.add('hidden');
            resultsContainer.classList.remove('hidden');
        }
    });

    function updateUI(data) {
        const origVuln = data.original_image_predictions.vulnerable_model.top_prediction;
        const origRob = data.original_image_predictions.robust_model.top_prediction;
        const atkVuln = data.attacked_image_predictions.vulnerable_model.top_prediction;
        const atkRob = data.attacked_image_predictions.robust_model.top_prediction;

        const allOrigVuln = data.original_image_predictions.vulnerable_model.all_predictions;
        const allOrigRob = data.original_image_predictions.robust_model.all_predictions;
        const allAtkVuln = data.attacked_image_predictions.vulnerable_model.all_predictions;
        const allAtkRob = data.attacked_image_predictions.robust_model.all_predictions;

        labelOrigVuln.innerText = origVuln.label;
        confOrigVuln.innerText = origVuln.confidence;
        labelOrigRob.innerText = origRob.label;
        confOrigRob.innerText = origRob.confidence;
        labelAtkVuln.innerText = atkVuln.label;
        confAtkVuln.innerText = atkVuln.confidence;
        labelAtkRob.innerText = atkRob.label;
        confAtkRob.innerText = atkRob.confidence;

        fillPredictionList(predsOrigVuln, allOrigVuln);
        fillPredictionList(predsOrigRob, allOrigRob);
        fillPredictionList(predsAtkVuln, allAtkVuln);
        fillPredictionList(predsAtkRob, allAtkRob);

        if (origVuln.label === atkVuln.label) {
            cardAtkVuln.classList.add('success');
        } else {
            cardAtkVuln.classList.add('fail');
        }
        if (origRob.label === atkRob.label) {
            cardAtkRob.classList.add('success');
        } else {
            cardAtkRob.classList.add('fail');
        }
    }

    function fillPredictionList(element, predictions) {
        element.innerHTML = '';
        const ul = document.createElement('ul');
        predictions.forEach(pred => {
            const li = document.createElement('li');
            
            const labelSpan = document.createElement('span');
            labelSpan.className = 'label';
            labelSpan.innerText = pred.label + ' - ';
            
            const confSpan = document.createElement('span');
            confSpan.className = 'conf';
            confSpan.innerText = pred.confidence;
            
            li.appendChild(labelSpan);
            li.appendChild(confSpan);
            ul.appendChild(li);
        });
        element.appendChild(ul);
    }

    function resetCardStyles() {
        [cardOrigVuln, cardOrigRob, cardAtkVuln, cardAtkRob].forEach(card => {
            card.classList.remove('success', 'fail');
        });
        [predsOrigVuln, predsOrigRob, predsAtkVuln, predsAtkRob].forEach(el => {
            el.innerHTML = '';
        });
    }
});