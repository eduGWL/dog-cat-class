from flask import Flask, request, render_template_string, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
import io
from PIL import Image
import os

app = Flask(__name__)

# Load model
model_path = 'best_model_xception.keras'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    # Get input shape from the model
    input_shape = model.input_shape[1:3] if model.input_shape[1] is not None else (299, 299)
    print("Model expected input shape:", input_shape)
else:
    model = None
    input_shape = (299, 299)
    print("Model not found. Please make sure best_model_xception.keras is in the same directory.")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>강아지 vs 고양이 분류기</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; margin-top: 50px; background-color: #f4f4f9; color: #333; }
        .container { max-width: 600px; margin: auto; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        h1 { color: #5a67d8; }
        input[type="file"] { margin: 20px 0; }
        .btn { background-color: #5a67d8; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; transition: background-color 0.3s; }
        .btn:hover { background-color: #434ce6; }
        #result { margin-top: 20px; font-size: 24px; font-weight: bold; color: #333; }
        img { max-width: 100%; border-radius: 10px; margin-top: 20px; display: none; margin-left: auto; margin-right: auto; }
        .loading { color: #888; font-style: italic; font-size: 18px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐱 강아지 vs 고양이 분류기 🐶</h1>
        <p>이미지를 업로드하면 최신 AI(Xception) 모델이 분석하여 결과를 알려줍니다!</p>
        <form id="upload-form" enctype="multipart/form-data">
            <input type="file" id="image-input" name="file" accept="image/*" required>
            <br>
            <button type="submit" class="btn">분류하기</button>
        </form>
        <img id="preview" src="#" alt="미리보기 이미지" />
        <div id="result"></div>
    </div>

    <script>
        const imageInput = document.getElementById('image-input');
        const preview = document.getElementById('preview');
        const resultDiv = document.getElementById('result');
        
        imageInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    resultDiv.innerHTML = '';
                }
                reader.readAsDataURL(file);
            }
        });

        document.getElementById('upload-form').onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            resultDiv.innerHTML = '<span class="loading">분석 중... ⏳</span>';
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (data.error) {
                    resultDiv.innerHTML = '<span style="color: red;">에러: ' + data.error + '</span>';
                } else {
                    let predColor = data.prediction.includes('강아지') ? '#2b6cb0' : '#c53030';
                    resultDiv.innerHTML = `결과: <span style="color: ${predColor};">${data.prediction}</span><br><span style="font-size: 16px; font-weight: normal; color: #718096;">AI 확신도: ${(data.confidence * 100).toFixed(2)}%</span>`;
                }
            } catch (err) {
                resultDiv.innerHTML = '<span style="color: red;">서버 오류가 발생했습니다.</span>';
            }
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': '모델이 로드되지 않았습니다. 서버 설정을 확인하세요.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': '유효한 파일이 전송되지 않았습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다.'}), 400
    
    try:
        img_bytes = file.read()
        # 이미지를 RGB로 변환하고 크기 조정
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize(input_shape)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 모델에 맞게 전처리 (Xception 모델 기준)
        img_array = preprocess_input(img_array)
        
        # 예측 수행
        preds = model.predict(img_array)
        score = float(preds[0][0])
        
        # 보통 0이 고양이, 1이 강아지인 경우가 많습니다. (flow_from_directory 기본값 알파벳순 배치의 경우: cats=0, dogs=1)
        if score > 0.5:
            prediction = '강아지 🐶'
            confidence = score
        else:
            prediction = '고양이 🐱'
            confidence = 1.0 - score
            
        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Flask 서버를 시작합니다...")
    app.run(host='0.0.0.0', port=5000, debug=True)
