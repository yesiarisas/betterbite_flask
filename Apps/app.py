import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Nonaktifkan GPU, pakai CPU saja

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model (gunakan CPU)
model = tf.keras.models.load_model('food101_mobilenetv2_final.keras')

# Daftar kelas makanan
class_names = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'cheese_plate',
    'cheesecake', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake',
    'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes',
    'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts',
    'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel',
    'filet_mignon', 'foie_gras', 'french_fries', 'french_toast',
    'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread',
    'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza',
    'hamburger', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
    'lobster_bisque', 'lobster_roll_sandwich', 'macarons', 'miso_soup', 'mussels',
    'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella',
    'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop',
    'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'red_velvet_cake',
    'risotto', 'samosa', 'sashimi', 'seaweed_salad', 'spaghetti_bolognese',
    'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]

# Fungsi prediksi makanan
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with tf.device('/cpu:0'):  # Pastikan prediksi dilakukan di CPU
        predictions = model.predict(img_array)
    
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index])

    return predicted_class, confidence

# Fungsi ambil data nutrisi dari file JSON
def get_nutrition_data(food_snake_case):
    food_title_case = food_snake_case.replace('_', ' ').title()

    with open('foodnutrition.json', 'r') as f:
        nutrition_data = json.load(f)

    for item in nutrition_data:
        if item['food_name'].lower() == food_title_case.lower():
            return item
    return None

# Endpoint prediksi
@app.route('/predict', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded!'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    label_snake, confidence = predict_image(filepath)
    label_title = label_snake.replace('_', ' ').title()

    nutrition = get_nutrition_data(label_snake)

    response_data = {
        'food_name': label_title,
        'confidence': confidence,
        'image_url': f'/static/uploads/{filename}',
        'nutrition_info': nutrition if nutrition else 'Not found'
    }

    return jsonify(response_data), 200

# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True, port=8000)
