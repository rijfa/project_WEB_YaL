from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash, generate_password_hash
import torch
import base64
import io
from PIL import Image
import numpy as np
import onnxruntime as ort
from datetime import datetime
import os

from models import db, User, Drawing

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/')
@login_required
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))

        flash('Invalid username or password', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not username or not password:
            flash('Username and password are required.', 'warning')
        elif password != confirm_password:
            flash('Passwords do not match.', 'warning')
        elif User.query.filter_by(username=username).first():
            flash('Username already exists.', 'warning')
        else:
            user = User(username=username, password_hash=generate_password_hash(password))
            db.session.add(user)
            db.session.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

session = ort.InferenceSession("model.onnx")
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes))

    img = img.convert('L').resize((28, 28))
    img_array = np.array(img).astype(np.float32)
    
    img_array = (img_array / 255.0 - 0.1307) / 0.3081
    
    input_tensor = img_array[np.newaxis, np.newaxis, :, :]

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    prediction = int(np.argmax(outputs[0]))
    exp_logits = np.exp(outputs[0] - np.max(outputs[0]))
    probs = exp_logits / np.sum(exp_logits)
    confidence = float(np.max(probs))
    
    # Сохранить изображение в файл
    filename = f"drawing_{current_user.id}_{datetime.utcnow().timestamp()}.png"
    filepath = os.path.join('static/uploads', filename)
    img.save(filepath)
    
    # Сохранить в БД
    drawing = Drawing(
        user_id=current_user.id,
        image_path=filepath,
        prediction_result=str(prediction),
        confidence=confidence
    )
    db.session.add(drawing)
    db.session.commit()
    
    return jsonify({'prediction': prediction, 'confidence': confidence})

@app.route('/history')
@login_required
def history():
    user_drawings = Drawing.query.filter_by(user_id=current_user.id).order_by(Drawing.id.desc()).all()
    return render_template('history.html', drawings=user_drawings)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=8080)