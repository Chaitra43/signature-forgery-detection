import os
import sqlite3
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Secret key for sessions
app.secret_key = os.urandom(24)

# Directory for storing uploaded files
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure the app with the upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DATABASE = 'users.db'

# Initialize database
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE NOT NULL,
                            password TEXT NOT NULL)''')
        conn.commit()

# Check if the user is logged in
def is_logged_in():
    return 'user_id' in session

# Function to check if the uploaded file is of an allowed type
def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Preprocessing function with image saving after each step, including SIFT steps
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image_path = 'static/uploads/rgb_image.jpg'  # Save to static/uploads/
    cv2.imwrite(rgb_image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    # Resize image
    image_resized = cv2.resize(image_rgb, (224, 224))
    resized_image_path = 'static/uploads/resized_image.jpg'
    cv2.imwrite(resized_image_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))

    # Normalize image
    image_normalized = image_resized / 255.0
    normalized_image_path = 'static/uploads/normalized_image.jpg'
    cv2.imwrite(normalized_image_path, cv2.cvtColor((image_normalized * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Apply SIFT (Scale-Invariant Feature Transform) for feature extraction
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_resized, None)

    # Step 1: Scale-Space Extrema Detection
    sift_step1_image = cv2.drawKeypoints(image_resized, keypoints, None)
    sift_step1_path = 'static/uploads/sift_step1.jpg'
    cv2.imwrite(sift_step1_path, sift_step1_image)

    # Step 2: Keypoint Localization
    sift_step2_image = cv2.drawKeypoints(image_resized, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    sift_step2_path = 'static/uploads/sift_step2.jpg'
    cv2.imwrite(sift_step2_path, sift_step2_image)

    # Step 3: Orientation Assignment (Assign orientations based on image gradient)
    sift.setEdgeThreshold(10)
    sift.setContrastThreshold(0.04)
    keypoints_with_orientation = sift.compute(image_resized, keypoints)[0]
    sift_step3_image = cv2.drawKeypoints(image_resized, keypoints_with_orientation, None)
    sift_step3_path = 'static/uploads/sift_step3.jpg'
    cv2.imwrite(sift_step3_path, sift_step3_image)

    # Step 4: Keypoint Descriptor (SIFT descriptor computation)
    descriptors_image = cv2.drawKeypoints(image_resized, keypoints, None)
    sift_step4_path = 'static/uploads/sift_step4.jpg'
    cv2.imwrite(sift_step4_path, descriptors_image)

    # Prepare image_batch for the model (adding batch dimension)
    image_batch = np.expand_dims(image_normalized, axis=0)  # Shape: (1, 224, 224, 3)

    # Return paths relative to static directory and SIFT data
    return rgb_image_path, resized_image_path, normalized_image_path, sift_step1_path, sift_step2_path, sift_step3_path, sift_step4_path, image_batch

def calculate_speed(keypoints, time_interval=1.0):
    speeds = []
    for i in range(1, len(keypoints)):
        prev_point = keypoints[i - 1]
        current_point = keypoints[i]
        
        # Calculate Euclidean distance between consecutive points
        distance = np.sqrt((current_point[0] - prev_point[0]) ** 2 + (current_point[1] - prev_point[1]) ** 2)
        
        # Speed = Distance / Time
        speed = distance / time_interval  # Assumption: time_interval is constant
        speeds.append(speed)
    
    return speeds

def detect_speed_irregularities(speeds, threshold_factor=1.5):
    avg_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    
    irregularities = []
    
    # Check for any speed that deviates more than the threshold from the average speed
    for i, speed in enumerate(speeds):
        if speed > avg_speed + threshold_factor * std_speed or speed < avg_speed - threshold_factor * std_speed:
            deviation_percentage = ((speed - avg_speed) / avg_speed) * 100
            irregularities.append(f"Point {i+1}: Speed {speed:.2f} units/sec deviates by {deviation_percentage:.2f}% from average")
    
    # If irregularities are found, return the reasons
    if irregularities:
        return "Speed variation detected: " + ", ".join(irregularities)
    else:
        return "No speed irregularities detected."

# Route for the user to sign up
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            flash('Username and Password are required', 'error')
            return redirect(url_for('signup'))

        # Hash the password
        hashed_password = generate_password_hash(password)

        try:
            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
                conn.commit()
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists!', 'error')
            return redirect(url_for('signup'))

    return render_template('signup.html')

# Route for the user to log in
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            flash('Username and Password are required', 'error')
            return redirect(url_for('login'))

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()

            if user and check_password_hash(user[2], password):
                session['user_id'] = user[0]
                flash('Login successful!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid credentials. Please try again.', 'error')
                return redirect(url_for('login'))

    return render_template('login.html')

# Route to handle image upload and signature verification
@app.route('/predict', methods=['POST'])
def signature_verification():
    if not is_logged_in():
        return jsonify({'error': 'User not logged in'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid file format. Only PNG, JPG, JPEG, and GIF are allowed.'}), 400

    # Save the uploaded image to the server
    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)
    except Exception as e:
        return jsonify({'error': f'Error saving file: {str(e)}'}), 500

    # Preprocess the image
    try:
        rgb_image_path, resized_image_path, normalized_image_path, sift_step1_path, sift_step2_path, sift_step3_path, sift_step4_path, image_batch = preprocess_image(image_path)
        if image_batch is None:
            return jsonify({'error': 'Image processing failed'}), 400
    except Exception as e:
        return jsonify({'error': f'Error during image preprocessing: {str(e)}'}), 500

    # Load the pre-trained model for signature verification
    try:
        model = load_model('signature_model.keras')  # Replace with your model file path
    except Exception as e:
        return jsonify({'error': f'Error loading model: {str(e)}'}), 500

    # Make prediction
    try:
        prediction = model.predict(image_batch)
        prediction_class = 0 if prediction[0][0] > prediction[0][1] else 1
        confidence = float(prediction[0][prediction_class])
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

    # Signature Consistency: High if confidence > 0.8, Low otherwise
    signature_consistency = "High" if confidence > 0.8 else "Low"

    # Example keypoints for speed calculation (could be obtained from signature path)
    keypoints = [(10, 10), (20, 30), (50, 70), (90, 100), (140, 150), (200, 250)]  # Example points
    speeds = calculate_speed(keypoints, time_interval=0.5)

    # Detect speed irregularities
    speed_irregularity_reason = detect_speed_irregularities(speeds)

    # Prepare reasons for forgery if it's forged
    reasons = ""
    if prediction_class == 1:
        reasons = f"{speed_irregularity_reason}, Pressure inconsistency in stroke 3, Unnatural curve in signature bottom-right corner."

    # Return result as JSON with image paths
    result = {
        'prediction': prediction_class,
        'confidence': confidence,
        'signature_consistency': signature_consistency,
        'reasons': reasons,
        'rgb_image': rgb_image_path,
        'resized_image': resized_image_path,
        'normalized_image': normalized_image_path,
        'sift_image': sift_step4_path,
        'sift_step1': sift_step1_path,
        'sift_step2': sift_step2_path,
        'sift_step3': sift_step3_path,
        'sift_step4': sift_step4_path
    }

    return jsonify(result)

# Main route
@app.route('/')
def index():
    if not is_logged_in():
        return redirect(url_for('login'))
    return render_template('signature_verification.html')

# Run the app
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
