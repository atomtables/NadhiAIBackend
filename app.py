from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import io
import base64
import json
import cv2
import requests
from pillow_heif import register_heif_opener
import pillow_heif

# Register HEIF opener to allow PIL to open HEIC files
register_heif_opener()

app = Flask(__name__)
CORS(app)

# Class labels
CLASS_LABELS = [
    'car', 'flooding', 'house_0', 'house_1', 'house_2', 'house_3',
    'house_4', 'house_5', 'house_6', 'house_7', 'house_8', 'person', 'scooter'
]

# Colors for visualization (RGB)
COLORS = [
    (255, 0, 0),      # car - red
    (255, 165, 0),    # flooding - orange
    (139, 69, 19),    # house_0 - brown
    (165, 42, 42),    # house_1
    (210, 105, 30),   # house_2
    (244, 164, 96),   # house_3
    (222, 184, 135),  # house_4
    (188, 143, 143),  # house_5
    (205, 133, 63),   # house_6
    (160, 82, 45),    # house_7
    (184, 134, 11),   # house_8
    (0, 255, 0),      # person - green
    (0, 0, 255),      # scooter - blue
]

# Load YOLO model
try:
    model = YOLO('model_weights.pt')
    print("✓ Model loaded successfully!")
    print(f"✓ Model type: {type(model)}")
    print(f"✓ Number of classes: {len(model.names)}")
    print(f"✓ Class names: {model.names}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None


def polygon_from_mask(mask):
    """Convert binary mask to polygon coordinates"""
    import cv2
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        # Simplify the contour
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        # Convert to list of [x, y] coordinates
        polygon = [[int(point[0][0]), int(point[0][1])] for point in approx]
        return polygon
    
    return []

def check_intersection(mask1, mask2):
    """Check if two masks intersect"""
    intersection = np.logical_and(mask1, mask2)
    return np.sum(intersection) > 0

def calculate_danger_level(detections):
    """Calculate danger level based on flooding intersections"""
    danger_level = 0
    
    # Find flooding masks
    flooding_indices = [i for i, det in enumerate(detections) if det['class'] == 'flooding']
    
    if not flooding_indices:
        return 0
    
    # Check intersections with other objects
    has_person_intersection = False
    has_car_house_intersection = False
    has_scooter_intersection = False
    
    for flood_idx in flooding_indices:
        flood_mask = detections[flood_idx]['mask']
        
        for i, det in enumerate(detections):
            if i == flood_idx:
                continue
            
            if check_intersection(flood_mask, det['mask']):
                if det['class'] == 'person':
                    has_person_intersection = True
                elif det['class'] in ['car'] + [f'house_{i}' for i in range(9)]:
                    has_car_house_intersection = True
                elif det['class'] == 'scooter':
                    has_scooter_intersection = True
    
    # Determine danger level
    if has_person_intersection:
        danger_level = 3
    elif has_scooter_intersection:
        danger_level = 2
    elif has_car_house_intersection:
        danger_level = 1
    
    return danger_level

def create_visualization(image, detections, danger_level):
    """Create annotated image with bounding boxes and masks"""
    img_array = np.array(image)
    img_copy = Image.fromarray(img_array).convert('RGBA')
    
    # Create overlay for masks
    overlay = Image.new('RGBA', img_copy.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    # Create draw object for boxes and text
    draw = ImageDraw.Draw(img_copy)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw masks and boxes
    for det in detections:
        # Get color for this class
        class_name = det['class']
        
        # Map class name to color
        color_map = {
            'car': (255, 0, 0),
            'flooding': (255, 165, 0),
            'person': (0, 255, 0),
            'scooter': (0, 0, 255)
        }
        
        # For houses, use brown shades
        if class_name.startswith('house'):
            color = (139, 69, 19)
        else:
            color = color_map.get(class_name, (255, 255, 255))
        
        # Draw semi-transparent mask
        mask_np = det['mask']
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
        
        # Create colored overlay for this mask
        mask_overlay = Image.new('RGBA', img_copy.size, color + (100,))
        temp = Image.new('RGBA', img_copy.size, (0, 0, 0, 0))
        temp.paste(mask_overlay, (0, 0), mask_img)
        overlay = Image.alpha_composite(overlay, temp)
        
        # Draw bounding box
        bbox = det['bbox']
        draw.rectangle(bbox, outline=color, width=3)
        
        # Draw label
        label = f"{det['class']}: {det['confidence']:.2f}"
        
        # Get text size for background
        bbox_text = draw.textbbox((bbox[0], bbox[1] - 25), label, font=small_font)
        draw.rectangle(bbox_text, fill=color + (220,))
        draw.text((bbox[0], bbox[1] - 25), label, fill='white', font=small_font)
    
    # Composite the overlay onto the image
    img_copy = Image.alpha_composite(img_copy, overlay)
    img_copy = img_copy.convert('RGB')
    
    # Draw danger level indicator on the final image
    draw_final = ImageDraw.Draw(img_copy)
    
    danger_colors = {
        0: (0, 255, 0),      # Green
        1: (255, 255, 0),    # Yellow
        2: (255, 165, 0),    # Orange
        3: (255, 0, 0)       # Red
    }
    danger_texts = {
        0: "SAFE",
        1: "CAUTION",
        2: "WARNING",
        3: "DANGER"
    }
    
    danger_color = danger_colors[danger_level]
    danger_text = f"DANGER LEVEL: {danger_level} - {danger_texts[danger_level]}"
    
    # Draw danger indicator in top-left corner
    bbox_danger = draw_final.textbbox((10, 10), danger_text, font=font)
    draw_final.rectangle(
        [(bbox_danger[0] - 5, bbox_danger[1] - 5), 
         (bbox_danger[2] + 5, bbox_danger[3] + 5)],
        fill=danger_color
    )
    draw_final.text((10, 10), danger_text, fill='white', font=font)
    
    return img_copy

def get_gps_info(image):
    """Extract GPS coordinates and altitude from image EXIF data"""
    try:
        # Try standard PIL getexif first
        exif_data = image.getexif()
        
        if not exif_data:
            return None
        
        gps_info = {}
        
        # Look for GPSInfo tag (tag 34853)
        if 34853 in exif_data:
            gps_ifd = exif_data.get_ifd(34853)
            for gps_tag, value in gps_ifd.items():
                sub_decoded = GPSTAGS.get(gps_tag, gps_tag)
                gps_info[sub_decoded] = value
        
        if not gps_info:
            return None
        
        # Extract latitude
        lat = None
        if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
            lat = gps_info['GPSLatitude']
            lat_ref = gps_info['GPSLatitudeRef']
            lat = float(lat[0]) + float(lat[1])/60 + float(lat[2])/3600
            if lat_ref == 'S':
                lat = -lat
        
        # Extract longitude
        lon = None
        if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
            lon = gps_info['GPSLongitude']
            lon_ref = gps_info['GPSLongitudeRef']
            lon = float(lon[0]) + float(lon[1])/60 + float(lon[2])/3600
            if lon_ref == 'W':
                lon = -lon
        
        # Extract altitude
        altitude = None
        if 'GPSAltitude' in gps_info:
            altitude = float(gps_info['GPSAltitude'])
            if 'GPSAltitudeRef' in gps_info and gps_info['GPSAltitudeRef'] == 1:
                altitude = -altitude
        
        if lat is not None and lon is not None:
            return {
                'latitude': lat,
                'longitude': lon,
                'altitude': altitude
            }
        
    except Exception as e:
        print(f"Error extracting GPS info: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def get_elevation_from_coords(lat, lon):
    """Get elevation (above sea level) from coordinates using Open-Elevation API"""
    try:
        # Using Open-Elevation API (free, no API key required)
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                elevation = data['results'][0]['elevation']
                return elevation
    except Exception as e:
        print(f"Error getting elevation: {e}")
    
    return None

def geocode_address(address):
    """Convert address to coordinates using Nominatim (OpenStreetMap)"""
    try:
        # Using Nominatim API (free, no API key required)
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': address,
            'format': 'json',
            'limit': 1
        }
        headers = {
            'User-Agent': 'FloodDetectionApp/1.0'
        }
        response = requests.get(url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                return {'latitude': lat, 'longitude': lon}
    except Exception as e:
        print(f"Error geocoding address: {e}")
    
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert-heic', methods=['POST'])
def convert_heic():
    """Convert HEIC to JPEG for preview and extract metadata"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    try:
        print(f"Converting HEIC file: {file.filename}")
        
        # Open HEIC file
        image = Image.open(file.stream).convert('RGB')
        print(f"Image loaded: format={image.format}, size={image.size}")
        
        # Extract GPS metadata using getexif()
        print("Attempting to extract GPS data...")
        gps_data = get_gps_info(image)
        
        if gps_data:
            print(f"✓ GPS data found: {gps_data}")
        else:
            print("✗ No GPS data found")
        
        # Convert to JPEG for preview
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        print(f"✓ Image converted to JPEG for preview")
        
        return jsonify({
            'preview': f'data:image/jpeg;base64,{img_str}',
            'has_gps': gps_data is not None,
            'gps_data': gps_data
        })
        
    except Exception as e:
        import traceback
        print("Error converting HEIC:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    file = request.files['image']
    confidence_threshold = float(request.form.get('confidence', 0.25))
    location_input = request.form.get('location', None)
    
    try:
        # Load image
        file.stream.seek(0)  # Reset file pointer
        image = Image.open(file.stream).convert('RGB')
        
        # Extract GPS info from image
        print(f"Processing image: {file.filename}")
        print(f"Image format: {image.format}")
        print(f"Image size: {image.size}")
        
        gps_data = get_gps_info(image)
        has_metadata = gps_data is not None
        
        if gps_data:
            print(f"GPS data extracted: {gps_data}")
        else:
            print("No GPS data found in image")
        
        # Calculate elevation data
        elevation_data = None
        if gps_data:
            # Image has GPS metadata
            lat, lon = gps_data['latitude'], gps_data['longitude']
            camera_altitude = gps_data.get('altitude')
            ground_elevation = get_elevation_from_coords(lat, lon)
            
            if ground_elevation is not None and camera_altitude is not None:
                height_above_ground = camera_altitude - ground_elevation
                elevation_data = {
                    'has_metadata': True,
                    'latitude': lat,
                    'longitude': lon,
                    'camera_altitude': round(camera_altitude, 2),
                    'ground_elevation': round(ground_elevation, 2),
                    'height_above_ground': round(height_above_ground, 2)
                }
        elif location_input:
            # User provided location
            coords = geocode_address(location_input)
            if coords:
                lat, lon = coords['latitude'], coords['longitude']
                ground_elevation = get_elevation_from_coords(lat, lon)
                
                if ground_elevation is not None:
                    elevation_data = {
                        'has_metadata': False,
                        'latitude': lat,
                        'longitude': lon,
                        'ground_elevation': round(ground_elevation, 2),
                        'location_name': location_input
                    }
        
        img_array = np.array(image)
        
        # Run YOLO inference with custom confidence
        results = model(img_array, conf=confidence_threshold, iou=0.45, verbose=False)
        result = results[0]
        
        # Process predictions
        detections = []
        
        if result.masks is not None:
            for i in range(len(result.boxes)):
                box = result.boxes[i]
                mask = result.masks[i]
                
                # Get class info
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                
                # Get class name from model
                class_name = model.names[class_id]
                
                # Get bounding box
                bbox_coords = box.xyxy[0].cpu().numpy()
                bbox = [int(bbox_coords[0]), int(bbox_coords[1]), 
                       int(bbox_coords[2]), int(bbox_coords[3])]
                
                # Get mask
                mask_array = mask.data[0].cpu().numpy()
                
                # Resize mask to image size if needed
                if mask_array.shape != (image.height, image.width):
                    mask_array = cv2.resize(mask_array, (image.width, image.height))
                
                # Convert to binary mask
                binary_mask = (mask_array > 0.5).astype(np.uint8)
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'mask': binary_mask,
                    'polygon': polygon_from_mask(binary_mask)
                })
        
        # Calculate danger level
        danger_level = calculate_danger_level(detections)
        
        # Create visualization
        annotated_image = create_visualization(image, detections, danger_level)
        
        # Convert annotated image to base64
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare JSON response (remove mask arrays)
        response_detections = []
        for det in detections:
            response_detections.append({
                'class': det['class'],
                'confidence': float(det['confidence']),
                'bbox': det['bbox'],
                'polygon': det['polygon']
            })
        
        response = {
            'detections': response_detections,
            'danger_level': danger_level,
            'object_count': len(detections),
            'annotated_image': f'data:image/png;base64,{img_str}',
            'has_metadata': has_metadata,
            'elevation_data': elevation_data,
            'summary': {
                'total_objects': len(detections),
                'classes_detected': list(set([d['class'] for d in detections])),
                'danger_status': ['SAFE', 'CAUTION', 'WARNING', 'DANGER'][danger_level]
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
