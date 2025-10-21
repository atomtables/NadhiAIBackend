# 🌊 Flood Detection & Instance Segmentation

A web-based application for instance segmentation using YOLOv8 that detects flooding, vehicles, people, and buildings, with automatic danger level assessment and GPS/elevation analysis.

## ✨ Features

### Detection & Analysis
- 🤖 **Instance Segmentation**: YOLOv8-based detection of 13 object classes
- 🎨 **Visual Annotations**: Colored masks and bounding boxes like Roboflow
- 🎚️ **Live Confidence Slider**: Adjust detection threshold (0.1-0.9) with auto re-analysis
- ⚠️ **Danger Assessment**: 4-level system based on flooding intersections
  - **Level 3 (RED)**: Flooding + Person = DANGER
  - **Level 2 (ORANGE)**: Flooding + Scooter = WARNING
  - **Level 1 (YELLOW)**: Flooding + Car/House = CAUTION
  - **Level 0 (GREEN)**: Safe

### Location & Elevation
- 📱 **HEIC Support**: Full iPhone/Android HEIC/HEIF support with preview
- � **GPS Auto-Detection**: Extracts coordinates & altitude from image EXIF
- 🗺️ **Manual Location**: Address input with geocoding (OpenStreetMap)
- �️ **Elevation Calculation**: Ground elevation via Open-Elevation API
- 📐 **Height Above Ground**: Camera altitude - Ground elevation (m & ft)

### Interface
- 🎯 **Drag & Drop**: Easy image upload
- 📏 **Fixed Image Sizes**: Consistent 600px height display
- � **JSON Output**: Complete detection data with metadata
- � **Beautiful UI**: Purple gradient, color-coded detections

## Detected Classes

- `car` - Vehicles (Red)
- `flooding` - Flooded areas (Orange)
- `house_0` to `house_8` - Buildings (Brown)
- `person` - People (Green)
- `scooter` - Scooters/motorcycles (Blue)

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test HEIC support (optional)
python test_heic.py

# 3. Start the server
python app.py

# 4. Open browser
open http://localhost:5000
```

## 📖 Usage

1. **Upload Image**
   - Drag & drop or click to browse
   - Supports: JPG, PNG, HEIC/HEIF, WebP
   - Preview appears (HEIC auto-converts)

2. **Location (if no GPS)**
   - Enter address if image has no GPS metadata
   - Used to calculate ground elevation

3. **Adjust Confidence**
   - Move slider (0.1-0.9, default 0.25)
   - Lower = more detections
   - Higher = only confident detections
   - Auto re-analyzes when changed

4. **Analyze**
   - Click "Analyze Image"
   - View results:
     - Annotated image with highlights
     - Danger level indicator
     - GPS coordinates & elevation
     - Height above ground
     - Detection list
     - JSON output

## 🎯 Workflow

```
Upload Image (JPG/PNG/HEIC)
         ↓
HEIC? → Convert to JPEG for preview
         ↓
Extract GPS from EXIF (or request location)
         ↓
Calculate ground elevation & height
         ↓
Run YOLOv8 detection (with confidence threshold)
         ↓
Calculate danger level (flooding intersections)
         ↓
Generate annotated image
         ↓
Display: Results + Elevation Data + JSON
```

## 📊 JSON Output Example

```json
{
  "detections": [
    {"class": "flooding", "confidence": 0.89, "bbox": [100, 200, 500, 600]},
    {"class": "person", "confidence": 0.95, "bbox": [300, 150, 450, 550]}
  ],
  "danger_level": 3,
  "object_count": 2,
  "annotated_image": "data:image/png;base64,...",
  "elevation_data": {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "camera_altitude": 15.5,
    "ground_elevation": 10.2,
    "height_above_ground": 5.3
  },
  "summary": {
    "danger_status": "DANGER",
    "classes_detected": ["flooding", "person"]
  }
}
```

## 🔧 Technical Details

### Stack
- **Backend**: Flask + YOLOv8 (Ultralytics)
- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing**: PIL, pillow-heif, OpenCV, NumPy
- **APIs**: Open-Elevation (elevation), Nominatim (geocoding)

### Dependencies
```
flask==3.1.0
ultralytics>=8.0.0
Pillow>=10.4.0
pillow-heif>=0.13.0  # HEIC support
opencv-python>=4.8.0
numpy>=1.26.0
shapely>=2.0.6
```

## 🧪 Testing HEIC Support

```bash
# Test HEIC installation
python test_heic.py

# Test with your HEIC file
python test_heic.py /path/to/photo.heic
```

Expected output:
```
✓ pillow-heif installed
✓ HEIC support is working!
✓ GPS data found: {'latitude': 40.74, 'longitude': -73.98, 'altitude': 15.5}
```

## 🎨 Customization

### Adjust Confidence (via UI)
Use the slider in the web interface (0.1-0.9)

### Change Detection Colors
Edit `create_visualization()` in `app.py`:
```python
color_map = {
    'car': (255, 0, 0),      # Red
    'flooding': (255, 165, 0), # Orange
    'person': (0, 255, 0),    # Green
    'scooter': (0, 0, 255)    # Blue
}
```

### Modify Danger Logic
Edit `calculate_danger_level()` in `app.py`

## 🐛 Troubleshooting

**HEIC preview not showing?**
- Run: `python test_heic.py`
- Check: `pip install pillow-heif`

**GPS not detected?**
- Check if image has EXIF data
- Try with iPhone photo

**No detections found?**
- Lower confidence threshold (slider)
- Check model weights file exists
- Verify objects are in trained classes

**Elevation API timeout?**
- Check internet connection
- APIs may be temporarily unavailable
- Try again in a moment

## 📝 License

MIT License - Free to use and modify
