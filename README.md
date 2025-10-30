# Nadhi: The AI Machine-learning model

This part of the app is responsible for the image segmentation model. This model detects flooding severity and height through the use of anchors like cars, houses, and people, allowing it to analyse images during a flood and require minimal intervention from the user. This model was measured to have up to 90% accuracy, and could most likely be higher if we had access to more powerful hardware.

Run it like this:
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
The AI model works best with JPGs. This model is based on the YOLOv8 architecture and was trained on 30,000 images of floods. It's an open-weight model (obviously there's a github page). It uses Flask for the testing interface, which allows you to upload pictures locally and test the output that comes out. This connects with the FastAPI backend to process tasks in the background.

### P.S. Nadhi, translated from Tamil, means: "river". It's also the acronym
