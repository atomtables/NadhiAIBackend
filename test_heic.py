#!/usr/bin/env python3
"""
Test script to verify HEIC support and GPS extraction
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pillow_heif import register_heif_opener
import sys

# Register HEIF opener
register_heif_opener()

def test_heic_support():
    """Test if HEIC files can be opened"""
    print("=" * 60)
    print("Testing HEIC Support")
    print("=" * 60)
    
    # Check if pillow-heif is installed
    try:
        import pillow_heif
        print("✓ pillow-heif installed")
        print(f"  Version: {pillow_heif.__version__}")
    except ImportError:
        print("✗ pillow-heif NOT installed")
        return False
    
    return True

def test_gps_extraction(image_path):
    """Test GPS extraction from an image"""
    print("\n" + "=" * 60)
    print(f"Testing GPS Extraction: {image_path}")
    print("=" * 60)
    
    try:
        # Open image
        img = Image.open(image_path)
        print(f"✓ Image opened successfully")
        print(f"  Format: {img.format}")
        print(f"  Size: {img.size}")
        print(f"  Mode: {img.mode}")
        
        # Get EXIF data using getexif()
        exif_data = img.getexif()
        
        if not exif_data:
            print("✗ No EXIF data found")
            return None
        
        print(f"✓ EXIF data found: {len(exif_data)} tags")
        
        # Look for GPS info (tag 34853)
        if 34853 in exif_data:
            print("✓ GPS data found in EXIF")
            gps_ifd = exif_data.get_ifd(34853)
            
            print(f"\nGPS Tags:")
            for tag, value in gps_ifd.items():
                tag_name = GPSTAGS.get(tag, f"Unknown({tag})")
                print(f"  {tag_name}: {value}")
            
            # Extract coordinates
            if 2 in gps_ifd and 4 in gps_ifd:  # Latitude and Longitude
                lat = gps_ifd[2]
                lat_ref = gps_ifd[1]
                lon = gps_ifd[4]
                lon_ref = gps_ifd[3]
                
                lat_decimal = float(lat[0]) + float(lat[1])/60 + float(lat[2])/3600
                if lat_ref == 'S':
                    lat_decimal = -lat_decimal
                
                lon_decimal = float(lon[0]) + float(lon[1])/60 + float(lon[2])/3600
                if lon_ref == 'W':
                    lon_decimal = -lon_decimal
                
                print(f"\n📍 Coordinates:")
                print(f"  Latitude: {lat_decimal:.6f}")
                print(f"  Longitude: {lon_decimal:.6f}")
                
                # Extract altitude if present
                if 6 in gps_ifd:  # GPSAltitude
                    altitude = float(gps_ifd[6])
                    print(f"  Altitude: {altitude:.2f} meters")
        else:
            print("✗ No GPS data in EXIF")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Test HEIC support
    if not test_heic_support():
        print("\nInstall pillow-heif: pip install pillow-heif")
        return
    
    print("\n✓ HEIC support is working!")
    
    # If a file path is provided, test GPS extraction
    if len(sys.argv) > 1:
        test_gps_extraction(sys.argv[1])
    else:
        print("\nTo test GPS extraction from an image:")
        print("  python test_heic.py /path/to/image.heic")

if __name__ == "__main__":
    main()
