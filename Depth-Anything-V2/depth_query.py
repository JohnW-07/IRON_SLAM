import cv2
import numpy as np

def get_depth_at_pixel(depth_map_path, x, y):
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    depth_value = depth_map[y, x]  # pixel brightness = relative depth
    return depth_value

def pixel_to_3d(x, y, depth_value, image_width, image_height, ceiling_height_ft=10.0):
    # Normalize pixel to -1..1 space
    norm_x = (x - image_width / 2) / (image_width / 2)
    norm_y = (y - image_height / 2) / (image_height / 2)
    
    # Scale depth to real-world units using ceiling as reference
    scale = ceiling_height_ft / 255.0
    z = depth_value * scale
    
    return {"x": round(norm_x * z, 2), "y": round(norm_y * z, 2), "z": round(z, 2)}

# Test it
if __name__ == "__main__":
    result = pixel_to_3d(x=320, y=240, depth_value=180, image_width=640, image_height=480)
    print(f"3D coordinates: {result}")