import ollama
import os
import cv2
import numpy as np

def preprocess_image(image_path, processed_path="preprocessed.jpg"):
    print(f"🛠️ Preprocessing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("❌ ERROR: Image file not found or unreadable!")
        return None

    # Optionally, resize for better OCR accuracy (keep color)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    # Add white padding to reduce edge text loss
    border_size = 32
    # If image is color, use [255,255,255] for white
    image = cv2.copyMakeBorder(
        image, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    cv2.imwrite(processed_path, image)
    print(f"✅ Preprocessed image saved as: {processed_path}")
    return processed_path

def extract_text_with_llama32_vision(image_path):
    print(f"📂 Checking if image exists at path: {image_path}")
    if not os.path.exists(image_path):
        print("❌ ERROR: Image file not found!")
        return

    # Preprocess the image
    processed_path = preprocess_image(image_path)
    if processed_path is None:
        return

    try:
        print("📥 Reading preprocessed image file...")
        with open(processed_path, "rb") as f:
            image_bytes = f.read()
        print(f"✅ Image loaded. Size: {len(image_bytes)} bytes")

        print("📤 Sending request to Llama 3.2-Vision model via Ollama...")
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[
                {
                    'role': 'user',
                    'content': (
                        "Extract all text from this image. "
                        "Return only the text, including anything at the edges or margins."
                    ),
                    'images': [image_bytes]
                }
            ]
        )

        print("📬 Response received from Ollama.")
        print("\n🔍 Extracted Text:\n" + "-" * 30)
        print(response['message']['content'])

    except Exception as e:
        print(f"❌ ERROR during processing: {e}")

# 👇 Replace with your test image
image_path = "image.png"

extract_text_with_llama32_vision(image_path)
