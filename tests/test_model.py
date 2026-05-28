# test_model.py - Test CNN eye classifier
import os
import sys
import cv2

tests_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(tests_dir)
sys.path.append(os.path.join(root_dir, "src"))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from eye_classifier import EyeClassifier

classifier = EyeClassifier()

test_image_path = os.path.join(tests_dir, "test.jpg")
test_img = cv2.imread(test_image_path)

if test_img is None:
    print(f"Error: Could not load test image at {test_image_path}")
else:
    label, confidence = classifier.predict(test_img)
    print("----- CNN MODEL TEST RESULTS -----")
    print("Predicted Label:", label)
    print("Confidence Score:", f"{confidence * 100:.2f}%")
