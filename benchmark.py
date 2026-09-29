import time
import os
import io
from model import extract_embedding_for_image, load_model_if_exists, predict_with_model

def run_benchmark():
    clf = load_model_if_exists()
    if not clf:
        print("No model found!")
        return

    dataset_dir = "dataset"
    if not os.path.isdir(dataset_dir):
        print("No dataset folder found.")
        return

    test_images = []
    for sid in os.listdir(dataset_dir):
        student_dir = os.path.join(dataset_dir, sid)
        if os.path.isdir(student_dir):
            for f in os.listdir(student_dir):
                if f.endswith('.jpg') or f.endswith('.jpeg'):
                    test_images.append(os.path.join(student_dir, f))
                    if len(test_images) >= 50: # collect up to 50 images
                        break
        if len(test_images) >= 50:
            break
            
    if not test_images:
        print("No images found.")
        return

    print(f"Benchmarking with {len(test_images)} images...")
    
    total_time = 0
    count = 0
    
    for img_path in test_images:
        with open(img_path, 'rb') as f:
            data = f.read()
            
        t0 = time.perf_counter()
        stream = io.BytesIO(data)
        emb = extract_embedding_for_image(stream)
        if emb is not None:
            predict_with_model(clf, emb)
            count += 1
        t1 = time.perf_counter()
        total_time += (t1 - t0)
        
    if count > 0:
        print(f"Successfully processed {count} faces out of {len(test_images)} images.")
        print(f"Total processing time (excluding file I/O): {total_time:.4f} seconds")
        print(f"Average time per successful recognition: {(total_time / count) * 1000:.2f} ms")
    else:
        print("No faces detected in the test sample.")

if __name__ == '__main__':
    run_benchmark()
