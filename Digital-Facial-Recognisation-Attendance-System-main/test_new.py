import os, pickle, cv2
from model import get_face_bbox, crop_face_and_embed, predict_with_model

clf = pickle.load(open('model.pkl', 'rb'))
res = []
for s in ['12', '13']:
    folder = f'dataset/{s}'
    if not os.path.exists(folder):
        continue
    files = os.listdir(folder)[:5]
    for f in files:
        p = os.path.join(folder, f)
        img = cv2.imread(p)
        bbox = get_face_bbox(img) if img is not None else None
        emb = crop_face_and_embed(img, bbox) if bbox is not None else None
        rec = predict_with_model(clf, emb) if emb is not None else None
        res.append({'s': s, 'pred': rec})

for r in res:
    print(r)
