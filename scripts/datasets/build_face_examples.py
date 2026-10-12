"""Illustrate real MTCNN crops/alignment and AdaFace vectors for the README."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from face_recognition.config import Settings
from face_recognition.core.backend import FaceBackend


def main():
    settings = Settings.from_env()
    backend = FaceBackend(settings)
    video_path = ROOT / 'input/chokepoint/P2E_S5_C2.mp4'
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, 12000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f'Cannot read example frame from {video_path}')
    faces = backend.detect_faces(frame)
    faces = sorted(faces, key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]), reverse=True)[:3]
    if len(faces) != 3:
        raise RuntimeError('Need three detected faces for the illustration')
    samples = backend.embed_faces(frame, faces)
    destination = ROOT / 'docs/media/faces'
    destination.mkdir(parents=True, exist_ok=True)
    examples = []
    width, height = 1140, 690
    canvas = Image.new('RGB', (width, height), '#0b1220')
    draw = ImageDraw.Draw(canvas)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 23) if font_path.exists() else ImageFont.load_default(size=23)
    small = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default(size=17)
    draw.text((28,20), 'From a detected face to a searchable identity embedding', font=font, fill='#e7edf7')
    for x,label in ((30,'1 · Detected crop'),(325,'2 · Landmark-aligned input'),(730,'3 · AdaFace embedding')):
        draw.text((x,70), label, font=small, fill='#64dec7')
    for index, (detection, sample) in enumerate(zip(faces, samples), 1):
        x1,y1,x2,y2 = detection.bbox
        x1,x2 = max(0,x1),min(frame.shape[1],x2)
        y1,y2 = max(0,y1),min(frame.shape[0],y2)
        cropped = Image.fromarray(frame[y1:y2,x1:x2,::-1])
        aligned = detection.aligned.convert('RGB')
        cropped.save(destination/f'face-{index}-crop.jpg',quality=95)
        aligned.save(destination/f'face-{index}-aligned.png')
        y = 120+(index-1)*177
        for image,x in ((cropped,30),(aligned,350)):
            preview = image.copy()
            preview.thumbnail((150,150))
            canvas.paste(preview,(x+(150-preview.width)//2,y+(150-preview.height)//2))
        draw.text((222,y+55),'→',font=font,fill='#64dec7')
        draw.text((565,y+55),'→',font=font,fill='#64dec7')
        vector = sample.embedding
        if vector.shape != (512,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1):
            raise RuntimeError('Invalid AdaFace example embedding')
        draw.text((730,y+20),'112 × 112 · BGR · [-1, 1]',font=small,fill='#9fb0c8')
        draw.text((730,y+60),'512 float values · L2 norm = 1',font=small,fill='#e7edf7')
        snippet = ', '.join(f'{v:+.3f}' for v in vector[:4])
        draw.text((730,y+100),f'[{snippet}, …]',font=small,fill='#bda0ff')
        examples.append({'example':index,'bbox':list(detection.bbox),'embedding_dimensions':512,
                         'embedding_norm':float(np.linalg.norm(vector)),
                         'first_eight_values':[round(float(v),5) for v in vector[:8]]})
    draw.text((30,655),'Real ChokePoint detections · crop → align → normalize → AdaFace → cosine search',font=small,fill='#9fb0c8')
    canvas.save(ROOT/'docs/media/face-preprocessing.png')
    (destination/'examples.json').write_text(json.dumps({'source_sequence':'P2E_S5_C2','source_seconds':12,
        'model':settings.model_name,'device':backend.device,'examples':examples},indent=2)+'\n')
    print(f'Created three actual face crop/alignment pairs and normalized GPU embeddings in {destination}')


if __name__ == '__main__':
    main()
