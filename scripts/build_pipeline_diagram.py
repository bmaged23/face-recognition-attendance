"""Generate a matching SVG and animated GIF from one reviewed pipeline layout."""
from html import escape
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'docs/media'
W, H = 1440, 1160
BACKGROUND, TEXT, MUTED = '#0b1220', '#e7edf7', '#9fb0c8'
PURPLE, CYAN, ORANGE = '#bda0ff', '#64dec7', '#ffbe76'

# Identifier, rectangle, title, subtitle, category color.
NODES = [
    ('photo', (60, 120, 280, 210), 'One photo + name', 'English display name', PURPLE),
    ('validate', (340, 120, 560, 210), 'MTCNN alignment', 'Require exactly one face', PURPLE),
    ('enroll', (620, 120, 840, 210), 'AdaFace embedding', '112×112 BGR → 512-D', PURPLE),
    ('store', (1100, 120, 1360, 210), 'Qdrant gallery', 'UUID + name + embedding', PURPLE),
    ('input', (60, 370, 280, 460), 'Video / live camera', 'Source time / wall time', CYAN),
    ('detect', (340, 370, 560, 460), 'MTCNN detection', 'Align every detected face', CYAN),
    ('iou', (620, 370, 840, 460), 'IoU association', 'Current frame boxes · ≥0.5', CYAN),
    ('due', (930, 370, 1190, 460), 'Recognition due?', 'New track / ≥0.5 seconds', CYAN),
    ('embed', (930, 550, 1190, 640), 'AdaFace on GPU', 'Normalized 512-D embedding', CYAN),
    ('search', (620, 550, 840, 640), 'Cosine retrieval', 'Top-1 match + runner-up', CYAN),
    ('gate', (340, 550, 560, 640), 'Accept identity?', 'Score ≥0.4 · gap ≥0.05', CYAN),
    ('match', (60, 550, 280, 640), 'Match / Unknown', 'Retain current face box', CYAN),
    ('recognition', (60, 875, 340, 965), 'Recognition JSON', 'id · name · first-seen time', CYAN),
    ('attendance', (490, 875, 750, 965), 'Attendance mode', 'Explicit action + timestamp', ORANGE),
    ('checkin', (870, 790, 1090, 875), 'Check-in', 'First confirmed sighting', ORANGE),
    ('checkout', (870, 970, 1090, 1055), 'Check-out', 'First confirmed sighting', ORANGE),
    ('state', (1180, 870, 1410, 970), 'Attendance JSON', 'id · name · checkin · checkout', ORANGE),
]
EDGES = [
    ([(280,165),(340,165)], PURPLE), ([(560,165),(620,165)], PURPLE),
    ([(840,165),(1100,165)], PURPLE),
    ([(280,415),(340,415)], CYAN), ([(560,415),(620,415)], CYAN),
    ([(840,415),(930,415)], CYAN), ([(1060,460),(1060,550)], CYAN),
    ([(930,595),(840,595)], CYAN), ([(620,595),(560,595)], CYAN),
    ([(340,595),(280,595)], CYAN),
    # Gallery vectors are read during cosine search; no enrollment on video frames.
    ([(1230,210),(1230,285),(875,285),(875,495),(730,495),(730,550)], PURPLE),
    # Successful IoU reuse bypasses embedding/search, but preserves current boxes.
    ([(1190,415),(1380,415),(1380,710),(30,710),(30,595),(60,595)], MUTED),
    ([(170,640),(170,805),(200,805),(200,875)], CYAN),
    ([(170,805),(620,805),(620,875)], ORANGE),
    ([(750,920),(810,920),(810,833),(870,833)], ORANGE),
    ([(810,920),(810,1013),(870,1013)], ORANGE),
    ([(1090,833),(1135,833),(1135,900),(1180,900)], ORANGE),
    ([(1090,1013),(1135,1013),(1135,940),(1180,940)], ORANGE),
]
LABELS = [
    (60, 65, 'ENROLL ONCE · ONE REFERENCE PHOTO PER PERSON', PURPLE, 25),
    (60, 335, 'RECOGNIZE · DETECT EVERY FRAME', CYAN, 25),
    (865, 145, 'Generate UUID', PURPLE, 18),
    (750, 263, 'Read enrolled references', PURPLE, 18),
    (1070, 504, 'Yes', CYAN, 20),
    (1000, 695, 'No: reuse previous name + score', MUTED, 20),
    (185, 777, 'Known identities only · Unknown is omitted from JSON and attendance', MUTED, 20),
    (60, 1097, 'ANNOTATION: current boxes + names + optional score %; attendance adds a rolling name panel.', TEXT, 22),
    (60, 1131, 'Check-in / check-out is selected by the caller; the system does not infer entry or exit direction.', MUTED, 20),
]


def fonts(size):
    for candidate in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                      '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def arrow(points, size=10):
    x, y = points[-1]
    a, b = points[-2]
    angle = math.atan2(y-b, x-a)
    return [(x,y), (x-size*math.cos(angle-.5), y-size*math.sin(angle-.5)),
            (x-size*math.cos(angle+.5), y-size*math.sin(angle+.5))]


def point_along(points, fraction):
    lengths = [math.dist(a,b) for a,b in zip(points,points[1:])]
    remaining = fraction * sum(lengths)
    for a,b,length in zip(points,points[1:],lengths):
        if remaining <= length:
            return (a[0]+(b[0]-a[0])*remaining/length, a[1]+(b[1]-a[1])*remaining/length)
        remaining -= length
    return points[-1]


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', (W,H), BACKGROUND)
    draw = ImageDraw.Draw(image)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
           '<title id="title">Face recognition and optional attendance pipeline</title>',
           '<desc id="desc">One photo is enrolled as a UUID and AdaFace embedding in Qdrant. Video or live frames are detected and aligned with MTCNN. IoU reuses identities between half-second recognition refreshes. Fresh AdaFace embeddings are compared with the two closest references. Accepted identities are exported once per run; unknowns are omitted. Optional explicit check-in or check-out records the first accepted sighting in a shared attendance file. Current face boxes and optional percentages are annotated.</desc>',
           f'<rect width="{W}" height="{H}" fill="{BACKGROUND}"/>']
    for points,color in EDGES:
        draw.line(points, fill=color, width=3)
        draw.polygon(arrow(points), fill=color)
        svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in arrow(points))}" fill="{color}"/>')
    for _,(x1,y1,x2,y2),title,subtitle,color in NODES:
        title_size = 23
        while draw.textlength(title, font=fonts(title_size)) > x2-x1-24:
            title_size -= 1
        draw.rounded_rectangle((x1,y1,x2,y2),radius=12,fill='#142036',outline=color,width=2)
        draw.text(((x1+x2)/2,y1+21),title,font=fonts(title_size),anchor='mt',fill=TEXT)
        draw.text(((x1+x2)/2,y1+56),subtitle,font=fonts(15),anchor='mt',fill=MUTED)
        svg.extend([f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" rx="12" fill="#142036" stroke="{color}" stroke-width="2"/>',
                    f'<text x="{(x1+x2)/2}" y="{y1+39}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="{title_size}" fill="{TEXT}">{escape(title)}</text>',
                    f'<text x="{(x1+x2)/2}" y="{y1+70}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="15" fill="{MUTED}">{escape(subtitle)}</text>'])
    for x,y,label,color,size in LABELS:
        draw.text((x,y),label,font=fonts(size),anchor='ls',fill=color)
        svg.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans, sans-serif" font-size="{size}" fill="{color}">{escape(label)}</text>')
    svg.append('</svg>')
    (DESTINATION/'pipeline.svg').write_text('\n'.join(svg)+'\n')
    frames = []
    for frame_index in range(32):
        frame = image.copy()
        painter = ImageDraw.Draw(frame)
        for points,color in EDGES:
            x,y = point_along(points, frame_index/32)
            painter.ellipse((x-5,y-5,x+5,y+5),fill=color)
        frames.append(frame)
    frames[0].save(DESTINATION/'pipeline.gif',save_all=True,append_images=frames[1:],duration=80,loop=0,optimize=True)
    print(f'Generated animated and static diagrams in {DESTINATION}')


if __name__ == '__main__':
    main()
