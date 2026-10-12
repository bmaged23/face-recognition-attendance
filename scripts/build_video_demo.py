"""Combine approved check-in/check-out videos and build a looping README preview."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'output/chokepoint_accuracy/improved'
    media = ROOT / 'docs/media'
    media.mkdir(parents=True, exist_ok=True)
    video = media / 'attendance-demo.mp4'
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(source / 'P2E_S5_C2_annotated.mp4'),
        '-i', str(source / 'P2L_S5_C2_annotated.mp4'),
        '-filter_complex', '[0:v]trim=start=3,setpts=PTS-STARTPTS[a];'
        '[1:v]trim=start=28,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0[v]',
        '-map', '[v]', '-c:v', 'libx264', '-preset', 'slow', '-crf', '16',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', str(video),
    ], check=True)
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', str(video),
        '-filter_complex', 'fps=8,split[a][b];'
        '[a]palettegen=max_colors=256:stats_mode=single[p];[b][p]paletteuse=new=1:dither=bayer:bayer_scale=3:diff_mode=rectangle',
        '-loop', '0', str(media / 'attendance-demo.gif'),
    ], check=True)
    # Regular GitHub repositories reject files over 100 MiB.
    for artifact in (video, media / 'attendance-demo.gif'):
        if artifact.stat().st_size >= 100 * 1024 * 1024:
            raise RuntimeError(f'{artifact.name} exceeds the GitHub file-size budget')
    print(f'Updated combined video and looping preview under {media}')


if __name__ == '__main__':
    main()
