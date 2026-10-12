'use strict';

// Relative URLs work with the localhost server and with a local HTML file.
const clips = [
  { source: '../../output/chokepoint_accuracy/improved/P2E_S5_C2_annotated.mp4', start: 3,
    label: 'Check-in · starts at 00:03' },
  { source: '../../output/chokepoint_accuracy/improved/P2L_S5_C2_annotated.mp4', start: 28,
    label: 'Check-out · starts at 00:28' }
];
const video = document.getElementById('demo-video');
const status = document.getElementById('status');
let index = 0;
let waitingForSeek = false;

function loadClip(nextIndex) {
  index = nextIndex;
  waitingForSeek = true;
  document.getElementById('clip-label').textContent = clips[index].label;
  clips.forEach((clip, i) => document.getElementById(`step-${i}`).classList.toggle('active', i === index));
  status.textContent = `Loading ${clips[index].label.toLowerCase()}…`;
  video.pause();
  // Autoplay only after the requested seek, so the first seconds do not flash.
  video.autoplay = false;
  video.src = clips[index].source;
  video.load();
}

video.addEventListener('loadedmetadata', () => {
  if (clips[index].start >= video.duration) {
    status.textContent = 'The requested start time is outside this video.';
    waitingForSeek = false;
    return;
  }
  video.currentTime = clips[index].start;
});

video.addEventListener('seeked', async () => {
  if (!waitingForSeek) return;
  waitingForSeek = false;
  try {
    await video.play();
    status.textContent = 'Playing · check-in → check-out → repeat';
  } catch {
    status.textContent = 'Press Play to start. The playlist will repeat automatically.';
  }
});
video.addEventListener('ended', () => loadClip((index + 1) % clips.length));
video.addEventListener('error', () => {
  status.textContent = 'Cannot load the video. Start python scripts/serve_demo.py from the project root and open http://localhost:8080/.';
});
document.getElementById('next').addEventListener('click', () => loadClip((index + 1) % clips.length));
document.getElementById('restart').addEventListener('click', () => loadClip(0));
loadClip(0);
