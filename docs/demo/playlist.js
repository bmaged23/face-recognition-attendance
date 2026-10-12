'use strict';

// The approved clips are trimmed and combined, so playback has no source switch.
const video = document.getElementById('demo-video');
const status = document.getElementById('status');
const checkoutStart = 25.0666667;
video.addEventListener('timeupdate', () => {
  const checkout = video.currentTime >= checkoutStart;
  document.getElementById('step-0').classList.toggle('active', !checkout);
  document.getElementById('step-1').classList.toggle('active', checkout);
  document.getElementById('clip-label').textContent = checkout ? 'Check-out · repeats into check-in' : 'Check-in · followed by check-out';
});
video.addEventListener('playing', () => { status.textContent = 'Playing · check-in → check-out → repeat'; });
video.addEventListener('error', () => { status.textContent = 'Cannot load the video. Try reloading the page or downloading the MP4 from the README.'; });
async function seekAndPlay(time) {
  video.currentTime = time;
  try { await video.play(); }
  catch { status.textContent = 'Press Play to start. The video repeats automatically.'; }
}
document.getElementById('next').addEventListener('click', () => seekAndPlay(checkoutStart));
document.getElementById('restart').addEventListener('click', () => seekAndPlay(0));
