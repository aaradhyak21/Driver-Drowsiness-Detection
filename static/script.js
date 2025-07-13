const video = document.getElementById('video');
const statusEl = document.getElementById('status');
const earEl = document.getElementById('ear');
const marEl = document.getElementById('mar');
const tiltEl = document.getElementById('tilt');

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  });

setInterval(() => {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const image = canvas.toDataURL('image/jpeg');

  fetch('/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image })
  })
  .then(response => response.json())
  .then(data => {
    statusEl.textContent = data.status;
    earEl.textContent = data.ear;
    marEl.textContent = data.mar;
    tiltEl.textContent = data.tilt;
  });
}, 1000);