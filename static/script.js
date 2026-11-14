const COLORS = ['#3c8d7b', '#b8a33a', '#d9922e', '#cf5b33', '#a3243b'];

const fileInput = document.getElementById('file');
const lens = document.getElementById('lens');
const preview = document.getElementById('preview');
const hint = document.getElementById('hint');
const busy = document.getElementById('busy');
const runBtn = document.getElementById('run');
const resetBtn = document.getElementById('reset');
const errorEl = document.getElementById('error');
const empty = document.getElementById('empty');
const result = document.getElementById('result');

let chosen = null;

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.hidden = !msg;
}

function setFile(file) {
  showError('');
  if (!file) return;
  if (!/^image\/(png|jpeg)$/.test(file.type)) {
    showError('Use a PNG or JPG image.');
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showError('Image is larger than 16 MB.');
    return;
  }
  chosen = file;
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  hint.hidden = true;
  runBtn.disabled = false;
  resetBtn.hidden = false;
  result.hidden = true;
  empty.hidden = false;
}

function reset() {
  chosen = null;
  fileInput.value = '';
  preview.hidden = true;
  hint.hidden = false;
  runBtn.disabled = true;
  resetBtn.hidden = true;
  result.hidden = true;
  empty.hidden = false;
  showError('');
}

lens.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
resetBtn.addEventListener('click', reset);

['dragenter', 'dragover'].forEach(ev =>
  lens.addEventListener(ev, e => { e.preventDefault(); lens.classList.add('drag'); }));
['dragleave', 'drop'].forEach(ev =>
  lens.addEventListener(ev, e => { e.preventDefault(); lens.classList.remove('drag'); }));
lens.addEventListener('drop', e => setFile(e.dataTransfer.files[0]));

runBtn.addEventListener('click', async () => {
  if (!chosen) return;
  showError('');
  runBtn.disabled = true;
  busy.hidden = false;

  const form = new FormData();
  form.append('image', chosen);

  try {
    const res = await fetch('/predict', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Something went wrong. Try again.');
    render(data);
  } catch (err) {
    showError(err.message || 'Could not reach the server.');
  } finally {
    busy.hidden = true;
    runBtn.disabled = false;
  }
});

function render(data) {
  document.getElementById('gradeNum').textContent = data.grade;
  document.getElementById('gradeName').textContent = data.label;
  result.style.setProperty('--grade', COLORS[data.grade]);

  document.querySelectorAll('#ladder li').forEach(li =>
    li.classList.toggle('on', Number(li.dataset.g) === data.grade));

  const bars = document.getElementById('bars');
  bars.innerHTML = '';
  data.probs.forEach((p, i) => {
    const li = document.createElement('li');
    li.style.setProperty('--c', COLORS[i]);
    if (i === data.grade) li.classList.add('top');
    li.innerHTML = '<span></span><span class="track"><span class="fill"></span></span><span class="pct"></span>';
    li.children[0].textContent = data.classes[i];
    li.children[2].textContent = (p * 100).toFixed(1) + '%';
    bars.appendChild(li);
    requestAnimationFrame(() => { li.querySelector('.fill').style.width = (p * 100) + '%'; });
  });

  empty.hidden = true;
  result.hidden = false;
}
