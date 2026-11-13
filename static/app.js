const $ = id => document.getElementById(id);
const fileIn = $('file'), drop = $('drop'), go = $('go'), err = $('err');
let chosen = null;

function pick(f) {
  err.hidden = true;
  if (!f) return;
  if (!/^image\/(png|jpe?g)$/.test(f.type)) return showErr('Please choose a JPG or PNG image.');
  if (f.size > 8 * 1024 * 1024) return showErr('File too large (max 8 MB).');
  chosen = f;
  $('preview').src = URL.createObjectURL(f);
  $('preview').hidden = false;
  $('dropText').hidden = true;
  go.disabled = false;
}
function showErr(m) { err.textContent = m; err.hidden = false; }
function show(which) { ['empty', 'loading', 'result'].forEach(k => $(k).hidden = k !== which); }

fileIn.onchange = () => pick(fileIn.files[0]);
['dragover', 'dragenter'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', ev => pick(ev.dataTransfer.files[0]));

go.onclick = async () => {
  if (!chosen) return;
  go.disabled = true; err.hidden = true; show('loading');
  const fd = new FormData();
  fd.append('image', chosen);
  fd.append('patient', $('patient').value);
  try {
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    if (res.status === 401) { location.href = '/login?next=/analyze'; return; }
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'Something went wrong.');
    render(d);
  } catch (e) {
    show('empty'); showErr(e.message);
  } finally { go.disabled = false; }
};

function render(d) {
  const v = $('verdict');
  v.className = 'verdict ' + (d.tumor ? 't' : 'n');
  v.textContent = d.headline;
  const s = document.createElement('small');
  s.textContent = 'Confidence ' + d.confidence + '%';
  v.appendChild(s);
  $('warn').hidden = !d.uncertain;
  $('info').textContent = d.info;
  const bars = $('bars'); bars.textContent = '';
  const fills = [];
  d.probs.forEach(p => {
    const row = document.createElement('div'); row.className = 'bar';
    const name = document.createElement('span'); name.textContent = p.name;
    const track = document.createElement('div'); track.className = 'track';
    const fill = document.createElement('i'); if (p.tumor) fill.className = 't';
    track.appendChild(fill);
    const val = document.createElement('em'); val.textContent = p.pct + '%';
    row.append(name, track, val); bars.appendChild(row); fills.push([fill, p.pct]);
  });
  setTimeout(() => fills.forEach(([f, pct]) => f.style.width = pct + '%'), 50);
  $('imgO').src = d.original_url; $('imgH').src = d.heatmap_url;
  $('rep').href = d.report_url;
  $('secs').textContent = 'Analysed in ' + d.seconds + ' s.';
  show('result');
  if (window.innerWidth < 860) $('resultCard').scrollIntoView({ behavior: 'smooth' });
}

$('again').onclick = () => {
  chosen = null; fileIn.value = ''; $('patient').value = '';
  $('preview').hidden = true; $('dropText').hidden = false; go.disabled = true;
  show('empty'); window.scrollTo({ top: $('tool').offsetTop - 80, behavior: 'smooth' });
};
