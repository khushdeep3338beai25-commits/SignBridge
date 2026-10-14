import './style.css';
import { GestureRecognizer, FilesetResolver } from '@mediapipe/tasks-vision';
import * as tf from '@tensorflow/tfjs';

const $ = s => document.querySelector(s);
const state = { landmarker:null, stream:null, raf:0, samples:[], sentence:[], history:[],
  model:null, labels:[], active:false, lastLabel:'', stable:[], lastAdded:'', muted:false,
  threshold:0.8, speechRate:1, latency:[], fps:0, frames:0, lastFps:performance.now(),
  hand:false, accepted:0, rejected:0, total:0, modelType:'Not trained' };
const vocab = ['HELLO','HELP','YES','NO','THANK YOU','PLEASE','STOP','WATER','I LOVE YOU','PEACE'];
const cannedMap = { Open_Palm:{label:'HELLO',text:'Hello!'}, Thumb_Up:{label:'YES',text:'Yes'}, Thumb_Down:{label:'NO',text:'No'}, Closed_Fist:{label:'STOP',text:'Stop'}, Pointing_Up:{label:'HELP',text:'I need help'}, Victory:{label:'PEACE',text:'Peace'}, ILoveYou:{label:'I LOVE YOU',text:'I love you'} };

const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

$('#app').innerHTML = `
<header class="topbar"><div class="brand"><div class="logo">🤟</div><div><b>SignBridge <span>AI</span></b><small>COMMUNICATION, CONNECTED</small></div></div>
<div class="top-status"><i id="statusDot"></i><span id="statusText">Camera inactive</span><span class="local-tag">● LOCAL-FIRST</span></div></header>
<div class="shell"><aside class="sidebar"><div class="side-label">WORKSPACE</div>
${[['dashboard','▦','Dashboard'],['recognition','◎','Live recognition'],['builder','☷','Sentence builder'],['lab','⌘','AI Training Lab'],['analytics','▥','Performance'],['about','ⓘ','About project']].map(x=>`<button class="nav ${x[0]==='dashboard'?'active':''}" data-page="${x[0]}"><span>${x[1]}</span>${x[2]}</button>`).join('')}
<div class="side-note"><div class="pulse"></div><b>AI system</b><p id="modelStatus">Hand tracker not initialized</p></div></aside>
<main class="main">
<section class="page active" id="page-dashboard"><div class="eyebrow">ACCESSIBILITY • EDGE AI</div><h1>Every gesture deserves<br><span>to be understood.</span></h1><p class="lead">A real-time hand landmark and sign-classification workspace. Your camera processing stays on this device.</p>
<div class="hero-card"><div><div class="eyebrow light">READY WHEN YOU ARE</div><h2>Turn hand signs into<br>words and voice.</h2><p>Track landmarks, recognize trained signs, and build a spoken message.</p><button class="primary" data-goto="recognition">Start recognition <span>→</span></button></div><div class="hero-art"><div class="orbit o1"></div><div class="orbit o2"></div><div class="hand">✋</div><div class="tag tag1">21 landmarks</div><div class="tag tag2">Edge AI</div></div></div>
<div class="stats"><div class="stat"><small>SUPPORTED LABELS</small><b id="labelCount">7</b><span>Configurable vocabulary</span></div><div class="stat"><small>SESSION PREDICTIONS</small><b id="predCount">0</b><span>Since page opened</span></div><div class="stat"><small>AVG. INFERENCE</small><b id="avgLatency">—</b><span>Live measurement</span></div><div class="stat"><small>MODEL STATUS</small><b class="small-stat" id="dashModel">Not trained</b><span>Custom model required</span></div></div>
<div class="section-head"><div><div class="eyebrow">QUICK ACCESS</div><h2>Your workspace</h2></div></div><div class="quick-grid"><button class="quick" data-goto="recognition"><b>◎ Live recognition</b><span>Camera, landmarks & predictions →</span></button><button class="quick" data-goto="lab"><b>⌘ Collect training data</b><span>Capture labeled landmark samples →</span></button><button class="quick" data-goto="analytics"><b>▥ Evaluate performance</b><span>Review live system measurements →</span></button></div>
</section>
<section class="page" id="page-recognition"><div class="page-head"><div><div class="eyebrow">LIVE COMPUTER VISION</div><h1>Live recognition</h1><p class="lead">Start the camera and show one of the supported hand poses. Built-in recognition works without training a custom model.</p></div><span class="pill" id="cameraPill">● Camera off</span></div>
<div class="recognition-grid"><div class="panel camera-panel"><div class="panel-head"><div><b>Camera workspace</b><small>MediaPipe Gesture Recognizer · built-in model</small></div><span class="live-badge">LIVE</span></div><div class="video-wrap"><video id="video" playsinline muted></video><canvas id="overlay"></canvas><div class="camera-placeholder" id="placeholder"><div class="cam-icon">◎</div><b>Your camera preview</b><span>Camera frames are processed locally</span></div><div class="video-label" id="handStatus">No hand detected</div></div><div class="camera-actions"><button class="primary" id="startCamera">Start camera</button><button class="secondary" id="stopCamera" disabled>Stop</button></div><p class="hint" id="cameraMessage">Camera access requires permission and a secure context (localhost is supported).</p></div>
<div class="panel prediction-panel"><div class="panel-head"><div><b>Recognition result</b><small>Built-in gesture model + optional custom model</small></div><span class="pill neutral" id="predStatus">Waiting</span></div><div class="prediction"><div class="prediction-icon">✧</div><div><small>RECOGNIZED SIGN</small><h2 id="predLabel">—</h2><span id="predDescription">Start the camera to detect hand landmarks.</span></div></div><div class="confidence"><div><span>Model confidence</span><b id="confidenceText">—</b></div><div class="bar"><i id="confidenceBar"></i></div></div><label class="field-label">Acceptance threshold <b id="thresholdText">80%</b></label><input id="threshold" type="range" min="50" max="99" value="80"><div class="divider"></div><div class="speech-row"><label class="toggle"><input type="checkbox" id="autoSpeak"><span></span> Auto-speak accepted signs</label><button class="secondary" id="speakLast">Speak result</button></div><div class="speech-row"><label class="toggle"><input type="checkbox" id="muted"><span></span> Mute speech</label><button class="secondary" id="stopSpeech">Stop speaking</button></div><label class="field-label">Speech rate <b id="rateText">1.0×</b></label><input id="speechRate" type="range" min="0.6" max="1.5" value="1" step="0.1"></div></div>
<div class="panel sentence-panel"><div class="panel-head"><div><b>Live sentence</b><small>Accepted signs become words</small></div><label class="toggle"><input id="autoBuild" type="checkbox" checked><span></span> Auto-build</label></div><textarea id="sentenceText" placeholder="Your recognized words will appear here…"></textarea><div class="button-row"><button class="secondary" id="undoWord">Undo</button><button class="secondary" id="removeWord">Remove last</button><button class="secondary" id="clearSentence">Clear</button><button class="secondary" id="copySentence">Copy</button><button class="primary" id="speakSentence">Speak sentence</button><button class="secondary" id="downloadText">Download .txt</button></div></div>
</section>
<section class="page" id="page-builder"><div class="eyebrow">COMMUNICATION</div><h1>Sentence builder</h1><p class="lead">Edit, copy, download, or speak a message built from recognized signs.</p><div class="panel"><textarea id="builderText" class="large-text" placeholder="Type or build a sentence…"></textarea><div class="button-row"><button class="secondary" id="builderCopy">Copy text</button><button class="secondary" id="builderDownload">Download .txt</button><button class="primary" id="builderSpeak">Speak sentence</button></div></div><div class="panel"><div class="panel-head"><b>Recent recognized signs</b><small>Current session</small></div><div id="history" class="history"><p class="empty">No accepted predictions yet.</p></div></div></section>
<section class="page" id="page-lab"><div class="eyebrow">MODEL DEVELOPMENT</div><h1>AI Training Lab</h1><p class="lead">Collect labeled landmark samples with your webcam, export a dataset, then train a classifier using the included Python pipeline.</p><div class="notice"><b>Model status: no trained model bundled</b><p>This project does not pretend to recognize signs before a real classifier is trained. Landmark tracking and data collection work independently.</p></div><div class="lab-grid"><div class="panel"><h3>Collect a sample</h3><label class="field-label">Sign label</label><select id="sampleLabel">${vocab.map(v=>`<option>${v}</option>`).join('')}<option value="__custom">＋ Custom label…</option></select><input id="customLabel" class="hidden" placeholder="Enter label"><p class="hint">Start the camera first. Keep your hand visible and capture varied examples.</p><button class="primary" id="captureSample">Capture current hand</button><button class="secondary" id="exportCsv">Export CSV</button><button class="secondary" id="exportJson">Export JSON</button><button class="danger" id="clearSamples">Clear dataset</button><p id="sampleMessage" class="hint">0 samples collected in this session.</p></div><div class="panel"><div class="panel-head"><h3>Dataset overview</h3><span class="pill neutral" id="sampleTotal">0 samples</span></div><div id="sampleTable"></div><p class="hint">Samples are stored in this browser's local storage. Clear them whenever you wish.</p></div></div><div class="panel"><h3>Train your classifier</h3><p>Export the JSON dataset, place it at <code>datasets/landmarks.json</code>, then follow <code>training/README.md</code>. The script produces evaluation artifacts and a TensorFlow.js model.</p><pre><code>python3 -m venv .venv\nsource .venv/bin/activate\npip install -r training/requirements.txt\npython training/train_classifier.py --data datasets/landmarks.json</code></pre><label class="field-label">Load a trained TensorFlow.js model URL (model.json)</label><input id="modelUrl" placeholder="/models/sign-classifier/model.json"><button class="primary" id="loadModel">Load model</button><p class="hint" id="modelMessage">Expected input: 63 normalized landmark values. See training documentation.</p></div></section>
<section class="page" id="page-analytics"><div class="eyebrow">MEASUREMENT, NOT GUESSWORK</div><h1>Performance analytics</h1><p class="lead">Live system metrics are measured in this session. Classification accuracy requires labeled held-out test data and is not fabricated.</p><div class="stats"><div class="stat"><small>AVERAGE LATENCY</small><b id="aLatency">—</b><span>Per inference / tracking call</span></div><div class="stat"><small>PROCESSING FPS</small><b id="aFps">—</b><span>Approximate camera loop rate</span></div><div class="stat"><small>ACCEPTED</small><b id="aAccepted">0</b><span>Model-accepted predictions</span></div><div class="stat"><small>REJECTED</small><b id="aRejected">0</b><span>Low-confidence/unstable</span></div></div><div class="panel"><div class="panel-head"><b>Session metrics</b><span class="pill">LIVE</span></div><div id="metricRows"></div><p class="hint">A high softmax score is not proof of correctness. Use the held-out evaluation report generated by the training pipeline for accuracy, precision, recall, F1 and confusion matrix.</p></div><div class="panel"><h3>Threshold experiment</h3><p>Change the confidence threshold in Live Recognition and compare acceptance/rejection counts across sessions. Record the threshold, accepted count, rejected count, and errors confirmed against labeled examples.</p></div></section>
<section class="page" id="page-about"><div class="eyebrow">PROJECT NOTES</div><h1>Technology with a human purpose.</h1><p class="lead">SignBridge AI explores how local computer vision can support basic communication.</p><div class="panel prose"><h3>Problem and approach</h3><p>People who use sign language may encounter communication barriers with people who do not understand their signs. This prototype tracks 21 hand landmarks, normalizes them, and can classify a limited set of trained static poses into text and speech.</p><h3>Built-in gesture labels</h3><p>The built-in model recognizes Open Palm, Closed Fist, Pointing Up, Thumb Up, Thumb Down, Victory, and I Love You. For this demo these map to HELLO, STOP, HELP, YES, NO, PEACE, and I LOVE YOU respectively. These are convenient prototype mappings, not a validated translation of Indian Sign Language or ASL.</p><h3>Important limitations</h3><ul><li>This is a limited-vocabulary prototype, not complete sign-language translation.</li><li>Sign languages have distinct vocabularies and grammar. Do not assume the example labels are universal signs.</li><li>Many signs require movement, facial expression, and context. This first version focuses on static hand poses.</li><li>Lighting, camera quality, hand orientation, and individual signing differences affect performance.</li><li>Softmax confidence alone cannot guarantee detection of unknown signs.</li><li>Do not use this prototype instead of a qualified interpreter in critical situations.</li></ul><h3>Privacy</h3><p>Camera frames are processed locally in the browser. Captured training samples are saved in this browser's local storage and can be exported or cleared. The application does not upload camera frames.</p><h3>Accessibility</h3><p>Use keyboard-accessible controls, visible status text, editable text output, and browser speech synthesis. Speech availability depends on your browser and operating system.</p></div></section>
<footer>SignBridge AI <span>•</span> Local-first accessibility prototype <span>•</span> Results depend on your trained model and data</footer>
</main></div>`;

function setStatus(text, kind='') { $('#statusText').textContent=text; $('#statusDot').className=kind; $('#cameraPill').textContent='● '+(state.active?'Camera active':'Camera off'); }
function page(name){document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id===`page-${name}`));document.querySelectorAll('.nav').forEach(n=>n.classList.toggle('active',n.dataset.page===name));}
document.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>page(b.dataset.page));
document.querySelectorAll('[data-goto]').forEach(b=>b.onclick=()=>page(b.dataset.goto));
function normalize(landmarks){
 const wrist=landmarks[0], shifted=landmarks.map(p=>[p.x-wrist.x,p.y-wrist.y,p.z-wrist.z]);
 const scale=Math.max(...shifted.map(p=>Math.hypot(...p)),1e-6);
 return shifted.flatMap(p=>p.map(v=>v/scale));
}
async function initTracker(){
 if(state.landmarker)return;
 $('#cameraMessage').textContent='Loading built-in gesture recognition model…';
 const vision=await FilesetResolver.forVisionTasks('/mediapipe/wasm');
 const options={baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'},runningMode:'VIDEO',numHands:1,minHandDetectionConfidence:0.45,minHandPresenceConfidence:0.45,minTrackingConfidence:0.45};
 state.landmarker=await GestureRecognizer.createFromOptions(vision,options);
 state.modelType='Built-in MediaPipe gesture model';
 $('#modelStatus').textContent='Built-in gesture model ready';
 $('#dashModel').textContent='Built-in model';
 $('#cameraMessage').textContent='Gesture model ready. Show one supported hand pose.';
}
$('#startCamera').onclick=async()=>{
 const button=$('#startCamera'), video=$('#video');
 button.disabled=true;
 try {
   setStatus('Initializing AI');
   $('#cameraMessage').textContent='Checking browser camera access…';
   if (!window.isSecureContext) throw new Error('InsecureContext: Open this app on localhost or an HTTPS URL.');
   if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera API unavailable: use a recent version of Chrome, Edge, or Safari and open this app on localhost or HTTPS.');
   await initTracker();
   // Avoid strict resolution constraints: let the browser select a supported webcam mode.
   let stream;
   try {
     stream = await navigator.mediaDevices.getUserMedia({
       video: { facingMode: { ideal: 'user' }, width: { ideal: 1280 }, height: { ideal: 720 } },
       audio: false
     });
   } catch (firstError) {
     // Some cameras reject ideal/facingMode constraints; retry with the simplest request.
     if (firstError?.name === 'NotAllowedError' || firstError?.name === 'PermissionDeniedError' ||
         firstError?.name === 'NotFoundError' || firstError?.name === 'NotReadableError') throw firstError;
     stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
   }
   state.stream=stream;
   video.srcObject=stream;
   await video.play();
   state.active=true;
   $('#placeholder').classList.add('hidden');
   $('#stopCamera').disabled=false;
   button.textContent='Camera running';
   setStatus('Camera active','ok');
   $('#cameraMessage').textContent='Camera connected. Keep your hand visible in the frame.';
   loop();
 } catch(error) {
   console.error('SignBridge startup error:', error);
   // DOM events can be rejected by some browser implementations; don't show "[object Event]".
   const err = error instanceof Error ? error : new Error(
     error?.message || error?.name || 'The browser returned an unknown camera error. Check site permissions and try again.'
   );
   if (state.stream) state.stream.getTracks().forEach(track=>track.stop());
   state.stream=null;
   video.srcObject=null;
   state.active=false;
   setStatus('Camera error','error');
   let tip='Check that the page is on localhost or HTTPS and allow camera access in browser settings.';
   if (/InsecureContext/.test(err.message) || err.name==='SecurityError') tip='Camera access requires localhost or HTTPS. Run `npm run dev` and open the localhost URL.';
   else if (err.name==='NotAllowedError' || err.name==='PermissionDeniedError') tip='Camera permission was denied. In Safari on Mac, open Safari > Settings > Websites > Camera, allow this site, then reload.';
   else if (err.name==='NotFoundError' || err.name==='DevicesNotFoundError') tip='No camera was found. Connect or enable your webcam and try again.';
   else if (err.name==='NotReadableError' || err.name==='TrackStartError') tip='The camera may be in use by another app. Close FaceTime, Zoom, or other camera apps and retry.';
   else if (err.name==='OverconstrainedError') tip='The selected camera does not support the requested settings. Try another camera.';
   else if (err.message.includes('Camera API unavailable')) tip=err.message;
   $('#cameraMessage').textContent=`Could not start camera. ${err.message || err.name || 'Unknown error'}. ${tip}`;
   button.disabled=false;
   button.textContent='Retry camera';
 }
};
$('#stopCamera').onclick=()=>{state.active=false;cancelAnimationFrame(state.raf);if(state.stream)state.stream.getTracks().forEach(t=>t.stop());state.stream=null;lastBuiltinAccepted='';$('#video').srcObject=null;$('#placeholder').classList.remove('hidden');$('#stopCamera').disabled=true;$('#startCamera').disabled=false;$('#startCamera').textContent='Start camera';$('#overlay').getContext('2d').clearRect(0,0,$('#overlay').width,$('#overlay').height);setStatus('Camera inactive');$('#handStatus').textContent='No hand detected';};
function draw(landmarks){
 const c=$('#overlay'),v=$('#video'),r=v.getBoundingClientRect();if(c.width!==Math.round(r.width*devicePixelRatio)||c.height!==Math.round(r.height*devicePixelRatio)){c.width=Math.round(r.width*devicePixelRatio);c.height=Math.round(r.height*devicePixelRatio);}
 const ctx=c.getContext('2d');ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);ctx.clearRect(0,0,r.width,r.height);
 if(!landmarks)return;
 const links=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
 ctx.lineWidth=2;ctx.strokeStyle='#22d3ee';for(const [a,b] of links){ctx.beginPath();ctx.moveTo(landmarks[a].x*r.width,landmarks[a].y*r.height);ctx.lineTo(landmarks[b].x*r.width,landmarks[b].y*r.height);ctx.stroke();}
 for(const p of landmarks){ctx.beginPath();ctx.arc(p.x*r.width,p.y*r.height,3.2,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();ctx.strokeStyle='#4f46e5';ctx.lineWidth=2;ctx.stroke();}
}
async function loop(){
 if(!state.active)return;const v=$('#video');if(v.readyState>=2){
 const t=performance.now(),result=state.landmarker.recognizeForVideo(v,t),elapsed=performance.now()-t;state.latency.push(elapsed);if(state.latency.length>120)state.latency.shift();state.frames++;
 const lm=result.landmarks?.[0];state.hand=!!lm;draw(lm);$('#handStatus').textContent=lm?'Hand detected · 21 landmarks':'No hand detected';setStatus(lm?'Hand detected':'Camera active',lm?'ok':'');
 if(lm){window.currentFeatures=normalize(lm);const category=result.gestures?.[0]?.[0];if(category&&category.categoryName&&category.categoryName!=='None'){const mapped=cannedMap[category.categoryName];if(mapped)showBuiltinPrediction(mapped.label,mapped.text,category.score||0);else resetBuiltinPrediction();}else resetBuiltinPrediction();if(state.model)await predict(window.currentFeatures);}
 if(t-state.lastFps>=1000){state.fps=state.frames*1000/(t-state.lastFps);state.frames=0;state.lastFps=t;updateMetrics();}
 }
 state.raf=requestAnimationFrame(loop);
}
let lastBuiltinAccepted='';
function resetBuiltinPrediction(){if(state.model)return;$('#predLabel').textContent='—';$('#predStatus').textContent='Waiting';$('#predDescription').textContent='Hand detected. Try an open palm, thumbs up/down, pointing up, fist, V sign, or I-love-you handshape.';$('#confidenceText').textContent='—';$('#confidenceBar').style.width='0%';}
function showBuiltinPrediction(label,spoken,confidence){
 if(state.model)return;
 state.total++;const accepted=confidence>=state.threshold;$('#predLabel').textContent=accepted?label:'LOW CONFIDENCE';$('#confidenceText').textContent=`${(confidence*100).toFixed(1)}%`;$('#confidenceBar').style.width=`${confidence*100}%`;$('#predStatus').textContent=accepted?'Recognized':'Try again';$('#predDescription').textContent=accepted?`${spoken} · built-in gesture mapping`:'Hold the pose clearly and try again.';
 if(accepted&&label!==lastBuiltinAccepted){state.accepted++;if($('#autoBuild').checked)addWord(label);if($('#autoSpeak').checked&&!state.muted)speak(spoken);state.lastLabel=label;addHistory(label,confidence);lastBuiltinAccepted=label;}
 if(!accepted)state.rejected++;updateMetrics();
}
async function predict(features){
 const start=performance.now();try{
 const tensor=tf.tensor2d([features]);const out=state.model.predict(tensor);const probs=Array.from(await out.data());tensor.dispose();out.dispose();
 const idx=probs.indexOf(Math.max(...probs)),confidence=probs[idx],label=state.labels[idx]||`CLASS ${idx+1}`;
 state.total++;const accepted=confidence>=state.threshold;state.stable.push({label,confidence,accepted});if(state.stable.length>5)state.stable.shift();
 const stable=state.stable.length>=3&&state.stable.slice(-3).every(x=>x.label===label&&x.accepted);
 $('#predLabel').textContent=accepted?label:'UNKNOWN';$('#confidenceText').textContent=`${(confidence*100).toFixed(1)}%`;$('#confidenceBar').style.width=`${confidence*100}%`;
 $('#predStatus').textContent=!accepted?'Rejected':stable?'Accepted':'Stabilizing';$('#predDescription').textContent=!accepted?'Confidence below threshold. Try a clearer pose.':stable?'Stable prediction.':'Waiting for a stable prediction…';
 if(!accepted)state.rejected++;else if(stable){state.accepted++;if(state.lastLabel!==label){if($('#autoBuild').checked)addWord(label);if($('#autoSpeak').checked&&!state.muted)speak(label);state.lastLabel=label;addHistory(label,confidence);}}
 state.latency.push(performance.now()-start);updateMetrics();
 }catch(e){console.error(e);$('#modelMessage').textContent='Prediction error: '+e.message;}
}
function addWord(label){state.sentence.push(label);syncSentence();}
function syncSentence(){const text=state.sentence.join(' ').replaceAll(' THANK YOU',' THANK YOU');$('#sentenceText').value=text;$('#builderText').value=text;}
function addHistory(label,conf){state.history.unshift({label,conf,time:new Date().toLocaleTimeString()});state.history=state.history.slice(0,12);$('#history').innerHTML=state.history.map(h=>`<div class="history-row"><b>${esc(h.label)}</b><span>${(h.conf*100).toFixed(1)}%</span><small>${h.time}</small></div>`).join('');}
function speak(text){if(!('speechSynthesis'in window)){alert('Speech synthesis is unavailable in this browser.');return;}speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.rate=state.speechRate;speechSynthesis.speak(u);}
$('#threshold').oninput=e=>{state.threshold=+e.target.value/100;$('#thresholdText').textContent=e.target.value+'%';};
$('#speechRate').oninput=e=>{state.speechRate=+e.target.value;$('#rateText').textContent=state.speechRate.toFixed(1)+'×';};
$('#muted').onchange=e=>state.muted=e.target.checked;
$('#speakLast').onclick=()=>speak($('#predLabel').textContent==='—'?'':$('#predLabel').textContent);
$('#stopSpeech').onclick=()=>speechSynthesis?.cancel();
$('#sentenceText').oninput=e=>{state.sentence=e.target.value.trim()?e.target.value.trim().split(/\s+/):[];$('#builderText').value=e.target.value;};
$('#builderText').oninput=e=>{$('#sentenceText').value=e.target.value;state.sentence=e.target.value.trim()?e.target.value.trim().split(/\s+/):[];};
$('#undoWord').onclick=()=>{state.sentence.pop();syncSentence();};$('#removeWord').onclick=()=>{state.sentence.pop();syncSentence();};
$('#clearSentence').onclick=()=>{state.sentence=[];syncSentence();};$('#copySentence').onclick=()=>navigator.clipboard.writeText($('#sentenceText').value);
$('#speakSentence').onclick=()=>speak($('#sentenceText').value);$('#builderSpeak').onclick=()=>speak($('#builderText').value);$('#builderCopy').onclick=()=>navigator.clipboard.writeText($('#builderText').value);
function download(name,content,type='text/plain'){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([content],{type}));a.download=name;a.click();URL.revokeObjectURL(a.href);}
$('#downloadText').onclick=()=>download('signbridge-conversation.txt',$('#sentenceText').value);
$('#builderDownload').onclick=()=>download('signbridge-conversation.txt',$('#builderText').value);
$('#sampleLabel').onchange=e=>$('#customLabel').classList.toggle('hidden',e.target.value!=='__custom');
function selectedLabel(){return $('#sampleLabel').value==='__custom'?$('#customLabel').value.trim().toUpperCase():$('#sampleLabel').value;}
function saveSamples(){localStorage.setItem('signbridge-samples',JSON.stringify(state.samples));renderSamples();}
function renderSamples(){const counts={};state.samples.forEach(s=>counts[s.label]=(counts[s.label]||0)+1);$('#sampleTotal').textContent=`${state.samples.length} samples`;$ ('#sampleMessage').textContent=`${state.samples.length} samples collected in this session.`;$('#sampleTable').innerHTML=Object.keys(counts).length?`<table><thead><tr><th>Class</th><th>Samples</th></tr></thead><tbody>${Object.entries(counts).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${v}</td></tr>`).join('')}</tbody></table>`:'<p class="empty">No samples yet. Capture a hand pose.</p>';}
try{state.samples=JSON.parse(localStorage.getItem('signbridge-samples')||'[]');}catch{}renderSamples();
$('#captureSample').onclick=()=>{const label=selectedLabel();if(!label){$('#sampleMessage').textContent='Enter a custom label first.';return;}if(!window.currentFeatures){$('#sampleMessage').textContent='No hand landmarks detected. Start the camera and show your hand.';return;}state.samples.push({label,features:window.currentFeatures,session:new Date().toISOString()});saveSamples();};
$('#clearSamples').onclick=()=>{if(confirm('Delete all locally stored training samples?')){state.samples=[];saveSamples();}};
$('#exportJson').onclick=()=>download('signbridge-landmarks.json',JSON.stringify({feature_schema:'wrist-relative xyz, max-radius normalized, 21 landmarks, 63 values',samples:state.samples},null,2),'application/json');
$('#exportCsv').onclick=()=>{const head=['label',...Array.from({length:63},(_,i)=>`f${i}`),'session'];const rows=state.samples.map(s=>[s.label,...s.features,s.session]);download('signbridge-landmarks.csv',[head,...rows].map(r=>r.map(x=>`"${String(x).replaceAll('"','""')}"`).join(',')).join('\\n'),'text/csv');};
$('#loadModel').onclick=async()=>{try{const url=$('#modelUrl').value.trim();if(!url)throw Error('Enter a model.json URL.');state.model=await tf.loadLayersModel(url);const meta=await fetch(url.replace(/model\\.json(?:\\?.*)?$/,'metadata.json'));if(meta.ok){const m=await meta.json();state.labels=m.labels||m.class_names||[];}if(!state.labels.length)throw Error('Model loaded, but metadata.json must include a labels array.');state.modelType='Custom TensorFlow.js model';$('#modelStatus').textContent=state.modelType;$('#dashModel').textContent='Loaded';$('#modelMessage').textContent=`Loaded model with ${state.labels.length} labels.`;}catch(e){state.model=null;$('#modelMessage').textContent='Could not load model: '+e.message;}};
function updateMetrics(){const avg=state.latency.length?state.latency.reduce((a,b)=>a+b,0)/state.latency.length:0;const val=avg?avg.toFixed(1)+' ms':'—';$('#avgLatency').textContent=val;$('#aLatency').textContent=val;$('#aFps').textContent=state.fps?state.fps.toFixed(1):'—';$('#predCount').textContent=state.total;$('#aAccepted').textContent=state.accepted;$('#aRejected').textContent=state.rejected;$('#metricRows').innerHTML=[['Model','MediaPipe hand tracker'+(state.model?' + classifier':'')],['Mean measured processing time',val],['Approx. camera-loop FPS',state.fps?state.fps.toFixed(1):'—'],['Accepted predictions',state.accepted],['Rejected predictions',state.rejected],['Tracked hands',state.hand?'Currently detected':'Not detected']].map(([a,b])=>`<div class="metric-row"><span>${a}</span><b>${b}</b></div>`).join('');}
window.addEventListener('beforeunload',()=>{if(state.stream)state.stream.getTracks().forEach(t=>t.stop());});
