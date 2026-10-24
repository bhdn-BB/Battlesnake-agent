const $ = id => document.getElementById(id);
const canvas = $('board'), ctx = canvas.getContext('2d');
let game = null, busy = false, checkpoints = [];
let watchTimer = null, watching = false;
const names = {up:'↑',down:'↓',left:'←',right:'→'};

async function api(path, body, method = body ? 'POST' : 'GET') {
  const response = await fetch(path, {method, headers:{'Content-Type':'application/json'}, ...(body ? {body:JSON.stringify(body)} : {})});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Check the entered values.');
  return data;
}
function error(message = '') { $('error').textContent = message; $('error').hidden = !message; }
function setBusy(value) {
  busy = value;
  $('start').disabled = value; $('overlay-start').disabled = value;
  $('watch').disabled = value || !checkpoints.length;
  $('slot').disabled = value;
  $('pause').disabled = !game || game.done;
  document.querySelectorAll('[data-move]').forEach(button => button.disabled = value || !game || game.done || game.mode==='watch');
}
function draw(board) {
  const n = board?.width || 11, cell = canvas.width / n;
  ctx.fillStyle = '#203530'; ctx.fillRect(0,0,canvas.width,canvas.height);
  for (let y=0;y<n;y++) for(let x=0;x<n;x++) {
    ctx.fillStyle = (x+y)%2 ? '#213730' : '#233932';
    ctx.fillRect(x*cell+1,y*cell+1,cell-2,cell-2);
    ctx.fillStyle = '#375046'; ctx.fillRect((x+.5)*cell-1,(y+.5)*cell-1,2,2);
  }
  if (!board) return;
  const pos = p => [(p.x+.5)*cell,(n-p.y-.5)*cell];
  for (const food of board.food || []) {
    const [x,y]=pos(food); ctx.fillStyle='#f1e3b3'; ctx.beginPath();ctx.arc(x,y,cell*.115,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle='#f1e3b340';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(x,y,cell*.20,0,Math.PI*2);ctx.stroke();
  }
  for (const snake of board.snakes) {
    const own=snake.id===game.player, dead=Boolean(snake.eliminated_cause);
    ctx.globalAlpha=dead ? .23 : 1;
    ctx.strokeStyle=own ? '#b6ed8b' : '#f7a971';ctx.fillStyle=ctx.strokeStyle;
    ctx.lineWidth=cell*.57;ctx.lineJoin='round';ctx.lineCap='round';ctx.beginPath();
    snake.body.forEach((p,i)=>{const [x,y]=pos(p); if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();
    for (const p of snake.body) {const [x,y]=pos(p);ctx.beginPath();ctx.arc(x,y,cell*.285,0,Math.PI*2);ctx.fill();}
    const [hx,hy]=pos(snake.body[0]); const next=snake.body[1] || snake.body[0];
    let dx=snake.body[0].x-next.x, dy=-(snake.body[0].y-next.y);if(!dx&&!dy)dy=-1;
    ctx.fillStyle='#23352b';
    for (const side of [-1,1]) {ctx.beginPath();ctx.arc(hx+dx*cell*.12-dy*side*cell*.115,hy+dy*cell*.12+dx*side*cell*.115,cell*.043,0,Math.PI*2);ctx.fill();}
    ctx.globalAlpha=1;
  }
}
function render() {
  const watch=game?.mode==='watch';
  const result={
    win:watch ? ['PPO WINS','PPO defeated the Teacher.'] : ['YOU WIN','You defeated your opponent.'],
    loss:watch ? ['TEACHER WINS','The Teacher defeated PPO.'] : ['YOU LOSE','Your opponent won.'],
    draw:['DRAW','Both snakes were eliminated.'],
    truncation:['TURN LIMIT REACHED','Neither snake won before the turn limit.']
  }[game?.outcome] || ['DUEL FINISHED','The duel has ended.'];
  draw(game?.board);
  $('turn').textContent=String(game?.board.turn || 0).padStart(3,'0');
  $('phase').textContent=!game ? 'READY TO DUEL' : game.done ? result[0] : watch ? (watching ? 'PPO × TEACHER · LIVE' : 'PPO × TEACHER · PAUSED') : 'YOUR TURN';
  $('duel-mode').textContent=watch ? 'PPO × TEACHER · SPECTATOR' : 'HUMAN × MODEL';
  $('player-legend').textContent=watch ? 'TEACHER' : 'YOU';
  $('model-legend').textContent=watch ? 'PPO' : 'MODEL';
  $('player-name').textContent=watch ? 'Teacher · algorithm' : 'You';
  $('model-name').textContent=watch ? 'PPO · neural network' : 'Model';
  $('slot-label').textContent=watch ? 'Teacher position' : 'Your position';
  $('controls-title').textContent=watch ? 'Watch the agents play' : 'One keypress, one move';
  $('controls-help').textContent=watch ? 'Green: Teacher. Orange: PPO. Pause or change the playback speed above.' : 'Arrow keys or WASD. Food increases length and restores health. Avoid walls and snake bodies.';
  $('watch-controls').hidden=!watch;
  $('pause').textContent=watching ? 'Pause' : 'Resume';
  if (game) {
    for (const [prefix,id] of [['human',game.player],['model',game.opponent]]) {
      const snake=game.board.snakes.find(s=>s.id===id);
      $(prefix+'-health').style.width=`${snake.eliminated_cause ? 0 : Math.max(0,snake.health)}%`;
      $(prefix+'-length').textContent=`${snake.body.length} cells / ${Math.max(0,snake.health)} HP`;
    }
    $('last-move').textContent=game.model_move ? (watch ? `Teacher: ${names[game.teacher_move]} · PPO: ${names[game.model_move]}` : `Last model move: ${names[game.model_move]}`) : 'Both moves are resolved simultaneously.';
    $('model-latency').textContent=game.model_latency_ms!=null ? `Model decision time: ${game.model_latency_ms.toFixed(3)} ms` : game.checkpoint==='teacher' ? 'Scripted opponent · no neural inference' : 'Model decision time: —';
  }
  $('overlay').hidden=Boolean(game && !game.done);
  if(game?.done) {
    $('overlay-title').textContent=result[0];
    $('overlay-text').textContent=`${result[1]} ${game.board.turn} turns.`;
    $('overlay-start').textContent=watch ? 'Watch another game ↗' : 'Play again ↗';
  }
  setBusy(busy);
}
function modelMeta() {
  const item=checkpoints.find(row=>row.id===$('checkpoint').value);
  $('model-meta').textContent=item ? `${item.algorithm.toUpperCase()} · ${item.transitions==null ? 'steps unknown' : Number(item.transitions).toLocaleString('en-US')+' steps'} · ${(item.bytes/1048576).toFixed(1)} MB` : 'A scripted opponent for warm-up games.';
}
async function refreshModels(initial=false) {
  try {
    const selected=$('checkpoint').value;
    checkpoints=(await api('/api/checkpoints')).checkpoints;
    $('checkpoint').replaceChildren();
    // Put the new duel run first, followed by trained Kaggle models.
    const priority=row=>row.id.endsWith('selected.ts') ? 0 : row.id.endsWith('champion.ts') ? 1 : row.id.endsWith('best.ts') ? 2 : 3;
    checkpoints.sort((a,b)=>priority(a)-priority(b));
    for(const row of checkpoints) $('checkpoint').add(new Option(row.id,row.id));
    $('checkpoint').add(new Option('Teacher · scripted opponent','teacher'));
    const preferred=initial ? checkpoints.find(row=>priority(row)<=2)?.id : selected;
    if(preferred&&[...$('checkpoint').options].some(o=>o.value===preferred)) $('checkpoint').value=preferred;
    modelMeta();
    setBusy(busy);
  } catch(e) {error(e.message);}
}
function scheduleWatch() {
  clearTimeout(watchTimer);watchTimer=null;
  if(watching&&game?.mode==='watch'&&!game.done&&!busy)
    watchTimer=setTimeout(watchStep,Number($('watch-speed').value)||400);
}
async function watchStep() {
  watchTimer=null;
  if(!watching||busy||!game||game.done||game.mode!=='watch')return;
  setBusy(true);error();
  try {
    game=await api(`/api/games/${game.id}/watch`,{turn:game.board.turn});
    if(game.done)watching=false;
  } catch(e) {watching=false;error(e.message);}
  finally {setBusy(false);render();scheduleWatch();}
}
async function start({keepSeed=false,mode='play'}={}) {
  if(mode==='watch'&&$('checkpoint').value==='teacher') {
    if(!checkpoints.length){error('Load a model checkpoint to watch PPO vs Teacher.');return;}
    $('checkpoint').value=checkpoints[0].id;modelMeta();
  }
  if(busy)return;setBusy(true);error();
  const old=game;
  clearTimeout(watchTimer);watchTimer=null;
  try {
    const body={checkpoint:$('checkpoint').value,slot:Number($('slot').value),mode};
    if(keepSeed&&old?.seed!=null)body.seed=old.seed;
    else if($('seed').value)body.seed=Number($('seed').value);
    game=await api('/api/games',body);watching=mode==='watch';render();canvas.focus({preventScroll:true});
    if(old) api('/api/games/'+old.id,null,'DELETE').catch(()=>{});
  } catch(e) {
    if(old)$('slot').value=old.player==='s1' ? '1' : '0';
    error(e.message);
  } finally {setBusy(false);scheduleWatch();}
}
async function move(direction) {
  if(busy||!game||game.done||game.mode==='watch')return;
  setBusy(true);error();
  try {game=await api(`/api/games/${game.id}/moves`,{direction,turn:game.board.turn});render();}
  catch(e){error(e.message);}finally{setBusy(false);}
}
async function refreshTraining() {
  try {
    const result=await api('/api/training'),status=result.status;
    if(!status){
      const remote=result.remote;
      $('training-status').textContent=remote ? (remote.status==='monitoring' ? 'Kaggle: session running. Waiting for the first saved checkpoint…' : 'Kaggle: '+remote.status) : result.config ? 'Training is starting. Waiting for the first checkpoint…' : 'Training has not started yet.';
      $('training-detail').textContent='Imitation → Masked PPO · 1v1';
      return;
    }
    const percent=100*status.transitions/status.target_transitions;
    $('train-percent').textContent=percent.toFixed(1)+'%';$('train-progress').style.width=percent+'%';
    $('training-status').textContent=`${Number(status.transitions).toLocaleString('en-US')} / ${Number(status.target_transitions).toLocaleString('en-US')} steps · ${status.status==='trained' ? 'complete' : 'latest checkpoint'}`;
    const validation=[...result.metrics].reverse().find(row=>row.event==='validation');
    $('training-detail').textContent=`Imitation → Masked PPO · 1v1${validation ? ' · win rate '+Math.round(validation.win_rate*100)+'%' : ''}`;
  } catch { $('training-status').textContent='Could not update training progress.'; }
}
$('start').onclick=()=>start();$('overlay-start').onclick=()=>start({mode:game?.mode||'play'});$('refresh').onclick=()=>refreshModels();$('checkpoint').onchange=modelMeta;
$('watch').onclick=()=>start({mode:'watch'});
$('pause').onclick=()=>{watching=!watching;render();scheduleWatch();};
$('watch-speed').onchange=scheduleWatch;
$('slot').onchange=()=>{if(game&&!busy)return start({keepSeed:true,mode:game.mode||'play'});};
document.querySelectorAll('[data-move]').forEach(button=>button.onclick=()=>move(button.dataset.move));
document.addEventListener('keydown',event=>{
  if(event.repeat||event.ctrlKey||event.metaKey||event.altKey||['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;
  const key=event.key.length===1 ? event.key.toLowerCase() : event.key;
  const direction={ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right',w:'up',s:'down',a:'left',d:'right',ц:'up',і:'down',ф:'left',в:'right'}[key];
  if(direction){event.preventDefault();move(direction);}
});
draw();setBusy(false);refreshModels(true);refreshTraining();setInterval(refreshTraining,15000);
