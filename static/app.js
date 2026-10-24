'use strict';
const $ = id => document.getElementById(id);
let ready = false, lastDigest = null, revision = 0;
const status = (text, error=false) => { $('status').textContent=text; $('status').className=error?'error':''; };
const el = (tag, text, className) => { const node=document.createElement(tag); if(text!==undefined)node.textContent=text; if(className)node.className=className; return node; };
$('threshold').addEventListener('input',()=>{$('threshold-value').value=$('threshold').value;revision++;if(lastDigest){resetResults();status('Similarity changed. Make a new digest to update the groups.');}});
function resetResults(){lastDigest=null;$('results').replaceChildren();$('stats').hidden=true;$('export').disabled=true;}
$('notes').addEventListener('input',()=>{revision++;if(lastDigest){resetResults();status('Notes changed. Make a new digest to update the groups.');}});
$('clear').addEventListener('click',()=>{revision++;$('notes').value='';resetResults();status('Cleared. Add a new pile whenever you’re ready.');$('notes').focus();});
function render(data){
  lastDigest=data;$('results').replaceChildren();$('stats').replaceChildren();$('stats').hidden=false;
  for(const [value,label] of [[data.message_count,'messages'],[data.group_count,'groups to review'],[data.repeated_count,'repeats grouped']]){const box=el('div',undefined,'stat');box.append(el('strong',String(value)),el('span',label));$('stats').append(box);}
  if(data.truncated_count)$('results').append(el('p',`${data.truncated_count} long message(s) used only their first 256 tokens for grouping. Their full text is preserved below. Review carefully.`,'warning'));
  data.groups.forEach((group,index)=>{
    const card=el('article',undefined,'group');
    card.append(el('div',`Group ${index+1} · ${group.messages.length} source message${group.messages.length===1?'':'s'}`,'group-meta'),el('p',group.representative,'group-title'));
    const details=el('details');details.append(el('summary','See every original'));const list=el('ol');
    group.messages.forEach((text,i)=>list.append(el('li',`[${group.source_ids[i]}] ${text}`)));details.append(list);card.append(details);
    const review=el('label',undefined,'review'),checkbox=el('input');checkbox.type='checkbox';checkbox.setAttribute('aria-label',`Mark group ${index+1} reviewed`);checkbox.addEventListener('change',()=>card.classList.toggle('reviewed',checkbox.checked));review.append(checkbox,document.createTextNode('Reviewed'));card.append(review);$('results').append(card);
  });
  status(`Done in ${(data.inference_ms/1000).toFixed(2)}s on your CPU. These are suggestions; check each original before acting.`);$('export').disabled=false;
}
$('digest').addEventListener('click',async()=>{
  if(!ready)return;const requestRevision=revision;$('digest').disabled=true;$('digest').textContent='Making it quieter…';$('export').disabled=true;status('Grouping messages with the local model…');
  try{const response=await fetch('/api/digest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:$('notes').value,threshold:Number($('threshold').value)})});const data=await response.json();if(requestRevision!==revision){status('Notes or settings changed. Make a new digest when ready.');return;}if(!response.ok)throw Error(data.error||'Could not make a digest.');render(data);}catch(error){if(requestRevision===revision){resetResults();status(error.message,true);}}finally{$('digest').disabled=false;$('digest').textContent='Make a steady digest →';}
});
$('export').addEventListener('click',()=>{
  if(!lastDigest)return;const lines=['# Steady Inbox digest','',`Model: ${lastDigest.model} @ ${lastDigest.revision}`,'',`${lastDigest.message_count} messages · ${lastDigest.group_count} suggested groups. All originals retained. Review before acting.`,''];
  lastDigest.groups.forEach((group,i)=>{lines.push(`## Group ${i+1}`,'',`> ${group.representative.replaceAll('\n',' ')}`,'');group.messages.forEach((message,j)=>lines.push(`- [${group.source_ids[j]}] ${message}`));lines.push('');});
  if(lastDigest.truncated_count)lines.push('Some long messages were truncated to 256 tokens for inference; the originals above remain complete.');
  const url=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/markdown;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='steady-inbox-digest.md';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
async function boot(){try{const response=await fetch('/api/status');if(!response.ok)throw Error('Local model is not ready.');const data=await response.json();ready=data.ready;$('digest').disabled=!ready;$('digest').textContent='Make a steady digest →';status('Your local model is ready. Try the example or add your own notes.');$('sample').addEventListener('click',()=>{revision++;$('notes').value=data.sample.join('\n');resetResults();status('Example loaded. These are synthetic notes; make a digest to try them.');});}catch(error){status(error.message,true);$('digest').textContent='Restart the local server';}}
boot();
