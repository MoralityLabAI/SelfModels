// QSM-03B: finite-history action certificate. No dependencies, no QPU.
import fs from 'node:fs';
function qsm03window(){
 const N=128,SEEDS=[101,103,107],H=32,SPLIT=8,SWITCH=16,LENS=[1,2,4,6,8,10,12],theta=Math.PI/3,c=Math.cos(theta)**2;
 const F=[[.80,.03,.03,.14],[.55,.17,.17,.11]],hazard=.06,probes=[2,0,2,1],strength=[Math.sin(theta),Math.sin(theta),Math.sin(theta)**2],scenarios=["matched","self_switch","decoy_switch"];
 const sum=a=>a.reduce((s,x)=>s+x,0),mean=a=>sum(a)/a.length,unif=()=>Array(8).fill(1/8),sign=(i,b)=>((i&b)?1:-1),feat=(i,k)=>k===0?sign(i,2):k===1?sign(i,1):sign(i,2)*sign(i,1);
 const norm=p=>p.map(v=>v/sum(p));
 const T=Array.from({length:8},(_,i)=>Array.from({length:8},(_,j)=>((i>>2)===(j>>2)?1-hazard:hazard)*F[i>>2][(i^j)&3]));
 const mv=(M,p)=>M.map(r=>sum(r.map((v,j)=>v*p[j]))),mm=(M,Q)=>M.map(r=>Array.from({length:8},(_,j)=>sum(r.map((v,k)=>v*Q[k][j]))));
 const matrices=probes.map(k=>[-1,1].map(y=>T.map((r,i)=>r.map(v=>v*(1+y*strength[k]*feat(i,k))/2))));
 const g=(k,y,i)=>(1+y*strength[k]*feat(i,k))/2;
 const rewards=Array.from({length:4},(_,a)=>Array.from({length:8},(_,i)=>((i&2)===(a&2)?1:c)*((i&1)===(a&1)?1:c)));
 const val=p=>rewards.map(r=>sum(r.map((x,i)=>x*p[i])));
 const argmax=v=>v.indexOf(Math.max(...v));
 const cache=new Map();
 function entry(phase,L,mask){
  const key=(L<<16)|(phase<<12)|mask;let e=cache.get(key);if(e)return e;
  const cur=matrices[phase][mask&1],M=L===1?cur:mm(cur,entry((phase+3)%4,L-1,mask>>1).M);
  const pred=norm(M.map(sum)),q=val(pred),a=argmax(q);
  const col=Array.from({length:8},(_,j)=>val(norm(M.map(r=>r[j]))));
  // Exact action-dependent worst-case regret over all possible preceding priors.
  const robust=Array.from({length:4},(_,i)=>{
   let gap=Math.min(...col.flatMap(v=>v.map((x,j)=>i===j?Infinity:v[i]-x)));
   return {strict:gap>1e-12,regret:Math.max(0,-gap),margin:gap};
  });
  e={M,pred,q,a,robust};cache.set(key,e);return e;
 }
 function rng(seed){let a=seed>>>0;return ()=>{a=(a+0x6D2B79F5)|0;let t=Math.imul(a^(a>>>15),1|a);t^=t+Math.imul(t^(t>>>7),61|t);return ((t^(t>>>14))>>>0)/4294967296;};}
 function draw(p,u){let s=0;for(let i=0;i<p.length;i++){s+=p[i];if(u<s)return i;}return p.length-1;}
 const binary=(v,k,u)=>u<g(k,1,v)?1:-1;
 let rows=[],checks=0,certs=0,maxRegretExcess=0,maxEarlyTV=0;
 for(const scenario of scenarios)for(const seed of SEEDS)for(let ep=0;ep<N;ep++){
  const rand=rng((seed*1000003+ep*65537)>>>0);let r0=+(rand()<.5),r=r0,b=draw([.25,.25,.25,.25],rand()),p=unif(),hist=0,exactReward=0;
  let st=Object.fromEntries(LENS.map(L=>[L,{reward:0,late:0,agreement:0,cert:0,post_warm_cert:0,post_warm_steps:0,regret:0,regret_bound:0,actual_tv:0,regime_brier:0}]));
  for(let t=0;t<H;t++){
   let u=Array.from({length:8},rand);
   if(t>0){if(scenario==="self_switch")r=t<SWITCH?r0:1-r0;else if(u[0]<hazard)r^=1;b^=draw(F[r],u[1]);}
   let k=probes[t%4],y=binary(b,k,u[4]),hidden=4*r+b;
   hist=((hist<<1)|+(y>0))&((1<<Math.max(...LENS))-1);
   p=norm((t===0?p:mv(T,p)).map((v,i)=>v*g(k,y,i)));let exact=val(p),best=argmax(exact);exactReward+=rewards[best][hidden];
   for(let L of LENS){
    let l=Math.min(L,t+1),e=entry(t%4,l,hist&((1<<l)-1)),a=e.a,z=st[L],actual=.5*sum(e.pred.map((v,i)=>Math.abs(v-p[i]))),gap=e.q[a]-Math.max(...e.q.filter((_,i)=>i!==a));
    // During warmup all observations since known initial prior are retained.
    // T leaves uniform invariant, including the artificial first transition.
    let robust=t<L?{strict:gap>1e-12,regret:0}:e.robust[a];
    if(t<L)maxEarlyTV=Math.max(maxEarlyTV,actual);
    let regret=Math.max(0,exact[best]-exact[a]);
    maxRegretExcess=Math.max(maxRegretExcess,regret-robust.regret);
    if(robust.strict){certs++;if(regret>1e-10)throw Error("invalid window certificate");}
    checks++;z.reward+=rewards[a][hidden];if(t>=SWITCH)z.late+=rewards[a][hidden];z.agreement+=+(a===best);z.cert+=+robust.strict;z.regret+=regret;z.regret_bound+=robust.regret;z.actual_tv+=actual;z.regime_brier+=(sum(e.pred.slice(4))-r)**2;
    if(t>=L){z.post_warm_steps++;z.post_warm_cert+=+robust.strict;}
   }
  }
  for(let L of LENS){let z=st[L];rows.push({scenario,seed,ep,window_bits:L,exact_reward:exactReward/H,reward:z.reward/H,late:z.late/(H-SWITCH),agreement:z.agreement/H,cert:z.cert/H,post_warm_cert:z.post_warm_cert/z.post_warm_steps,regret:z.regret/H,regret_bound:z.regret_bound/H,actual_tv:z.actual_tv/H,regime_brier:z.regime_brier/H});}
 }
 if(maxRegretExcess>1e-10||maxEarlyTV>1e-10)throw Error("window validation failed");
 let metrics={};for(let s of scenarios){metrics[s]={};for(let L of LENS){let v=rows.filter(r=>r.scenario===s&&r.window_bits===L);metrics[s][L]=Object.fromEntries(["exact_reward","reward","late","agreement","cert","post_warm_cert","regret","regret_bound","actual_tv","regime_brier"].map(k=>[k,mean(v.map(r=>r[k]))]));}}
 const paired={};for(let s of scenarios){paired[s]={};for(let L of LENS){let d=rows.filter(r=>r.scenario===s&&r.window_bits===L).map(r=>r.exact_reward-r.reward),rand=rng(58241),b=[];for(let j=0;j<4000;j++){let z=0;for(let i=0;i<d.length;i++)z+=d[Math.floor(rand()*d.length)];b.push(z/d.length);}b.sort((a,b)=>a-b);paired[s][L]={exact_minus_window:mean(d),ci95:[b[99],b[3899]],n:d.length};}}
 return {label:"exact finite-window controller and online action certificate; same synthetic trajectories as QSM-03A",config:{N,SEEDS,H,SPLIT,SWITCH,LENS,theta,F,hazard,scenarios,probes:["XX","XI","XX","IX"]},validation:{checks,certs,maxRegretExcess,maxEarlyTV,cache_entries:cache.size,initial_stationary_error:Math.max(...mv(T,unif()).map(x=>Math.abs(x-1/8)))},metrics,paired,rows};
}
const r=qsm03window();fs.writeFileSync(process.argv[2]||new URL('qsm03b_results.json',import.meta.url),JSON.stringify(r,null,2)+'\n');console.log(JSON.stringify({validation:r.validation,metrics:r.metrics,paired:r.paired},null,2));
