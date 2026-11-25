// QSM-03: hidden-regime HMM and online belief-compression certificates; no dependencies.
import fs from 'node:fs';
function qsm03(){
 const N=128,SEEDS=[101,103,107],H=32,SPLIT=8,SWITCH=16,W=4,theta=Math.PI/3,c=Math.cos(theta)**2,A=(1+c)/2,B=(1-c)/2;
 const F=[[.80,.03,.03,.14],[.55,.17,.17,.11]],hazard=.06,probes=[2,0,2,1],strength=[Math.sin(theta),Math.sin(theta),Math.sin(theta)**2],scenarios=["matched","self_switch","decoy_switch"];
 const arms=[
 {id:"adaptive_exact",kind:"adaptive"},{id:"informed_regime",kind:"informed"},{id:"static_mixture",kind:"static",law:F[0].map((x,i)=>(x+F[1][i])/2)},{id:"static_slow",kind:"static",law:F[0]},{id:"static_fast",kind:"static",law:F[1]},
 {id:"product_exact",kind:"product",cert:true},{id:"decoy",kind:"adaptive",decoy:true},{id:"memoryless",kind:"memoryless"},
 ...[14,21,28,42].flatMap(b=>[{id:"joint_"+b,kind:"joint",bits:b,cert:true},{id:"product_"+b,kind:"product",bits:b,cert:true}])];
 const sum=a=>a.reduce((s,x)=>s+x,0),mean=a=>sum(a)/a.length,uniform=()=>Array(8).fill(1/8),sign=(i,b)=>((i&b)?1:-1),bs=i=>[sign(i,2),sign(i,1)],feat=(i,k)=>k===0?sign(i,2):k===1?sign(i,1):sign(i,2)*sign(i,1),chi=(i,k)=>{let s=1;for(let b of [1,2,4])if(k&b)s*=sign(i,b);return s;};
 const norm=p=>{let z=sum(p);if(!(z>0)||!Number.isFinite(z))throw Error("bad normalization");return p.map(x=>x/z);};
 const T=Array.from({length:8},(_,i)=>Array.from({length:8},(_,j)=>((i>>2)===(j>>2)?1-hazard:hazard)*F[i>>2][(i^j)&3]));
 const staticMatrix=f=>Array.from({length:8},(_,i)=>Array.from({length:8},(_,j)=>.5*f[(i^j)&3]));
 const matvec=(M,p)=>M.map(r=>sum(r.map((v,j)=>v*p[j])));
 const matmul=(M,Q)=>M.map(r=>Array.from({length:8},(_,j)=>sum(r.map((v,k)=>v*Q[k][j]))));
 const eye=()=>Array.from({length:8},(_,i)=>Array.from({length:8},(_,j)=>+(i===j)));
 const g=(k,y)=>Array.from({length:8},(_,i)=>(1+y*strength[k]*feat(i,k))/2);
 function update(p,k,y){let w=p.map((v,i)=>v*g(k,y)[i]);return {p:norm(w),z:sum(w)};}
 const hdist=(p,q)=>{if(p.some(x=>!(x>0))||q.some(x=>!(x>0)))throw Error("nonpositive Hilbert point");let r=p.map((x,i)=>Math.log(x/q[i]));return Math.max(...r)-Math.min(...r);};
 const tv=(p,q)=>.5*sum(p.map((v,i)=>Math.abs(v-q[i])));
 function birkhoff(M){let log=M.map(r=>r.map(x=>{if(!(x>0))throw Error("nonpositive operator");return Math.log(x);})),D=0;for(let i=0;i<8;i++)for(let j=i+1;j<8;j++){let a=log[i].map((x,k)=>x-log[j][k]);D=Math.max(D,Math.max(...a)-Math.min(...a));}return Math.tanh(D/4);}
 let tauTV=0;for(let i=0;i<8;i++)for(let j=i+1;j<8;j++)tauTV=Math.max(tauTV,.5*sum(T.map(r=>Math.abs(r[i]-r[j]))));
 const tau1=birkhoff(T),lookup={};
 for(let phase=0;phase<4;phase++)for(let L=1;L<=W;L++)for(let mask=0;mask<(1<<L);mask++){let M=eye();for(let j=0;j<L;j++){let k=probes[(phase-L+1+j+4*W)%4],y=((mask>>(L-1-j))&1)?1:-1,lik=g(k,y);M=matmul(T.map((r,i)=>r.map(v=>v*lik[i])),M);}lookup[[phase,L,mask].join("/")]=birkhoff(M);}
 const tau=(t,L,hist)=>L===0?1:lookup[[t%4,L,hist&((1<<L)-1)].join("/")];
 function product(p){let m=[sum(p.slice(0,4)),sum(p.slice(4,8))],b=Array.from({length:4},(_,i)=>p[i]+p[i+4]);return p.map((_,i)=>m[i>>2]*b[i&3]);}
 function allocate(bits,count){let q=Math.floor(bits/count),extra=bits%count;return Array.from({length:count},(_,i)=>q+(i<extra?1:0));}
 function compress(p,arm){
  if(arm.kind==="product")p=product(p);
  if(!arm.bits)return p;
  let ks=arm.kind==="joint"?[1,2,3,4,5,6,7]:[1,2,3,4],bits=allocate(arm.bits,ks.length),logs=p.map(Math.log);
  const co=ks.map((k,j)=>{let x=mean(logs.map((v,i)=>v*chi(i,k))),q=(1<<(bits[j]-1))-1;if(q<1)throw Error("invalid bit budget");return 2*Math.round(Math.max(-2,Math.min(2,x))*q/2)/q;});
  const out=p.map((_,i)=>Math.exp(sum(co.map((v,j)=>v*chi(i,ks[j])))));
  return norm(out);
 }
 const rewards=Array.from({length:4},(_,a)=>Array.from({length:8},(_,i)=>((i&2)===(a&2)?1:c)*((i&1)===(a&1)?1:c)));
 function values(p){return rewards.map(r=>sum(r.map((v,i)=>v*p[i])));}
 function decide(p){let v=values(p);return {a:v.indexOf(Math.max(...v)),v};}
 function rng(seed){let a=seed>>>0;return ()=>{a=(a+0x6D2B79F5)|0;let t=Math.imul(a^(a>>>15),1|a);t^=t+Math.imul(t^(t>>>7),61|t);return ((t^(t>>>14))>>>0)/4294967296;};}
 function draw(p,u){let s=0;for(let i=0;i<p.length;i++){s+=p[i];if(u<s)return i;}return p.length-1;}
 const binary=(v,k,u)=>u<(1+strength[k]*feat(v,k))/2?1:-1;
 function informedPrior(p,r,t){let marginal=Array.from({length:4},(_,i)=>p[i]+p[i+4]);let b=t===0?[.25,.25,.25,.25]:marginal.map((_,i)=>sum(marginal.map((v,j)=>v*F[r][i^j])));return Array.from({length:8},(_,i)=>(i>>2)===r?b[i&3]:0);}
 function newCert(){return {one:0,block:[],deltas:[],tvOnly:0,hybrid:0,hist:0};}
 function envelope(st,pre,post,z,k,y,t){
   let d=hdist(pre,post),dTV=tv(pre,post);
   st.hist=((st.hist<<1)|+(y>0))&((1<<W)-1);
   const old=st.block,delta=[...st.deltas,d].slice(-W);
   let h1=t===0?d:tau1*st.one+d, hb=h1;
   if(t>0){for(let L=1;L<=Math.min(W,t);L++){let b=tau(t,L,st.hist)*old[old.length-L];for(let j=0;j<L;j++)b+=tau(t,j,st.hist)*delta[delta.length-1-j];hb=Math.min(hb,b);}}
   const amp=(1+strength[k])/2/z, tvOnly=t===0?dTV:Math.min(1,amp*tauTV*st.tvOnly+dTV), hybrid=t===0?dTV:Math.min(1,amp*tauTV*st.hybrid+dTV,Math.tanh(hb/4));
   // Add a conservative numerical tolerance in comparisons, not a formal rounding certificate.
   st.one=h1;st.block=[...old,hb].slice(-W);st.deltas=delta;st.tvOnly=tvOnly;st.hybrid=hybrid;
   return {h1,hb,tvOnly,hybrid,tvH1:Math.tanh(h1/4),tvHB:Math.tanh(hb/4),local:d};
 }
 function gapCert(p,a,E){
   let q=values(p),okTV=true,okH=true,regretBound=0;
   for(let b=0;b<4;b++){if(b===a)continue;let dif=rewards[a].map((x,i)=>x-rewards[b][i]);let osc=Math.max(...dif)-Math.min(...dif),margin=q[a]-q[b];
     let pos=sum(p.map((x,i)=>x*Math.max(0,dif[i]))),neg=sum(p.map((x,i)=>x*Math.max(0,-dif[i])));
     okTV&&=margin>osc*E.hybrid+1e-12;okH&&=pos>0&&(neg===0||Math.log(pos/neg)>E.hb+1e-12);
     regretBound=Math.max(regretBound,-margin+osc*E.hybrid);
   }return {tv:okTV,projective:okH,either:okTV||okH,regretBound:Math.min(1,Math.max(0,regretBound))};
 }
 let rows=[],maxTVExcess=0,maxHExcess=0,maxRegretExcess=0,certChecks=0,certWins=0,regimeExactAccuracy=0,steps=0;
 const fields=["reward","pre","post","late","regret","prediction_mse","regime_brier","regime_acc","agreement","tv_only_cert","h1_cert","block_cert","online_cert","online_radius","actual_tv","local_h","regret_bound"];
 function newStats(){return Object.fromEntries(fields.map(k=>[k,0]));}
 for(let scenario of scenarios)for(let seed of SEEDS)for(let ep=0;ep<N;ep++){
  const rand=rng((seed*1000003+ep*65537)>>>0);
  let r0=+(rand()<.5),r=r0,b=draw([.25,.25,.25,.25],rand()),dr=0,db=0,decoyInitial=0;
  const p=Object.fromEntries(arms.map(a=>[a.id,uniform()])),st=Object.fromEntries(arms.filter(a=>a.cert).map(a=>[a.id,newCert()])),stats=Object.fromEntries(arms.map(a=>[a.id,newStats()]));
  const staticT=Object.fromEntries(arms.filter(a=>a.kind==="static").map(a=>[a.id,staticMatrix(a.law)]));
  for(let t=0;t<H;t++){
    // Fixed random tape per episode across all intervention worlds, even when draws are unused.
    let u=Array.from({length:8},rand);
    if(t>0){if(scenario==="self_switch")r=t<SWITCH?r0:1-r0;else if(u[0]<hazard)r^=1;b^=draw(F[r],u[1]);}
    if(t<SPLIT){dr=r;db=b;}else if(t===SPLIT){decoyInitial=+(u[2]<.5);dr=decoyInitial;db=draw([.25,.25,.25,.25],u[3]);}else{
      if(scenario==="decoy_switch")dr=t<SWITCH?decoyInitial:1-decoyInitial;else if(u[2]<hazard)dr^=1;db^=draw(F[dr],u[3]);
    }
    let k=probes[t%4],y=binary(b,k,u[4]),dy=t<SPLIT?y:binary(db,k,u[5]),hidden=4*r+b;
    let exact=update(t===0?uniform():matvec(T,p.adaptive_exact),k,y).p,best=decide(exact);
    for(let arm of arms){
      let pre;
      if(arm.id==="adaptive_exact")pre={p:exact,z:1};
      else if(arm.kind==="informed")pre=update(informedPrior(p[arm.id],r,t),k,y);
      else if(arm.kind==="memoryless")pre=update(uniform(),k,y);
      else pre=update(t===0?uniform():matvec(staticT[arm.id]||T,p[arm.id]),k,arm.decoy?dy:y);
      let q=arm.cert?compress(pre.p,arm):pre.p;
      p[arm.id]=q;
      let choice=decide(q),a=choice.a,reward=rewards[a][hidden],z=stats[arm.id],regret=Math.max(0,best.v[best.a]-best.v[a]),pr=sum(q.slice(4));
      z.reward+=reward;z.regret+=regret;z.prediction_mse+=(choice.v[a]-reward)**2;z.regime_brier+=(pr-r)**2;z.regime_acc+=+(+(pr>=.5-1e-12)===r);z.agreement+=+(a===best.a);
      if(t<SPLIT)z.pre+=reward;else z.post+=reward;if(t>=SWITCH)z.late+=reward;
      if(arm.cert){
        let E=envelope(st[arm.id],pre.p,q,pre.z,k,y,t),actual=tv(exact,q),h=hdist(exact,q),cert=gapCert(q,a,E);
        maxTVExcess=Math.max(maxTVExcess,actual-E.hybrid,actual-E.tvOnly);
        maxHExcess=Math.max(maxHExcess,h-E.hb,h-E.h1);
        maxRegretExcess=Math.max(maxRegretExcess,regret-cert.regretBound);
        if(cert.either){certWins++;if(regret>1e-10)throw Error("invalid online certificate");}
        certChecks++;
        let v=choice.v,margin=v[a]-Math.max(...v.filter((_,i)=>i!==a));
        z.tv_only_cert+=+(margin>2*E.tvOnly+1e-12);z.h1_cert+=+(margin>2*E.tvH1+1e-12);z.block_cert+=+(margin>2*E.tvHB+1e-12);z.online_cert+=+cert.either;z.online_radius+=E.hybrid;z.actual_tv+=actual;z.local_h+=E.local;z.regret_bound+=cert.regretBound;
      }
    }steps++;
  }
  for(let arm of arms){let z=stats[arm.id];let record={scenario,seed,ep,arm:arm.id};for(let k of fields){record[k]=k==="pre"?z[k]/SPLIT:k==="post"?z[k]/(H-SPLIT):k==="late"?z[k]/(H-SWITCH):(!arm.cert&&["tv_only_cert","h1_cert","block_cert","online_cert","online_radius","actual_tv","local_h","regret_bound"].includes(k))?null:z[k]/H;}rows.push(record);}
 }
 if(maxTVExcess>1e-9||maxHExcess>1e-9||maxRegretExcess>1e-9)throw Error("online bound violation "+JSON.stringify({maxTVExcess,maxHExcess,maxRegretExcess}));
 const metrics={};for(let s of scenarios){metrics[s]={};for(let arm of arms){let r=rows.filter(x=>x.scenario===s&&x.arm===arm.id);metrics[s][arm.id]={memory_bits:arm.bits||null,...Object.fromEntries(fields.map(k=>[k,r[0][k]===null?null:mean(r.map(x=>x[k]))]))};}}
 function paired(s1,a,s2,b,field){let x=rows.filter(r=>r.scenario===s1&&r.arm===a),y=rows.filter(r=>r.scenario===s2&&r.arm===b);if(x.length!==y.length||x.some((v,i)=>v.seed!==y[i].seed||v.ep!==y[i].ep))throw Error("pair mismatch");let d=x.map((v,i)=>v[field]-y[i][field]);let rand=rng(58241),boot=[];for(let j=0;j<4000;j++){let z=0;for(let i=0;i<d.length;i++)z+=d[Math.floor(rand()*d.length)];boot.push(z/d.length);}boot.sort((a,b)=>a-b);return {mean:mean(d),ci95:[boot[99],boot[3899]],n:d.length};}
 const comparisons={};for(let s of scenarios){comparisons[s]={};for(let [a,b,field] of [["adaptive_exact","static_mixture","reward"],["adaptive_exact","static_mixture","late"],["adaptive_exact","static_mixture","regime_brier"],["informed_regime","adaptive_exact","reward"],["adaptive_exact","decoy","post"],["adaptive_exact","decoy","pre"],["joint_14","product_14","reward"],["joint_21","product_21","reward"],["joint_28","product_28","reward"],["joint_42","product_42","reward"],["adaptive_exact","joint_28","reward"],["adaptive_exact","joint_42","reward"]])comparisons[s][a+"_minus_"+b+"_"+field]=paired(s,a,s,b,field);}
 const crossed={self_under_decoy_switch:paired("decoy_switch","adaptive_exact","matched","adaptive_exact","reward"),decoy_under_decoy_switch:paired("decoy_switch","decoy","matched","decoy","late"),self_under_self_switch:paired("self_switch","adaptive_exact","matched","adaptive_exact","late"),decoy_under_self_switch:paired("self_switch","decoy","matched","decoy","late")};
 // Decoy-only intervention must leave every self-input policy's realized trajectory unchanged.
 for(let arm of arms.filter(a=>!a.decoy)){let x=rows.filter(r=>r.scenario==="matched"&&r.arm===arm.id),y=rows.filter(r=>r.scenario==="decoy_switch"&&r.arm===arm.id);if(x.some((r,i)=>fields.some(k=>r[k]!==y[i][k])))throw Error("decoy intervention contaminated self input");}
 if(comparisons.matched.adaptive_exact_minus_decoy_pre.mean!==0)throw Error("pre-split mismatch");
 return {label:"synthetic hidden-regime identification; all controllers classical; no learning of unknown laws; no hardware",config:{N,SEEDS,H,SPLIT,SWITCH,W,theta,F,hazard,scenarios,arms,probes:["XX","XI","XX","IX"],log_coefficient_range:2},validation:{episodes:scenarios.length*SEEDS.length*N,steps,decisions:steps*arms.length,certChecks,certWins,maxTVExcess,maxHExcess,maxRegretExcess,tau1,tauTV,blockTau4Range:[Math.min(...Object.entries(lookup).filter(([k])=>k.split("/")[1]==="4").map(([,v])=>v)),Math.max(...Object.entries(lookup).filter(([k])=>k.split("/")[1]==="4").map(([,v])=>v))]},metrics,comparisons,crossed,rows};
}
const result=qsm03();fs.writeFileSync(process.argv[2]||new URL('qsm03_results.json',import.meta.url),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({validation:result.validation,metrics:result.metrics,comparisons:result.comparisons,crossed:result.crossed},null,2));
