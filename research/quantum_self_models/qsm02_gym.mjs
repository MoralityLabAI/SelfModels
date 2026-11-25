// QSM-02 exact synthetic recurrent quantum-channel gym. Node 18+, no dependencies.
import fs from 'node:fs';
function qsm02(){
  const states=[[-1,-1],[-1,1],[1,-1],[1,1]],theta=Math.PI/3,c=Math.cos(theta)**2,A=(1+c)/2,B=(1-c)/2;
  const strengths=[Math.sin(theta),Math.sin(theta),Math.sin(theta)**2],feature=(b,k)=>k===0?b[0]:k===1?b[1]:b[0]*b[1];
  const BASE=[.80,.03,.03,.14],DRIFT=[.55,.17,.17,.11],T=32,SPLIT=8,SHIFT=16,SEEDS=[41,43,47],N=256;
  const arms=[
    {id:"bayes",mode:"full"},{id:"bayes_frozen",mode:"full",frozen:true},
    {id:"decision_factorized",mode:"decision_factorized"},{id:"factorized",mode:"factorized"},
    {id:"full_6bit",mode:"full",bits:2},{id:"factor_6bit",mode:"factorized",bits:3},
    {id:"full_12bit",mode:"full",bits:4},{id:"factor_12bit",mode:"factorized",bits:6},
    {id:"full_24bit",mode:"full",bits:8},{id:"memoryless",mode:"memoryless"},
    {id:"decoy",mode:"full",decoy:true},{id:"blind",mode:"blind"},{id:"oracle",mode:"oracle"}
  ];
  const uniform=()=>[.25,.25,.25,.25];
  const sum=x=>x.reduce((s,v)=>s+v,0),mean=x=>sum(x)/x.length;
  const moment=p=>[0,1,2].map(k=>sum(p.map((v,i)=>v*feature(states[i],k))));
  function decode(m){let p=states.map(b=>Math.max(0,(1+m[0]*b[0]+m[1]*b[1]+m[2]*b[0]*b[1])/4));let z=sum(p);return p.map(v=>v/z);}
  const quant=(v,bits)=>{const k=(1<<(bits-1))-1;return Math.round(Math.max(-1,Math.min(1,v))*k)/k;};
  function project(p,arm){let m=moment(p);if(arm.mode==="factorized"){if(arm.bits)m=m.map(v=>quant(v,arm.bits));return decode([m[0],m[1],m[0]*m[1]]);}if(arm.bits){return decode(m.map(v=>quant(v,arm.bits)));}return p;}
  function transit(p,f){return states.map((_,i)=>sum(p.map((v,j)=>v*f[i^j])));}
  function update(p,k,y){let w=p.map((v,i)=>v*(1+y*strengths[k]*feature(states[i],k))/2),z=sum(w);if(!(z>0))throw Error("zero likelihood");return w.map(v=>v/z);}
  function value(p,a){const m=moment(p);return A*A+A*B*(a[0]*m[0]+a[1]*m[1])+B*B*a[0]*a[1]*m[2];}
  function reward(b,a){return (b[0]===a[0]?1:c)*(b[1]===a[1]?1:c);}
  function decide(p){let q=states.map(a=>value(p,a)),i=q.indexOf(Math.max(...q));return {i,q};}
  function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)/4294967296);}
  function draw(f,u){let z=0;for(let i=0;i<f.length;i++){z+=f[i];if(u<z)return i;}return f.length-1;}
  function probe(b,k,u){return u<(1+strengths[k]*feature(b,k))/2?1:-1;}
  const storage=arm=>arm.bits?(arm.mode==="factorized"?2:3)*arm.bits:null;
  // Analytic validation of the action polynomial for all pure hypotheses.
  for(let i=0;i<4;i++)for(let a of states){let p=[0,0,0,0];p[i]=1;if(Math.abs(value(p,a)-reward(states[i],a))>1e-12)throw Error("reward mismatch");}
  let rows=[],maxCertificateViolation=0,maxValueViolation=0,maxDecodeError=0,certificateChecks=0;
  // No fit; all parameters of the data-generating law are known to Bayesian arms.
  for(let regime of ["stationary","shift"])for(let seed of SEEDS){
    const rand=rng(seed+(regime==="shift"?100003:0));
    for(let ep=0;ep<N;ep++){
      let hidden=draw(uniform(),rand()),other=hidden;
      let belief=Object.fromEntries(arms.map(a=>[a.id,uniform()]));
      let stats=Object.fromEntries(arms.map(a=>[a.id,{reward:0,regret:0,predSq:0,pre:0,post:0,late:0,cert:0,agree:0}]));
      for(let t=0;t<T;t++){
        let f=regime==="shift"&&t>=SHIFT?DRIFT:BASE;
        if(t>0)hidden^=draw(f,rand());
        if(t<SPLIT)other=hidden;else if(t===SPLIT)other=draw(uniform(),rand());else other^=draw(f,rand());
        const k=[2,0,2,1][t%4],y=probe(states[hidden],k,rand()),dy=t<SPLIT?y:probe(states[other],k,rand());
        const exact=update(t===0?uniform():transit(belief.bayes,f),k,y);
        const best=decide(exact);
        // All finite rewards are [0,1]. Exact reference is used only for retrospective audits.
        for(let arm of arms){
          let p;
          if(arm.mode==="oracle"){p=[0,0,0,0];p[hidden]=1;}
          else if(arm.mode==="blind"){p=uniform();}
          else if(arm.id==="bayes")p=exact;
          else if(arm.mode==="decision_factorized")p=exact;
          else {
            let prev=belief[arm.id];
            let prior=arm.mode==="memoryless"||t===0?uniform():transit(prev,arm.frozen?BASE:f);
            p=project(update(prior,k,arm.decoy?dy:y),arm);
          }
          belief[arm.id]=p;
          let read=p;
          if(arm.mode==="decision_factorized"){let m=moment(p);read=decode([m[0],m[1],m[0]*m[1]]);}
          let chosen=decide(read),a=states[chosen.i],r=reward(states[hidden],a),s=stats[arm.id];
          let exactRegret=Math.max(0,best.q[best.i]-best.q[chosen.i]);
          s.reward+=r;s.predSq+=(chosen.q[chosen.i]-r)**2;
          s.regret+=exactRegret;
          if(t<SPLIT)s.pre+=r;else s.post+=r;if(t>=SHIFT)s.late+=r;
          s.agree+=chosen.i===best.i?1:0;
          let m=moment(exact),v=moment(read),tv=.5*sum(exact.map((x,i)=>Math.abs(x-read[i])));
          let eps=Math.min(tv,A*B*(Math.abs(m[0]-v[0])+Math.abs(m[1]-v[1]))+B*B*Math.abs(m[2]-v[2]));
          let alt=Math.max(...chosen.q.filter((_,i)=>i!==chosen.i)),margin=chosen.q[chosen.i]-alt;
          maxValueViolation=Math.max(maxValueViolation,...chosen.q.map((q,i)=>Math.abs(q-best.q[i])-eps));
          maxDecodeError=Math.max(maxDecodeError,Math.abs(sum(read)-1));
          maxCertificateViolation=Math.max(maxCertificateViolation,exactRegret-2*eps);
          if(margin>2*eps+1e-12){s.cert++;certificateChecks++;if(exactRegret>1e-12)throw Error("bad certificate");}
        }
      }
      for(let arm of arms){let s=stats[arm.id];rows.push({regime,seed,ep,arm:arm.id,reward:s.reward/T,conditional_reference_regret:s.regret/T,prediction_mse:s.predSq/T,pre:s.pre/SPLIT,post:s.post/(T-SPLIT),late:s.late/(T-SHIFT),agreement:s.agree/T,certified:s.cert/T});}
    }
  }
  if(maxValueViolation>1e-10||maxCertificateViolation>1e-10||maxDecodeError>1e-10)throw Error("bound violation");
  const metrics={};const columns=["reward","conditional_reference_regret","prediction_mse","pre","post","late","agreement","certified"];
  for(let regime of ["stationary","shift"]){metrics[regime]={};for(let arm of arms){let a=rows.filter(r=>r.regime===regime&&r.arm===arm.id);metrics[regime][arm.id]={memory_bits:storage(arm),...Object.fromEntries(columns.map(k=>[k,mean(a.map(x=>x[k]))]))};}}
  function paired(regime,a,b,field){let u=rows.filter(r=>r.regime===regime&&r.arm===a),v=rows.filter(r=>r.regime===regime&&r.arm===b);if(u.some((x,i)=>x.ep!==v[i].ep||x.seed!==v[i].seed))throw Error("bad pairing");let d=u.map((x,i)=>x[field]-v[i][field]);const rand=rng(58241),boot=[];for(let i=0;i<4000;i++){let s=0;for(let j=0;j<d.length;j++)s+=d[Math.floor(rand()*d.length)];boot.push(s/d.length);}boot.sort((a,b)=>a-b);return {mean:mean(d),ci95:[boot[99],boot[3899]],n:d.length};}
  const comparisons={};
  for(let regime of ["stationary","shift"]){comparisons[regime]={};for(let [a,b,field] of [["bayes","decoy","post"],["bayes","decoy","pre"],["bayes","memoryless","reward"],["bayes","factorized","reward"],["full_6bit","factor_6bit","reward"],["full_12bit","factor_12bit","reward"],["bayes","full_12bit","reward"],["bayes","full_24bit","reward"],["bayes","bayes_frozen","late"]])comparisons[regime][a+"_minus_"+b+"_"+field]=paired(regime,a,b,field);}
  return {label:"synthetic two-qubit recurrent channel identification; classical posterior controllers; no quantum advantage claim",config:{theta,BASE,DRIFT,T,SPLIT,SHIFT,SEEDS,N,probe_schedule:["XX","XI","XX","IX"],arms,task_reward:"exact expected terminal |00> restoration probability; no reward observation",diagnostic:"one fresh two-qubit channel use per step"},validation:{trajectories:2*SEEDS.length*N,steps:2*SEEDS.length*N*T,decision_checks:rows.length*T,certificateChecks,maxValueViolation,maxCertificateViolation,maxDecodeError},metrics,comparisons,rows};
}
const r=qsm02();const out=process.argv[2]||new URL('qsm02_results.json',import.meta.url);fs.writeFileSync(out,JSON.stringify(r,null,2)+'\n');console.log(JSON.stringify({validation:r.validation,metrics:r.metrics,comparisons:r.comparisons},null,2));
