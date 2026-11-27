// QSM-04: active self-diagnostics with consequential channel regimes.
import fs from 'node:fs';
function qsm04(){
 const seeds=[211,223,227],N=128,H=64,h=.08,eta=.8,worlds=[["matched",.08],["fast_drift",.20]];
 const specs=[["oracle","oracle"],["blind","blind"],["current_only","current"],["always_probe","always"],["scheduled","scheduled"],["myopic","myopic"],["lookahead3","look"],["decoy3","decoy"],["quant3","quant",3],["quant5","quant",5],["window2","window",2],["window4","window",4],["window8","window",8]];
 const eps=1e-12,pred=(p,a=h)=>a+(1-2*a)*p,conf=p=>Math.max(p,1-p),act=p=>+(p>=.5-eps);
 function rng(seed){let a=seed>>>0;return ()=>{a=(a+0x6D2B79F5)|0;let t=Math.imul(a^(a>>>15),1|a);t^=t+Math.imul(t^(t>>>7),61|t);return ((t^(t>>>14))>>>0)/4294967296;};}
 const z1=p=>(1-eta)+(2*eta-1)*p;
 const update=(p,y)=>y===-1?p:y===1?eta*p/z1(p):(1-eta)*p/(1-z1(p));
 const quant=(p,b)=>{let k=2**b-2;return Math.round(k*p+eps)/k;};
 function value(p,t,n,ws,cs){
   if(!n||t>=H)return 0;
   const f=(q)=>ws[t]*conf(q)+value(pred(q),t+1,n-1,ws,cs);
   const v=f(p),z=z1(p),u=-cs[t]+z*f(update(p,1))+(1-z)*f(update(p,0));
   return Math.max(v,u);
 }
 function probeGap(p,t,n,ws,cs){
   const f=q=>ws[t]*conf(q)+value(pred(q),t+1,n-1,ws,cs);
   return -cs[t]+z1(p)*f(update(p,1))+(1-z1(p))*f(update(p,0))-f(p);
 }
 function windowPrior(hist,L){let p=.5;for(const y of hist.slice(-L))p=pred(update(p,y));return p;}
 // All p in the incoming interval and all per-step hazards in [loH,hiH].
 function predictInterval(l,u,loH,hiH){let a=[pred(l,loH),pred(l,hiH),pred(u,loH),pred(u,hiH)];return [Math.min(...a),Math.max(...a)];}
 function intervalPrior(hist,L,t,loH,hiH){
   let l=t<=L?.5:0,u=t<=L?.5:1;
   for(const y of hist.slice(-L)){[l,u]=predictInterval(update(l,y),update(u,y),loH,hiH);}
   return [l,u];
 }
 function posteriorInterval(hist,L,t,y,loH,hiH){const [l,u]=intervalPrior(hist,L,t,loH,hiH);return [update(l,y),update(u,y)];}
 const keys=["opportunities","success","probes","probe_cost","net","preprobe","strict_nom","strict_robust","agree_nom","agree_true","probe_cert","probe_eligible","probe_agree","brier"];
 let rows=[],checks={voi:0,interval:0,nomCert:0,robustCert:0,probeCert:0,violations:0,maxIntervalExcess:0},maxVoiError=0;
 const fail=(m)=>{checks.violations++;throw Error(m);};
 for(let i=0;i<=1000;i++){const p=i/1000,z=z1(p),a=z*conf(update(p,1))+(1-z)*conf(update(p,0))-conf(p),b=Math.max(0,eta-conf(p));maxVoiError=Math.max(maxVoiError,Math.abs(a-b));checks.voi++;if(Math.abs(a-b)>eps)fail("VOI identity");}
 for(const [world,actualH] of worlds)for(const seed of seeds)for(let ep=0;ep<N;ep++){
   const random=rng((seed*1000003+ep*65537)>>>0);
   let r=+(random()<.5),d=+(random()<.5),op=+(random()<.4),rs=[],ds=[],ys=[],dys=[],ws=[],cs=[];
   for(let t=0;t<H;t++){
     const a=Array.from({length:6},random);
     if(t){if(a[0]<actualH)r^=1;if(a[1]<actualH)d^=1;op=+(a[2]<(op?.65:.23));}
     rs.push(r);ds.push(d);ys.push(a[3]<eta?r:1-r);dys.push(a[4]<eta?d:1-d);ws.push(op);cs.push(a[5]<.5?.04:.14);
   }
   for(const [id,kind,L] of specs){
     let q=.5,audit=.5,truth=.5,hist=[],s=Object.fromEntries(keys.map(k=>[k,0]));
     for(let t=0;t<H;t++){
       const w=ws[t],c=cs[t];let p=t===0?.5:pred(q),use=false;
       if(kind==="window")p=windowPrior(hist,L);
       if(kind==="current"||kind==="blind")p=.5;
       if(kind==="oracle")p=rs[t];
       if(kind==="always")use=true;
       else if(kind==="scheduled")use=!!(w||(ws[t+1]&&c===.04));
       else if(kind==="current"||kind==="myopic")use=probeGap(p,t,1,ws,cs)>eps;
       else if(["look","decoy","quant","window"].includes(kind))use=probeGap(p,t,3,ws,cs)>eps;
       if(use){s.probes++;s.probe_cost+=c;s.preprobe+=+(!w);}
       const y=use?(kind==="decoy"?dys[t]:ys[t]):-1;
       let post=update(p,y);if(kind==="quant")post=quant(post,L);q=post;
       const a=act(post),success=+(a===rs[t]);
       s.opportunities+=w;s.success+=w*success;s.net+=w*success-(use?c:0);s.brier+=(post-rs[t])**2;
       // Audit using only the realized diagnostic schedule of this policy.
       if(kind==="window"){
         const priorNom=intervalPrior(hist,L,t,h,h),priorRob=intervalPrior(hist,L,t,h,.20);
         const exactPrior=audit,truePrior=truth;
         const R=ws.slice(t,t+3).reduce((a,b)=>a+b,0)+cs.slice(t,t+3).reduce((a,b)=>a+b,0);
         const gap=probeGap(p,t,3,ws,cs),rad=Math.max(Math.abs(p-priorNom[0]),Math.abs(p-priorNom[1]));
         const referenceProbe=probeGap(exactPrior,t,3,ws,cs)>eps;
         s.probe_eligible++;s.probe_agree+=+(referenceProbe===use);
         if(Math.abs(gap)>2*R*rad+eps){s.probe_cert++;checks.probeCert++;if(referenceProbe!==use)fail("probe cert");}
         const nom=posteriorInterval(hist,L,t,y,h,h),rob=posteriorInterval(hist,L,t,y,h,.20);
         audit=update(audit,y);truth=update(truth,y);
         const excess=Math.max(nom[0]-audit,audit-nom[1],rob[0]-truth,truth-rob[1],priorRob[0]-truePrior,truePrior-priorRob[1]);
         checks.maxIntervalExcess=Math.max(checks.maxIntervalExcess,excess);checks.interval++;
         if(excess>eps)fail("interval");
         if(w){
           s.agree_nom+=+(a===act(audit));s.agree_true+=+(a===act(truth));
           const nc=a?nom[0]>.5+eps:nom[1]<.5-eps;
           const rc=a?rob[0]>.5+eps:rob[1]<.5-eps;
           if(nc){s.strict_nom++;checks.nomCert++;if(a!==act(audit))fail("nom cert");}
           if(rc){s.strict_robust++;checks.robustCert++;if(a!==act(truth))fail("rob cert");}
         }
         audit=pred(audit);truth=pred(truth,actualH);
       }
       hist.push(y);
     }
     rows.push({world,seed,ep,arm:id,...s});
   }
 }
 const mean=x=>x.reduce((a,b)=>a+b,0)/x.length;
 const metrics={},comparisons={};for(const [world] of worlds){
   metrics[world]={};comparisons[world]={};
   for(const [id,kind,L] of specs){const rr=rows.filter(r=>r.world===world&&r.arm===id),tot=k=>rr.reduce((a,b)=>a+b[k],0),n=rr.length*H,opp=tot("opportunities"),probes=tot("probes");
     metrics[world][id]={net_per_step:tot("net")/n,net_per_opportunity:tot("net")/opp,opportunity_accuracy:tot("success")/opp,probe_rate:probes/n,probe_cost_per_step:tot("probe_cost")/n,preprobe_rate:tot("preprobe")/n,brier:tot("brier")/n,strict_nom:kind==="window"?tot("strict_nom")/opp:null,strict_robust:kind==="window"?tot("strict_robust")/opp:null,agree_nom:kind==="window"?tot("agree_nom")/opp:null,agree_true:kind==="window"?tot("agree_true")/opp:null,probe_cert:kind==="window"?tot("probe_cert")/n:null,probe_agree:kind==="window"?tot("probe_agree")/n:null,persistent_code_bits:kind==="quant"?L:kind==="window"?Math.ceil(L*Math.log2(3)):null};
   }
   for(const [a,b] of [["lookahead3","blind"],["lookahead3","current_only"],["lookahead3","always_probe"],["lookahead3","scheduled"],["lookahead3","myopic"],["lookahead3","decoy3"],["lookahead3","quant3"],["lookahead3","quant5"],["lookahead3","window2"],["lookahead3","window4"],["lookahead3","window8"],["oracle","lookahead3"]]){
     const x=rows.filter(r=>r.world===world&&r.arm===a),y=rows.filter(r=>r.world===world&&r.arm===b),d=x.map((v,i)=>{if(v.seed!==y[i].seed||v.ep!==y[i].ep)fail("pair");return (v.net-y[i].net)/H;});
     const rand=rng(49231),boot=[];for(let j=0;j<4000;j++){let v=0;for(let i=0;i<d.length;i++)v+=d[Math.floor(rand()*d.length)];boot.push(v/d.length);}boot.sort((a,b)=>a-b);
     comparisons[world][a+"_minus_"+b]={mean:mean(d),ci95:[boot[99],boot[3899]],n:d.length};
   }
 }
 return {label:"exploratory classical channel model; public three-step opportunity and cost forecast; no hardware",config:{seeds,N,H,h,eta,worlds,specs,opportunity_transition:[.23,.65],costs:[.04,.14]},validation:{world_episodes:worlds.length*seeds.length*N,decisions:worlds.length*seeds.length*N*H*specs.length,...checks,maxVoiError},metrics,comparisons,rows};
}
const result=qsm04();
fs.writeFileSync(process.argv[2]||"qsm04_results.json",JSON.stringify(result,null,2)+"\n");
console.log(JSON.stringify(result.validation));
