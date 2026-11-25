import fs from 'node:fs';
import assert from 'node:assert/strict';
function experiment() {
 const H=[.08,.28,.08],E=[.76,.76,.54],eta=.86,sw=.025,N=192,T=96,alpha=.05;
 const arms=["fixed","static","adaptive","mode_oracle","current","gate","q7","q31","q7_audit4","window3","window8"],worlds=["nominal","hazard_shift","sensor_shift","offbank"];
 const init=[.5,.5,0,0,0,0],mix=Array(6).fill(1/6),net=x=>(x.success-x.cents/100)/T;
 function rng(seed){let x=seed>>>0;return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296;};}
 function kernel(s){return Array.from({length:6},(_,i)=>Array.from({length:6},(_,j)=>((i>>1)===(j>>1)?1-s:s/2)*(i%2===j%2?1-H[j>>1]:H[j>>1])));}
 const K=kernel(sw),S=kernel(0);
 function prior(b,k){const c=Array(6).fill(0);for(let i=0;i<6;i++)for(let j=0;j<6;j++)c[j]+=b[i]*k[i][j];return c;}
 function prob(b){return b[1]+b[3]+b[5];}
 function likelihood(i,y,paid){const a=paid?eta:E[i>>1];return y===i%2?a:1-a;}
 function obs(b,y,paid=false){const a=b.map((v,i)=>v*likelihood(i,y,paid)),z=a.reduce((s,v)=>s+v,0);return[a.map(v=>v/z),z];}
 function move(b,k,y,z=-1){const a=obs(prior(b,k),y)[0];return z<0?a:obs(a,z,true)[0];}
 function choose(p){return p>=.5-1e-12?1:0;}
 function buy(p,w,c){return w*Math.max(0,eta-Math.max(p,1-p))>c+1e-12;}
 function quant(b,n){const v=b.map(x=>x*n),a=v.map(Math.floor),rem=n-a.reduce((s,x)=>s+x,0),order=v.map((x,i)=>i).sort((i,j)=>(v[j]-a[j])-(v[i]-a[i])||i-j);for(let j=0;j<rem;j++)a[order[j]]++;return a.map(x=>x/n);}
 function suffix(hist,y,L){let a=init;for(const[v,z]of hist.slice(-L))a=move(a,K,v,z);return obs(prior(a,K),y)[0];}
 function bounds(hist,y,L){let d=Array(6).fill(1),n=Array.from({length:6},(_,i)=>i%2);const seq=[...hist.slice(-L),[y,-1]];for(let t=seq.length-1;t>=0;t--){const[v,z]=seq[t],nd=Array(6).fill(0),nn=Array(6).fill(0);for(let j=0;j<6;j++){const l=likelihood(j,v,false)*(z<0?1:likelihood(j,z,true));d[j]*=l;n[j]*=l;}for(let i=0;i<6;i++)for(let j=0;j<6;j++){nd[i]+=K[i][j]*d[j];nn[i]+=K[i][j]*n[j];}d=nd;n=nn;}const p=n.map((x,i)=>x/d[i]);return[Math.min(...p),Math.max(...p)];}
 function scalarObs(p,z,e){const a=p*(z?e:1-e),b=(1-p)*(z?1-e:e);return a/(a+b);}
 function config(w,t){if(w===0||t<32)return[0,H[0],E[0]];if(w===1)return[1,H[1],E[1]];if(w===2)return[2,H[2],E[2]];return[-1,.36,.60];}
 function tape(w,e){const rand=rng((0x9e3779b9^Math.imul(e+1,0x85ebca6b))>>>0);let r=rand()<.5?1:0,op=0;const a=[];for(let t=0;t<T;t++){const[mode,h,s]=config(w,t);if(rand()<h)r=1-r;const y=rand()<s?r:1-r,z=rand()<eta?r:1-r;op=rand()<(op?.65:.23)?1:0;const c=rand()<.5?.04:.14;a.push({r,y,z,w:op,c,mode});}return a;}
 function policy(arm,a){
 let b=arm==="static"?mix:init,ref=init,fixed=init,alt=init,hist=[],lr=1,alarm=-1,op=0,success=0,probes=0,cents=0,brier=0,mistakes=0,over=0,audit=0,violations=0,cert=0,recoveryCert=0;
 for(let t=0;t<a.length;t++){
 const {r,y,z,w,c,mode}=a[t];op+=w;
 const[f,lf]=obs(prior(fixed,S),y),[g,lg]=obs(prior(alt,K),y);lr*=lg/lf;if(alarm<0&&lr>=1/alpha)alarm=t;
 let p,tmp,lo=0,hi=1;
 if(arm==="fixed"){tmp=f;p=prob(tmp);}
 else if(arm==="static"){tmp=obs(prior(b,S),y)[0];p=prob(tmp);}
 else if(arm==="gate"){tmp=alarm<0?f:g;p=prob(tmp);}
 else if(arm==="current"){p=y?E[0]:1-E[0];}
 else if(arm==="mode_oracle"){const m=mode<0?1:mode,h=mode<0?.36:H[m],s=mode<0?.60:E[m];p=scalarObs(h+(1-2*h)*prob(b),y,s);}
 else if(arm.startsWith("window")){tmp=suffix(hist,y,Number(arm.slice(6)));p=prob(tmp);}
 else{tmp=obs(prior(b,K),y)[0];p=prob(tmp);}
 const rf=obs(prior(ref,K),y)[0],rp=prob(rf);let probe=buy(p,w,c);
 if(arm.startsWith("q")&&w&&!probe&&buy(rp,w,c))over++;
 if(arm==="q7_audit4"){
 [lo,hi]=bounds(hist,y,4);if(rp<lo-1e-10||rp>hi+1e-10)violations++;
 const minV=Math.max(.5,lo,1-hi),maxV=Math.max(hi,1-lo),minGain=w*Math.max(0,eta-maxV)-c,maxGain=w*Math.max(0,eta-minV)-c;
 const forced=minGain>1e-12?true:maxGain< -1e-12?false:null;
 if(forced!==null){cert++;if(forced!==buy(rp,w,c))violations++;if(probe!==forced){probe=forced;audit++;}}
 }
 let post=p;
 if(probe){probes++;cents+=Math.round(c*100);const[ff,zf]=obs(f,z,true),[gg,zg]=obs(g,z,true);lr*=zg/zf;if(alarm<0&&lr>=1/alpha)alarm=t;fixed=ff;alt=gg;ref=obs(rf,z,true)[0];if(arm==="current"||arm==="mode_oracle")post=scalarObs(p,z,eta);else if(arm==="gate"){tmp=alarm<0?ff:gg;post=prob(tmp);}else{tmp=obs(tmp,z,true)[0];post=prob(tmp);}if(arm==="q7_audit4"){lo=scalarObs(lo,z,eta);hi=scalarObs(hi,z,eta);}}
 else{fixed=f;alt=g;ref=rf;}
 if(arm==="gate"){tmp=alarm<0?fixed:alt;post=prob(tmp);}
 let action=choose(post);
 if(arm==="q7_audit4"){const q=prob(ref);if(q<lo-1e-10||q>hi+1e-10)violations++;const forced=lo>.5+1e-12?1:hi<.5-1e-12?0:null;if(forced!==null){recoveryCert++;if(forced!==choose(q))violations++;if(action!==forced){action=forced;audit++;}}}
 success+=w*(action===r);brier+=(post-r)**2;mistakes+=w*(choose(prob(ref))!==action);
 if(arm==="mode_oracle"||arm==="current")b=[1-post,post,0,0,0,0];else if(arm==="q7"||arm==="q7_audit4")b=quant(tmp,7);else if(arm==="q31")b=quant(tmp,31);else b=tmp;
 hist.push([y,probe?z:-1]);if(hist.length>8)hist.shift();
 }
 return{op,success,probes,cents,brier,mistakes,over,audit,violations,cert,recoveryCert,alarm};
 }
 const rows=[],summaries={},round=x=>Math.round(x*1e8)/1e8;
 function stats(x){const m=x.reduce((s,v)=>s+v,0)/x.length,se=Math.sqrt(x.reduce((s,v)=>s+(v-m)**2,0)/(x.length*(x.length-1)));return[m,m-1.96*se,m+1.96*se].map(round);}
 for(let w=0;w<worlds.length;w++){
 const by=Object.fromEntries(arms.map(a=>[a,[]]));
 for(let e=0;e<N;e++){const data=tape(w,e);for(const a of arms){const x=policy(a,data);by[a].push(x);rows.push({world:worlds[w],episode:e,arm:a,...x,brier:round(x.brier)});}}
 const summary={};for(const a of arms){const rs=by[a],sum=k=>rs.reduce((s,v)=>s+v[k],0),delays=rs.filter(x=>x.alarm>=32).map(x=>x.alarm-32).sort((a,b)=>a-b);summary[a]={net:stats(rs.map(net))[0],success:round(sum("success")/sum("op")),probeRate:round(sum("probes")/(N*T)),brier:round(sum("brier")/(N*T)),alarms:rs.filter(x=>x.alarm>=0).length,earlyAlarms:rs.filter(x=>x.alarm>=0&&x.alarm<32).length,afterShiftAlarms:delays.length,medianAlarm:delays.length?delays[Math.floor((delays.length-1)/2)]:null,over:sum("over"),audit:sum("audit"),violations:sum("violations"),cert:sum("cert"),recoveryCert:sum("recoveryCert"),mistakes:sum("mistakes")};}
 const pairs=[["adaptive","fixed"],["adaptive","static"],["adaptive","current"],["gate","fixed"],["gate","adaptive"],["q7_audit4","q7"],["q7_audit4","window8"],["q7","window3"],["q31","adaptive"]];
 summary.paired=Object.fromEntries(pairs.map(([a,b])=>[a+"_minus_"+b,stats(by[a].map((x,i)=>net(x)-net(by[b][i])))]));summaries[worlds[w]]=summary;
 }
 let mass=0,weighted=0,falseMass=0,leaves=0;
 function enumerate(t,f,g,pnull,palt,hit){if(t===4){mass+=pnull;weighted+=palt;falseMass+=hit?pnull:0;leaves++;return;}for(let y=0;y<2;y++){const[ff,lf]=obs(prior(f,S),y),[gg,lg]=obs(prior(g,K),y),n=pnull*lf,a=palt*lg,h=hit||a/n>=20;if(!buy(prob(gg),1,.04))enumerate(t+1,ff,gg,n,a,h);else for(let z=0;z<2;z++){const[fz,lz]=obs(ff,z,true),[gz,mz]=obs(gg,z,true);enumerate(t+1,fz,gz,n*lz,a*mz,h||a*mz/(n*lz)>=20);}}}
 enumerate(0,init,init,1,1,false);
 if(Math.abs(mass-1)>1e-10||Math.abs(weighted-1)>1e-10||falseMass>alpha+1e-12)throw Error("likelihood audit");
 const count=k=>worlds.reduce((s,w)=>s+summaries[w].q7_audit4[k],0);if(count("violations"))throw Error("certificate violation");
 return{summary:{episodes:N*worlds.length,steps:T,decisions:N*worlds.length*T*arms.length,arms,worlds,summaries,audit:{violations:count("violations"),probeCertificates:count("cert"),recoveryCertificates:count("recoveryCert"),likelihoodTreeLeaves:leaves,nullMass:round(mass),altMass:round(weighted),falseAlarmMass:round(falseMass)}},rows};
}
const {summary,rows}=experiment();
const out=JSON.stringify(summary,null,2)+'\n';
const keys=Object.keys(rows[0]);
const csv=[keys.join(','),...rows.map(r=>keys.map(k=>r[k]).join(','))].join('\n')+'\n';
if(process.argv.includes('--verify')){
 const expected=JSON.parse(fs.readFileSync(new URL('./qsm05_summary.json',import.meta.url),'utf8'));
 function compare(a,b,path=''){if(typeof a==='number'&&typeof b==='number'){assert.ok(Math.abs(a-b)<=1e-7,path);return;}if(a&&b&&typeof a==='object'&&typeof b==='object'){assert.deepEqual(Object.keys(a),Object.keys(b));for(const k of Object.keys(a))compare(a[k],b[k],path+'/'+k);return;}assert.equal(a,b,path);}
 compare(summary,expected);const saved=fs.readFileSync(new URL('./qsm05_trajectories.csv',import.meta.url),'utf8').trim().split('\n'),actual=csv.trim().split('\n');assert.equal(saved.length,actual.length);assert.equal(saved[0],actual[0]);for(let i=1;i<saved.length;i++){const a=actual[i].split(','),b=saved[i].split(',');assert.equal(a.length,b.length);for(let j=0;j<a.length;j++){if(keys[j]==='brier')assert.ok(Math.abs(Number(a[j])-Number(b[j]))<=1e-7);else assert.equal(a[j],b[j],'row '+i+' col '+keys[j]);}}
 console.log('QSM-05 receipt replay passed: '+rows.length+' rows');
}else{const target=process.argv[2]||'.';fs.writeFileSync(target+'/qsm05_summary.json',out);fs.writeFileSync(target+'/qsm05_trajectories.csv',csv);}
