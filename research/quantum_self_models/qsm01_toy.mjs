// QSM-01 exact toy checks, Node 18+, no dependencies, no QPU.
function run(){
 const rows=[];
 for(const theta of [Math.PI/12,Math.PI/6,Math.PI/4,Math.PI/3]){
   let baseline=0,self=0,decoy=0,oracle=0,flip=0;
   for(const b of [-1,1])for(const y of [-1,1])for(const d of [-1,1]){
     const p=(1+y*b*Math.sin(theta))/8;
     const reward=a=>Math.cos((b-a)*theta/2)**2;
     baseline+=p*reward(1); self+=p*reward(y); decoy+=p*reward(d);oracle+=p*reward(b);flip+=p*reward(-y);
   }
   const expected=(1+Math.cos(theta)**2)/2, uplift=Math.sin(theta)**3/2;
   if(Math.abs(baseline-expected)>1e-12||Math.abs(self-baseline-uplift)>1e-12||Math.abs(decoy-baseline)>1e-12||Math.abs(oracle-1)>1e-12||Math.abs(flip-baseline+uplift)>1e-12)throw Error("analytic mismatch");
   rows.push({theta,blind:baseline,self_probe:self,decoy,swapped_sign:flip,oracle,causal_gain:self-decoy});
 }
 // Direct real-amplitude statevector / density-matrix check of reference-entanglement control.
 const theta=Math.PI/3, R=8, idx=(r,m,s)=>4*r+2*m+s, psi=Array(R).fill(0);
 for(let b=0;b<2;b++){psi[idx(b,b,0)]=Math.cos(theta/2)/Math.sqrt(2);psi[idx(b,b,1)]=(b? -1:1)*Math.sin(theta/2)/Math.sqrt(2);}
 function correction(v){let out=v.slice();for(let r=0;r<2;r++)for(let m=0;m<2;m++){let i=idx(r,m,0),j=i+1,a=(m?1:-1)*theta/2,c=Math.cos(a),s=Math.sin(a);out[i]=c*v[i]-s*v[j];out[j]=s*v[i]+c*v[j];}return out;}
 function outer(v){return v.map(a=>v.map(b=>a*b));}
 const fixed=correction(psi),rho=outer(fixed);
 function measures(density){let s0=0,bell=0;for(let i=0;i<R;i++)if(i%2===0)s0+=density[i][i];for(let s=0;s<2;s++)for(let r=0;r<2;r++)for(let t=0;t<2;t++)bell+=density[idx(r,r,s)][idx(t,t,s)]/2;return {restore:s0,reference_bell_fidelity:bell};}
 const coherence=[];
 for(const retained of [0,.25,.5,.75,1]){const noisy=rho.map((row,i)=>row.map((v,j)=>((i>>1&1)===(j>>1&1)?v:retained*v)));const m=measures(noisy);if(Math.abs(m.restore-1)>1e-12||Math.abs(m.reference_bell_fidelity-(1+retained)/2)>1e-12)throw Error("coherence mismatch");coherence.push({retained,...m});}
 return {label:"exact toy calculations; no training or hardware",classical_decision:rows,reference_coherence:coherence};
}
console.log(JSON.stringify(run(),null,2));
