"""three.js ground view of the three real sites, built from the orthophotos and qest.detect's output.

Ground: the Land Board orthophoto of each site on a plane in metres (east x, north -z). On it, at the detected
positions: one tilted module table per 2.5 m module cell of the Pihlaka mask (south-facing, 25 deg), Hertz 1's 54
containers on the detected compound's axis, and the 14 Paldiski turbines (hub / rotor from OSM) at their detected
rotor positions, blades turning. Orbit with the mouse; the buttons switch site.
usage: .venv/bin/python -m qest.scene [out_dir]   -> <out_dir>/scene.html + scene_<site>.jpg (default results/figs/scene)
"""
import json
import math
import sys
from pathlib import Path

from PIL import Image

from qest.paths import DERIVED, RAW, RESULTS

SITES = ('solar_pihlaka', 'battery_kiisa', 'wind_paldiski')
TEX = 2048


def local(M, lat, lon):
    lat0, lon0 = (M['north'] + M['south']) / 2, (M['west'] + M['east']) / 2
    return round((lon - lon0) * 111_320 * math.cos(math.radians(lat0)), 1), round(-(lat - lat0) * 111_320, 1)


def data():
    D = json.loads((DERIVED / 'assets_detected.json').read_text()); I = D['imagery']; out = {}
    for k in SITES:
        M = I[k]; w = M['m_per_px'] * M['width']; h = M['m_per_px'] * M['height']
        out[k] = dict(w=round(w), h=round(h), tex=f'scene_{k}.jpg')
    s, b, wd = D['solar'], D['battery'], D['wind']
    out['solar_pihlaka']['cells'] = [c for ll in s['cells'] for c in local(I['solar_pihlaka'], *ll)]
    out['solar_pihlaka']['info'] = [['Park (detected)', f"{s['park_ha']:.1f} ha"], ['Module area', f"{s['module_m2'] / 1e4:.1f} ha"],
                                    ['DC rating, 21 % modules', f"{s['mwp']['0']:.0f}–{s['mwp']['35']:.0f} MWp"], ['Dataset', '42.3 MW, 4 devices']]
    cx, cz = local(I['battery_kiisa'], *b['center'])
    out['battery_kiisa'].update(center=[cx, cz], axis=b['axis_deg'], n=b['containers'],
                                outline=[c for ll in b['outline'] for c in local(I['battery_kiisa'], *ll)],
                                info=[['Compound (detected)', f"{b['compound_m2'] / 1e4:.2f} ha"], ['Containers', f"{b['containers']} (operator)"],
                                      ['Rating', '100 MW / 200 MWh'], ['Imagery', 'before completion']])
    out['wind_paldiski'].update(turbines=[[*local(I['wind_paldiski'], t['lat'], t['lon']), t['hub'], t['rotor']] for t in wd['turbines']],
                                info=[['Turbines confirmed', f"{wd['detected']} / {len(wd['turbines'])}"],
                                      ['Offset to OSM', f"{min(t['offset_m'] for t in wd['turbines'] if t['detected']):.0f}–{max(t['offset_m'] for t in wd['turbines'] if t['detected']):.0f} m"],
                                      ['Rated', '45 MW'], ['Hub / rotor', '100 m / 100 m']])
    return out


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Real Sites 3D</title>
<style>
:root{--bg:#fcfcfb;--ink:#14213d;--muted:#4a5568;--line:#d9dde3;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#11161f;--ink:#e8ecf2;--muted:#a3adbb;--line:#2b3442}}
:root[data-theme="dark"]{--bg:#11161f;--ink:#e8ecf2;--muted:#a3adbb;--line:#2b3442}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);font:15px/1.4 Inter,system-ui,sans-serif}
#wrap{display:flex;height:100%}#view{flex:1;min-width:0;position:relative}canvas{display:block}
#side{width:280px;padding:20px;border-left:1px solid var(--line);display:flex;flex-direction:column;gap:14px;overflow:auto}
button{font:inherit;padding:8px 10px;border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:8px;cursor:pointer;text-align:left}
button.on{border-color:var(--accent);color:var(--accent);font-weight:600}
table{border-collapse:collapse;width:100%}td{padding:6px 0;border-bottom:1px solid var(--line)}td:last-child{text-align:right;font-weight:600}
small{color:var(--muted)}
@media (max-width:700px){#wrap{flex-direction:column}#side{width:auto;border-left:0;border-top:1px solid var(--line)}}
</style></head><body><div id="wrap"><div id="view"></div><div id="side">
<div id="btns"></div><table id="info"></table>
<small>Orthophoto: Maa-amet (CC BY 4.0). Modules, compound and turbines detected in the image (qest.detect); drag to orbit, scroll to zoom.</small>
</div></div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D=__DATA__;const NAMES={solar_pihlaka:'Pihlaka solar · 42.3 MW',battery_kiisa:'Hertz 1 battery · Kiisa',wind_paldiski:'Paldiski wind · 45 MW'};
const view=document.getElementById('view');const R=new THREE.WebGLRenderer({antialias:true});R.setPixelRatio(Math.min(devicePixelRatio,2));view.appendChild(R.domElement);
const cam=new THREE.PerspectiveCamera(50,1,1,20000);const ctl=new OrbitControls(cam,R.domElement);ctl.maxPolarAngle=Math.PI*0.49;ctl.enableDamping=true;
let scene,blades=[];const tl=new THREE.TextureLoader();
function build(k){const d=D[k];scene=new THREE.Scene();scene.background=new THREE.Color(0xbfd3e6);scene.fog=new THREE.Fog(0xbfd3e6,d.w*0.9,d.w*3);blades=[];
 scene.add(new THREE.HemisphereLight(0xffffff,0x667755,1.1));const sun=new THREE.DirectionalLight(0xffffff,1.6);sun.position.set(-300,600,800);scene.add(sun);
 const tex=tl.load(d.tex);tex.colorSpace=THREE.SRGBColorSpace;tex.anisotropy=8;
 const g=new THREE.Mesh(new THREE.PlaneGeometry(d.w,d.h),new THREE.MeshLambertMaterial({map:tex}));g.rotation.x=-Math.PI/2;scene.add(g);
 const m4=new THREE.Matrix4(),q=new THREE.Quaternion(),s=new THREE.Vector3(1,1,1),p=new THREE.Vector3();
 if(d.cells){const n=d.cells.length/2;const geo=new THREE.BoxGeometry(2.4,0.06,2.0);const im=new THREE.InstancedMesh(geo,new THREE.MeshStandardMaterial({color:0x1d2f5a,metalness:.5,roughness:.35}),n);
  q.setFromEuler(new THREE.Euler(-25*Math.PI/180,0,0));for(let i=0;i<n;i++){p.set(d.cells[2*i],1.4,d.cells[2*i+1]);m4.compose(p,q,s);im.setMatrixAt(i,m4)}scene.add(im);
  let mx=0,mz=0;for(let i=0;i<n;i++){mx+=d.cells[2*i];mz+=d.cells[2*i+1]}mx/=n;mz/=n;cam.position.set(mx-160,70,mz+230);ctl.target.set(mx,0,mz)}
 if(d.center){const a=-d.axis*Math.PI/180;const geo=new THREE.BoxGeometry(12.2,2.9,2.6);const im=new THREE.InstancedMesh(geo,new THREE.MeshStandardMaterial({color:0xf2f2ee,roughness:.6}),d.n);
  const cols=9,rows=Math.ceil(d.n/cols);q.setFromEuler(new THREE.Euler(0,a+Math.PI/2,0));let i=0;  // long side across the compound axis, as the pads
  for(let r=0;r<rows;r++)for(let c=0;c<cols&&i<d.n;c++,i++){const u=(c-(cols-1)/2)*6.5,v=(r-(rows-1)/2)*16;p.set(d.center[0]+u*Math.cos(a)+v*Math.sin(a),1.45,d.center[1]-u*Math.sin(a)+v*Math.cos(a));m4.compose(p,q,s);im.setMatrixAt(i,m4)}
  scene.add(im);const pts=[];for(let j=0;j<d.outline.length;j+=2)pts.push(new THREE.Vector3(d.outline[j],0.6,d.outline[j+1]));pts.push(pts[0].clone());
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0xeb6834})));
  cam.position.set(d.center[0]-90,30,d.center[1]+110);ctl.target.set(d.center[0],0,d.center[1])}
 if(d.turbines){const mat=new THREE.MeshStandardMaterial({color:0xf4f4f4,roughness:.5});
  for(const [x,z,hub,rot] of d.turbines){const t=new THREE.Mesh(new THREE.CylinderGeometry(1.6,2.6,hub,16),mat);t.position.set(x,hub/2,z);scene.add(t);
   const nac=new THREE.Mesh(new THREE.BoxGeometry(4,4,10),mat);nac.position.set(x,hub,z);scene.add(nac);
   const hubg=new THREE.Group();hubg.position.set(x,hub,z+5.5);for(let b=0;b<3;b++){const bl=new THREE.Mesh(new THREE.BoxGeometry(1.8,rot/2,0.5),mat);bl.position.y=rot/4;const arm=new THREE.Group();arm.rotation.z=b*2*Math.PI/3;arm.add(bl);hubg.add(arm)}
   scene.add(hubg);blades.push(hubg)}
  const [x0,z0]=d.turbines[0];cam.position.set(x0-400,90,z0+650);ctl.target.set(x0,60,z0)}
 if(new URLSearchParams(location.search).get('top')){cam.position.set(ctl.target.x,d.w*0.45,ctl.target.z+1)}  // plan view (checks)
 document.getElementById('info').innerHTML=d.info.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('');
 document.querySelectorAll('#btns button').forEach(b=>b.classList.toggle('on',b.dataset.k===k));ctl.update()}
const bt=document.getElementById('btns');for(const k of Object.keys(D)){const b=document.createElement('button');b.textContent=NAMES[k];b.dataset.k=k;b.onclick=()=>build(k);bt.appendChild(b);bt.appendChild(document.createElement('br'))}
function size(){const w=view.clientWidth,h=view.clientHeight||480;R.setSize(w,h);cam.aspect=w/h;cam.updateProjectionMatrix()}addEventListener('resize',size);
build(new URLSearchParams(location.search).get('site')||'solar_pihlaka');size();
R.setAnimationLoop(()=>{blades.forEach(b=>b.rotation.z+=0.012);ctl.update();R.render(scene,cam)});
</script></body></html>"""


def main(out=None):
    out = Path(out or RESULTS / 'figs' / 'scene'); out.mkdir(parents=True, exist_ok=True)
    for k in SITES:
        im = Image.open(RAW / f'ortho_{k}.jpg'); im.thumbnail((TEX, TEX)); im.save(out / f'scene_{k}.jpg', quality=85)
    (out / 'scene.html').write_text(HTML.replace('__DATA__', json.dumps(data(), separators=(',', ':'))))
    print('wrote', out / 'scene.html')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
