"""Build the isolated twin view from the exact HTML supplied for this project.

The source is a bundled React application, so every targeted replacement is
checked exactly once. A changed upstream bundle must be reviewed explicitly.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'ship_fault/web/vendor/digital_twin_original.html'
OUTPUT = ROOT / 'ship_fault/web/digital_twin.html'
SOURCE_SHA256 = 'd631b07437e0eab8778bcf867159ea8f98b19dd7f0b29cff12b583433767dfec'


def once(document: str, old: str, new: str) -> str:
    count = document.count(old)
    if count != 1:
        raise ValueError(f'Expected exactly one bundle anchor, found {count}: {old[:90]!r}')
    return document.replace(old, new, 1)


def build() -> None:
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('Supplied digital twin source changed; review the adapter before rebuilding')
    html = raw.decode('utf-8')
    boot = '''<script>
      window.__twinEmbed = true;
      window.__twinNonce = new URLSearchParams(location.search).get('nonce') || '';
      window.__twinRegions = Object.create(null);
      window.__twinFaults = [];
      document.documentElement.classList.add('embedded');
    </script>'''
    html = once(html, '<script type="module" crossorigin>', boot + '<script type="module" crossorigin>')
    # The embedded scene uses only explicit parent-supplied component regions.
    html = once(html, 'const j=q>.02;Qs(',
                'if(window.__twinEmbed)q=window.__twinRegions?.[n]?1:0;const j=q>.02;Qs(')
    html = once(html, 'meshBasicMaterial",{color:"#ff2244",transparent:!0,opacity:Se',
                'meshBasicMaterial",{color:window.__twinMode==="manual"?"#ff8797":"#ff2244",transparent:!0,opacity:Se')
    html = once(html, 'if(j)me=fr.faultRed,tt=fr.faultGlow,it=1.35',
                'if(j)me=window.__twinMode==="manual"?"#ff8797":fr.faultRed,tt=window.__twinMode==="manual"?"#7a2436":fr.faultGlow,it=1.35')
    html = once(html, 'color:j&&d?"#ff8ca9":Ce',
                'color:j&&d?(window.__twinMode==="manual"?"#ffb2be":"#ff8ca9"):Ce')
    # Drei's Html labels follow world positions as the camera auto-rotates. Use
    # the supplied twin's existing annotation anchors, but never its telemetry.
    callouts = r'''function TwinFaultLabels({revision:unused}) {
      const faults=window.__twinFaults||[];
      if(!faults.length)return null;
      const point=id=>FP_LABELS.find(item=>item.id===id);
      const de=point("pde"),nde=point("pnde"),rotor=point("rot"),foundation=point("fnd");
      const slots=[
        {region:"bearings",title:"BEARINGS",spot:de,anchors:[de.anchor,nde.anchor]},
        {region:"bearingHousings",title:"BEARING HOUSINGS",spot:nde,offset:[-.16,.02,0],anchors:[de.anchor,nde.anchor]},
        {region:"rotorDiscs",title:"ROTOR DISCS",spot:rotor,anchors:[rotor.anchor]},
        {region:"rotorBase",title:"ROTOR BASE",spot:foundation,offset:[-.10,0,0],anchors:[foundation.anchor]}
      ];
      const color=window.__twinMode==="manual"?"#ff8797":"#ff3956";
      return L.jsx("group",{children:slots.filter(slot=>faults.some(f=>f.regions.includes(slot.region))).map(slot=>{
        const matches=faults.filter(f=>f.regions.includes(slot.region));
        const labelPos=slot.offset?slot.spot.pos.map((value,index)=>value+slot.offset[index]):slot.spot.pos;
        const leaders=slot.anchors.flatMap((anchor,index)=>{
          const offset=[anchor[0]-labelPos[0],anchor[1]-labelPos[1],anchor[2]-labelPos[2]];
          const points=new Float32Array([0,0,0,...offset]);
          return [
            L.jsxs("line",{children:[L.jsx("bufferGeometry",{children:L.jsx("bufferAttribute",{attach:"attributes-position",args:[points,3]})}),L.jsx("lineBasicMaterial",{color})]},slot.region+"-line-"+index),
            L.jsxs("mesh",{position:offset,children:[L.jsx("sphereGeometry",{args:[.009,12,12]}),L.jsx("meshBasicMaterial",{color})]},slot.region+"-dot-"+index)
          ];
        });
        const first=matches[0];
        const card=L.jsxs("div",{className:"twin-fault-callout",style:{borderColor:color},"aria-label":slot.title+": "+matches.map(f=>f.subtype).join(", "),children:[
          L.jsx("strong",{className:"twin-fault-callout-title",children:slot.title}),
          L.jsx("span",{className:"twin-fault-callout-name",children:first.shortSubtype||first.subtype}),
          matches.length>1?L.jsx("small",{className:"twin-fault-callout-more",children:"+"+(matches.length-1)+" more faults"}):null,
          matches.length===1&&first.count>1?L.jsx("small",{className:"twin-fault-callout-more",children:first.count+" recordings"}):null,
          matches.some(f=>f.review)?L.jsx("small",{className:"twin-fault-callout-review",children:"Review"}):null
        ]});
        return L.jsxs("group",{position:labelPos,children:[...leaders,L.jsx(vB,{center:!0,distanceFactor:1,style:{pointerEvents:"none"},children:card})]},slot.region);
      })});
    }'''
    html = once(html, 'function E4(n){', callouts + 'function E4(n){')
    html = once(html, 'L.jsx(ZB,{rpm:n.rpm,running:n.running,timeScale:n.timeScale,showTransmission:n.showTransmission}),L.jsx(u4,',
                'L.jsx(ZB,{rpm:n.rpm,running:n.running,timeScale:n.timeScale,showTransmission:n.showTransmission}),L.jsx(TwinFaultLabels,{revision:n.twinRevision}),L.jsx(u4,')
    # React's memoized viewer context must refresh when only the union of regions changes.
    html = once(html, 'loaded:!!n.showTransmission}),[e,h,f,d,p,g,y,t,i,r,s,o,a,c,n.fault',
                'loaded:!!n.showTransmission,twinRevision:n.twinRevision}),[e,h,f,d,p,g,y,t,i,r,s,o,a,c,n.twinRevision,n.fault')
    html = once(html, 'showAnnotations:ce,selPoint:fpPt,ubExtra:fpUbX',
                'showAnnotations:ce,selPoint:fpPt,twinRevision:fpTwinRevision,ubExtra:fpUbX')
    html = once(html, 'const[fpFam,fpSetFam]=$.useState("normal")',
                'const[fpTwinRevision,fpSetTwinRevision]=$.useState(0),[fpFam,fpSetFam]=$.useState("normal")')
    html = once(html, '[ce,k]=$.useState(!0),[fpPt', '[ce,k]=$.useState(!1),[fpPt')
    html = once(html, '[a,c]=$.useState(!1),[h,f]', '[a,c]=$.useState(!0),[h,f]')
    html = once(html, '[w,_]=$.useState({name:"iso",nonce:0})',
                '[w,_]=$.useState({name:"front",nonce:0})')
    html = once(html, 'camera:{position:J2.iso.position,fov:40',
                'camera:{position:J2.front.position,fov:40')
    html = once(html, 'front:{label:"Front",position:[0,.3,1.6]}',
                'front:{label:"Front",position:[0,.62,1.6]}')
    html = once(html, 'autoRotateSpeed:1.2,minDistance:.35',
                'autoRotateSpeed:.65,minDistance:.35')
    old = 'window.setPredictedFault=go;const mh=ev=>{const d=ev.data;go(typeof d=="string"?d:d&&(d.label||d.fault||d.prediction))};window.addEventListener("message",mh);return()=>window.removeEventListener("message",mh)'
    new = '''window.setPredictedFault=go;const mh=ev=>{
      const d=ev.data;
      if(!window.__twinEmbed){go(typeof d==="string"?d:d&&(d.label||d.fault||d.prediction));return}
      if(ev.source!==window.parent||!d||d.type!=="simulator-twin/update"||d.version!==1||d.nonce!==window.__twinNonce)return;
      const allowed=["bearings","bearingHousings","rotorBase","rotorDiscs"];
      window.__twinRegions=Object.fromEntries(allowed.map(key=>[key,d.regions?.includes(key)===true]));
      window.__twinFaults=Array.isArray(d.faults)?d.faults.slice(0,33).map(item=>({
        subtype:String(item.subtype||"").slice(0,180),
        shortSubtype:String(item.shortSubtype||"").slice(0,40),
        regions:allowed.filter(key=>Array.isArray(item.regions)&&item.regions.includes(key)),
        review:item.review===true,
        count:Math.max(1,Math.min(999,Math.floor(Number(item.count)||1)))
      })).filter(item=>item.subtype&&item.regions.length):[];
      window.__twinMode=d.mode==="live"?"live":"manual";
      const nextPreset=FP_FAM.some(item=>item.id===d.presetId)?d.presetId:"normal";
      fpPickFam(nextPreset);
      const faultKey=window.__twinMode+":"+nextPreset+":"+window.__twinFaults.map(item=>item.subtype).join("|");
      if(faultKey!==window.__twinLastKey){window.__twinLastKey=faultKey;Te("front");c(true)}
      if(window.__twinMode==="live")r(true);
      fpSetTwinRevision(value=>value+1);
    };window.addEventListener("message",mh);
    if(window.__twinEmbed)window.parent.postMessage({type:"simulator-twin/ready",version:1,nonce:window.__twinNonce},"*");
    return()=>window.removeEventListener("message",mh)'''
    html = once(html, old, new)
    # Remove controls/readouts which would imply measured temperature, vibration or RPM.
    html = once(html, 'Object.keys(Gc).map(e=>L.jsx("button"',
                'Object.keys(Gc).filter(e=>e!=="thermal").map(e=>L.jsx("button"')
    html = once(html, 'L.jsx("button",{className:n.showAnnotations?"on":"",onClick:()=>n.setShowAnnotations(!n.showAnnotations),title:"Toggle 3D floating telemetry annotation tags",children:"Tags"})',
                'null')
    html = once(html, 'L.jsx("button",{className:n.running?"":"on",onClick:()=>n.setRunning(!n.running),children:n.running?"Stop":"Run"})',
                'null')
    html = once(html, 'L.jsx(FpInfo,{point:n.selPoint})', 'null')
    html = once(html,
                'L.jsx(P4,{rpm:t,timeScale:s,showTransmission:h,setShowTransmission:f,selectedId:d,activeViews:n,running:i,bloomEnabled:v,setBloomEnabled:S,bloomStrength:M,setBloomStrength:T,bloomRadius:I,setBloomRadius:C,thermalRange:P,setThermalRange:N,fault:H,setFault:Oe,fpFam:fpFam,fpPickFam:fpPickFam,faultSeverity:R,decayingFault:X,decayingSeverity:se}),L.jsx(I4,{activeViews:n,rpm:t,running:i,bloomEnabled:v,fault:H})',
                'null,null')
    embedded_css = '''<style>
      html.embedded .app { grid-template-columns: minmax(0,1fr); grid-template-rows: 56px minmax(0,1fr); }
      html.embedded .sidebar.right, html.embedded .statusbar { display: none !important; }
      html.embedded .topbar .brand .sub { display: none; }
      html.embedded .topbar { overflow-x: auto; }
      html.embedded .topbar .group { flex-shrink: 0; }
      html.embedded .viewport { overflow: hidden; }
      html.embedded .twin-fault-callout { width:145px; max-width:145px; padding:6px 8px; border:1px solid #ff8797; border-radius:6px; color:#fff4f5; background:rgba(31,9,21,.84); box-shadow:0 4px 14px rgba(0,0,0,.32); white-space:normal; text-align:left; }
      html.embedded .twin-fault-callout-title { display:block; margin-bottom:3px; color:#ffbdc7; font-size:9px; line-height:1.15; letter-spacing:.04em; }
      html.embedded .twin-fault-callout-name { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:11px; font-weight:700; line-height:1.2; }
      html.embedded .twin-fault-callout-more, html.embedded .twin-fault-callout-review { display:block; margin-top:2px; color:#ffced4; font-size:9px; line-height:1.15; }
    </style>'''
    html = once(html, '</head>', embedded_css + '</head>')
    OUTPUT.write_text(html, encoding='utf-8')


if __name__ == '__main__':
    build()
