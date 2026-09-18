/* =========================================================================
   Aquifer Sense — client-side groundwater prediction & early-warning
   prototype. Every number shown in this app is computed in-browser from
   whatever dataset is currently loaded (uploaded CSV or the labeled
   synthetic demo set). Nothing is hard-coded.
   ========================================================================= */

/* ---------------------------- Global state ---------------------------- */
const STATE = {
  view: "landing",
  projects: [],            // {id,name,location,state,country,description,wells,createdAt}
  activeProjectId: null,
  datasets: {},            // projectId -> {records:[], meta:{}, isDemo:bool, fileName}
  validation: {},          // projectId -> validation result object
  features: {},            // projectId -> engineered feature rows
  models: {},              // projectId -> {results:[{name,mae,rmse,r2,model}], chosenModel, testSplit}
  predictions: {},         // projectId -> {byWell: {wellId: {history:[], forecast:[]}}}
  wqi: {},                 // projectId -> {byWell: {...}, overall}
  risk: {},                // projectId -> {byWell:{...}, overall}
  alerts: {},              // projectId -> [alert,...]
  recommendations: {},     // projectId -> [text,...]
  ui: { predictWell: null, predictHorizon: 30, wqiWeights: null, thresholds: null }
};

const REQUIRED_COLUMNS = ["Date","Well_ID","Latitude","Longitude","Groundwater_Level","Rainfall","Temperature","Soil_Moisture","pH","TDS","Fluoride","Nitrate","Chloride"];

// Configurable, user-adjustable reference ranges. These are NOT sourced from
// a specific cited regulatory standard — they exist so the prototype can
// flag values for review. Replace with an applicable standard before any
// real decision-making use.
const DEFAULT_THRESHOLDS = {
  Groundwater_Level: {min:0, max:200, unit:"m"},
  Rainfall: {min:0, max:600, unit:"mm"},
  Temperature: {min:-10, max:55, unit:"°C"},
  Soil_Moisture: {min:0, max:100, unit:"%"},
  pH: {min:0, max:14, unit:""},
  TDS: {min:0, max:3000, unit:"mg/L"},
  Fluoride: {min:0, max:10, unit:"mg/L"},
  Nitrate: {min:0, max:200, unit:"mg/L"},
  Chloride: {min:0, max:1500, unit:"mg/L"}
};

// Reference values used only for the WQI sub-index calculation (Qi = Vi/Si*100).
// Configurable — swap in whatever standard applies to your context.
const DEFAULT_WQI_STANDARDS = {
  pH: {ideal:7.0, standard:8.5, weightFactor:0.20},
  TDS: {ideal:0, standard:500, weightFactor:0.20},
  Nitrate: {ideal:0, standard:45, weightFactor:0.25},
  Fluoride: {ideal:0, standard:1.5, weightFactor:0.15},
  Chloride: {ideal:0, standard:250, weightFactor:0.20}
};

/* ------------------------------- Utils --------------------------------- */
const fmt = (n, d=2) => (n===null||n===undefined||Number.isNaN(n)) ? "—" : Number(n).toFixed(d);
const uid = () => Math.random().toString(36).slice(2,10);
const todayISO = () => new Date().toISOString().slice(0,10);

function parseDateLoose(v){
  if(v===null||v===undefined||v==="") return null;
  if(v instanceof Date) return isNaN(v.getTime())?null:v;
  const s = String(v).trim();
  // try native
  let d = new Date(s);
  if(!isNaN(d.getTime())) return d;
  // try DD/MM/YYYY or MM/DD/YYYY
  const m = s.match(/^(\d{1,4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,4})$/);
  if(m){
    let [_,a,b,c] = m;
    if(a.length===4){ d = new Date(+a, +b-1, +c); }
    else { d = new Date(+c, +b-1, +a); } // assume DD/MM/YYYY
    if(!isNaN(d.getTime())) return d;
  }
  return null;
}

function toNum(v){
  if(v===null||v===undefined||v==="") return null;
  const n = parseFloat(v);
  return Number.isNaN(n) ? null : n;
}

function mean(arr){ const a=arr.filter(v=>v!=null && !Number.isNaN(v)); return a.length? a.reduce((s,v)=>s+v,0)/a.length : null; }
function std(arr){ const a=arr.filter(v=>v!=null && !Number.isNaN(v)); if(a.length<2) return null; const m=mean(a); return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/(a.length-1)); }
function quantile(sortedArr, q){
  if(!sortedArr.length) return null;
  const pos = (sortedArr.length-1)*q;
  const base = Math.floor(pos), rest = pos-base;
  if(sortedArr[base+1]!==undefined) return sortedArr[base] + rest*(sortedArr[base+1]-sortedArr[base]);
  return sortedArr[base];
}
function iqrBounds(values){
  const a = values.filter(v=>v!=null && !Number.isNaN(v)).slice().sort((x,y)=>x-y);
  if(a.length<4) return null;
  const q1 = quantile(a,0.25), q3 = quantile(a,0.75), iqr = q3-q1;
  return {q1,q3,iqr, low:q1-1.5*iqr, high:q3+1.5*iqr};
}
function seasonOf(month){ // 0-indexed month
  if([11,0,1].includes(month)) return "Winter";
  if([2,3,4].includes(month)) return "Summer";
  if([5,6,7,8].includes(month)) return "Monsoon";
  return "Post-Monsoon";
}

/* ---------------------- Simple linear algebra (OLS) --------------------- */
function transpose(A){ return A[0].map((_,c)=>A.map(r=>r[c])); }
function matMul(A,B){
  const r=A.length, k=A[0].length, c=B[0].length;
  const out = Array.from({length:r},()=>new Array(c).fill(0));
  for(let i=0;i<r;i++) for(let j=0;j<c;j++){ let s=0; for(let x=0;x<k;x++) s+=A[i][x]*B[x][j]; out[i][j]=s; }
  return out;
}
function matInv(M){
  // Gauss-Jordan on augmented matrix. M is n x n.
  const n = M.length;
  const A = M.map((row,i)=>[...row, ...Array.from({length:n},(_,j)=>i===j?1:0)]);
  for(let col=0; col<n; col++){
    let pivot = col;
    for(let r=col+1;r<n;r++) if(Math.abs(A[r][col])>Math.abs(A[pivot][col])) pivot=r;
    if(Math.abs(A[pivot][col])<1e-10){ A[pivot][col]+=1e-8; } // regularize singular
    [A[col],A[pivot]]=[A[pivot],A[col]];
    const pv = A[col][col];
    for(let j=0;j<2*n;j++) A[col][j]/=pv;
    for(let r=0;r<n;r++){
      if(r===col) continue;
      const factor = A[r][col];
      for(let j=0;j<2*n;j++) A[r][j]-=factor*A[col][j];
    }
  }
  return A.map(row=>row.slice(n));
}
// Multiple linear regression via normal equations with L2 regularization for stability.
function trainLinearRegression(X, y){
  const n = X.length, p = X[0].length;
  const Xb = X.map(row=>[1,...row]); // add intercept
  const Xt = transpose(Xb);
  const XtX = matMul(Xt, Xb);
  for(let i=0;i<XtX.length;i++) XtX[i][i]+=1e-6; // ridge regularization
  const XtXinv = matInv(XtX);
  const Xty = matMul(Xt, y.map(v=>[v]));
  const beta = matMul(XtXinv, Xty).map(r=>r[0]);
  return {
    predict(row){
      let s = beta[0];
      for(let i=0;i<row.length;i++) s += beta[i+1]*row[i];
      return s;
    },
    beta
  };
}

// Simple CART regression tree (real recursive splitting on real data).
function trainRegressionTree(X, y, maxDepth=4, minLeaf=5){
  function variance(idxs){
    const vals = idxs.map(i=>y[i]);
    const m = mean(vals);
    return vals.reduce((s,v)=>s+(v-m)**2,0);
  }
  function buildNode(idxs, depth){
    if(depth>=maxDepth || idxs.length<minLeaf*2){
      return {leaf:true, value:mean(idxs.map(i=>y[i]))};
    }
    let best=null;
    const nFeat = X[0].length;
    for(let f=0; f<nFeat; f++){
      const sortedIdx = idxs.slice().sort((a,b)=>X[a][f]-X[b][f]);
      for(let t=minLeaf; t<sortedIdx.length-minLeaf; t++){
        const thresh = (X[sortedIdx[t-1]][f]+X[sortedIdx[t]][f])/2;
        const left = idxs.filter(i=>X[i][f]<=thresh);
        const right = idxs.filter(i=>X[i][f]>thresh);
        if(left.length<minLeaf||right.length<minLeaf) continue;
        const score = variance(left)+variance(right);
        if(!best || score<best.score) best = {score, f, thresh, left, right};
      }
    }
    if(!best) return {leaf:true, value:mean(idxs.map(i=>y[i]))};
    return {leaf:false, f:best.f, thresh:best.thresh,
      left:buildNode(best.left, depth+1), right:buildNode(best.right, depth+1)};
  }
  const root = buildNode(X.map((_,i)=>i), 0);
  function predictRow(node, row){
    if(node.leaf) return node.value;
    return row[node.f] <= node.thresh ? predictRow(node.left,row) : predictRow(node.right,row);
  }
  return { predict: row => predictRow(root,row) };
}

function trainMovingAverageBaseline(X, y, windowFeatureIdx){
  // Baseline: predicts the previous-level feature (assumed to be X column 0) unchanged (persistence model).
  return { predict: row => row[0] };
}

function evalMetrics(yTrue, yPred){
  const n = yTrue.length;
  const errors = yTrue.map((t,i)=>t-yPred[i]);
  const mae = mean(errors.map(Math.abs));
  const rmse = Math.sqrt(mean(errors.map(e=>e*e)));
  const yMean = mean(yTrue);
  const ssTot = yTrue.reduce((s,t)=>s+(t-yMean)**2,0);
  const ssRes = errors.reduce((s,e)=>s+e*e,0);
  const r2 = ssTot>0 ? 1 - ssRes/ssTot : null;
  return {mae, rmse, r2, n};
}

/* ----------------------- Synthetic demo data generator ----------------------
   Clearly-labeled synthetic data so the full pipeline can be demonstrated
   without an upload. Generated with a seeded PRNG so a given demo project
   is reproducible within a session. */
function seededRandom(seed){
  let s = seed % 2147483647; if (s<=0) s += 2147483646;
  return () => (s = s*16807 % 2147483647) / 2147483647;
}
function generateSyntheticDataset(project){
  const rand = seededRandom(project.wells*7919 + project.name.length*31 + 17);
  const wells = Math.max(3, Math.min(20, project.wells||6));
  const startDate = new Date(); startDate.setDate(startDate.getDate() - 730); // 2 years
  const records = [];
  const baseLat = 18.5 + (rand()-0.5)*4, baseLon = 73.8 + (rand()-0.5)*4;
  for(let w=0; w<wells; w++){
    const wellId = "W-"+String(w+1).padStart(2,"0");
    const lat = baseLat + (rand()-0.5)*0.6;
    const lon = baseLon + (rand()-0.5)*0.6;
    let level = 6 + rand()*6; // starting groundwater level (m below reference)
    const declineTendency = (rand()-0.35)*0.006; // some wells trend down
    let ph = 6.8+rand()*1.0, tds = 300+rand()*500, fl = 0.3+rand()*1.0, no3 = 5+rand()*40, cl = 80+rand()*250;
    for(let d=0; d<730; d+=7){ // weekly records
      const date = new Date(startDate); date.setDate(date.getDate()+d);
      const month = date.getMonth();
      const season = seasonOf(month);
      const rainfall = season==="Monsoon" ? 40+rand()*160 : season==="Summer" ? rand()*8 : 5+rand()*30;
      const temperature = season==="Summer" ? 30+rand()*8 : season==="Monsoon" ? 24+rand()*5 : 15+rand()*8;
      const soilMoisture = Math.min(100, Math.max(2, 20 + rainfall*0.3 - (season==="Summer"?10:0) + (rand()-0.5)*8));
      // groundwater responds to rainfall (recharge) and slow decline/pumping trend
      level = level - declineTendency*7 + (rainfall>60 ? rand()*0.4 : -rand()*0.15) + (rand()-0.5)*0.15;
      level = Math.max(0.5, level);
      // slow water-quality drift correlated with declining level (concentration effect)
      const stressFactor = Math.max(0, (8-level))*0.01;
      ph = Math.min(9.5, Math.max(6.0, ph + (rand()-0.5)*0.05));
      tds = Math.max(80, tds + stressFactor*15 + (rand()-0.5)*20);
      fl = Math.max(0.1, fl + stressFactor*0.02 + (rand()-0.5)*0.05);
      no3 = Math.max(0.5, no3 + stressFactor*1.2 + (rand()-0.5)*2);
      cl = Math.max(20, cl + stressFactor*8 + (rand()-0.5)*10);
      records.push({
        Date: date.toISOString().slice(0,10), Well_ID: wellId,
        Latitude: +lat.toFixed(5), Longitude: +lon.toFixed(5),
        Groundwater_Level: +level.toFixed(2), Rainfall: +rainfall.toFixed(1),
        Temperature: +temperature.toFixed(1), Soil_Moisture: +soilMoisture.toFixed(1),
        pH: +ph.toFixed(2), TDS: +tds.toFixed(0), Fluoride: +fl.toFixed(2),
        Nitrate: +no3.toFixed(1), Chloride: +cl.toFixed(0)
      });
    }
  }
  // sprinkle in a few realistic data problems so validation has something to find
  if(records.length>20){
    for(let i=0;i<Math.max(3,Math.floor(records.length*0.01));i++){
      const idx = Math.floor(rand()*records.length);
      const field = ["Groundwater_Level","pH","TDS","Rainfall"][Math.floor(rand()*4)];
      records[idx][field] = null; // missing value
    }
    records.push({...records[3]}); // exact duplicate
    records.push({...records[10]});
    const outIdx = Math.floor(rand()*records.length);
    records[outIdx].Groundwater_Level = records[outIdx].Groundwater_Level + 25; // outlier
  }
  return records;
}

/* ----------------------------- CSV handling ----------------------------- */
function parseCSVFile(file, callback){
  Papa.parse(file, {
    header:true, dynamicTyping:false, skipEmptyLines:true,
    complete: (res) => callback(res.data, res.meta),
    error: (err) => callback(null, null, err)
  });
}

function normalizeRecords(rawRows){
  // Coerce raw parsed rows (string values) into typed records matching REQUIRED_COLUMNS.
  return rawRows.map(r=>{
    const out = {};
    REQUIRED_COLUMNS.forEach(col=>{
      let v = r[col];
      if(v===undefined){ // try case-insensitive match
        const key = Object.keys(r).find(k=>k.toLowerCase()===col.toLowerCase());
        v = key ? r[key] : undefined;
      }
      if(col==="Date"){ out[col] = v; }
      else if(col==="Well_ID"){ out[col] = v===undefined||v===""?null:String(v).trim(); }
      else { out[col] = (v===undefined||v==="") ? null : toNum(v); }
    });
    return out;
  });
}

/* ============================== VALIDATION ============================== */
function runValidation(records){
  const total = records.length;
  const missing = {};
  REQUIRED_COLUMNS.forEach(c=> missing[c]=0);
  const seenKeys = new Map();
  let duplicateCount = 0;
  const duplicateIdxs = new Set();
  const invalidDates = [];
  const futureDates = [];
  const badLocation = [];
  const rangeIssues = {}; // field -> count
  Object.keys(DEFAULT_THRESHOLDS).forEach(f=>rangeIssues[f]=0);
  const parsedDates = [];

  records.forEach((r, i)=>{
    REQUIRED_COLUMNS.forEach(c=>{ if(r[c]===null || r[c]===undefined) missing[c]++; });
    const key = (r.Well_ID||"?")+"__"+(r.Date||"?");
    if(seenKeys.has(key)){ duplicateCount++; duplicateIdxs.add(i); duplicateIdxs.add(seenKeys.get(key)); }
    else seenKeys.set(key,i);

    const d = parseDateLoose(r.Date);
    parsedDates.push(d);
    if(!d) invalidDates.push(i);
    else if(d.getTime() > Date.now()+86400000) futureDates.push(i);

    if(r.Latitude===null || r.Longitude===null || r.Latitude<-90||r.Latitude>90||r.Longitude<-180||r.Longitude>180){
      badLocation.push(i);
    }
    Object.entries(DEFAULT_THRESHOLDS).forEach(([field,{min,max}])=>{
      const v = r[field];
      if(v!==null && (v<min || v>max)) rangeIssues[field]++;
    });
  });

  // outlier detection per well per numeric field (IQR method)
  const outlierIdxs = new Set();
  const numericFields = ["Groundwater_Level","Rainfall","Temperature","Soil_Moisture","pH","TDS","Fluoride","Nitrate","Chloride"];
  const byWell = {};
  records.forEach((r,i)=>{ const w=r.Well_ID||"UNKNOWN"; (byWell[w]=byWell[w]||[]).push(i); });
  const outlierDetail = {};
  numericFields.forEach(f=>outlierDetail[f]=0);
  Object.values(byWell).forEach(idxs=>{
    numericFields.forEach(field=>{
      const vals = idxs.map(i=>records[i][field]).filter(v=>v!=null);
      const b = iqrBounds(vals);
      if(!b) return;
      idxs.forEach(i=>{
        const v = records[i][field];
        if(v!=null && (v<b.low || v>b.high)){ outlierIdxs.add(i); outlierDetail[field]++; }
      });
    });
  });

  // chronological ordering check (per well)
  let outOfOrderCount = 0;
  Object.values(byWell).forEach(idxs=>{
    let lastT = -Infinity;
    idxs.forEach(i=>{
      const d = parsedDates[i];
      if(d){ if(d.getTime()<lastT) outOfOrderCount++; lastT = Math.max(lastT, d.getTime()); }
    });
  });

  // per-record quality flag
  const flags = records.map((r,i)=>{
    const isInvalid = invalidDates.includes(i) || badLocation.includes(i) || REQUIRED_COLUMNS.some(c=>c!=="Date"&&r[c]===null && ["Well_ID","Groundwater_Level"].includes(c));
    if(isInvalid) return "Invalid";
    if(duplicateIdxs.has(i) || outlierIdxs.has(i) || futureDates.includes(i) ||
       Object.keys(rangeIssues).some(f=> r[f]!=null && (r[f]<DEFAULT_THRESHOLDS[f].min||r[f]>DEFAULT_THRESHOLDS[f].max))){
      return "Review";
    }
    return "Valid";
  });

  const wells = Object.keys(byWell).filter(w=>w!=="UNKNOWN");
  const validDates = parsedDates.filter(Boolean).map(d=>d.getTime());

  return {
    total,
    columns: REQUIRED_COLUMNS,
    dateRange: validDates.length ? [new Date(Math.min(...validDates)), new Date(Math.max(...validDates))] : null,
    wellCount: wells.length,
    missing, missingTotal: Object.values(missing).reduce((a,b)=>a+b,0),
    duplicateCount,
    invalidDateCount: invalidDates.length,
    futureDateCount: futureDates.length,
    outOfOrderCount,
    badLocationCount: badLocation.length,
    rangeIssues, rangeIssueTotal: Object.values(rangeIssues).reduce((a,b)=>a+b,0),
    outlierDetail, outlierCount: outlierIdxs.size,
    flags,
    validCount: flags.filter(f=>f==="Valid").length,
    reviewCount: flags.filter(f=>f==="Review").length,
    invalidCount: flags.filter(f=>f==="Invalid").length,
    reviewDecisions: {} // idx -> 'accept'|'ignore', filled in by user in UI
  };
}

/* ========================== FEATURE ENGINEERING ========================== */
function engineerFeatures(records, validation){
  // Use only records not flagged Invalid; Review records are kept (flagged) unless the
  // user has explicitly excluded them via reviewDecisions.
  const usable = records.map((r,i)=>({r,i})).filter(({i})=>{
    if(validation.flags[i]==="Invalid") return false;
    if(validation.flags[i]==="Review" && validation.reviewDecisions[i]==="ignore") return false;
    return true;
  });
  const byWell = {};
  usable.forEach(({r,i})=>{ const w=r.Well_ID||"UNKNOWN"; (byWell[w]=byWell[w]||[]).push({...r, _i:i, _date: parseDateLoose(r.Date)}); });

  const rows = [];
  Object.entries(byWell).forEach(([wellId, recs])=>{
    const sorted = recs.filter(r=>r._date && r.Groundwater_Level!=null).sort((a,b)=>a._date-b._date);
    for(let k=0;k<sorted.length;k++){
      const cur = sorted[k];
      const prev1 = sorted[k-1];
      // lag-7 / lag-30 by index proxy (nearest record at least N days back)
      const findLag = (days) => {
        for(let j=k-1;j>=0;j--){ if((cur._date-sorted[j]._date)/86400000 >= days) return sorted[j]; }
        return null;
      };
      const lag7 = findLag(7), lag30 = findLag(30);
      const within = (days) => sorted.filter(s=> s._date<=cur._date && (cur._date-s._date)/86400000<=days);
      const cum7 = within(7).reduce((s,r)=>s+(r.Rainfall||0),0);
      const cum30 = within(30).reduce((s,r)=>s+(r.Rainfall||0),0);
      // short trend: slope of last up-to-5 prior levels
      const recentWindow = sorted.slice(Math.max(0,k-5),k);
      let trend = 0;
      if(recentWindow.length>=2){
        const xs = recentWindow.map((_,idx)=>idx);
        const ys = recentWindow.map(r=>r.Groundwater_Level);
        const mx = mean(xs), my = mean(ys);
        const num = xs.reduce((s,x,idx)=>s+(x-mx)*(ys[idx]-my),0);
        const den = xs.reduce((s,x)=>s+(x-mx)**2,0);
        trend = den>0 ? num/den : 0;
      }
      rows.push({
        wellId, date: cur._date, target: cur.Groundwater_Level,
        prevLevel: prev1 ? prev1.Groundwater_Level : cur.Groundwater_Level,
        lag7Level: lag7 ? lag7.Groundwater_Level : (prev1?prev1.Groundwater_Level:cur.Groundwater_Level),
        lag30Level: lag30 ? lag30.Groundwater_Level : (prev1?prev1.Groundwater_Level:cur.Groundwater_Level),
        rainfall: cur.Rainfall||0, cumRain7: cum7, cumRain30: cum30,
        temperature: cur.Temperature||0, soilMoisture: cur.Soil_Moisture||0,
        month: cur._date.getMonth(), season: seasonOf(cur._date.getMonth()), trend,
        lat: cur.Latitude, lon: cur.Longitude,
        ph: cur.pH, tds: cur.TDS, fluoride: cur.Fluoride, nitrate: cur.Nitrate, chloride: cur.Chloride
      });
    }
  });
  return rows;
}

function featureRowToVector(fr){
  return [fr.prevLevel, fr.lag7Level, fr.lag30Level, fr.cumRain7, fr.cumRain30, fr.temperature, fr.soilMoisture, fr.trend, fr.month];
}

/* ================================ ML TRAINING ============================ */
function trainAndEvaluate(featureRows){
  if(featureRows.length < 20){
    return {insufficient:true, count:featureRows.length};
  }
  // time-aware split: sort by date, first 80% train, last 20% test
  const sorted = featureRows.slice().sort((a,b)=>a.date-b.date);
  const splitAt = Math.floor(sorted.length*0.8);
  const train = sorted.slice(0,splitAt), test = sorted.slice(splitAt);
  const Xtr = train.map(featureRowToVector), ytr = train.map(r=>r.target);
  const Xte = test.map(featureRowToVector), yte = test.map(r=>r.target);

  const candidates = [];
  try{
    const lr = trainLinearRegression(Xtr, ytr);
    const preds = Xte.map(x=>lr.predict(x));
    candidates.push({name:"Linear Regression", model:lr, metrics: evalMetrics(yte, preds)});
  }catch(e){ /* skip on numerical failure */ }
  try{
    // Cap tree training set size so very large uploads stay responsive in-browser;
    // the O(n^2) split search is fine at prototype scale but not at unbounded scale.
    const CAP = 3000;
    let XtrTree = Xtr, ytrTree = ytr;
    if(Xtr.length > CAP){
      const step = Xtr.length / CAP;
      const idxs = Array.from({length:CAP}, (_,i)=>Math.floor(i*step));
      XtrTree = idxs.map(i=>Xtr[i]); ytrTree = idxs.map(i=>ytr[i]);
    }
    const tree = trainRegressionTree(XtrTree, ytrTree, 5, Math.max(3,Math.floor(XtrTree.length*0.02)));
    const preds = Xte.map(x=>tree.predict(x));
    candidates.push({name:"Decision Tree", model:tree, metrics: evalMetrics(yte, preds)});
  }catch(e){}
  try{
    const base = trainMovingAverageBaseline(Xtr, ytr);
    const preds = Xte.map(x=>base.predict(x));
    candidates.push({name:"Persistence Baseline", model:base, metrics: evalMetrics(yte, preds)});
  }catch(e){}

  candidates.sort((a,b)=> (a.metrics.rmse ?? Infinity) - (b.metrics.rmse ?? Infinity));
  return {
    insufficient:false,
    trainCount: train.length, testCount: test.length,
    featuresUsed: ["Previous level","7-day lag level","30-day lag level","7-day cum. rainfall","30-day cum. rainfall","Temperature","Soil moisture","Recent trend","Month"],
    candidates,
    chosen: candidates[0] || null
  };
}

function forecastWell(featureRows, wellId, model, horizonDays){
  const wellRows = featureRows.filter(r=>r.wellId===wellId).sort((a,b)=>a.date-b.date);
  if(!wellRows.length || !model) return {history:[], forecast:[]};
  const history = wellRows.map(r=>({date:r.date, level:r.target}));
  // iterative forecasting: step forward ~7 days at a time (matches weekly cadence assumption),
  // re-using the last known feature context and updating lag/prev with predicted values.
  const stepDays = 7;
  const steps = Math.max(1, Math.round(horizonDays/stepDays));
  let last = wellRows[wellRows.length-1];
  let recentLevels = wellRows.slice(-5).map(r=>r.target);
  const forecast = [];
  let curDate = new Date(last.date);
  let prevLevel = last.target, lag7 = last.lag7Level, lag30 = last.lag30Level;
  for(let s=0;s<steps;s++){
    curDate = new Date(curDate); curDate.setDate(curDate.getDate()+stepDays);
    const xs = recentLevels.map((_,idx)=>idx), ys=recentLevels;
    const mx=mean(xs), my=mean(ys);
    const num = xs.reduce((sum,x,idx)=>sum+(x-mx)*(ys[idx]-my),0);
    const den = xs.reduce((sum,x)=>sum+(x-mx)**2,0);
    const trend = den>0 ? num/den : 0;
    const vec = [prevLevel, lag7, lag30, last.cumRain7, last.cumRain30, last.temperature, last.soilMoisture, trend, curDate.getMonth()];
    const pred = model.predict(vec);
    forecast.push({date:new Date(curDate), level:pred});
    lag30 = lag7; lag7 = prevLevel; prevLevel = pred;
    recentLevels = [...recentLevels.slice(1), pred];
  }
  return {history, forecast};
}

/* ================================== WQI ================================== */
function calcWQI(sample, standards){
  // sample: {pH, TDS, Fluoride, Nitrate, Chloride}
  const fieldMap = {pH:"ph", TDS:"tds", Fluoride:"fluoride", Nitrate:"nitrate", Chloride:"chloride"};
  let wSum=0, wqSum=0;
  const contributions = [];
  Object.entries(standards).forEach(([field,{ideal,standard,weightFactor}])=>{
    const v = sample[fieldMap[field]];
    if(v==null) return;
    const qi = ((v-ideal)/(standard-ideal))*100;
    const unitWeight = weightFactor/standard; // relative unit weight
    wSum += unitWeight;
    wqSum += unitWeight*Math.max(0,qi);
    contributions.push({field, value:v, qi, weight:unitWeight});
  });
  const wqi = wSum>0 ? wqSum/wSum : null;
  let classification, color;
  if(wqi==null){ classification="Insufficient data"; color="neutral"; }
  else if(wqi<=50){ classification="Good"; color="safe"; }
  else if(wqi<=100){ classification="Moderate"; color="warning"; }
  else { classification="Poor"; color="critical"; }
  return {wqi, classification, color, contributions};
}

function paramStatus(field, value){
  const t = DEFAULT_THRESHOLDS[field];
  if(value==null || !t) return "neutral";
  if(value<t.min || value>t.max) return "critical";
  // soft warning zone within 15% of bound
  const span = t.max-t.min, warnMargin = span*0.1;
  if(value > t.max-warnMargin || value < t.min+warnMargin) return "warning";
  return "safe";
}

/* ================================ RISK ENGINE ============================= */
function assessWellRisk(wellId, featureRows, forecastResult, wqiResult){
  const wellRows = featureRows.filter(r=>r.wellId===wellId).sort((a,b)=>a.date-b.date);
  if(!wellRows.length) return {level:"SAFE", reasons:[], predictedChangePct:0};
  const currentLevel = wellRows[wellRows.length-1].target;
  const forecastEnd = forecastResult.forecast.length ? forecastResult.forecast[forecastResult.forecast.length-1].level : currentLevel;
  const changePct = currentLevel>0 ? ((forecastEnd-currentLevel)/currentLevel)*100 : 0;

  // historical trend over last ~90 days of records
  const cutoff = new Date(wellRows[wellRows.length-1].date); cutoff.setDate(cutoff.getDate()-90);
  const recent = wellRows.filter(r=>r.date>=cutoff);
  let histTrend = 0;
  if(recent.length>=2){
    const xs=recent.map((_,i)=>i), ys=recent.map(r=>r.target);
    const mx=mean(xs), my=mean(ys);
    const num=xs.reduce((s,x,i)=>s+(x-mx)*(ys[i]-my),0), den=xs.reduce((s,x)=>s+(x-mx)**2,0);
    histTrend = den>0?num/den:0;
  }

  const reasons = [];
  let score = 0; // higher = worse
  if(changePct < -15){ score+=2; reasons.push("Forecast shows a significant projected decline over the horizon."); }
  else if(changePct < -5){ score+=1; reasons.push("Forecast shows a moderate projected decline over the horizon."); }
  if(histTrend < -0.05){ score+=1; reasons.push("Recent recorded levels show a declining trend."); }
  const lastRain = wellRows[wellRows.length-1].cumRain30;
  if(lastRain!=null && lastRain < 10){ score+=0.5; reasons.push("Low recent cumulative rainfall (30-day) may limit natural recharge."); }
  if(wqiResult && wqiResult.classification==="Poor"){ score+=2; reasons.push("Water Quality Index is classified as Poor."); }
  else if(wqiResult && wqiResult.classification==="Moderate"){ score+=1; reasons.push("Water Quality Index is classified as Moderate."); }

  let level = "SAFE";
  if(score>=3) level="CRITICAL";
  else if(score>=1) level="WARNING";
  return {level, reasons, predictedChangePct:changePct, histTrend, currentLevel, forecastEnd};
}

/* ============================ ALERTS & RECOMMENDATIONS ==================== */
function generateAlerts(riskByWell){
  const alerts = [];
  Object.entries(riskByWell).forEach(([wellId, risk])=>{
    if(risk.level==="CRITICAL"){
      alerts.push({
        wellId, severity:"CRITICAL",
        title:"Critical risk detected",
        body:`The prediction engine identifies significant deterioration for well ${wellId} requiring investigation. ${risk.reasons[0]||""}`
      });
    } else if(risk.level==="WARNING"){
      const isDecline = risk.reasons.some(r=>r.toLowerCase().includes("declin"));
      const isQuality = risk.reasons.some(r=>r.toLowerCase().includes("water quality"));
      if(isDecline){
        alerts.push({wellId, severity:"WARNING", title:"Groundwater decline", body:`Predicted or recent groundwater level shows a declining trend at well ${wellId}.`});
      }
      if(isQuality){
        alerts.push({wellId, severity:"WARNING", title:"Water quality concern", body:`One or more monitored parameters at well ${wellId} require review.`});
      }
      if(!isDecline && !isQuality){
        alerts.push({wellId, severity:"WARNING", title:"Condition requires review", body:`Well ${wellId} shows early indicators that warrant closer monitoring.`});
      }
    }
  });
  return alerts;
}

function generateRecommendations(riskByWell, wqiByWell){
  const recs = [];
  const criticalWells = Object.entries(riskByWell).filter(([,r])=>r.level==="CRITICAL").map(([w])=>w);
  const warningWells = Object.entries(riskByWell).filter(([,r])=>r.level==="WARNING").map(([w])=>w);
  const poorWQI = Object.entries(wqiByWell||{}).filter(([,w])=>w.classification==="Poor").map(([w])=>w);

  if(criticalWells.length){
    recs.push(`Wells ${criticalWells.join(", ")} show critical projected conditions. Consider increasing monitoring frequency and investigating local extraction patterns and rainfall conditions as soon as practical.`);
  }
  if(warningWells.length){
    recs.push(`Wells ${warningWells.join(", ")} show early warning indicators. Continue routine monitoring with closer attention to upcoming readings.`);
  }
  if(poorWQI.length){
    recs.push(`Water quality at ${poorWQI.join(", ")} is classified as Poor based on the configured index. Review the affected parameters and consider additional monitoring or laboratory verification.`);
  }
  if(!criticalWells.length && !warningWells.length){
    recs.push("Overall risk is low across monitored wells. Continue routine monitoring and maintain the current observation schedule.");
  }
  recs.push("These are decision-support suggestions generated by a prototype model and are not a replacement for professional hydrological advice or laboratory testing.");
  return recs;
}

/* ============================ PIPELINE ORCHESTRATION ======================= */
function runFullPipeline(projectId){
  const ds = STATE.datasets[projectId];
  if(!ds) return;
  const validation = runValidation(ds.records);
  STATE.validation[projectId] = validation;

  const featureRows = engineerFeatures(ds.records, validation);
  STATE.features[projectId] = featureRows;

  const training = trainAndEvaluate(featureRows);
  STATE.models[projectId] = training;

  const wells = [...new Set(featureRows.map(r=>r.wellId))];
  const predByWell = {}, wqiByWell = {}, riskByWell = {};
  const model = training.chosen ? training.chosen.model : null;
  const horizon = STATE.ui.predictHorizon || 30;

  wells.forEach(wellId=>{
    const fc = model ? forecastWell(featureRows, wellId, model, horizon) : {history:featureRows.filter(r=>r.wellId===wellId).map(r=>({date:r.date,level:r.target})), forecast:[]};
    predByWell[wellId] = fc;
    const wellRows = featureRows.filter(r=>r.wellId===wellId);
    const last = wellRows[wellRows.length-1];
    const sample = last ? {ph:last.ph, tds:last.tds, fluoride:last.fluoride, nitrate:last.nitrate, chloride:last.chloride} : {};
    const wqi = calcWQI(sample, DEFAULT_WQI_STANDARDS);
    wqiByWell[wellId] = wqi;
    riskByWell[wellId] = assessWellRisk(wellId, featureRows, fc, wqi);
  });

  STATE.predictions[projectId] = {byWell: predByWell};
  STATE.wqi[projectId] = {byWell: wqiByWell};
  STATE.risk[projectId] = {byWell: riskByWell};
  STATE.alerts[projectId] = generateAlerts(riskByWell);
  STATE.recommendations[projectId] = generateRecommendations(riskByWell, wqiByWell);

  if(!STATE.ui.predictWell || !wells.includes(STATE.ui.predictWell)) STATE.ui.predictWell = wells[0]||null;
}

function overallRiskForProject(projectId){
  const risk = STATE.risk[projectId];
  if(!risk) return "SAFE";
  const levels = Object.values(risk.byWell).map(r=>r.level);
  if(levels.includes("CRITICAL")) return "CRITICAL";
  if(levels.includes("WARNING")) return "WARNING";
  return "SAFE";
}
function overallWQIForProject(projectId){
  const wqi = STATE.wqi[projectId];
  if(!wqi) return null;
  const vals = Object.values(wqi.byWell).map(w=>w.wqi).filter(v=>v!=null);
  return vals.length ? mean(vals) : null;
}

/* ============================== PROJECT MANAGEMENT ========================= */
function createProject({name, location, stateName, country, description, wells}){
  const p = {id:uid(), name, location, state:stateName, country, description, wells:+wells||1, createdAt: new Date()};
  STATE.projects.push(p);
  STATE.activeProjectId = p.id;
  return p;
}
function loadDemoIntoProject(projectId){
  const project = STATE.projects.find(p=>p.id===projectId);
  const records = generateSyntheticDataset(project);
  STATE.datasets[projectId] = {records, isDemo:true, fileName:"synthetic_demo_dataset.csv", uploadedAt:new Date()};
  runFullPipeline(projectId);
}
function loadUploadedRecords(projectId, rawRows, fileName){
  const records = normalizeRecords(rawRows);
  STATE.datasets[projectId] = {records, isDemo:false, fileName, uploadedAt:new Date()};
  runFullPipeline(projectId);
}

/* ================================== UI LAYER =============================== */
const NAV_ITEMS = [
  {id:"landing", label:"Home", icon:"⌂"},
  {id:"dashboard", label:"Dashboard", icon:"▤"},
  {id:"projects", label:"Projects", icon:"◧"},
  {id:"upload", label:"Data Upload", icon:"⇧"},
  {id:"validation", label:"Validation", icon:"✓"},
  {id:"prediction", label:"Prediction", icon:"↝"},
  {id:"waterquality", label:"Water Quality", icon:"◍"},
  {id:"risk", label:"Risk & Alerts", icon:"⚠"},
  {id:"map", label:"Map", icon:"⚲"},
  {id:"reports", label:"Reports", icon:"▥"},
];
let activeCharts = []; // track Chart.js instances to destroy on re-render
let leafletMap = null;

function el(html){ const t=document.createElement("template"); t.innerHTML=html.trim(); return t.content.firstElementChild; }
function destroyCharts(){ activeCharts.forEach(c=>c.destroy()); activeCharts=[]; }

function activeProject(){ return STATE.projects.find(p=>p.id===STATE.activeProjectId) || null; }
function requireProjectGate(viewName){
  const p = activeProject();
  if(!p){
    return `<div class="empty panel"><h3>No project selected</h3><p>Create or select a project to use ${viewName}.</p>
      <div class="btnrow" style="justify-content:center;margin-top:14px;"><button class="btn" onclick="navigate('projects')">Go to Projects</button></div></div>`;
  }
  return null;
}
function requireDataGate(){
  const p = activeProject();
  const ds = STATE.datasets[p.id];
  if(!ds){
    return `<div class="empty panel"><h3>No dataset loaded</h3><p>Upload a dataset or load the demo dataset for <b>${p.name}</b>.</p>
      <div class="btnrow" style="justify-content:center;margin-top:14px;">
        <button class="btn" onclick="navigate('upload')">Go to Data Upload</button>
        <button class="btn secondary" onclick="handleLoadDemo()">Load Demo Project</button>
      </div></div>`;
  }
  return null;
}

function renderSidebar(){
  const navlist = document.getElementById("navlist");
  navlist.innerHTML = NAV_ITEMS.map(item=>`
    <li><button class="${STATE.view===item.id?'active':''}" onclick="navigate('${item.id}')">
      <span class="icn">${item.icon}</span>${item.label}
    </button></li>`).join("");
  const picker = document.getElementById("projectPicker");
  if(!STATE.projects.length){
    picker.innerHTML = `<option>No projects yet</option>`;
  } else {
    picker.innerHTML = STATE.projects.map(p=>`<option value="${p.id}" ${p.id===STATE.activeProjectId?'selected':''}>${p.name}</option>`).join("");
  }
  picker.onchange = (e)=>{ STATE.activeProjectId = e.target.value; render(); };
}

function navigate(view){ STATE.view = view; render(); window.scrollTo(0,0); }

function render(){
  renderSidebar();
  const container = document.getElementById("views");
  const fn = VIEW_RENDERERS[STATE.view] || VIEW_RENDERERS.landing;
  destroyCharts();
  container.innerHTML = `<div class="view active">${fn()}</div>`;
  const post = POST_RENDER[STATE.view];
  if(post) post();
}

const VIEW_RENDERERS = {};
const POST_RENDER = {};

/* --------------------------------- LANDING -------------------------------- */
VIEW_RENDERERS.landing = () => `
  <div id="landing">
    <div class="hero">
      <div class="hero-inner">
        <h1>AI-Powered Groundwater Intelligence</h1>
        <p class="tag">Predict groundwater conditions. Detect risks early. Support better decisions.</p>
        <div class="btnrow">
          <button class="btn" style="background:#fff;color:var(--deep-teal);" onclick="navigate('dashboard')">Open Dashboard</button>
          <button class="btn secondary" onclick="navigate('projects')">Create Project</button>
        </div>
      </div>
    </div>
    <div class="capgrid">
      <div class="capcard"><h4>Groundwater Prediction</h4><p>AI-powered future level forecasting built from your own historical readings.</p></div>
      <div class="capcard"><h4>Water Quality</h4><p>WQI-based quality assessment across pH, TDS, fluoride, nitrate and chloride.</p></div>
      <div class="capcard"><h4>Early Warning</h4><p>Identify potential risks before they escalate, with plain-language recommendations.</p></div>
    </div>
    <div class="landing-body">
      <div class="panel">
        <h3>How it works</h3>
        <p style="color:var(--ink-soft);font-size:13.5px;">Every screen in this prototype is wired to a real calculation running in your browser — nothing here is a static mockup.</p>
        <div class="pipeline" style="margin-top:16px;">
          ${["Project","Upload","Validate","Engineer features","Train models","Predict","WQI & Risk","Dashboard"].map(s=>`<div class="pipestep pending">${s}</div>`).join("")}
        </div>
        <div class="note limit">This is a client-side prototype: data validation, feature engineering, model training/evaluation, WQI and risk scoring all run locally in your browser session. A production deployment would move the ML and storage layers to a server (FastAPI + PostgreSQL) as described in the accompanying README, so multiple users can share persisted data.</div>
      </div>
    </div>
  </div>`;

/* --------------------------------- PROJECTS -------------------------------- */
VIEW_RENDERERS.projects = () => {
  const cards = STATE.projects.map(p=>{
    const ds = STATE.datasets[p.id];
    const risk = ds ? overallRiskForProject(p.id) : null;
    const wqi = ds ? overallWQIForProject(p.id) : null;
    const lastLevel = ds && STATE.predictions[p.id] ? (()=>{
      const wells = Object.values(STATE.predictions[p.id].byWell);
      const levels = wells.map(w=>w.history.length?w.history[w.history.length-1].level:null).filter(v=>v!=null);
      return levels.length?mean(levels):null;
    })() : null;
    return `<div class="projectcard ${p.id===STATE.activeProjectId?'selected':''}" onclick="selectProject('${p.id}')">
      <h4>${p.name}</h4>
      <div class="loc">${p.location}${p.state?', '+p.state:''}${p.country?', '+p.country:''}</div>
      <div class="stats">
        <span>Wells <b>${p.wells}</b></span>
        <span>Last update <b>${ds?new Date(ds.uploadedAt).toLocaleDateString():'—'}</b></span>
        <span>Level <b>${lastLevel!=null?fmt(lastLevel,1)+' m':'—'}</b></span>
        <span>WQI <b>${wqi!=null?fmt(wqi,0):'—'}</b></span>
        ${risk?`<span class="status-pill status-${risk.toLowerCase()}">${risk}</span>`:''}
        ${ds && ds.isDemo?'<span class="demotag">Demo data</span>':''}
      </div>
    </div>`;
  }).join("");

  return `
    <div class="pagehead"><div><h2>Projects</h2><div class="ctx">Create and manage groundwater monitoring projects.</div></div></div>
    <div class="grid cols-2">
      <div>
        <div class="grid" style="grid-template-columns:1fr;gap:14px;">
          ${cards || `<div class="empty panel"><h3>No projects yet</h3><p>Create your first monitoring project using the form.</p></div>`}
        </div>
      </div>
      <div class="panel" style="align-self:start;">
        <h3>New project</h3>
        <div class="field"><label>Project name</label><input id="npName" placeholder="e.g. Coastal Aquifer Monitoring"></div>
        <div class="grid cols-2">
          <div class="field"><label>Location</label><input id="npLocation" placeholder="City / district"></div>
          <div class="field"><label>State / region</label><input id="npState" placeholder="State"></div>
        </div>
        <div class="grid cols-2">
          <div class="field"><label>Country</label><input id="npCountry" placeholder="Country"></div>
          <div class="field"><label>Monitoring wells</label><input id="npWells" type="number" min="1" value="6"></div>
        </div>
        <div class="field"><label>Description</label><textarea id="npDesc" rows="3" placeholder="Optional notes about this project"></textarea></div>
        <button class="btn" onclick="handleCreateProject()">Create project</button>
        <div class="note">After creating a project, load the demo dataset or go to Data Upload to bring in your own CSV/XLSX file.</div>
      </div>
    </div>`;
};

function handleCreateProject(){
  const name = document.getElementById("npName").value.trim();
  if(!name){ alert("Please enter a project name."); return; }
  const p = createProject({
    name,
    location: document.getElementById("npLocation").value.trim() || "Unspecified",
    stateName: document.getElementById("npState").value.trim(),
    country: document.getElementById("npCountry").value.trim(),
    description: document.getElementById("npDesc").value.trim(),
    wells: document.getElementById("npWells").value
  });
  render();
}
function selectProject(id){ STATE.activeProjectId=id; render(); }
function handleLoadDemo(){
  const p = activeProject(); if(!p) return;
  loadDemoIntoProject(p.id);
  navigate("dashboard");
}

/* --------------------------------- UPLOAD -------------------------------- */
VIEW_RENDERERS.upload = () => {
  const gate = requireProjectGate("Data Upload"); if(gate) return gate;
  const p = activeProject();
  const ds = STATE.datasets[p.id];
  const val = STATE.validation[p.id];

  let previewBlock = "";
  if(ds){
    const dr = val && val.dateRange ? `${val.dateRange[0].toLocaleDateString()} – ${val.dateRange[1].toLocaleDateString()}` : "—";
    previewBlock = `
      <div class="panel">
        <div class="panel-title-row">
          <h3>Dataset preview ${ds.isDemo?'<span class="demotag">Demo data</span>':''}</h3>
          <span class="ctx" style="font-size:12px;color:var(--ink-soft);">${ds.fileName}</span>
        </div>
        <div class="grid cols-4">
          <div class="kpi"><div class="label">Records</div><div class="value">${ds.records.length.toLocaleString()}</div></div>
          <div class="kpi"><div class="label">Monitoring wells</div><div class="value">${val?val.wellCount:'—'}</div></div>
          <div class="kpi"><div class="label">Missing values</div><div class="value">${val?val.missingTotal:'—'}</div></div>
          <div class="kpi"><div class="label">Potential outliers</div><div class="value">${val?val.outlierCount:'—'}</div></div>
        </div>
        <div class="note">Date range: ${dr}</div>
        <div class="tablewrap" style="margin-top:16px;">
          <table><thead><tr>${REQUIRED_COLUMNS.map(c=>`<th>${c}</th>`).join("")}</tr></thead>
          <tbody>${ds.records.slice(0,8).map(r=>`<tr>${REQUIRED_COLUMNS.map(c=>`<td>${r[c]??'<span style="color:var(--rust)">—</span>'}</td>`).join("")}</tr>`).join("")}</tbody></table>
        </div>
        <div class="btnrow" style="margin-top:16px;">
          <button class="btn" onclick="navigate('validation')">Continue to Validation</button>
        </div>
      </div>`;
  }

  return `
    <div class="pagehead"><div><h2>Data Upload</h2><div class="ctx">Project: ${p.name}</div></div></div>
    <div class="panel">
      <div id="dropzone" class="dropzone">
        <div class="big">Drag &amp; drop your CSV or XLSX file here</div>
        <div class="small">or click to browse — data flows straight into validation and prediction</div>
        <input type="file" id="fileInput" accept=".csv" style="display:none;">
        <div class="btnrow" style="justify-content:center;margin-top:16px;">
          <button class="btn" onclick="document.getElementById('fileInput').click()">Upload dataset</button>
          <button class="btn secondary" onclick="downloadTemplate()">Download sample template</button>
          <button class="btn ghost" onclick="handleLoadDemo()">Load demo project</button>
        </div>
        <div class="note" style="margin-top:16px;text-align:left;">Expected columns: ${REQUIRED_COLUMNS.join(", ")}. XLSX files: export/save as CSV first — this browser-based prototype parses CSV directly.</div>
      </div>
    </div>
    ${previewBlock}
  `;
};

POST_RENDER.upload = () => {
  const dz = document.getElementById("dropzone");
  const input = document.getElementById("fileInput");
  if(!dz) return;
  dz.addEventListener("click", ()=>input.click());
  ["dragenter","dragover"].forEach(evt=>dz.addEventListener(evt, e=>{e.preventDefault(); dz.classList.add("drag");}));
  ["dragleave","drop"].forEach(evt=>dz.addEventListener(evt, e=>{e.preventDefault(); dz.classList.remove("drag");}));
  dz.addEventListener("drop", e=>{
    const f = e.dataTransfer.files[0];
    if(f) handleFile(f);
  });
  input.addEventListener("change", e=>{
    const f = e.target.files[0];
    if(f) handleFile(f);
  });
};

function handleFile(file){
  const p = activeProject(); if(!p) return;
  if(!/\.csv$/i.test(file.name)){
    alert("This prototype parses CSV directly. Please export your spreadsheet as CSV, or use Download Sample Template as a starting point.");
    return;
  }
  parseCSVFile(file, (rows, meta, err)=>{
    if(err || !rows){ alert("Could not read this file: "+(err?err.message:"unknown error")); return; }
    if(!rows.length){ alert("The uploaded file has no data rows."); return; }
    loadUploadedRecords(p.id, rows, file.name);
    navigate("validation");
  });
}

function downloadTemplate(){
  const header = REQUIRED_COLUMNS.join(",");
  const sampleRow = "2026-01-05,W-01,18.52,73.85,7.4,12.0,24.5,38.0,7.1,410,0.6,18.2,120";
  const csv = header+"\n"+sampleRow+"\n";
  const blob = new Blob([csv], {type:"text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "groundwater_sample_template.csv"; a.click();
}

/* ------------------------------- VALIDATION ------------------------------- */
VIEW_RENDERERS.validation = () => {
  const gate = requireProjectGate("Validation"); if(gate) return gate;
  const p = activeProject();
  const dataGate = requireDataGate(); if(dataGate) return dataGate;
  const val = STATE.validation[p.id];

  const steps = [
    {label:"Upload", ok:true},
    {label:"Missing Values", ok: val.missingTotal===0, warn: val.missingTotal>0},
    {label:"Duplicates", ok: val.duplicateCount===0, warn: val.duplicateCount>0},
    {label:"Dates", ok: val.invalidDateCount===0 && val.futureDateCount===0, warn: val.invalidDateCount>0||val.futureDateCount>0},
    {label:"Locations", ok: val.badLocationCount===0, warn: val.badLocationCount>0},
    {label:"Range Checks", ok: val.rangeIssueTotal===0, warn: val.rangeIssueTotal>0},
    {label:"Outliers", ok: val.outlierCount===0, warn: val.outlierCount>0},
  ];
  const pipelineHtml = steps.map(s=>`<div class="pipestep ${s.ok?'ok':(s.warn?'warn':'pending')}">${s.label}<div class="n">${s.ok?'✓':(s.warn?'⚠':'')}</div></div>`).join("");

  const missingRows = Object.entries(val.missing).filter(([,v])=>v>0)
    .map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join("") || `<tr><td colspan="2" style="color:var(--ink-soft);">No missing values detected.</td></tr>`;
  const rangeRows = Object.entries(val.rangeIssues).filter(([,v])=>v>0)
    .map(([k,v])=>`<tr><td>${k}</td><td>${v}</td><td class="mono">${DEFAULT_THRESHOLDS[k].min}–${DEFAULT_THRESHOLDS[k].max} ${DEFAULT_THRESHOLDS[k].unit}</td></tr>`).join("")
    || `<tr><td colspan="3" style="color:var(--ink-soft);">No range issues detected against configured thresholds.</td></tr>`;
  const outlierRows = Object.entries(val.outlierDetail).filter(([,v])=>v>0)
    .map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join("") || `<tr><td colspan="2" style="color:var(--ink-soft);">No statistical outliers (IQR method) detected.</td></tr>`;

  return `
    <div class="pagehead"><div><h2>Data Validation</h2><div class="ctx">Project: ${p.name} · ${val.total.toLocaleString()} records</div></div></div>
    <div class="panel">
      <h3>Validation pipeline</h3>
      <div class="pipeline">${pipelineHtml}</div>
      <div class="grid cols-3" style="margin-top:14px;">
        <div class="kpi"><div class="label">Valid records</div><div class="value" style="color:var(--good)">${val.validCount.toLocaleString()}</div></div>
        <div class="kpi"><div class="label">Requiring review</div><div class="value" style="color:var(--amber)">${val.reviewCount.toLocaleString()}</div></div>
        <div class="kpi"><div class="label">Invalid</div><div class="value" style="color:var(--rust)">${val.invalidCount.toLocaleString()}</div></div>
      </div>
    </div>
    <div class="grid cols-2">
      <div class="panel"><h3>Missing values by field</h3><div class="tablewrap"><table><thead><tr><th>Field</th><th>Count</th></tr></thead><tbody>${missingRows}</tbody></table></div></div>
      <div class="panel"><h3>Duplicate &amp; date checks</h3>
        <table><tbody>
          <tr><td>Duplicate records (Well_ID + Date)</td><td class="mono">${val.duplicateCount}</td></tr>
          <tr><td>Invalid / unparseable dates</td><td class="mono">${val.invalidDateCount}</td></tr>
          <tr><td>Future dates</td><td class="mono">${val.futureDateCount}</td></tr>
          <tr><td>Out-of-order records</td><td class="mono">${val.outOfOrderCount}</td></tr>
          <tr><td>Invalid coordinates</td><td class="mono">${val.badLocationCount}</td></tr>
        </tbody></table>
      </div>
    </div>
    <div class="grid cols-2">
      <div class="panel"><h3>Range checks <span style="font-weight:400;font-size:11.5px;color:var(--ink-soft);">(configurable thresholds, not a cited regulatory standard)</span></h3>
        <div class="tablewrap"><table><thead><tr><th>Field</th><th>Flagged</th><th>Expected range</th></tr></thead><tbody>${rangeRows}</tbody></table></div>
      </div>
      <div class="panel"><h3>Outlier detection <span style="font-weight:400;font-size:11.5px;color:var(--ink-soft);">(IQR method, per well)</span></h3>
        <div class="tablewrap"><table><thead><tr><th>Field</th><th>Flagged</th></tr></thead><tbody>${outlierRows}</tbody></table>
        <div class="note">Outliers are flagged for review, not deleted automatically — an unusual reading can reflect a real environmental event.</div>
      </div>
    </div>
    <div class="panel">
      <h3>Record-level quality flags</h3>
      <div class="tablewrap"><table><thead><tr><th>#</th><th>Well</th><th>Date</th><th>Level</th><th>Flag</th></tr></thead>
      <tbody>${STATE.datasets[p.id].records.map((r,i)=>({r,i})).filter(({i})=>val.flags[i]!=="Valid").slice(0,25).map(({r,i})=>
        `<tr><td class="mono">${i}</td><td>${r.Well_ID??'—'}</td><td>${r.Date??'—'}</td><td class="mono">${r.Groundwater_Level??'—'}</td><td><span class="badge ${val.flags[i].toLowerCase()}">${val.flags[i]}</span></td></tr>`
      ).join("") || `<tr><td colspan="5" style="color:var(--ink-soft);">All records passed validation cleanly.</td></tr>`}</tbody></table></div>
      <div class="note">Showing up to 25 flagged records. Review and Invalid records can be handled via a data-steward workflow in a full deployment; this prototype currently keeps Review records in the pipeline (flagged) and excludes Invalid records from model training.</div>
    </div>
    <div class="btnrow"><button class="btn" onclick="navigate('prediction')">Continue to Prediction</button></div>
  `;
};

/* ------------------------------- PREDICTION ------------------------------- */
VIEW_RENDERERS.prediction = () => {
  const gate = requireProjectGate("Prediction"); if(gate) return gate;
  const p = activeProject();
  const dataGate = requireDataGate(); if(dataGate) return dataGate;
  const training = STATE.models[p.id];
  const featureRows = STATE.features[p.id];
  const wells = [...new Set(featureRows.map(r=>r.wellId))];

  if(training.insufficient){
    return `
    <div class="pagehead"><div><h2>AI Prediction</h2><div class="ctx">Project: ${p.name}</div></div></div>
    <div class="panel empty"><h3>Prediction unavailable</h3>
      <p>This dataset does not yet have enough historical records (${training.count} usable rows) to generate a reliable forecast.<br>Please upload additional historical observations.</p>
    </div>`;
  }

  const modelRows = training.candidates.map(c=>`
    <tr class="${training.chosen===c?'style="font-weight:600"':''}">
      <td>${c.name} ${training.chosen===c?'<span class="badge valid" style="margin-left:6px;">Selected</span>':''}</td>
      <td class="mono">${fmt(c.metrics.mae)}</td>
      <td class="mono">${fmt(c.metrics.rmse)}</td>
      <td class="mono">${c.metrics.r2!=null?fmt(c.metrics.r2,3):'—'}</td>
    </tr>`).join("");

  const wellOptions = wells.map(w=>`<option value="${w}" ${w===STATE.ui.predictWell?'selected':''}>${w}</option>`).join("");
  const fc = STATE.predictions[p.id].byWell[STATE.ui.predictWell] || {history:[],forecast:[]};
  const curLevel = fc.history.length ? fc.history[fc.history.length-1].level : null;
  const predLevel = fc.forecast.length ? fc.forecast[fc.forecast.length-1].level : null;
  const change = (curLevel!=null && predLevel!=null) ? predLevel-curLevel : null;

  return `
    <div class="pagehead"><div><h2>AI Prediction</h2><div class="ctx">Project: ${p.name}</div></div></div>
    <div class="panel">
      <h3>Model comparison <span style="font-weight:400;font-size:11.5px;color:var(--ink-soft);">— trained on this dataset, time-aware split</span></h3>
      <div class="grid cols-4" style="margin-bottom:14px;">
        <div class="kpi"><div class="label">Training records</div><div class="value">${training.trainCount}</div></div>
        <div class="kpi"><div class="label">Testing records</div><div class="value">${training.testCount}</div></div>
        <div class="kpi"><div class="label">Features used</div><div class="value" style="font-size:14px;line-height:1.4;">${training.featuresUsed.length}</div></div>
        <div class="kpi"><div class="label">Prediction horizon</div><div class="value" style="font-size:22px;">${STATE.ui.predictHorizon} days</div></div>
      </div>
      <div class="tablewrap"><table><thead><tr><th>Model</th><th>MAE (m)</th><th>RMSE (m)</th><th>R²</th></tr></thead><tbody>${modelRows}</tbody></table></div>
      <div class="note">Metrics are computed on a held-out, time-ordered test split — later dates only, never shuffled into training. Random Forest / XGBoost would typically be run server-side (see README); this browser prototype substitutes a genuine Decision Tree regressor and a persistence baseline so every number stays real rather than illustrative.</div>
      <div class="note limit">Features considered: ${training.featuresUsed.join(", ")}.</div>
    </div>

    <div class="panel">
      <div class="panel-title-row">
        <h3>Forecast</h3>
        <div class="btnrow">
          <select id="wellSelect" style="padding:7px 10px;border-radius:4px;border:1px solid var(--line);">${wellOptions}</select>
          <select id="horizonSelect" style="padding:7px 10px;border-radius:4px;border:1px solid var(--line);">
            ${[7,30,90].map(h=>`<option value="${h}" ${h===STATE.ui.predictHorizon?'selected':''}>${h} days</option>`).join("")}
          </select>
        </div>
      </div>
      <div class="grid cols-3">
        <div class="kpi"><div class="label">Current groundwater level</div><div class="value">${fmt(curLevel,1)} m</div></div>
        <div class="kpi"><div class="label">Predicted level</div><div class="value">${fmt(predLevel,1)} m</div></div>
        <div class="kpi"><div class="label">Predicted change</div><div class="value delta ${change==null?'':(change<0?'up':'down')}" style="font-size:22px;">${change==null?'—':(change>0?'+':'')+fmt(change,1)+' m'}</div></div>
      </div>
      <div class="chartbox" style="margin-top:16px;"><canvas id="forecastChart"></canvas></div>
    </div>
    <div class="btnrow"><button class="btn" onclick="navigate('waterquality')">Continue to Water Quality</button></div>
  `;
};

POST_RENDER.prediction = () => {
  const p = activeProject(); if(!p || !STATE.predictions[p.id]) return;
  const wellSelect = document.getElementById("wellSelect");
  const horizonSelect = document.getElementById("horizonSelect");
  if(wellSelect) wellSelect.onchange = (e)=>{ STATE.ui.predictWell=e.target.value; render(); };
  if(horizonSelect) horizonSelect.onchange = (e)=>{ STATE.ui.predictHorizon=+e.target.value; runFullPipeline(p.id); render(); };

  const canvas = document.getElementById("forecastChart");
  if(!canvas) return;
  const fc = STATE.predictions[p.id].byWell[STATE.ui.predictWell] || {history:[],forecast:[]};
  const histLabels = fc.history.map(h=>h.date.toLocaleDateString());
  const fcLabels = fc.forecast.map(h=>h.date.toLocaleDateString());
  const labels = [...histLabels, ...fcLabels];
  const histData = [...fc.history.map(h=>h.level), ...new Array(fc.forecast.length).fill(null)];
  const fcData = [...new Array(Math.max(0,fc.history.length-1)).fill(null), fc.history.length?fc.history[fc.history.length-1].level:null, ...fc.forecast.map(h=>h.level)];
  const chart = new Chart(canvas, {
    type:"line",
    data:{ labels, datasets:[
      {label:"Historical level", data:histData, borderColor:"#175C5A", backgroundColor:"rgba(23,92,90,.08)", tension:.25, pointRadius:0, borderWidth:2},
      {label:"Predicted level", data:fcData, borderColor:"#C68A2E", borderDash:[6,4], tension:.25, pointRadius:0, borderWidth:2}
    ]},
    options:{ responsive:true, maintainAspectRatio:false, interaction:{mode:"index",intersect:false},
      scales:{ y:{title:{display:true,text:"Groundwater level (m)"}}, x:{ticks:{maxTicksLimit:10}} },
      plugins:{legend:{position:"bottom"}} }
  });
  activeCharts.push(chart);
};

/* ------------------------------ WATER QUALITY ------------------------------ */
VIEW_RENDERERS.waterquality = () => {
  const gate = requireProjectGate("Water Quality"); if(gate) return gate;
  const p = activeProject();
  const dataGate = requireDataGate(); if(dataGate) return dataGate;
  const wqiByWell = STATE.wqi[p.id].byWell;
  const wells = Object.keys(wqiByWell);
  const selectedWell = STATE.ui.predictWell && wells.includes(STATE.ui.predictWell) ? STATE.ui.predictWell : wells[0];
  const wqi = wqiByWell[selectedWell];
  const featureRows = STATE.features[p.id].filter(r=>r.wellId===selectedWell).sort((a,b)=>a.date-b.date);
  const last = featureRows[featureRows.length-1] || {};

  const params = [
    {key:"ph", label:"pH", val:last.ph, max:14},
    {key:"tds", label:"TDS", val:last.tds, max:1500, unit:"mg/L"},
    {key:"nitrate", label:"Nitrate", val:last.nitrate, max:100, unit:"mg/L"},
    {key:"fluoride", label:"Fluoride", val:last.fluoride, max:5, unit:"mg/L"},
    {key:"chloride", label:"Chloride", val:last.chloride, max:600, unit:"mg/L"},
  ];
  const fieldNameMap = {ph:"pH", tds:"TDS", nitrate:"Nitrate", fluoride:"Fluoride", chloride:"Chloride"};
  const rows = params.map(pr=>{
    const status = paramStatus(fieldNameMap[pr.key], pr.val);
    const icon = status==="safe"?"✓":status==="warning"?"⚠":"❌";
    const pct = pr.val!=null ? Math.min(100,(pr.val/pr.max)*100) : 0;
    const color = status==="safe"?"var(--good)":status==="warning"?"var(--amber)":"var(--rust)";
    return `<div class="wqirow"><div class="pname">${pr.label}</div><div class="pval">${pr.val!=null?fmt(pr.val, pr.key==='ph'?2:1):'—'} ${pr.unit||''}</div>
      <div class="bar"><div style="width:${pct}%;background:${color};"></div></div><div>${icon}</div></div>`;
  }).join("");

  const gaugeColor = wqi.color==="safe"?"var(--good-bg)":wqi.color==="warning"?"var(--amber-bg)":wqi.color==="critical"?"var(--rust-bg)":"#EEECE3";
  const gaugeText = wqi.color==="safe"?"var(--good)":wqi.color==="warning"?"var(--amber)":wqi.color==="critical"?"var(--rust)":"var(--ink-soft)";

  return `
    <div class="pagehead"><div><h2>Water Quality</h2><div class="ctx">Project: ${p.name}</div></div>
      <select id="wqWellSelect" style="padding:7px 10px;border-radius:4px;border:1px solid var(--line);">
        ${wells.map(w=>`<option value="${w}" ${w===selectedWell?'selected':''}>${w}</option>`).join("")}
      </select>
    </div>
    <div class="panel">
      <div class="sidebysidewqi">
        <div class="wqigauge" style="background:${gaugeColor};color:${gaugeText};">
          <div class="num">${wqi.wqi!=null?fmt(wqi.wqi,0):'—'}</div>
          <div class="lbl">${wqi.classification}</div>
        </div>
        <div style="flex:1;min-width:260px;">
          <h3 style="margin-bottom:6px;">Water Quality Index — Well ${selectedWell}</h3>
          <p style="font-size:13px;color:var(--ink-soft);margin-top:0;">Calculated from the weighted arithmetic WQI method using the parameters below against configurable reference values (see note). Classification: 🟢 Good (≤50) · 🟡 Moderate (51–100) · 🔴 Poor (&gt;100).</p>
          <div class="note">Reference values used for scoring are user-configurable defaults, not a cited regulatory standard. Replace them with an applicable local/national standard before using this for real decisions.</div>
        </div>
      </div>
    </div>
    <div class="grid cols-2">
      <div class="panel"><h3>Current parameters</h3>${rows}</div>
      <div class="panel"><h3>Historical trend</h3><div class="chartbox small"><canvas id="wqTrendChart"></canvas></div></div>
    </div>
    <div class="btnrow"><button class="btn" onclick="navigate('risk')">Continue to Risk &amp; Alerts</button></div>
  `;
};

POST_RENDER.waterquality = () => {
  const p = activeProject(); if(!p) return;
  const sel = document.getElementById("wqWellSelect");
  if(sel) sel.onchange = e=>{ STATE.ui.predictWell = e.target.value; render(); };
  const canvas = document.getElementById("wqTrendChart"); if(!canvas) return;
  const well = STATE.ui.predictWell || Object.keys(STATE.wqi[p.id].byWell)[0];
  const rows = STATE.features[p.id].filter(r=>r.wellId===well).sort((a,b)=>a.date-b.date);
  const labels = rows.map(r=>r.date.toLocaleDateString());
  const chart = new Chart(canvas, {
    type:"line",
    data:{labels, datasets:[
      {label:"TDS (mg/L)", data:rows.map(r=>r.tds), borderColor:"#175C5A", yAxisID:"y", pointRadius:0, borderWidth:2},
      {label:"Nitrate (mg/L)", data:rows.map(r=>r.nitrate), borderColor:"#C68A2E", yAxisID:"y1", pointRadius:0, borderWidth:2},
    ]},
    options:{responsive:true, maintainAspectRatio:false, interaction:{mode:"index",intersect:false},
      scales:{y:{position:"left",title:{display:true,text:"TDS"}}, y1:{position:"right",title:{display:true,text:"Nitrate"},grid:{drawOnChartArea:false}}, x:{ticks:{maxTicksLimit:8}}},
      plugins:{legend:{position:"bottom"}}}
  });
  activeCharts.push(chart);
};

/* -------------------------------- RISK & ALERTS ---------------------------- */
VIEW_RENDERERS.risk = () => {
  const gate = requireProjectGate("Risk & Alerts"); if(gate) return gate;
  const p = activeProject();
  const dataGate = requireDataGate(); if(dataGate) return dataGate;
  const riskByWell = STATE.risk[p.id].byWell;
  const alerts = STATE.alerts[p.id];
  const recs = STATE.recommendations[p.id];

  const wellRows = Object.entries(riskByWell).map(([w,r])=>`
    <tr><td>${w}</td><td><span class="status-pill status-${r.level.toLowerCase()}">${r.level}</span></td>
    <td class="mono">${r.predictedChangePct!=null?(r.predictedChangePct>0?'+':'')+fmt(r.predictedChangePct,1)+'%':'—'}</td>
    <td style="font-size:12.5px;color:var(--ink-soft);">${r.reasons[0]||'No elevated risk indicators.'}</td></tr>`).join("");

  return `
    <div class="pagehead"><div><h2>Risk &amp; Early Warning</h2><div class="ctx">Project: ${p.name}</div></div></div>
    <div class="panel">
      <h3>Well-level risk assessment</h3>
      <div class="tablewrap"><table><thead><tr><th>Well</th><th>Risk</th><th>Predicted change</th><th>Primary factor</th></tr></thead><tbody>${wellRows}</tbody></table></div>
      <div class="note">Risk levels (SAFE / WARNING / CRITICAL) are produced by this prototype's rule-based heuristic combining forecast trend, recent historical trend, rainfall, and water-quality classification. This is a decision-support heuristic, not a formal government or hydrological classification.</div>
    </div>
    <div class="grid cols-2">
      <div class="panel">
        <h3>Early-warning alerts</h3>
        ${alerts.length ? alerts.map(a=>`<div class="alertcard ${a.severity}"><div class="sev">${a.severity==='CRITICAL'?'🔴':'⚠'} ${a.severity}</div><h4>${a.title} — ${a.wellId}</h4><p>${a.body}</p></div>`).join("")
          : `<div class="note" style="text-align:center;padding:20px;">No active alerts. All monitored wells are within expected conditions.</div>`}
      </div>
      <div class="panel">
        <h3>Recommendations</h3>
        ${recs.map(r=>`<div class="reco">${r}</div>`).join("")}
      </div>
    </div>
    <div class="btnrow"><button class="btn" onclick="navigate('map')">Continue to Map</button></div>
  `;
};

/* ----------------------------------- MAP ----------------------------------- */
VIEW_RENDERERS.map = () => {
  const gate = requireProjectGate("Map Monitoring"); if(gate) return gate;
  const p = activeProject();
  const dataGate = requireDataGate(); if(dataGate) return dataGate;
  return `
    <div class="pagehead"><div><h2>Map Monitoring</h2><div class="ctx">Project: ${p.name}</div></div></div>
    <div class="panel">
      <div id="mapview"></div>
      <div class="note" style="margin-top:10px;">🟢 Safe · 🟡 Warning · 🔴 Critical — coordinates and status are taken from the uploaded/demo dataset and the risk engine, not hard-coded.</div>
    </div>
    <div class="panel" id="wellDetailPanel" style="display:none;"></div>
  `;
};

POST_RENDER.map = () => {
  const p = activeProject(); if(!p || !STATE.datasets[p.id]) return;
  if(leafletMap){ leafletMap.remove(); leafletMap=null; }
  const featureRows = STATE.features[p.id];
  const riskByWell = STATE.risk[p.id].byWell;
  const wells = [...new Set(featureRows.map(r=>r.wellId))];
  const points = wells.map(w=>{
    const rows = featureRows.filter(r=>r.wellId===w);
    const last = rows[rows.length-1];
    return {wellId:w, lat:last.lat, lon:last.lon, level:last.target, risk:riskByWell[w]};
  }).filter(pt=>pt.lat!=null && pt.lon!=null);

  const mapEl = document.getElementById("mapview");
  if(!mapEl) return;
  leafletMap = L.map(mapEl);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {attribution:'© OpenStreetMap contributors', maxZoom:18}).addTo(leafletMap);

  if(points.length){
    const bounds = L.latLngBounds(points.map(pt=>[pt.lat,pt.lon]));
    leafletMap.fitBounds(bounds.pad(0.3));
  } else {
    leafletMap.setView([20,78], 5);
  }
  const colorFor = level => level==="CRITICAL"?"#B4432F":level==="WARNING"?"#C68A2E":"#2E7D5B";
  points.forEach(pt=>{
    const marker = L.circleMarker([pt.lat,pt.lon], {radius:9, color:"#fff", weight:2, fillColor:colorFor(pt.risk.level), fillOpacity:0.95}).addTo(leafletMap);
    const wqi = STATE.wqi[p.id].byWell[pt.wellId];
    marker.bindPopup(`<b>${pt.wellId}</b><br>Level: ${fmt(pt.level,1)} m<br>WQI: ${wqi.wqi!=null?fmt(wqi.wqi,0):'—'} (${wqi.classification})<br>Risk: ${pt.risk.level}`);
    marker.on("click", ()=>showWellDetail(pt.wellId));
  });
  setTimeout(()=>leafletMap.invalidateSize(), 50);
};

function showWellDetail(wellId){
  const p = activeProject();
  const panel = document.getElementById("wellDetailPanel");
  const rows = STATE.features[p.id].filter(r=>r.wellId===wellId).sort((a,b)=>a.date-b.date);
  const last = rows[rows.length-1];
  const risk = STATE.risk[p.id].byWell[wellId];
  const wqi = STATE.wqi[p.id].byWell[wellId];
  const fc = STATE.predictions[p.id].byWell[wellId];
  const predLevel = fc.forecast.length ? fc.forecast[fc.forecast.length-1].level : null;
  const trendLabel = risk.histTrend>0.02?"Increasing":risk.histTrend<-0.02?"Declining":"Stable";
  panel.style.display = "block";
  panel.innerHTML = `
    <div class="panel-title-row"><h3>Well ${wellId}</h3><span class="status-pill status-${risk.level.toLowerCase()}">${risk.level}</span></div>
    <div class="grid cols-4">
      <div class="kpi"><div class="label">Location</div><div class="value" style="font-size:14px;">${fmt(last.lat,3)}, ${fmt(last.lon,3)}</div></div>
      <div class="kpi"><div class="label">Current level</div><div class="value">${fmt(last.target,1)} m</div></div>
      <div class="kpi"><div class="label">Predicted level</div><div class="value">${fmt(predLevel,1)} m</div></div>
      <div class="kpi"><div class="label">Trend</div><div class="value" style="font-size:18px;">${trendLabel}</div></div>
    </div>
    <div class="note">WQI ${wqi.wqi!=null?fmt(wqi.wqi,0):'—'} (${wqi.classification}). ${risk.reasons.length?risk.reasons.join(" "):"No elevated risk indicators for this well."}</div>
  `;
  panel.scrollIntoView({behavior:"smooth", block:"nearest"});
}

/* --------------------------------- DASHBOARD -------------------------------- */
VIEW_RENDERERS.dashboard = () => {
  const gate = requireProjectGate("the Dashboard"); if(gate) return gate;
  const p = activeProject();
  const dataGate = requireDataGate(); if(dataGate) return dataGate;
  const ds = STATE.datasets[p.id];
  const featureRows = STATE.features[p.id];
  const wells = [...new Set(featureRows.map(r=>r.wellId))];
  const levels = wells.map(w=>{ const rows=featureRows.filter(r=>r.wellId===w); return rows[rows.length-1].target; });
  const preds = wells.map(w=>{ const fc=STATE.predictions[p.id].byWell[w]; return fc.forecast.length?fc.forecast[fc.forecast.length-1].level:null; }).filter(v=>v!=null);
  const curAvg = mean(levels), predAvg = preds.length?mean(preds):null;
  const change = (curAvg!=null && predAvg!=null) ? predAvg-curAvg : null;
  const wqiAvg = overallWQIForProject(p.id);
  const risk = overallRiskForProject(p.id);

  return `
    <div class="pagehead">
      <div><h2>AI Groundwater Intelligence</h2><div class="ctx">Project: ${p.name} · ${p.location} · Last updated ${new Date(ds.uploadedAt).toLocaleString()} ${ds.isDemo?'<span class="demotag">Demo data</span>':''}</div></div>
      <button class="btn secondary" onclick="navigate('reports')">Generate report</button>
    </div>
    <div class="grid cols-4">
      <div class="kpi"><div class="label">Current level (avg)</div><div class="value">${fmt(curAvg,1)} m</div></div>
      <div class="kpi"><div class="label">Predicted level (avg)</div><div class="value">${fmt(predAvg,1)} m</div></div>
      <div class="kpi"><div class="label">Level change</div><div class="value delta ${change==null?'':(change<0?'up':'down')}" style="font-size:22px;">${change==null?'—':(change>0?'+':'')+fmt(change,1)+' m'}</div></div>
      <div class="kpi"><div class="label">Risk</div><div class="value" style="font-size:22px;"><span class="status-pill status-${risk.toLowerCase()}">${risk}</span></div></div>
    </div>
    <div class="grid cols-2">
      <div class="panel"><h3>Groundwater trend (avg across wells)</h3><div class="chartbox"><canvas id="dashTrendChart"></canvas></div></div>
      <div class="panel"><h3>Rainfall vs groundwater level</h3><div class="chartbox"><canvas id="dashRainChart"></canvas></div></div>
    </div>
    <div class="grid cols-2">
      <div class="panel"><h3>WQI trend</h3><div class="chartbox small"><canvas id="dashWqiChart"></canvas></div></div>
      <div class="panel"><h3>Risk by well</h3>
        <table><thead><tr><th>Well</th><th>Risk</th><th>WQI</th></tr></thead><tbody>
        ${wells.map(w=>{
          const r = STATE.risk[p.id].byWell[w]; const wq = STATE.wqi[p.id].byWell[w];
          return `<tr><td>${w}</td><td><span class="status-pill status-${r.level.toLowerCase()}">${r.level}</span></td><td class="mono">${wq.wqi!=null?fmt(wq.wqi,0):'—'}</td></tr>`;
        }).join("")}
        </tbody></table>
      </div>
    </div>
  `;
};

POST_RENDER.dashboard = () => {
  const p = activeProject(); if(!p || !STATE.features[p.id]) return;
  const featureRows = STATE.features[p.id];
  const wells = [...new Set(featureRows.map(r=>r.wellId))];
  // aggregate avg level & rainfall & wqi across wells by date bucket (weekly)
  const byDate = {};
  featureRows.forEach(r=>{
    const key = r.date.toISOString().slice(0,10);
    byDate[key] = byDate[key] || {levels:[], rain:[], tds:[], nitrate:[]};
    byDate[key].levels.push(r.target); byDate[key].rain.push(r.rainfall);
    byDate[key].tds.push(r.tds); byDate[key].nitrate.push(r.nitrate);
  });
  const dateKeys = Object.keys(byDate).sort();
  const sampleEvery = Math.max(1, Math.floor(dateKeys.length/60));
  const sampledKeys = dateKeys.filter((_,i)=>i%sampleEvery===0);
  const labels = sampledKeys.map(k=>new Date(k).toLocaleDateString());
  const avgLevel = sampledKeys.map(k=>mean(byDate[k].levels));
  const avgRain = sampledKeys.map(k=>mean(byDate[k].rain));
  const avgTds = sampledKeys.map(k=>mean(byDate[k].tds));

  const c1 = new Chart(document.getElementById("dashTrendChart"), {
    type:"line", data:{labels, datasets:[{label:"Avg groundwater level (m)", data:avgLevel, borderColor:"#175C5A", pointRadius:0, borderWidth:2, tension:.25}]},
    options:{responsive:true,maintainAspectRatio:false, scales:{x:{ticks:{maxTicksLimit:8}}}, plugins:{legend:{display:false}}}
  }); activeCharts.push(c1);

  const c2 = new Chart(document.getElementById("dashRainChart"), {
    type:"bar", data:{labels, datasets:[
      {label:"Rainfall (mm)", data:avgRain, backgroundColor:"rgba(46,158,150,.5)", yAxisID:"y"},
    ]},
    options:{responsive:true,maintainAspectRatio:false, scales:{x:{ticks:{maxTicksLimit:8}}, y:{title:{display:true,text:"mm"}}}, plugins:{legend:{position:"bottom"}}}
  }); activeCharts.push(c2);

  const c3 = new Chart(document.getElementById("dashWqiChart"), {
    type:"line", data:{labels, datasets:[{label:"Avg TDS (proxy for WQI trend)", data:avgTds, borderColor:"#C68A2E", pointRadius:0, borderWidth:2, tension:.25}]},
    options:{responsive:true,maintainAspectRatio:false, scales:{x:{ticks:{maxTicksLimit:8}}}, plugins:{legend:{display:false}}}
  }); activeCharts.push(c3);
};

/* --------------------------------- REPORTS ---------------------------------- */
VIEW_RENDERERS.reports = () => {
  const gate = requireProjectGate("Reports"); if(gate) return gate;
  const p = activeProject();
  const dataGate = requireDataGate(); if(dataGate) return dataGate;
  const val = STATE.validation[p.id];
  const training = STATE.models[p.id];
  const risk = STATE.risk[p.id].byWell;
  const wqi = STATE.wqi[p.id].byWell;
  const alerts = STATE.alerts[p.id];
  const recs = STATE.recommendations[p.id];
  const wells = Object.keys(risk);

  return `
    <div class="pagehead"><div><h2>Report</h2><div class="ctx">Project: ${p.name}</div></div>
      <div class="btnrow"><button class="btn" onclick="downloadReportPdf()">Download PDF</button></div>
    </div>
    <div class="panel" id="reportContent">
      <h3>Project information</h3>
      <table><tbody>
        <tr><td>Project</td><td>${p.name}</td></tr>
        <tr><td>Location</td><td>${p.location}${p.state?', '+p.state:''}${p.country?', '+p.country:''}</td></tr>
        <tr><td>Monitoring wells (configured)</td><td>${p.wells}</td></tr>
        <tr><td>Dataset</td><td>${STATE.datasets[p.id].fileName} ${STATE.datasets[p.id].isDemo?'(synthetic demo data)':''}</td></tr>
        <tr><td>Records</td><td>${val.total.toLocaleString()}</td></tr>
      </tbody></table>

      <h3 style="margin-top:22px;">Validation summary</h3>
      <table><tbody>
        <tr><td>Valid</td><td>${val.validCount}</td></tr>
        <tr><td>Requiring review</td><td>${val.reviewCount}</td></tr>
        <tr><td>Invalid</td><td>${val.invalidCount}</td></tr>
        <tr><td>Duplicates</td><td>${val.duplicateCount}</td></tr>
        <tr><td>Outliers flagged</td><td>${val.outlierCount}</td></tr>
      </tbody></table>

      <h3 style="margin-top:22px;">Groundwater prediction</h3>
      ${training.insufficient ? `<p>Insufficient historical data for prediction.</p>` : `
      <table><thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead><tbody>
        ${training.candidates.map(c=>`<tr><td>${c.name}${training.chosen===c?' (selected)':''}</td><td>${fmt(c.metrics.mae)}</td><td>${fmt(c.metrics.rmse)}</td><td>${c.metrics.r2!=null?fmt(c.metrics.r2,3):'—'}</td></tr>`).join("")}
      </tbody></table>`}

      <h3 style="margin-top:22px;">Water quality &amp; risk by well</h3>
      <table><thead><tr><th>Well</th><th>WQI</th><th>Classification</th><th>Risk</th></tr></thead><tbody>
        ${wells.map(w=>`<tr><td>${w}</td><td>${wqi[w].wqi!=null?fmt(wqi[w].wqi,0):'—'}</td><td>${wqi[w].classification}</td><td>${risk[w].level}</td></tr>`).join("")}
      </tbody></table>

      <h3 style="margin-top:22px;">Early warnings</h3>
      ${alerts.length ? `<ul>${alerts.map(a=>`<li>[${a.severity}] ${a.title} — ${a.wellId}: ${a.body}</li>`).join("")}</ul>` : `<p>No active alerts.</p>`}

      <h3 style="margin-top:22px;">Recommendations</h3>
      <ul>${recs.map(r=>`<li>${r}</li>`).join("")}</ul>

      <p style="margin-top:20px;font-size:12px;color:var(--ink-soft);">Generated by the Aquifer Sense prototype on ${new Date().toLocaleString()}. This report is a decision-support summary and is not a substitute for professional hydrological assessment or laboratory verification.</p>
    </div>
  `;
};

function downloadReportPdf(){
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({unit:"pt", format:"a4"});
  const content = document.getElementById("reportContent");
  const p = activeProject();
  doc.html(content, {
    callback: function(doc){ doc.save(`${p.name.replace(/\s+/g,'_')}_report.pdf`); },
    x:24, y:24, width:540, windowWidth:800
  });
}

/* ================================== INIT =================================== */
function handleGlobalDemoQuickstart(){
  if(!STATE.projects.length){
    const p = createProject({name:"Demo Groundwater Monitoring", location:"Sample Region", stateName:"", country:"", description:"Auto-created demo project.", wells:8});
    loadDemoIntoProject(p.id);
  }
}
render();
