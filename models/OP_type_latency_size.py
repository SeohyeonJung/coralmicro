import json, re, sys
from pathlib import Path
from collections import defaultdict

# ===== 설정 =====
OUTDIR = Path("/home/seohyeon/EAI/tflite_prof")
MODEL_BASENAME = "emo_mobilebert_int8"          # <- 모델 파일명에 들어가는 키워드
CSV_PATH = Path("/tmp/tflite_perop_emo.csv")    # <- benchmark_model이 만든 CSV

# ===== 모델 JSON 고르기 =====
cand = sorted(OUTDIR.glob("*.json"))
if not cand: sys.exit("[에러] OUTDIR에 json이 없습니다. flatc 단계 먼저 실행하세요.")
pref = [p for p in cand if MODEL_BASENAME in p.name]
mj_path = pref[0] if pref else cand[0]

# ===== JSON 로드: op 타입/입출력 크기 계산 =====
mj = json.load(open(mj_path))
sg = mj["subgraphs"][0]
tensors, ops = sg["tensors"], sg["operators"]
opcodes = mj["operator_codes"]

DT={"FLOAT32":4,"INT32":4,"INT8":1,"UINT8":1,"INT16":2,"FLOAT16":2,"INT64":8,"BOOL":1}
def tbytes(t):
    n=1
    for d in t.get("shape",[]) or []:
        try:n*=max(1,int(d))
        except:n*=1
    return n*DT.get(t.get("type","FLOAT32"),4)

tb=[tbytes(t) for t in tensors]
def io_bytes(op):
    ib=sum(tb[i] for i in (op.get("inputs",[])  or []) if isinstance(i,int) and i>=0)
    ob=sum(tb[i] for i in (op.get("outputs",[]) or []) if isinstance(i,int) and i>=0)
    return ib,ob

def op_type(op):
    oc = opcodes[op["opcode_index"]]
    if isinstance(oc.get("builtin_code"), str):
        return oc["builtin_code"]
    if oc.get("custom_code"):  # custom op
        return oc["custom_code"]
    return "UNKNOWN"

types=[op_type(o) for o in ops]

# ===== CSV 파싱: 마지막 Run Order 블록에서 avg_ms, name 수집 =====
if not CSV_PATH.exists() or CSV_PATH.stat().st_size==0:
    sys.exit(f"[에러] CSV가 없거나 비어 있음: {CSV_PATH}")
lines = CSV_PATH.read_text().splitlines()
marker = "============================== Run Order =============================="
starts=[i for i,l in enumerate(lines) if l.strip().startswith(marker)]
if not starts: sys.exit("[에러] CSV에서 'Run Order' 표를 찾지 못했습니다.")
start = starts[-1] + 1
header = [c.strip() for c in lines[start].split(",")]
rows=[]
for i in range(start+1,len(lines)):
    s=lines[i].strip()
    if not s or s.startswith("="): break
    rows.append([c.strip() for c in s.split(",")])

def fcol(h,keys):
    hl=[x.lower() for x in h]
    for k in keys:
        for i,c in enumerate(hl):
            if k in c: return i
    return None

c_avg = fcol(header, ["avg_ms","avg ms"])
c_name= fcol(header, ["name"])
if None in (c_avg, c_name):
    sys.exit(f"[에러] CSV 헤더 해석 실패: {header}")

def parse_node_index(name:str):
    m=re.search(r"#(\d+)",name)
    if m: return int(m.group(1))
    m=re.search(r"/(\d+)$",name)
    if m: return int(m.group(1))
    return None

# perf[i] = (name, avg_us); 인덱스 없으면 순서로 매칭
perf={}
seq=[]
for r in rows:
    try: avg_ms=float(r[c_avg])
    except: continue
    name=r[c_name]
    idx=parse_node_index(name)
    if idx is not None: perf[idx]=(name, avg_ms*1000.0)
    else: seq.append((name, avg_ms*1000.0))
if not perf:
    n=min(len(seq), len(ops))
    for i in range(n): perf[i]=seq[i]

# ===== 전체 op를 index 0..N-1 순서대로 출력 =====
hdr=f'{"idx":>6}  {"type":<18}  {"name":<60}  {"avg_us":>12}  {"in_KB":>10}  {"out_KB":>10}'
print(hdr); print("-"*len(hdr))
total=0.0
for i in range(len(ops)):
    t = types[i]
    name,avg_us = perf.get(i, (f"(no-csv-entry)", 0.0))
    ib,ob = io_bytes(ops[i])
    total += avg_us
    print(f'{i:6d}  {t:<18}  {name[:60]:<60}  {avg_us:12.3f}  {ib/1024:10.2f}  {ob/1024:10.2f}')
print("-"*len(hdr))
print(f"Sum of avg_us over all ops: {total:.3f}")

# ===== op type별 집계 =====
agg = defaultdict(lambda: {"count":0,"sum":0.0})
for i in range(len(ops)):
    t = types[i]
    _, avg_us = perf.get(i, ("",0.0))
    agg[t]["count"] += 1
    agg[t]["sum"]   += avg_us

hdr2 = f'{"op_type":<20}  {"count":>6}  {"total_us":>12}  {"avg_us":>12}'
print("\n" + hdr2)
print("-"*len(hdr2))
for t,(c,s) in sorted(((k,v["count"],v["sum"]) for k,v in agg.items()), key=lambda x:-x[2]):
    avg = s/c if c>0 else 0.0
    print(f'{t:<20}  {c:6d}  {s:12.3f}  {avg:12.3f}')
print("-"*len(hdr2))

