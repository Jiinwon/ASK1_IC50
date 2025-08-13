#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selenium + Headless Chrome로 PubChem→InChIKey→ChEMBL(IC50 for ASK1=CHEMBL5285) 수집

- 입력 CSV(고정): /home1/won0316/DACON/JUMP_AI_2025_EST/data/test.csv (열: ID, Smiles)
- 출력 CSV(고정): /home1/won0316/DACON/JUMP_AI_2025_EST/step2_advanced_jinwon/test_warn/results/ask1_ic50_human_selenium.csv
- 스냅샷/로그:   /home1/won0316/DACON/JUMP_AI_2025_EST/step2_advanced_jinwon/test_warn/selenium_logs
"""
import os, re, csv, json, time, math, pathlib, traceback, random
import pandas as pd
from urllib.parse import quote

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# ---------- 경로 고정 ----------
INPUT_CSV  = "/home1/won0316/DACON/JUMP_AI_2025_EST/data/test.csv"
OUTPUT_CSV = "/home1/won0316/DACON/JUMP_AI_2025_EST/step2_advanced_jinwon/test_warn/results/ask1_ic50_human_selenium.csv"
LOG_DIR    = "/home1/won0316/DACON/JUMP_AI_2025_EST/step2_advanced_jinwon/test_warn/selenium_logs"
pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# ---------- 대상/필터 ----------
ASK1_TGT    = "CHEMBL5285"  # Human ASK1
ASSAY_TYPES = {"B","F"}

# ---------- 브라우저 ----------
def make_driver():
    opts = Options()
    # 최신 headless 권장 플래그 + HPC 친화 옵션
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    # 필요 시 프록시(환경변수 HTTPS_PROXY/HTTP_PROXY 인식)
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    if proxy:
        opts.add_argument(f"--proxy-server={proxy}")
    # UA 살짝 지정(일부 CDN 차단 회피)
    opts.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36")
    return webdriver.Chrome(options=opts)  # Selenium Manager가 드라이버 자동관리

# ---------- 공통: JSON fetch with retry ----------
def get_json_via_browser(drv: webdriver.Chrome, url: str, max_retry=5, base_sleep=0.8):
    last_err = None
    for i in range(max_retry):
        try:
            drv.get(url)
            time.sleep(base_sleep + random.random()*0.5)
            # 대부분의 REST 응답은 body 또는 pre에 그대로 JSON 텍스트
            text = ""
            try:
                text = drv.find_element(By.TAG_NAME, "pre").text.strip()
            except Exception:
                pass
            if not text:
                text = drv.find_element(By.TAG_NAME, "body").text.strip()
            if not text:
                raise RuntimeError("Empty body")
            return json.loads(text)
        except Exception as e:
            last_err = e
            time.sleep(base_sleep*(2**i) + random.random()*0.5)
    # 디버그 스냅샷
    ts = int(time.time()*1000)
    drv.save_screenshot(f"{LOG_DIR}/json_err_{ts}.png")
    with open(f"{LOG_DIR}/json_err_{ts}.html","w",encoding="utf-8") as f:
        f.write(drv.page_source)
    raise last_err

# ---------- PubChem: SMILES -> InChIKey ----------
def pubchem_inchikey_from_smiles(drv, smiles: str):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        f"{quote(smiles, safe='')}/property/InChIKey/JSON"
    )
    js = get_json_via_browser(drv, url)
    props = (js or {}).get("PropertyTable", {}).get("Properties", []) or []
    if props:
        return props[0].get("InChIKey")
    return None

# ---------- ChEMBL: InChIKey -> CHEMBL 분자ID 목록 ----------
def chembl_ids_from_inchikey(drv, inchikey: str):
    url = (
        "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
        f"?molecule_structures__standard_inchi_key={quote(inchikey, safe='')}"
        "&limit=200"
    )
    js = get_json_via_browser(drv, url)
    mols = (js or {}).get("molecules", []) or []
    out = []
    for m in mols:
        mid = m.get("molecule_chembl_id")
        if mid:
            out.append(mid)
    return list(dict.fromkeys(out))  # unique

# ---------- 단위 변환/지표 ----------
def to_nM(val, unit):
    if val is None or unit is None:
        return None
    try:
        v = float(val); u = str(unit).strip().lower()
    except Exception:
        return None
    if u == "nm": return v
    if u in ("μm","um"): return v*1000.0
    if u == "mm": return v*1e6
    if u == "pm": return v*1e-3
    if u == "m":  return v*1e9
    return None

# ---------- ChEMBL: 분자ID -> ASK1(IC50) ----------
def activities_for_molecule(drv, molecule_id: str):
    url = (
        "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        f"?molecule_chembl_id={molecule_id}"
        f"&target_chembl_id={ASK1_TGT}"
        f"&standard_type=IC50"
        "&limit=1000"
    )
    js = get_json_via_browser(drv, url)
    acts = (js or {}).get("activities", []) or []
    rows = []
    for a in acts:
        if a.get("assay_type") not in ASSAY_TYPES:
            continue
        val  = a.get("standard_value")
        unit = a.get("standard_units")
        vnm  = to_nM(val, unit)
        p    = a.get("pchembl_value")
        if p is None and vnm and vnm>0 and (a.get("standard_relation") in (None,"","=")):
            p = -(math.log10(vnm*1e-9))
        rows.append({
            "molecule_chembl_id": molecule_id,
            "activity_id": a.get("activity_id"),
            "assay_chembl_id": a.get("assay_chembl_id"),
            "document_chembl_id": a.get("document_chembl_id"),
            "assay_type": a.get("assay_type"),
            "assay_desc": a.get("assay_description"),
            "standard_type": a.get("standard_type"),
            "relation": a.get("standard_relation"),
            "ic50_value_raw": val,
            "ic50_unit_raw": unit,
            "ic50_value_nM": vnm,
            "pIC50": p,
            "target_organism": a.get("target_organism"),
            "assay_tax_id": a.get("assay_tax_id"),
            "cell_id": a.get("cell_id"),
            "cell_name": a.get("cell_name"),
            "data_validity_comment": a.get("data_validity_comment"),
            "src": "ChEMBL",
            "source_url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{molecule_id}/",
        })
    return rows

# ---------- 메인 ----------
def main():
    drv = make_driver()
    all_rows, unmatched = [], []
    df = pd.read_csv(INPUT_CSV)
    assert {"ID","Smiles"}.issubset(df.columns), "INPUT_CSV는 ID, Smiles 열이 필요합니다."

    try:
        for _, r in df.iterrows():
            sid = r["ID"]; smi = str(r["Smiles"]).strip()
            if not smi or smi.lower()=="nan":
                unmatched.append({"ID":sid, "Smiles":smi, "reason":"EMPTY_SMILES"})
                continue

            # PubChem: SMILES -> InChIKey
            try:
                ikey = pubchem_inchikey_from_smiles(drv, smi)
            except Exception:
                ikey = None
            if not ikey:
                unmatched.append({"ID":sid, "Smiles":smi, "reason":"NO_PUBCHEM_INCHIKEY"})
                continue

            # ChEMBL: InChIKey -> CHEMBL IDs
            try:
                mol_ids = chembl_ids_from_inchikey(drv, ikey)
            except Exception:
                mol_ids = []
            if not mol_ids:
                unmatched.append({"ID":sid, "Smiles":smi, "InChIKey":ikey, "reason":"NO_MATCH_IN_CHEMBL"})
                continue

            # 각 CHEMBL ID에 대해 ASK1(IC50)
            got = False
            for mid in mol_ids:
                rows = []
                try:
                    rows = activities_for_molecule(drv, mid)
                except Exception:
                    rows = []
                time.sleep(0.2 + random.random()*0.2)
                if rows:
                    got = True
                    for row in rows:
                        row.update({"ID":sid, "Smiles":smi, "InChIKey":ikey})
                        all_rows.append(row)

            if not got:
                # 분자는 매칭됐으나 ASK1 IC50이 없을 때
                all_rows.append({
                    "ID": sid, "Smiles": smi, "InChIKey": ikey,
                    "molecule_chembl_id": ",".join(mol_ids),
                    "activity_id": None, "assay_chembl_id": None, "document_chembl_id": None,
                    "assay_type": None, "assay_desc": "NO_IC50_FOR_ASK1_FOUND_FOR_THIS_MOLECULE",
                    "standard_type": "IC50", "relation": None,
                    "ic50_value_raw": None, "ic50_unit_raw": None, "ic50_value_nM": None, "pIC50": None,
                    "target_organism": None, "assay_tax_id": None, "cell_id": None, "cell_name": None,
                    "data_validity_comment": None, "src":"ChEMBL",
                    "source_url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{mol_ids[0]}/" if mol_ids else None
                })

    finally:
        drv.quit()

    # 저장
    cols = ["ID","Smiles","InChIKey","molecule_chembl_id","activity_id","assay_chembl_id","document_chembl_id",
            "assay_type","assay_desc","standard_type","relation","ic50_value_raw","ic50_unit_raw",
            "ic50_value_nM","pIC50","target_organism","assay_tax_id","cell_id","cell_name",
            "data_validity_comment","src","source_url"]
    pd.DataFrame(all_rows).reindex(columns=cols).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    if unmatched:
        pd.DataFrame(unmatched).to_csv(
            os.path.join(LOG_DIR, "unmatched.csv"), index=False, encoding="utf-8"
        )

if __name__ == "__main__":
    main()
