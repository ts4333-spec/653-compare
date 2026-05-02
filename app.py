import re
import time
import unicodedata
import streamlit as st
import requests
from openai import OpenAI

# ─────────────────────────────────────────────
# 모델별 1M 토큰당 예상 비용 (달러) - 최신 정책에 따라 변동 가능
# ─────────────────────────────────────────────
PRICING = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o": {"input": 5.00, "output": 15.00}
}

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="KORMARC 653 모델 비교기",
    page_icon="📚",
    layout="wide", # 양방향 비교를 위해 넓은 화면 사용
)

# ─────────────────────────────────────────────
# 사이드바: API 키 입력
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API 키 설정")
    ttb_key    = st.text_input("Aladin TTB Key",  type="password", placeholder="ttbxxxxxxxxxxxxxxxx")
    openai_key = st.text_input("OpenAI API Key",  type="password", placeholder="sk-...")
    st.markdown("---")
    st.caption("알라딘 Open API: https://www.aladin.co.kr/ttb/wblog_guide.aspx")
    st.caption("OpenAI API: https://platform.openai.com/api-keys")

# ─────────────────────────────────────────────
# Session State 초기화
# ─────────────────────────────────────────────
_DEFAULTS: dict = {
    "meta_loaded":  False,
    "title":        "",
    "author":       "",
    "categoryName": "",
    "description":  "",
    "toc":          "",
    "compare_results": None,
    "last_isbn":    "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────
# 전처리 함수
# ─────────────────────────────────────────────
def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text.lower().strip()

def _clean_author_str(s: str) -> str:
    s = re.sub(r"[\(\（\[\【].*?[\)\）\]\】]", "", s)
    s = re.sub(r"(지음|글|그림|옮김|편저|엮음|저|역|편)", "", s)
    s = re.sub(r"[,·\|/\\]", " ", s)
    return s.strip()

def _build_forbidden_set(title: str, authors: list[str]) -> set[str]:
    forbidden: set[str] = set()
    for token in _norm(title).split():
        if len(token) >= 2:
            forbidden.add(token)
            for size in range(2, min(len(token) + 1, 5)):
                for start in range(len(token) - size + 1):
                    forbidden.add(token[start:start + size])
    for author in authors:
        for token in _norm(_clean_author_str(author)).split():
            if len(token) >= 2:
                forbidden.add(token)
    return forbidden

def _should_keep_keyword(kw: str, forbidden: set[str]) -> bool:
    normed = _norm(kw)
    if not normed: return False
    if normed in forbidden: return False
    for fb in forbidden:
        if fb and fb in normed: return False
    return True

# ─────────────────────────────────────────────
# 알라딘 API 호출
# ─────────────────────────────────────────────
def fetch_aladin_metadata(ttb_key: str, isbn: str) -> dict:
    url = "https://www.aladin.co.kr/ttb/api/ItemLookUp.aspx"
    params = {
        "TTBKey": ttb_key, "itemIdType": "ISBN13", "ItemId": isbn.strip(),
        "output": "js", "Version": "20131101", "OptResult": "description,toc",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data  = resp.json()
    items = data.get("item", [])
    if not items: raise ValueError("해당 ISBN의 도서를 알라딘에서 찾을 수 없습니다.")
    item = items[0]
    description = (item.get("description", "").strip() or item.get("fullDescription", "").strip())
    return {
        "title": item.get("title", "").strip(),
        "author": item.get("author", "").strip(),
        "categoryName": item.get("categoryName", "").strip(),
        "description": description,
        "toc": item.get("toc", "").strip(),
    }

# ─────────────────────────────────────────────
# GPT 생성 및 지표 측정 함수
# ─────────────────────────────────────────────
def generate_653_eval(openai_key: str, meta: dict, forbidden: set[str], model_name: str) -> dict:
    client = OpenAI(api_key=openai_key)
    system_prompt = """당신은 도서관 목록 전문가로, KORMARC 653 비통제 주제어 필드를 생성합니다.
[규칙]
1. 주제어는 반드시 띄어쓰기 없는 복합명사 형태로 작성하세요. (예: 감정조절, 아동문학)
2. 서명(제목)이나 저자명에서 그대로 따온 단어는 절대 사용하지 마세요.
3. 추상적·메타적 표현(연구, 의의, 현황, 소개, 개요, 분석, 방법, 이론) 사용 금지.
4. 최대 7개 이내, $a키워드 형식으로만 출력하세요.
5. 중복 키워드 없이 출력하세요.
6. 다른 설명이나 부연 없이 $a로 시작하는 문자열만 반환하세요."""

    user_content = f"""제목: {meta['title']}
저자: {meta['author']}
카테고리: {meta['categoryName']}
책 소개: {meta['description'][:800] if meta['description'] else '(없음)'}
목차: {meta['toc'][:600] if meta['toc'] else '(없음)'}"""

    start_time = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    elapsed_time = time.time() - start_time
    
    # 비용 계산
    usage = response.usage
    cost_input = (usage.prompt_tokens / 1_000_000) * PRICING[model_name]["input"]
    cost_output = (usage.completion_tokens / 1_000_000) * PRICING[model_name]["output"]
    total_cost = cost_input + cost_output

    raw = response.choices[0].message.content.strip()
    parts = [p.strip() for p in raw.split("$a") if p.strip()]
    filtered = [kw for kw in parts if _should_keep_keyword(kw, forbidden)]

    seen: set[str] = set()
    deduped: list[str] = []
    for kw in filtered:
        key = _norm(kw)
        if key not in seen:
            seen.add(key)
            deduped.append(kw)
            
    final_kws = deduped[:7]
    field_653 = f"=653  \\\\{''.join(f'$a{kw}' for kw in final_kws)}"

    return {
        "model": model_name,
        "field_653": field_653,
        "kw_list": final_kws,
        "time": elapsed_time,
        "cost": total_cost,
        "tokens": usage.total_tokens
    }

# ─────────────────────────────────────────────
# 메인 UI
# ─────────────────────────────────────────────
st.title("📚 KORMARC 653 AI 모델 비교기 (gpt-4o-mini vs gpt-4o)")
st.caption("동일한 서지 데이터로 두 모델의 색인어 품질, 소요 시간, API 비용을 비교합니다.")
st.markdown("---")

st.markdown("#### 1단계 · 도서 정보 가져오기")
isbn_col, fetch_col = st.columns([3, 1])
with isbn_col:
    isbn = st.text_input("ISBN", placeholder="13자리 숫자 입력", max_chars=13, label_visibility="collapsed")
with fetch_col:
    fetch_btn = st.button("📥 정보 가져오기", use_container_width=True)

if fetch_btn:
    if not ttb_key: st.warning("사이드바에서 Aladin TTB Key를 입력하세요.")
    elif not isbn or len(isbn.strip()) != 13: st.warning("올바른 ISBN 13자리를 입력하세요.")
    else:
        with st.spinner("가져오는 중..."):
            try:
                meta = fetch_aladin_metadata(ttb_key, isbn.strip())
                st.session_state.update(meta)
                st.session_state["meta_loaded"] = True
                st.session_state["compare_results"] = None
                st.success(f"✅ **{meta['title']}**")
            except Exception as e:
                st.error(f"오류: {e}")

st.markdown("---")
st.markdown("#### 2단계 · 메타데이터 확인 및 수정")
if not st.session_state["meta_loaded"]:
    st.info("ISBN을 입력해 정보를 불러오세요.")

col_l, col_r = st.columns(2)
with col_l: edit_title = st.text_input("제목", value=st.session_state["title"])
with col_r: edit_author = st.text_input("저자", value=st.session_state["author"])
edit_category = st.text_input("카테고리", value=st.session_state["categoryName"])
edit_description = st.text_area("초록 / 책 소개", value=st.session_state["description"], height=100)
edit_toc = st.text_area("목차", value=st.session_state["toc"], height=100)

st.markdown("---")
st.markdown("#### 3단계 · 모델 동시 실행 및 비교")
run_btn = st.button("⚖️ 양방향 비교 생성", type="primary", use_container_width=True)

if run_btn:
    if not openai_key: st.warning("사이드바에서 OpenAI API Key를 입력하세요.")
    elif not edit_title.strip(): st.warning("제목 정보가 필요합니다.")
    else:
        meta_to_use = {
            "title": edit_title.strip(), "author": edit_author.strip(),
            "categoryName": edit_category.strip(), "description": edit_description.strip(),
            "toc": edit_toc.strip(),
        }
        with st.spinner("두 모델을 동시 호출하여 분석 중입니다. 잠시만 기다려주세요..."):
            try:
                authors = [a.strip() for a in re.split(r"[,·|]", meta_to_use["author"]) if a.strip()]
                forbidden = _build_forbidden_set(meta_to_use["title"], authors)
                
                # 두 모델 순차적 호출 및 결과 저장
                res_mini = generate_653_eval(openai_key, meta_to_use, forbidden, "gpt-4o-mini")
                res_4o = generate_653_eval(openai_key, meta_to_use, forbidden, "gpt-4o")
                
                st.session_state["compare_results"] = {"mini": res_mini, "4o": res_4o}
            except Exception as e:
                st.error(f"비교 생성 중 오류 발생: {e}")

# ─────────────────────────────────────────────
# 결과 출력부 (화면 분할)
# ─────────────────────────────────────────────
if st.session_state["compare_results"]:
    res_mini = st.session_state["compare_results"]["mini"]
    res_4o = st.session_state["compare_results"]["4o"]
    
    st.success("✅ 비교 분석 완료!")
    
    col_m, col_o = st.columns(2)
    
    def display_result(col, data, title, color):
        with col:
            st.markdown(f"### {title}")
            # 지표 메트릭 표시
            m1, m2, m3 = st.columns(3)
            m1.metric("소요 시간", f"{data['time']:.2f} 초")
            m2.metric("예상 비용", f"${data['cost']:.5f}")
            m3.metric("사용 토큰", f"{data['tokens']} 개")
            
            st.markdown("**KORMARC 653 필드**")
            st.code(data['field_653'], language=None)
            
            st.markdown("**추출된 주제어**")
            if data['kw_list']:
                tag_html = " ".join(
                    f"<span style='background:{color};color:#ffffff;"
                    f"padding:5px 12px;border-radius:3px;font-size:0.88rem;"
                    f"margin:2px;display:inline-block;'>{kw}</span>"
                    for kw in data['kw_list']
                )
                st.markdown(tag_html, unsafe_allow_html=True)
            else:
                st.write("추출된 주제어가 없습니다.")
                
    display_result(col_m, res_mini, "⚡ GPT-4o-mini", "#2e7d32") # 초록색 계열
    display_result(col_o, res_4o, "🧠 GPT-4o", "#1565c0")      # 파란색 계열