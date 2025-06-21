import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
from openai import OpenAI
import time

# ───────────── 1. OpenAI 문자 생성 함수 ─────────────
def generate_ai_sms(
    client: OpenAI,
    target: str,
    category: str,
    content_details: str,
    date: str,
    school: str,
    additional_info: Dict[str, str],
    tone_guide: str = "",
    length_option: str = "표준",
    style_option: str = "기본"
) -> Tuple[str, bool]:
    """생성형 AI를 활용한 세계교육 표준"""
    
    # 대상별 톤 가이드
    tone_guides = {
        "학부모": "정중하고 상세하며 신뢰감을 주는 톤. 존댓말 사용. [학교명]으로 시작",
        "학생": "친근하고 이해하기 쉬운 톤. 존댓말 사용. 학생들의 눈높이에 맞춘 표현",
        "교직원": "간결하고 업무적이며 핵심만 전달하는 톤. 존댓말 사용. 담당 업무 명시"
    }
    
    # 카테고리별 포함 요소
    category_elements = {
        "안전": "안전 주의사항, 구체적인 행동 지침",
        "재난": "대응 방법, 비상 연락처, 준비물",
        "체험학습": "일시, 장소, 준비물, 주의사항",
        "행사 안내": "일시, 장소, 참여 방법, 준비사항",
        "상담": "상담 일정, 신청 방법, 준비 서류",
        "안내": "핵심 정보, 확인 사항, 문의처"
    }
    
    # 길이 옵션별 글자 수
    length_guides = {
        "매우 짧게": "40자 이내로 핵심만 간단히",
        "짧게": "60자 이내로 간결하게",
        "표준": "80자 내외로 적절하게",
        "길게": "120자 내외로 상세하게",
        "매우 길게": "180자 내외로 자세하게"
    }
    
    # 스타일 옵션별 가이드
    style_guides = {
        "기본": "표준적이고 격식 있는 문체",
        "친근함": "따뜻하고 친근한 문체, 이모티콘 포함 가능",
        "긴급함": "긴급하고 단호한 문체, 중요 내용 강조",
        "공식적": "매우 격식 있고 공식적인 문체",
        "안내형": "차분하고 설명적인 문체, 단계별 안내"
    }
    
    # 추가 정보 문자열 생성
    additional_str = ""
    for key, value in additional_info.items():
        if value:
            additional_str += f"- {key}: {value}\n"
    
    prompt = f"""학교에서 발송하는 문자 메시지를 작성해주세요.

대상: {target}
카테고리: {category}
학교명: {school}
날짜/시간: {date}
주요 내용: {content_details}

추가 정보:
{additional_str if additional_str else "없음"}

작성 지침:
1. 톤: {tone_guides.get(target, tone_guide)}
2. 필수 포함 요소: {category_elements.get(category, "핵심 정보")}
3. 길이: {length_guides.get(length_option, "80자 내외")}
4. 스타일: {style_guides.get(style_option, "표준적인 문체")}
5. 명확하고 구체적인 정보 전달
6. 불필요한 미사나 수식어 제외
7. 모든 대상에게 존댓말 사용

문자 메시지만 작성하고, 다른 설명은 포함하지 마세요."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 학교 행정 업무를 돕는 전문가입니다. 간결하고 명확한 문자 메시지를 작성합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        sms = response.choices[0].message.content.strip()
        return sms, True
        
    except Exception as e:
        return f"문자 생성 중 오류가 발생했습니다: {str(e)}", False

# ───────────── 2. 예제 템플릿 (참고용) ─────────────
EXAMPLE_TEMPLATES = {
    "학부모": {
        "안전": "[○○학교] 11월 15일 등하교 시 교통안전 지도 부탁드립니다. 횡단보도에서 좌우를 확인하도록 가정에서도 지도 부탁드립니다.",
        "체험학습": "[○○학교] 3학년 11월 20일 과학관 현장체험학습 안내입니다. 도시락, 물, 우산을 준비해 주세요. 참가 동의서는 11월 18일까지 제출 부탁드립니다."
    },
    "학생": {
        "안전": "[○○학교] 내일 등교할 때 빗길에 미끄러지지 않도록 조심하세요. 우산을 꼭 챙기고, 천천히 걸어오세요.",
        "행사 안내": "[○○학교] 11월 25일 오후 2시 운동장에서 가을 축제가 열립니다. 친구들과 함께 즐거운 시간 보내세요!"
    },
    "교직원": {
        "안전": "[○○학교] 11월 15일 우천 시 등하교 안전 지도 철저히 부탁드립니다. 담당 구역 확인 후 배치 부탁드립니다.",
        "행사 안내": "[○○학교] 11월 25일 14:00 가을축제 진행. 담당 부스 운영 교사는 13:30까지 준비 완료 부탁드립니다."
    }
}

# ───────────── 3. 일괄 생성을 위한 시나리오 ─────────────
BATCH_SCENARIOS = {
    "등하교 안전 안내": {
        "category": "안전",
        "targets": ["학부모", "학생", "교직원"],
        "base_content": "우천 시 등하교 안전 주의"
    },
    "현장체험학습 안내": {
        "category": "체험학습",
        "targets": ["학부모", "학생"],
        "base_content": "현장체험학습 실시 및 준비물 안내"
    },
    "학교 행사 안내": {
        "category": "행사 안내",
        "targets": ["학부모", "학생", "교직원"],
        "base_content": "학교 행사 개최 안내"
    },
    "상담 주간 안내": {
        "category": "상담",
        "targets": ["학부모", "교직원"],
        "base_content": "학부모 상담 주간 운영"
    }
}

# ───────────── 4. Streamlit UI ─────────────
st.set_page_config(page_title="🤖 AI 학교 문자 생성기", layout="wide")
st.title("🤖 AI 학교 문자 생성기")
st.markdown("경상북도교육청 맞춤형 학교 문자 자동 생성 시스템")

# API 키 확인
api_key = st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    st.error("⚠️ OpenAI API 키가 설정되지 않았습니다. Streamlit secrets에 OPENAI_API_KEY를 추가해주세요.")
    st.stop()

client = OpenAI(api_key=api_key)

# 사이드바 - 기본 정보
with st.sidebar:
    st.header("🏫 기본 정보 설정")
    school_name = st.text_input("학교명", value="○○초등학교")
    
    st.header("📅 날짜 설정")
    use_specific_date = st.checkbox("특정 날짜 사용")
    
    if use_specific_date:
        selected_date = st.date_input("날짜 선택", datetime.now())
        date_str = selected_date.strftime("%m월 %d일")
    else:
        date_options = {
            "오늘": datetime.now().strftime("%m월 %d일"),
            "내일": (datetime.now() + timedelta(days=1)).strftime("%m월 %d일"),
            "모레": (datetime.now() + timedelta(days=2)).strftime("%m월 %d일"),
            "이번 주 금요일": "이번 주 금요일",
            "다음 주 월요일": "다음 주 월요일"
        }
        date_str = st.selectbox("날짜 선택", list(date_options.values()))
    
    # 시간 설정 (선택사항)
    include_time = st.checkbox("시간 포함")
    if include_time:
        time_hour = st.selectbox("시", list(range(0, 24)), index=9)
        time_minute = st.selectbox("분", [0, 10, 20, 30, 40, 50])
        time_str = f" {time_hour:02d}:{time_minute:02d}"
        date_str += time_str

# 메인 영역
tab1, tab2, tab3, tab4 = st.tabs(["✨ AI 문자 생성", "🚀 시나리오별 일괄 생성", "📊 생성 이력", "❓ 도움말"])

with tab1:
    st.subheader("✨ AI 기반 스마트 문자 생성")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 기본 정보 입력
        st.markdown("### 📝 기본 정보")
        target = st.selectbox(
            "대상",
            ["학부모", "학생", "교직원"],
            help="문자를 받을 대상을 선택하세요"
        )
        
        category = st.selectbox(
            "카테고리",
            ["안전", "재난", "체험학습", "행사 안내", "상담", "안내"],
            help="문자의 주요 목적을 선택하세요"
        )
        
        # 주요 내용
        content_details = st.text_area(
            "주요 내용",
            placeholder="예: 내일 오전 강한 비 예상, 우산 준비 및 등하교 시 안전 주의 필요",
            height=100,
            help="전달하고자 하는 핵심 내용을 자유롭게 작성하세요"
        )
        
    with col2:
        # 추가 정보
        st.markdown("### 🔧 추가 정보 (선택사항)")
        
        additional_info = {}
        
        if category == "체험학습":
            additional_info["장소"] = st.text_input("장소", placeholder="예: 국립과학관", key="single_place")
            additional_info["준비물"] = st.text_input("준비물", placeholder="예: 도시락, 물, 우산", key="single_prep")
            additional_info["학년"] = st.text_input("대상 학년", placeholder="예: 3학년", key="single_grade")
            
        elif category == "행사 안내":
            additional_info["행사명"] = st.text_input("행사명", placeholder="예: 가을 축제", key="single_event")
            additional_info["장소"] = st.text_input("장소", placeholder="예: 운동장", key="single_event_place")
            additional_info["참가 대상"] = st.text_input("참가 대상", placeholder="예: 전교생", key="single_participants")
            
        elif category == "상담":
            additional_info["상담 유형"] = st.selectbox("상담 유형", ["학부모 상담", "진로 상담", "학습 상담"], key="single_consult_type")
            additional_info["신청 방법"] = st.text_input("신청 방법", placeholder="예: 담임교사에게 신청", key="single_apply")
            additional_info["기한"] = st.text_input("신청 기한", placeholder="예: 11월 20일까지", key="single_deadline")
            
        elif category == "안전":
            additional_info["위험 요소"] = st.text_input("위험 요소", placeholder="예: 빗길 미끄러움", key="single_danger")
            additional_info["주의 구역"] = st.text_input("주의 구역", placeholder="예: 정문 앞 횡단보도", key="single_caution")
            
        elif category == "재난":
            additional_info["재난 유형"] = st.selectbox("재난 유형", ["태풍", "폭우", "폭설", "지진", "화재"], key="single_disaster_type")
            additional_info["대응 방법"] = st.text_input("대응 방법", placeholder="예: 실내 대피", key="single_response")
            
        else:  # 일반 안내
            additional_info["문의처"] = st.text_input("문의처", placeholder="예: 교무실 02-123-4567", key="single_contact")
            additional_info["참고 사항"] = st.text_input("참고 사항", placeholder="예: 자세한 내용은 홈페이지 참조", key="single_reference")
        
        # 커스텀 톤 설정
        custom_tone = st.checkbox("커스텀 톤 사용")
        if custom_tone:
            tone_guide = st.text_input(
                "톤 가이드",
                placeholder="예: 긴급하고 단호한 톤으로 작성",
                key="single_tone"
            )
        else:
            tone_guide = ""
    
    # 문자 옵션 설정
    st.markdown("### ⚙️ 문자 옵션")
    option_col1, option_col2 = st.columns(2)
    
    with option_col1:
        length_option = st.selectbox(
            "문자 길이",
            ["매우 짧게", "짧게", "표준", "길게", "매우 길게"],
            index=2,  # 기본값: 표준
            help="매우 짧게(40자), 짧게(60자), 표준(80자), 길게(120자), 매우 길게(180자)"
        )
    
    with option_col2:
        style_option = st.selectbox(
            "문자 스타일",
            ["기본", "친근함", "긴급함", "공식적", "안내형"],
            help="상황에 맞는 문체를 선택하세요"
        )
    
    # 생성 버튼
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        generate_btn = st.button("🎯 AI 문자 생성", type="primary", use_container_width=True)
    
    if generate_btn:
        if content_details:
            with st.spinner("AI가 문자를 생성하고 있습니다..."):
                sms, success = generate_ai_sms(
                    client=client,
                    target=target,
                    category=category,
                    content_details=content_details,
                    date=date_str,
                    school=school_name,
                    additional_info=additional_info,
                    tone_guide=tone_guide,
                    length_option=length_option,
                    style_option=style_option
                )
                
            if success:
                st.success("✅ AI 문자가 생성되었습니다!")
                
                # 결과 표시
                result_col1, result_col2 = st.columns([2, 1])
                
                with result_col1:
                    st.text_area(
                        "생성된 문자",
                        value=sms,
                        height=150,
                        key="ai_generated_sms"
                    )
                
                with result_col2:
                    st.metric("글자 수", f"{len(sms)}자")
                    sms_type = "단문(SMS)" if len(sms) <= 80 else "장문(LMS)"
                    st.metric("문자 유형", sms_type)
                    
                    # 예상 비용 (참고용)
                    if sms_type == "단문(SMS)":
                        cost = "약 20원"
                    else:
                        cost = "약 30-50원"
                    st.metric("예상 비용", cost)
                
                # 추가 작업 옵션
                st.markdown("---")
                action_col1, action_col2, action_col3 = st.columns(3)
                
                with action_col1:
                    if st.button("🔄 다시 생성"):
                        st.rerun()
                
                with action_col2:
                    # 생성 이력 저장 (세션 스테이트)
                    if st.button("💾 이력 저장"):
                        if "sms_history" not in st.session_state:
                            st.session_state.sms_history = []
                        
                        st.session_state.sms_history.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "target": target,
                            "category": category,
                            "content": sms,
                            "length": len(sms),
                            "style": style_option,
                            "length_option": length_option
                        })
                        st.success("이력에 저장되었습니다!")
                
                with action_col3:
                    # 예제 보기
                    with st.expander("📋 예제 템플릿 보기"):
                        if target in EXAMPLE_TEMPLATES and category in EXAMPLE_TEMPLATES[target]:
                            st.text(EXAMPLE_TEMPLATES[target][category])
                        else:
                            st.text("해당 카테고리의 예제가 없습니다.")
            else:
                st.error(sms)
        else:
            st.warning("⚠️ 주요 내용을 입력해주세요.")

with tab2:
    st.subheader("🚀 시나리오별 일괄 생성")
    st.markdown("자주 사용하는 시나리오를 선택하면 대상별로 적절한 문자를 한 번에 생성합니다.")
    
    # 시나리오 선택
    scenario = st.selectbox(
        "시나리오 선택",
        list(BATCH_SCENARIOS.keys()),
        help="상황에 맞는 시나리오를 선택하세요"
    )
    
    scenario_info = BATCH_SCENARIOS[scenario]
    
    # 시나리오 정보 표시
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.info(f"📂 카테고리: {scenario_info['category']}")
    with info_col2:
        st.info(f"👥 대상: {', '.join(scenario_info['targets'])}")
    with info_col3:
        st.info(f"📝 기본 내용: {scenario_info['base_content']}")
    
    # 세부 내용 입력
    st.markdown("### 📝 세부 내용 입력")
    
    detail_content = st.text_area(
        "구체적인 내용",
        placeholder=f"{scenario_info['base_content']}에 대한 구체적인 내용을 입력하세요...",
        height=100
    )
    
    # 시나리오별 추가 정보
    batch_additional_info = {}
    
    # 일괄 생성 옵션 설정
    st.markdown("### ⚙️ 일괄 생성 옵션")
    batch_col1, batch_col2 = st.columns(2)
    
    with batch_col1:
        batch_length = st.selectbox(
            "문자 길이",
            ["매우 짧게", "짧게", "표준", "길게", "매우 길게"],
            index=2,
            key="batch_length"
        )
    
    with batch_col2:
        batch_style = st.selectbox(
            "문자 스타일",
            ["기본", "친근함", "긴급함", "공식적", "안내형"],
            key="batch_style"
        )
    
    if scenario == "등하교 안전 안내":
        batch_additional_info["날씨 상황"] = st.text_input("날씨 상황", placeholder="예: 강한 비, 눈", key="batch_weather")
        batch_additional_info["주의 사항"] = st.text_input("특별 주의 사항", placeholder="예: 우산 지참, 미끄러운 길 주의", key="batch_caution")
        
    elif scenario == "현장체험학습 안내":
        col1, col2 = st.columns(2)
        with col1:
            batch_additional_info["장소"] = st.text_input("체험학습 장소", placeholder="예: 국립과학관", key="batch_field_place")
            batch_additional_info["학년"] = st.text_input("대상 학년", placeholder="예: 3학년", key="batch_field_grade")
        with col2:
            batch_additional_info["준비물"] = st.text_input("준비물", placeholder="예: 도시락, 물", key="batch_field_prep")
            batch_additional_info["집합 시간"] = st.text_input("집합 시간", placeholder="예: 오전 8시 30분", key="batch_field_time")
    
    elif scenario == "학교 행사 안내":
        batch_additional_info["행사명"] = st.text_input("행사명", placeholder="예: 가을 축제", key="batch_event_name")
        batch_additional_info["장소"] = st.text_input("행사 장소", placeholder="예: 운동장", key="batch_event_place")
        
    elif scenario == "상담 주간 안내":
        batch_additional_info["상담 기간"] = st.text_input("상담 기간", placeholder="예: 11월 20일 ~ 24일", key="batch_consult_period")
        batch_additional_info["신청 방법"] = st.text_input("신청 방법", placeholder="예: 담임교사에게 신청", key="batch_consult_apply")
    
    # 일괄 생성 버튼
    if st.button("🚀 시나리오 일괄 생성", type="primary"):
        if detail_content:
            generated_messages = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            targets = scenario_info['targets']
            
            for i, target in enumerate(targets):
                status_text.text(f"{target}용 문자 생성 중...")
                
                sms, success = generate_ai_sms(
                    client=client,
                    target=target,
                    category=scenario_info['category'],
                    content_details=detail_content,
                    date=date_str,
                    school=school_name,
                    additional_info=batch_additional_info,
                    tone_guide="",
                    length_option=batch_length,
                    style_option=batch_style
                )
                
                if success:
                    generated_messages.append({
                        "target": target,
                        "content": sms,
                        "length": len(sms)
                    })
                
                progress_bar.progress((i + 1) / len(targets))
                time.sleep(0.5)  # API 호출 간격
            
            status_text.empty()
            progress_bar.empty()
            
            # 결과 표시
            if generated_messages:
                st.success(f"✅ {len(generated_messages)}개의 문자가 생성되었습니다!")
                
                # 각 대상별 문자 표시
                for msg in generated_messages:
                    with st.expander(f"📱 {msg['target']}용 문자 ({msg['length']}자)"):
                        st.text_area(
                            "",
                            value=msg['content'],
                            height=100,
                            key=f"batch_{msg['target']}"
                        )
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.caption(f"문자 유형: {'단문(SMS)' if msg['length'] <= 80 else '장문(LMS)'}")
                        with col2:
                            if st.button("📋 복사", key=f"copy_{msg['target']}"):
                                st.info("텍스트를 선택 후 Ctrl+C로 복사하세요")
                
                # 전체 다운로드
                all_messages = "\n\n".join([
                    f"[{msg['target']}용 문자]\n{msg['content']}\n(글자수: {msg['length']}자)"
                    for msg in generated_messages
                ])
                
                st.download_button(
                    label="📥 전체 문자 다운로드",
                    data=all_messages,
                    file_name=f"{scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        else:
            st.warning("⚠️ 구체적인 내용을 입력해주세요.")

with tab3:
    st.subheader("📊 생성 이력")
    
    if "sms_history" in st.session_state and st.session_state.sms_history:
        # 이력을 DataFrame으로 변환
        history_df = pd.DataFrame(st.session_state.sms_history)
        
        # 필터링 옵션
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_target = st.selectbox("대상 필터", ["전체"] + ["학부모", "학생", "교직원"])
        with col2:
            filter_category = st.selectbox("카테고리 필터", ["전체"] + ["안전", "재난", "체험학습", "행사 안내", "상담", "안내"])
        with col3:
            if st.button("🗑️ 이력 초기화"):
                st.session_state.sms_history = []
                st.rerun()
        
        # 필터링 적용
        filtered_df = history_df.copy()
        if filter_target != "전체":
            filtered_df = filtered_df[filtered_df['target'] == filter_target]
        if filter_category != "전체":
            filtered_df = filtered_df[filtered_df['category'] == filter_category]
        
        # 이력 표시
        for idx, row in filtered_df.iterrows():
            with st.expander(f"📅 {row['timestamp']} - {row['target']} ({row['category']})"):
                st.text_area(
                    "",
                    value=row['content'],
                    height=80,
                    key=f"history_{idx}"
                )
                st.caption(f"글자 수: {row['length']}자 | 스타일: {row.get('style', '기본')} | 길이: {row.get('length_option', '표준')}")
        
        # 통계 표시
        st.markdown("### 📈 통계")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("총 생성 수", f"{len(filtered_df)}개")
        with stat_col2:
            avg_length = filtered_df['length'].mean() if not filtered_df.empty else 0
            st.metric("평균 글자 수", f"{avg_length:.1f}자")
        with stat_col3:
            most_target = filtered_df['target'].mode()[0] if not filtered_df.empty else "없음"
            st.metric("가장 많은 대상", most_target)
        with stat_col4:
            most_category = filtered_df['category'].mode()[0] if not filtered_df.empty else "없음"
            st.metric("가장 많은 카테고리", most_category)
    else:
        st.info("아직 생성된 문자가 없습니다. AI 문자 생성 탭에서 문자를 생성해보세요!")

with tab4:
    st.subheader("❓ 사용 가이드")
    
    st.markdown("""
    ### 🤖 AI 문자 생성기 소개
    
    OpenAI의 GPT 모델을 활용하여 상황에 맞는 학교 문자를 자동으로 생성합니다.
    
    ### 📱 주요 기능
    
    #### 1. AI 문자 생성
    - **대상별 맞춤형**: 학부모, 학생, 교직원별로 적절한 톤과 내용
    - **카테고리별 최적화**: 안전, 재난, 체험학습 등 상황별 필수 요소 포함
    - **추가 정보 반영**: 장소, 시간, 준비물 등 세부 정보 자동 반영
    - **길이 조절**: 매우 짧게(40자)부터 매우 길게(180자)까지 5단계
    - **스타일 선택**: 기본, 친근함, 긴급함, 공식적, 안내형 중 선택
    
    #### 2. 시나리오별 일괄 생성
    - **자주 쓰는 상황**: 등하교 안전, 체험학습 등 미리 정의된 시나리오
    - **다중 대상 생성**: 한 번에 여러 대상용 문자 생성
    - **일관된 정보 전달**: 같은 내용을 대상별로 적절히 변환
    
    #### 3. 생성 이력 관리
    - **이력 저장**: 생성된 문자 자동 저장
    - **필터링**: 대상, 카테고리별 검색
    - **통계 확인**: 사용 패턴 분석
    
    ### 💡 활용 팁
    
    1. **구체적인 정보 입력**: AI가 더 정확한 문자를 생성합니다
    2. **추가 정보 활용**: 카테고리별 추가 정보를 입력하면 더 완성도 있는 문자 생성
    3. **문자 길이 조절**: 상황에 맞게 길이를 선택 (긴급 상황은 짧게, 상세 안내는 길게)
    4. **스타일 선택**: 
       - 기본: 일반적인 안내
       - 친근함: 행사나 축하 메시지
       - 긴급함: 재난이나 안전 관련
       - 공식적: 중요 공지나 행정 사항
       - 안내형: 단계별 설명이 필요한 경우
    5. **시나리오 활용**: 반복적인 상황은 시나리오 기능 사용
    6. **이력 참고**: 이전에 생성한 문자를 참고하여 일관성 유지
    
    ### ⚠️ 주의사항
    
    - AI가 생성한 문자는 반드시 검토 후 발송
    - 개인정보나 민감한 정보는 직접 입력하지 않기
    - 긴급 상황 시에는 미리 준비된 템플릿 사용 권장
    - 모든 대상에게 존댓말을 사용하도록 설정되어 있음
    
    ### 🔧 문제 해결
    
    - **API 오류**: 잠시 후 다시 시도하거나 API 키 확인
    - **생성 실패**: 입력 내용을 더 구체적으로 작성
    - **부적절한 내용**: 다시 생성하거나 직접 수정
    """)
    
    # FAQ
    with st.expander("자주 묻는 질문"):
        st.markdown("""
        **Q: AI가 생성한 문자를 그대로 발송해도 되나요?**
        A: 반드시 검토 후 발송하시기 바랍니다. AI는 도우미 역할일 뿐입니다.
        
        **Q: 생성된 문자가 너무 길어요.**
        A: 문자 길이 옵션에서 '짧게' 또는 '매우 짧게'를 선택하세요.
        
        **Q: 특정 형식으로 생성하고 싶어요.**
        A: 커스텀 톤 옵션을 사용하여 원하는 형식을 지정할 수 있습니다.
        
        **Q: 이력이 사라졌어요.**
        A: 이력은 세션 동안만 유지됩니다. 중요한 문자는 따로 저장하세요.
        
        **Q: 학생에게 반말로 보내고 싶어요.**
        A: 현재는 모든 대상에게 존댓말을 사용하도록 설정되어 있습니다. 필요시 생성 후 수정하세요.
        """)

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p> 경상북도교육청 학교 문자 생성기 v2.0 | Powered by OpenAI GPT-4</p>
    <p style='font-size: 0.8em; color: gray;'>효율적이고 정확한 학교-가정 소통을 지원합니다</p>
</div>
""", unsafe_allow_html=True)
