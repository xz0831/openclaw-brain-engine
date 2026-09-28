# 엔진 구조와 자료의 위치

![엔진 구조](assets/architecture-core.svg)

## 문서 경로

`knowledge/extraction/`이 사용자가 제공한 문서를 구조화된 블록과 청크로 만들고, 추출·원문 근거 대조를 수행한다. `knowledge/matching/`, `knowledge/reasoning/`과 pipeline이 기존 지식과의 관계를 정리하고 `knowledge/graph/`를 통해 저장한다. 원문·추출 결과·그래프 데이터는 사용자의 별도 저장소에 쌓인다.

## 질문 경로

MCP 도구가 `agent.py`와 각 처리 경로를 호출한다. 검색된 맥락으로 답변을 합성하고, `answer_contract.py`가 인용과 답변 표시 형식을 검사한다. `model_knowledge` 정책으로 답하는 경우 모델 지식 사용을 표시한다. 인용 검사를 통과했다는 사실과 답변의 의미가 옳다는 사실은 구분한다.

## 실행 경로

`knowledge/executable/`의 템플릿·레시피·실행기·판정기가 별도 시뮬레이션 경로를 구성한다. `corpus.py`는 이 결과를 저장하는 코드다. 저장된 실제 corpus, ClaimCard, Regularity 결과 파일은 이 소스 배포에 포함하지 않는다. 기본 검색 답변에 카드 판정이 자동 반영되는 경로는 아직 완성되지 않았다.

## 접근과 반출

MCP에는 전체·연구·읽기 전용 도구 프로필이 있다. `egress.py`의 local-only 정책은 앱이 관리하는 연결을 검사하며, 호스트 방화벽이나 클라이언트 자체의 통신까지 제어하지 않는다. 공개용 샘플 설정은 loopback과 local-only로 시작하고 모델 ID는 사용자가 지정하게 한다.

## 소스 분리

이 사본에는 엔진 Python 코드와 합성 입력을 쓰는 시험, 빈 데이터베이스를 위한 schema, 예제 설정을 넣었다. 기존 개발 저장소의 Git 이력·실험·운영 기록·문헌 원문·실제 지식 데이터는 가져오지 않았다. PyMuPDF를 직접 사용하던 미사용 legacy 두 경로는 제외했다. 라이브 모델·Neo4j·시뮬레이션의 배포 검증과 단위시험은 별개다.
