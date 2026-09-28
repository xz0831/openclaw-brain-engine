# 엔진 공개 범위와 검토 기록

이 저장소는 작성자가 공개를 승인한 엔진 소스 배포본이다. 기존 개발 저장소의 Git 이력과 실제 사용자 자료는 포함하지 않는다.

## 포함·제외 기준

- 포함: 직접 작성한 엔진 코드, 일반 예시·기본 템플릿·시드, PDK 연결 로직, 합성 입력 기반 unit test, 최소 샘플 설정, 직접 작성한 README·구조도.
- 제외: 문헌 PDF/HTML/추출문, 수집한 그래프·카드·embedding, 실제 실행 netlist·측정결과·모델 로그, 회사 자료·도입 구상, 개인 운영 설정, 원래 `.git`와 커밋·태그.
- `pdks.py`는 공개 PDK를 연결하는 코드이며 실제 foundry 모델 파일이 아니다. `corpus.py`는 자료 저장 코드이며 축적된 corpus 자체가 아니다.

## 라이선스 검토

작성자 소유 부분의 이용 조건은 열람·심사이며, 실행·수정·재배포는 별도 허락이다. 제3자 조건과 이미 부여한 과거 권리는 루트 LICENSE와 구분한다.

PyMuPDF를 직접 사용하던 미사용 legacy PDF 두 경로와 직접 의존은 이 후보에서 제외했다. PyMuPDF import를 차단한 전체 시험은 1,700개 통과·143개 건너뜀이다. 실제 주 인입은 선택적 MinerU 구조화 경로를 유지한다. 이 후보에는 의존성 패키지·모델·PDK 배포물을 넣지 않았으며, 라이선스 조건은 [THIRD_PARTY.md](THIRD_PARTY.md)에서 개별적으로 다룬다. 모든 버전·선택 의존성·운영 방식의 법률 검토가 끝났다는 뜻은 아니다.

코드를 공개한 GitHub 저장소에서는 플랫폼 약관상 열람·포크 등의 권리가 있다. 권리유보 문구로 복사를 기술적으로 막을 수 있다고 안내하지 않는다. [GitHub 라이선스 안내](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)

## 기존 저장소

기존 비공개 개발 저장소는 삭제하지 않고 보존한다. 새 후보는 그 저장소의 mirror나 fork가 아니며, 허용 목록으로 선택한 현재 파일만 담는다. 기존 이력을 지우거나 과거의 권리를 소급해서 바꾼 것으로 설명하지 않는다.
