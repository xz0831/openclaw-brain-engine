# 제3자 소프트웨어와 자료

이 저장소의 LICENSE는 작성자 소유 엔진 코드에 적용한다. 외부 라이브러리·도구·모델·PDK·사용자 문헌의 권리를 대신 부여하지 않는다. 각 구성요소는 자체 조건을 따른다.

## 배포에 포함하지 않는 것

이 저장소와 wheel에는 다른 패키지의 소스·바이너리, 모델 가중치, 실제 PDK 모델카드, 문헌 원문, 지식그래프 스냅샷을 넣지 않는다. `pyproject.toml`은 별도로 설치할 의존성을 선언한다. 패키지를 별도 설치한다는 사실만으로 라이선스 의무가 사라지는 것은 아니다.

## PDF 경로

- 엔진의 실제 PDF 인입은 선택 의존성인 MinerU의 구조화 출력 경로를 사용한다.
- **PyMuPDF/MuPDF를 직접 쓰던 미사용 legacy `chunk_pdf()`와 `vision.py`는 이 엔진판에 포함하지 않는다.** 직접 dependency와 해당 legacy 시험도 제외했다. 구조화 chunker·HTML 인입·검색·MCP·회로 실행 코드는 유지한다.
- 확인한 **MinerU 3.4.5**는 Apache-2.0에 추가 조건을 붙인 라이선스다. 일정 규모 이상의 상업적 사용에는 별도 조건이 있고 온라인 서비스를 제공할 때 MinerU 사용을 표시해야 한다. 공식 버전의 [LICENSE](https://github.com/opendatalab/MinerU/blob/mineru-3.4.5-released/LICENSE.md)와 [의존성 선언](https://github.com/opendatalab/MinerU/blob/mineru-3.4.5-released/pyproject.toml)을 따른다.
- 이 버전의 직접 의존성과 검증 환경의 설치 metadata에서 PyMuPDF를 요구하는 MinerU 경로는 확인되지 않았다. 미래 버전과 모든 선택·전이 의존성의 조건까지 인증했다는 뜻은 아니다. 배포·서비스 운영자는 실제 설치하는 버전과 모델의 조건을 별도로 확인한다.

## 회로 도구와 PDK

직접 작성한 템플릿·시드·PDK 연결 코드는 엔진에 포함한다. ngspice·iverilog 실행 파일이나 SkyWater/GF180/IHP의 실제 PDK 배포물은 포함하지 않는다. 도구와 PDK를 별도로 설치하고 공급자의 조건을 따른다. 일반적인 코드 예시는 논문·교재 원문을 제공하는 지식 코퍼스가 아니다.

## 과거 이용 허락과 공개 플랫폼

이전 버전에서 이미 유효하게 부여된 MIT 등 이용권은 현재 LICENSE로 소급 철회하지 않는다. GitHub 공개 시에는 플랫폼 약관이 허용하는 열람·포크 등의 권리가 있다. 권리유보 표시는 복사 방지 기술이나 제3자 권리의 해제 수단이 아니다.
