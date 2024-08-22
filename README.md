# 핵 미사일 시뮬레이터
![스크린샷 2024-08-22 142720](https://github.com/user-attachments/assets/e63ec4b0-6360-4b52-8e97-d340ba379a62)
## 개요
tensorflow를 이용해 핵에 들어가는 우라늄의 양에 따른 일정한 간격으로 얻은 핵 미사일의 범위로 발생하는 피해를 예측하는 프로그램입니다.  

## 기능
- 정해진 핵미사일 프리셋을 선택하거나, 직접 우라늄양을 지정합니다.
- 인구 수와 토지의 넓이를 지정합니다.
- 핵 미사일이 발사되었을 경우 발생하는 피해 정보를 확인합니다.

## 상세
### 피해 범위 상세 설명
- 공기 폭발 반경 (Airburst Radius, KM):
핵폭발이 공중에서 발생하는 경우 충격파와 열복사의 영향을 최대화하는 반경입니다.
- 화염구 반경 (Fireball Radius, KM):
폭발 직후 생성되는 고온의 불덩이 반경으로, 이 안에 있는 모든 물질이 증발하거나 완전히 소실됩니다.
- 강한 폭발 피해 반경 (20 psi, KM):
강한 충격파로 인해 건물 붕괴와 같은 치명적인 피해를 입는 범위입니다.
- 보통 폭발 피해 반경 (Moderate Blast Damage, KM):
중간 정도의 충격파로 건물의 구조적 손상과 창문 파손 등의 피해가 발생하는 반경입니다.
- 열 복사 반경 (Thermal Radiation Radius, 3rd Degree Burns, KM):
핵폭발로 발생하는 열 복사로 인해 3도 화상을 입을 수 있는 거리입니다.
- 약한 폭발 피해 반경 (1 psi, KM):
비교적 약한 충격파로 창문 파손이나 경미한 구조적 피해가 발생하는 반경입니다.
- 방사선 반경 (Radiation Radius, 500 rem, KM):
치명적인 양의 방사선을 방출하는 반경으로, 이 범위 내에서는 심각한 방사선 피폭이 발생할 수 있습니다.

## 요구사항
* 모듈
```
matplotlib
numpy
sklearn
tesorflow
```
* 기타
```
나눔고딕
```
## 출처
데이터셋: [NUKEMAP by Alex Wellerstein](https://nuclearsecrecy.com/nukemap/)

## 기타
- 분류: 2023 고등학교 동아리