# RDPG,PPO,A2C 앙상블 트레이더
 ## SP500 trading of SOTA reinforcement learning Ensemble Agent
 
- 도구

  [![파이썬 Badge](https://img.shields.io/badge/python-3776AB?style=flat-square&logo=python&logoColor=white&link=mailto:wjtls01@naver.com)](mailto:wjtls01@naver.com)

  [![파이토치 Badge](https://img.shields.io/badge/pytorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white&link=mailto:wjtls01@naver.com)](mailto:wjtls01@naver.com)

  [![주피터 Badge](https://img.shields.io/badge/jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white&link=mailto:wjtls01@naver.com)](mailto:wjtls01@naver.com)

- 목표: 앙상블을 사용함으로써 단일에이전트의 휴리스틱한 전략보다 하락장과 상승장에서 안정적으로 트레이딩

- 구현 이유: Deep Reinforcement Learning for Automated stock Trading :An Ensemble Strategy 라는 논문을 보게 됐고,
  이 프로젝트를 완료하게 된다면 A2C,RDPG,PPO 만 사용하는 것이 아니라 여러 SOTA 에이전트 사용 및 유동적인 팩터를 사용하는 트레이더를 구현할수 있을것이라 생각하여 진행
  
 
## 기능

- API와 csv를 사용하여 비트코인 또는 SP500데이터 호출
- Env 클래스 생성 (리워드 계산 및 전처리)
- A2C 에이전트 (강화학습)
- PPO 에이전트 (강화학습)
- DDPG 에이전트 (강화학습)

- optimize (over fitting 방지를 위한 clip gradient 사용)
- 에이전트 앙상블 (sharpe ratio 사용)


## 요약
- ![image](https://user-images.githubusercontent.com/60399060/145546012-46aebe4a-7ee4-4b54-8ff7-3de0866f640c.png)

- Train the three algorithms that take actions in the environment and ensemble the three agents together using the Sharpe ratio.
- 3개의 에이전트를 멀티프로세싱 방식(현 프로젝트에서는 동기적 학습실행)으로 각각 학습하여 가중치를 저장.
- 기간을 설정하고 구간별로 rolling 하여 샤프지수 계산.
- 매스탭마다 가장 높은 샤프지수를 가진 에이전트를 선정하여 해당 에이전트로 트레이딩.   
<br/>
<br/>
<br/>
<br/>


- <img src="https://user-images.githubusercontent.com/60399060/145567671-98bd6c89-daac-4c6b-b45a-f1b47df612d3.png" width="710px" height="820px" title="px(픽셀) 크기 설정" alt="RubberDuck"></img><br/>

- Discrete action space (PPO,A2C) 에서 action = [0=매도 ,1=관망 ,2= 매수]로 설정하였고, <br/> 
  Continuous action space (RDPG)에서 action = {-k,…-1,0,1,…,k} 로 정의 (k < maximum amount of shares for each buying action)
- Reward를 위와 같이 정의할 경우 next step에서 주가가 상승할때 매수 또는 홀딩으로 PV를 최대화 할수있으며 <br/>
  next step에서 주가가 하락하는 경우 이전 스탭에서 매도 하려는 성향을 가진다. 
- state 는 종가 하나만 사용한다.(PCA나 양질의 데이터 생성 등으로 curse of dimentionality문제를 완화한다면 가격 데이터 이외에 여러 지표를 사용가능)


## 본론

 - ## PPO
   PPO는 new policy가 old policy 와 크게 다르지않도록 Clipping 하기 때문에 논문에서 안정성이 높고 빠르다는 결과를 보인다. <br/>
   또한 상승구간에서 타 에이전트에 비해 수익률이 잘나오는 편이다. 그러나 하락구간에서 A2C보다 낮은 샤프지수를 보인다.
 

 - ## RDPG
 - DDPG 알고리즘에 RNN을 결합한 알고리즘.
 - DQN 알고리즘과 다르게 이산공간 보다는 연속행동공간에서 잘 작동하며 이로인해 더 큰 data set에서 트레이딩 할수있다. <br/>
 - 다른 SOTA 알고리즘에 비해 간단한 경향이 있으며 이러한 단순성 때문에 사용자는 좀더 트레이딩 전략에 초점을 맞출수 있으며 <br/>
   actor network에서 tanh 활성화 함수 사용으로 exploration을 더 잘하게된다.
 - Data의 상관관계 감소를 위해 random sampling
 
 - ## A2C
 
 
## 결론

## 한계 및 개선
