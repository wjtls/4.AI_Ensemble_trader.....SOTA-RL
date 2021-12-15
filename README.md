# RDPG,PPO,A2C 앙상블 트레이더
 ## SP500 trading of SOTA reinforcement learning Ensemble Agent
 
- 도구

  [![파이썬 Badge](https://img.shields.io/badge/python-3776AB?style=flat-square&logo=python&logoColor=white&link=mailto:wjtls01@naver.com)](mailto:wjtls01@naver.com)

  [![파이토치 Badge](https://img.shields.io/badge/pytorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white&link=mailto:wjtls01@naver.com)](mailto:wjtls01@naver.com)

  [![주피터 Badge](https://img.shields.io/badge/jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white&link=mailto:wjtls01@naver.com)](mailto:wjtls01@naver.com)

- 목표: 앙상블을 사용함으로써 단일에이전트의 휴리스틱한 전략보다 하락장과 상승장에서 안정적으로 트레이딩

- 구현 이유: Deep Reinforcement Learning for Automated stock Trading :An Ensemble Strategy 라는 논문을 보게 됐고,
  이 프로젝트를 완료하게 된다면 A2C,RDPG,PPO 만 사용하는 것이 아니라 여러 SOTA 에이전트 사용 및 더 많은 팩터를 사용하는 트레이더를 구현할수 있을것이라 생각하여 진행
  
 
## 기능

- API와 csv를 사용하여 비트코인 또는 SP500데이터 호출
- Env 클래스 생성 (리워드 계산 및 전처리)
- A2C 에이전트 (강화학습)
- PPO 에이전트 (강화학습)
- RDPG 에이전트 (강화학습)
- Replay buffer 사용
- optimize (over fitting 방지를 위한 clip gradient 사용)
- 에이전트 앙상블 (sharpe ratio 사용)


## 요약
- ![image](https://user-images.githubusercontent.com/60399060/145546012-46aebe4a-7ee4-4b54-8ff7-3de0866f640c.png)

- Train the three algorithms that take actions in the environment and ensemble the three agents together using the Sharpe ratio.
- 3개의 에이전트를 순차적으로으로 각각 학습하여 가중치를 저장(GPU 사양이 된다면 멀티프로세싱 사용).
- 기간을 설정하고 구간별로 rolling 하여 샤프지수 계산.
- 매스탭마다 가장 높은 샤프지수를 가진 에이전트를 선정하여 해당 에이전트로 1스탭 트레이딩.   
<br/>
<br/>
<br/>
<br/>


- <img src="https://user-images.githubusercontent.com/60399060/145567671-98bd6c89-daac-4c6b-b45a-f1b47df612d3.png" width="710px" height="820px" title="px(픽셀) 크기 설정" alt="RubberDuck"></img><br/>

- Discrete action space (PPO,A2C) 에서 action = [0=매도 ,1=관망 ,2= 매수]로 설정하였고, <br/> 
  Continuous action space (RDPG)에서 action = {-k,…-1,0,1,…,k} 로 정의 (k < maximum amount of shares for each buying action) 
  <br/>  
- Reward를 위와 같이 정의할 경우 next step에서 주가가 상승할때 매수 또는 홀딩으로 PV를 최대화 할수있으며 <br/>
  next step에서 주가가 하락하는 경우 현재의 step 에서 매도 하려는 성향을 가진다.
  <br/>   
- state 는 종가 하나만 사용.(PCA나 양질의 데이터 생성 등으로 curse of dimensionality 문제를 완화한다면 가격 데이터 이외에 여러 지표를 사용가능)


## 본론

 - ## PPO
 - DeepReinforcementLearninginQuantitative AlgorithmicTrading:AReview 에 따르면 PPO는 타 RL알고리즘 보다 복잡한 환경에서 잘작동한다. 기존 RL의 가장 큰 문제는 최근 폴리시에 일반적으로 의존하는데 이는 observation과 reward등의 트레젝터리 데이터 분포가 학습을 하는도중에도 지속적으로 변한다는 의미이다.
 이는 불안정성을 초래하는 가장큰 이유중 하나이다. RL은 또한 하이퍼파라미터에 매우 민감하다 
 이런 문제점이 존재한다면 PPO알고리즘에 사용된 바와 같이 region 을 서치공간에 사용하는 방식으로 완화할수있다. 이는 next batch에서 매우 안좋은 policy를 뽑고 다시는 재사용하지 않는다
 
   PPO는 new policy가 old policy 와 크게 다르지않도록 Clipping 하기 때문에 논문에서 안정성이 높고 빠르다는 결과를 보인다. <br/>
   또한 상승구간에서 타 에이전트에 비해 수익률이 잘나오는 편이다. 그러나 하락구간에서 A2C보다 낮은 샤프지수를 보인다.
 
 다. 정책 알고리즘으로서 PPO는 새로운 정책이 기존 정책에서 너무 멀리 바뀌는 것을 피하기 위해 대리목표를 활용하여 샘플 효율성 문제를 해결한다. PPO는 정책 업데이트를 정규화하고 교육 데이터를 재사용할 수 있기 때문에 대리 목표는 PPO의 핵심 기능이다.

 - ## RDPG
 - DDPG 알고리즘에 LSTM을 결합한 알고리즘.
 - 연속 행동공간에서 행동의 확률분포를 출력하지 않고 이산적인 값을 출력한다.
 - DQN,PPO와 같은 이산공간 알고리즘과 다르게 연속행동공간에서 잘 작동하며 이로인해 더 큰 data set에서 트레이딩 할수있다. <br/>
 - 다른 SOTA 알고리즘에 비해 간단한 경향이 있으며 이러한 단순성 때문에 사용자는 좀더 트레이딩 전략에 초점을 맞출수 있으며 <br/>
   actor network에서 tanh 활성화 함수 사용으로 exploration을 더 잘하게된다.
 - Data 효율성 증가를 위해 Replay buffer 사용, 샘플간 상관관계 감소를 위해 random sampling
 - ![image](https://user-images.githubusercontent.com/60399060/145927678-6fb737cd-989f-4c41-9199-85d22be65266.png)


 - 세부 
   Deterministic 하게 바꾸는 과정에서 r+감마*Q(s,u) 사용으로 환경만 영향을 미치므로 과거 action에 영향을 받지않는다
   따라서 off policy 방식으로 학습을 하며 Replay buffer를 사용할수 있게된다.
 
 
 - ## A2C
 - Acor와 Critic network 를 사용하며 Advantage function을 사용하여 policy gradient의 분산을 줄여준다.
 - 벨류 펑션만 추정하는 대신 크리틱 네트워크로 어드밴티지 펑션도 추정한다. 따라서 얼마나 좋은 액션인지 뿐만아니라 얼마나 더 좋아질수 있는지도 고려한다 그러므로 폴리시 네트워크 의 높은 분산을 감소시키는 방법은 강력
 - A2C는 스톡 트레이딩환경에서 안정적이다.
 
 
## 결론 (수정 필요 A2C 학습 제대로 안됨)

- kindex SP500 validation set 에서 백테스팅 결과
- 시장 수익률 :-0.3765702247619629 %
- Ensemble Agent return : 0.18998384475708008 %
- Ensemble Agent Alpha : 0.566554069519043 % <br/><br/>


- 앙상블 에이전트는 약 100- 200 step의 하락구간에서 주로 A2C와 DDPG 알고리즘으로 거래를 하였고 
- 200-350 step 까지의 상승구간에서 주로 PPO 알고리즘을 사용하여 거래하였다

## 한계 및 개선
- State의 정의 (차원의 저주 문제로 인해 1개의 feature만 사용)<br/>
    - 팩터들 사이에서 다중공선성 문제가 생길 수 있으므로 Feature Extraction 방법으로 차원의 저주와 높은 상관계수 문제 해결 예정<br/><br/>
    - continuous action space에서 효과적인 다른 SOTA 에이전트 사용 (TD3,SLAC,SAC,D4PG 등등)<br/><br/>
    - 
- 시장은 t시점에서 알파를 찾아도 향후 새로운 알파가 생겨난다. <br/> 
    - 단일 에이전트 보다는 유동적으로 전략을 찾을수 있지만 현 앙상블 에이전트에서도 여전히 전략간 상관계수와 편향이 있다.  <br/> 
    - 더많은 에이전트를 앙상블하거나 MARL(Multi-Agent-Reinforcement-learning) 사용 예정   <br/>
    - 각 에이전트가 알고리즘 자체를 스스로 개선하도록 하여 여러 에이전트들의 전략간 상관계수와 편향을 낮출 예정

    
    
    
    
