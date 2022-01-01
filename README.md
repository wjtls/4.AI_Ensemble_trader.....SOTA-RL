# RDPG,PPO,A2C 앙상블 트레이더
 ## SP500 trading of SOTA reinforcement learning Ensemble Agent
 
- 도구

    [![파이썬 Badge](https://img.shields.io/badge/python-3776AB?style=flat-square&logo=python&logoColor=white&link=mailto:wjtls01@naver.com)](mailto:wjtls01@naver.com)

    [![파이토치 Badge](https://img.shields.io/badge/pytorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white&link=mailto:wjtls01@naver.com)](mailto:wjtls01@naver.com)

    [![주피터 Badge](https://img.shields.io/badge/jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white&link=mailto:wjtls01@naver.com)](mailto:wjtls01@naver.com)

- 목표: 앙상블을 사용함으로써 인간의 퀀트 전략 및 단일 강화학습 에이전트보다 안정적으로 트레이딩

- 구현 이유: Deep Reinforcement Learning for Automated stock Trading :An Ensemble Strategy 라는 논문을 보게 됐고,
  이 프로젝트를 완료하게 된다면 A2C,RDPG,PPO 만 사용하는 것이 아니라 여러 SOTA 에이전트 사용 및 더 많은 팩터를 사용하는 트레이더를 구현할수 있을것이라 생각하여 진행
  
- DRL(Deep Reinforcement learning) 사용 이유
    - 1. 주식 거래 목표인 수익률 극대화 가능 : 강화학습 에이전트는 행동,전략에 대한 보상을 극대화하여 이러한 최적화 문제 해결
    - 2. 주식 거래 의사 결정 과정을 모델링 가능 : MDP(Markov Decision process)를 정의
    - 3. 다차원 데이터 처리 가능 : 극도로 큰 행동 공간 및 상태 공간을 가지는 다차원 데이터도 효과적으로 계산 가능
    - 4. 한 분야에서 인간을 능가 할 수도 있다.
  
  <br/><br/><br/><br/>


## 기능

  - API와 csv를 사용하여 비트코인 또는 SP500데이터 호출
  - Env 클래스 생성 (리워드 계산 및 전처리)
  - A2C 에이전트 (강화학습)
  - PPO 에이전트 (강화학습)
  - RDPG 에이전트 (강화학습)
  - Replay buffer 사용
  - OU process 사용
  - soft target update 사용
  - optimize 중 over fitting 방지를 위한 clip gradient 사용
  - 에이전트 앙상블 (sharpe ratio 사용)



  <br/><br/><br/><br/>


## 요약
  - 3개의 AI 에이전트를 앙상블 하여 트레이딩을 한다. <br/>
    각기 투자 성향이 다른 n명의 트레이더가 머리를 맞대고 트레이딩 하는 상황을 예로 들 수 있다.

  - MDP(Markov Decision process) 정의 <br/>
    마르코브 의사 결정 과정은 리워드, 상태, 행동, 벨류 등을 정의하는 과정이다. <br/>
    이는 트레이딩 환경에서 뿐만 아니라 강화학습을 수행 하는 모든 환경에서 가장 중요한 문제이다. <br/>
    예를 들면 인간이 주식을 할 때 목표(수익률)을 설정하고 매매 행동, 지표, 팩터를 선택하는 과정이라 볼 수 있다. <br/> <br/>
    
  - <img src="https://user-images.githubusercontent.com/60399060/145567671-98bd6c89-daac-4c6b-b45a-f1b47df612d3.png" width="710px" height="820px" title="px(픽셀) 크기 설정" alt="RubberDuck"></img><br/>

  - Discrete action space (PPO,A2C) 에서 action = [0=매도 ,1=관망 ,2= 매수]로 설정하였고, <br/> 
    Continuous action space (RDPG)에서 action = {-k,…-1,0,1,…,k} 로 정의 (k < maximum amount of shares for each buying action) 
    <br/>  
  - Reward를 위와 같이 정의할 경우 next step에서 주가가 상승할때 매수 또는 홀딩으로 PV를 최대화 할수있으며 <br/>
    next step에서 주가가 하락하는 경우 현재의 step 에서 매도 하려는 성향을 가진다.
    <br/>   
  - state 는 종가 하나만 사용.(PCA나 양질의 데이터 생성 등으로 차원의저주 문제를 완화한다면 가격 데이터 이외에 여러 지표를 사용가능)


  - ![image](https://user-images.githubusercontent.com/60399060/145546012-46aebe4a-7ee4-4b54-8ff7-3de0866f640c.png)

  - Train the three algorithms that take actions in the environment and ensemble the three agents together using the Sharpe ratio.
  - 3개의 에이전트를 순차적으로으로 각각 학습하여 가중치를 저장(GPU 사양이 된다면 멀티프로세싱 사용).
  - 기간을 설정하고 구간별로 rolling 하여 샤프지수 계산.
  - 매스탭마다 가장 높은 샤프지수를 가진 에이전트를 선정하여 해당 에이전트로 1스탭 트레이딩.<br/><br/><br/><br/>




  <br/><br/><br/><br/>


## 본론
 - 기존 RL의 가장 큰 문제는 최근 폴리시에 일반적으로 의존하는데 이는 observation과 reward등의 트레젝터리 데이터 분포가 학습을 하는도중에도 지속적으로 변한다.
   이는 학습의 불안정성을 초래하는 가장큰 이유중 하나이다. 또한 RL은 하이퍼파라미터에 매우 민감하다 
   따라서 PPO, A2C 알고리즘을 선정하여 학습의 속도와 안정성을 높이고 RDPG 알고리즘을 선정하여 샘플 효율성 및 수렴성을 높인다.

 - ## PPO 알고리즘
   - Deep Reinforcement Learning in Quantitative Algorithmic Trading: A Review 에 따르면 PPO는 타 RL알고리즘 보다 주식시장의 복잡한 환경에서 잘작동 하므로 선택했다.
   
   - ![image](https://user-images.githubusercontent.com/60399060/146135720-9f131c45-c616-4383-bf87-f9235cf7f55f.png)
   - new policy가 old policy 와 크게 다르지않도록 Clipping 하기 때문에 논문에서 안정성이 높고 빠르다는 결과를 보인다. <br/>
   
   - ![image](https://user-images.githubusercontent.com/60399060/146135945-5e1bd0e9-8ef7-49c2-9d41-b2ae8ebb9f25.png)
   - GAE Advantage를 사용하여 Advantage를 잘추산한다. 이로인해 분산을 더 적절하게 감소 시킬수 있다.
   - 신뢰 지역(Trust region) 에서 GAE를 구하고 r세타를 연산하는 덕에 buffer를 사용할수 있고 next batch에서 좋지않은 policy를 뽑을 경우 재사용하지 않는다
 
   - ![image](https://user-images.githubusercontent.com/60399060/146136194-aa3647e1-29a8-45f4-a21c-6d38884ab353.png)
   - PPO는 새로운 정책이 기존 정책에서 너무 멀리 바뀌는 것을 피하기 위해 대리 목표를 활용하여 min을 취함으로 샘플 효율성 문제를 해결한다.
   - PPO는 정책 업데이트를 정규화하고 교육 데이터를 재사용할 수 있기 때문에 대리 목표는 PPO의 핵심 기능이다. 따라서 
     ppo 는 on policy이지만 on policy 의 수렴성과 대리목표(Surrogate loss) 사용으로 off policy의 장점인 샘플 효율성을 가지게 된다.
   - 상승구간에서 타 에이전트에 비해 수익률이 잘나오는 편이다. 그러나 하락구간에서 A2C보다 낮은 샤프지수를 보인다.

 - ## RDPG 알고리즘
   - ![image](https://user-images.githubusercontent.com/60399060/145927678-6fb737cd-989f-4c41-9199-85d22be65266.png)
   - DDPG 알고리즘에 LSTM을 결합한 알고리즘.
   - actor network에서 tanh 활성화 함수를 사용하여 exploration을 더 잘하게된다.
   - Data 효율성 증가와 샘플간 상관관계 감소를 위해 Replay buffer 사용. <br/><br/>
 
   - ![image](https://user-images.githubusercontent.com/60399060/146115365-355181cf-c4b1-4ff4-8665-d31b2670c287.png)
   - 연속 행동공간에서 행동의 확률분포를 출력하지 않고 이산적인 값을 출력한다. (Q-learning method를 가져와서 연속행동공간에서의 off policy 알고리즘으로 활용)
   - 다른 SOTA 알고리즘에 비해 간단한 경향이 있으며 이러한 단순성 때문에 사용자는 좀더 트레이딩 전략에 초점을 맞출수 있다. <br/><br/>

   - ![image](https://user-images.githubusercontent.com/60399060/146114550-5d2ebba7-4a0a-4824-9ae9-99ab787aaeb3.png)
   - OU process는 과거 노이즈들과 시간적으로 연관된 확률 변수를 생성하는 exploration 기법이다 (시간적 연관성은 노이즈 값이 너무 커지거나 작아지는걸 방지하기 위함)
   - critic 네트워크가 이산 값을 출력하기 때문에 action에 OU프로세스 값을 추가하여 exploration 효과를 추가<br/><br/>

   - ![image](https://user-images.githubusercontent.com/60399060/146114471-6b1b63fa-c5e3-4974-80bd-efe2b5045f6a.png)
   - soft target update로 급격한 네트워크 변화를 방지하여 안정적인 수렴을 돕는다<br/><br/>

   - 세부 
     Deterministic 하게 바꾸는 과정에서 r+감마*Q(s,u) 사용으로 환경만 영향을 미치므로 과거 action에 영향을 받지않는다
     따라서 off policy 방식으로 학습을 하며 Replay buffer를 사용할수 있게된다.<br/><br/>



 - ## A2C 알고리즘
   - ![image](https://user-images.githubusercontent.com/60399060/146138922-687b1b5d-5da0-4912-b517-291bc4e84485.png)
   - actor network는 action을 선택하고 critic network는 Q(s,a) 를 평가 한다. 또한 A(Advantage function)을 사용하여 policy 네트워크의 분산을 줄여준다.
   - 크리틱 네트워크로 Advantage function을 추정하므로 얼마나 좋은 액션인지 뿐만아니라 얼마나 더 좋아질수 있는지도 고려한다.
   
   - ![image](https://user-images.githubusercontent.com/60399060/146139038-3a3b8890-7850-43d4-8e8c-52569167f674.png)
   - A2C는 주식 트레이딩환경에서 안정적이며 수렴이 빠르다.
   - 단점으로는 off policy 알고리즘에 비해 샘플 효율성이 낮고, 샘플간 상관관계가 높을수 있다.

  <br/><br/><br/><br/>

 
## 결론 
  -  ## 앙상블 이전 각 에이전트 학습<br/>


  - ** 앙상블의 효과를 확인 하기 위해 train set 에서 오버 피팅 시킨 에이전트를 사용 **
  - ** (과적합된 에이전트로도 유의미한 결과를 낼수있는지 확인하기 위함) **
 <br/><br/>
  - ![image](https://user-images.githubusercontent.com/60399060/147758677-6f82ea74-4610-41e1-9294-dcd4c4c49e85.png)
  - (Train data set 의 그래프)
  - 학습 데이터 셋의 시장 수익률 : 17.9672% 
  
 <br/><br/>
  
  - ![image](https://user-images.githubusercontent.com/60399060/147758732-c6fc8391-c796-4fdf-abad-cb504544f1b6.png)
  - 주황 그래프: Train set에서 학습시킨 A2C agent의 포트폴리오 가치(PV) (61.19% 수익)
  - 파랑 그래프: Train set에서 학습시칸 PPO agent의 포트폴리오 가치 (52.76% 수익)
  - 초록 그래프: Train set에서 학습시킨 RDPG agent의 포트폴리오 가치 (31.34% 수익)
  - 특이사항 : 이들은 모두 오버피팅 시키기 위해 가격 데이터 하나만으로 학습
<br/><br/>

  - ![image](https://user-images.githubusercontent.com/60399060/147759207-585757d8-14f5-4dd6-a31b-354ca86d0c67.png)
  - 검증 데이터 셋(70일)의 시장수익률: (8.12%)
  - 주황 그래프: Validation set에서 트레이딩 이후 A2C agent의 포트폴리오 가치  (6.67 % 수익)
  - 파랑 그래프: Validation set에서 트레이딩 이후 PPO agent의 포트폴리오 가치  (7.73% 수익)
  - 초록 그래프: Validation set에서 트레이딩 이후 RDPG agent의 포트폴리오 가치 (0% 수익)
  - 특이사항 : A2C 에이전트는 고점에서 관망하며 안정성을 추구했으며 PPO 에이전트는 공격적인 매매방식을,<br/>
              RDPG는 매매하지 않는 행동을 선택했다.
<br/><br/>  
  -  ## 앙상블 결과 <br/>
  -  ![image](https://user-images.githubusercontent.com/60399060/147761827-5294e01b-219d-4460-811c-a06c091d346a.png)
  -  앙상블 에이전트 수익률 : 9.46 %
  -  앙상블 에이전트 알파 수익률 : 1.33 %
  -  시장 수익률 : 8.12 %
  -  앙상블 에이전트는 오버피팅된 3개의 에이전트를 사용 했음에도 향상된 결과를 보인다.
  -  SOTA 에이전트들의 단점을 배제하고 장점이 반영 됐다. 
  -  (A2C의 안정성과 PPO의 수익률을 반영, RDPG의 매우 보수적인 매매방식은 거의 반영하지 않았다)
<br/><br/>  
  - 에이전트 선정 내역
  - ![image](https://user-images.githubusercontent.com/60399060/147762049-98bfbf1d-3c60-4905-aa9e-ea3ea876cbc8.png)
  - ![image](https://user-images.githubusercontent.com/60399060/147762175-bf62cc29-7bc2-4b06-947d-e11cd8811783.png)
  - 하락장 및 횡보구간(약40 ~ 60스탭) 에서 A2C 에이전트가 주로 선정되어 안정적인 트레이딩 전략을 채택했다.
  - 상승장 (0 ~ 40스탭 , 60 ~ 65스탭 ) 에서 PPO 에이전트가 주로 선정되어 수익률을 극대화 하는 결과를 보였다.

              


  <br/><br/><br/><br/>
 
  

## 문제점 및 개선방안
  - MDP(Markov decision process) 정의 문제
      - 현 프로젝트에서는 state= [price] 로 정의. 이는 노이즈가 심하고 비정상성을 보이는 데이터 이므로 오버피팅 가능성이 높다. 따라서 여러 팩터를 사용할 필요성이 있다.
      - 여러 팩터를 사용하면 팩터들 사이에서 다중공선성 문제가 생길 수 있으므로 Feature Extraction 방법으로 차원의 저주와 높은 상관계수 문제 해결 가능<br/><br/>

  - RDPG 알고리즘의 하이퍼 파라미터 튜닝 문제 
      - RDPG는 다른 RL알고리즘보다 파라미터 튜닝이 어려운 편이다.
      - Auto ML 방식 사용 (하이퍼 파라미터를 자동으로 튜닝)
      - continuous action space에서 효과적인 다른 SOTA 에이전트 사용 (TD3,SLAC,SAC,D4PG 등등)<br/><br/>

  - A2C 알고리즘의 낮은 샘플 효율성
      - on policy(behavior policy 와 target policy가 같음.) 학습을 하며 과거 샘플을 재사용 할수 없기 때문에 샘플 효율성이 낮다.
      - Replay buffer 를 사용하는 ACER(Actor Critic with Experience replay buffer) 를 사용할수 있다.<br/><br/>
  
  - 강화학습 에이전트들의 민감한 튜닝 문제
      - 데이터셋의 길이, 분포 , 표준편차 등이 조금만 바뀌어도 하이퍼 파라미터가 수정 돼야한다.
      - 앙상블을 진행 할 경우 더많은 하이퍼 파라미터들이 추가되는데, 학습 할때마다 이들을 튜닝 해야 한다.
      - Auto ML 방식을 사용하여 하이퍼 파라미터로 인한 과소,과대 적합 문제를 어느정도 완화

  - 시장은 t시점에서 알파를 찾아도 향후 새로운 알파가 생겨난다.
      - 단일 에이전트 보다는 유동적으로 전략을 찾을수 있지만 현 앙상블 에이전트에서도 여전히 전략간 상관계수와 편향이 있다.
      - 따라서 더많은 에이전트를 앙상블하거나 MARL(Multi-Agent-Reinforcement-learning) 사용 
      - 각 에이전트가 알고리즘 자체를 스스로 개선하도록 하여 여러 에이전트들의 전략간 상관계수와 편향을 낮출 수 있다.



  <br/><br/><br/><br/>
  
## 참고 
  -  LSTM-DDPG for Trading with Variable Positions [RDPG 논문]
  -  Learning to Trade with Deep Actor Critic Methods [A2C 논문]
  -  CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING [DDPG 논문]
  -  SAMPLE EFFICIENT ACTOR-CRITIC WITH EXPERIENCE REPLAY [ACER 논문]
  -  Proximal Policy Optimization Algorithms [PPO 논문]
  -  Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy [강화학습의 앙상블 논문] 
 




