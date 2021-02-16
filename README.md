# Samsumg-research

https://www.kaggle.com/jsrojas/ip-network-traffic-flows-labeled-with-87-apps
위 사이트에서 csv 파일을 다운로드 받은 후, dataset 폴더 안에 넣으세요.

그 후 python preprocess.py를 실행하여 dataset을 preprocessing 합니다

### Stochastic Gradient Descent 
python train.py --batch_size 1
### Batch Gradient Descent
python train.py --batch_size 7000
### Mini-Batch Gradient Descent
python train.py --batch_size 100

### momentum
python train.py --optimizer momentum
### NAG
python train.py --optimizer NAG
### Adagrad
python train.py --optimizer Adagrad
### RMSProp
python train.py --optimizer RMSProp --lr 0.001
### Adam
python train.py --optimizer Adam --lr 0.001

데이터 셋 중 10000개 사용. 전체 사용하고 싶을 시 
python train.py --max_len -1