데이터 증강에 대한 조사 및 적용 사례 연구
------------------------------------------ 
데이터 증강에 대해 조사한 뒤, 이를 직접 코드에 적용해보았습니다.

## How to Run
1. https://www.kaggle.com/jsrojas/ip-network-traffic-flows-labeled-with-87-apps  
위 사이트에서 csv 파일을 다운로드 받은 후, dataset 폴더 안에 넣습니다.  

2. preprocess.py를 실행하여 dataset을 preprocessing 합니다  
``` python
python preprocess.py
```  

3. train.py를 실행하여 모델을 학습합니다.  
아래 코드를 참고하여 원하는 옵션으로 실행할 수 있습니다.  
``` python
python train.py 
```

데이터 셋은 1000개를 사용합니다.  
데이터 셋의 크기를 늘리고 싶을 때는 --max_len으로 길이를 설정할 수 있는데,  
-1의 값을 넣어준다면 전체 데이터 셋을 사용할 수 있습니다.  

## Options  
``` python
### Mixup, make 200 images
python train.py --augmentation Mixup --new_data_len 200
### Delete with 20% probability, make 200 images
python train.py --augmentation Delete --proability 0.2 --new_data_len 200
### Modify with 20% probability, make 200 images
python train.py --augmentation Modify --proability 0.2 --new_data_len 200
```