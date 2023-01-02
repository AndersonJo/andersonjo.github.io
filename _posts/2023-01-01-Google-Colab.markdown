---
layout: post
title:  "Google Colab"
date:   2023-01-01 01:00:00
categories: "google-colab"
asset_path: /assets/images/
tags: []
---


# 1. Google Colab 

## 1.1 Session Timeout 우회하기

다음을 실행시켜서 주기적으로 자동으로 버튼을 누르게 만듭니다. 

```javascript
function ClickConnect(){
console.log("Working"); 
document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click();
}
var clicker = setInterval(ClickConnect,60000);
```

끌때는 다음과 같이 합니다.

```javascript
clearInterval(clicker);
```

## 1.2 Mounting Google Drive to Colab

Google Colab notebook이 google drive에 access 할 수 있도록 만듭니다.<br>
Google Colab의 특정 위치에 실제로 구글 드라이브를 USB 마운트 시키듯이 올리는 것 입니다.

튜토리얼 시작전, Google Drive에서 /data 디렉토리를 먼저 만들고 시작합니다.<br> 
drive.mount("/data") 라는 뜻은 현재 구글 코랩 서버에서 /data 디렉토리에 마운트 시키겠다는 뜻이지, 구글 드라이브의 위치를 가르키는게 아닙니다.

```python
from google.colab import drive
drive.mount('/data')
```


## 1.3 Kaggle API

구글 드라이브에 이미 kaggle.json 파일이 존재해야 합니다. <br> 
해당 파일은 Kaggle에서 secret으로 다운로드 받을 수 있습니다.<br>

```python
from google.colab import drive
drive.mount('/data')

!pip install kaggle -q
!mkdir -p ~/.kaggle
!cp /data/MyDrive/data/secrets/kaggle.json ~/.kaggle/
!chmod 400 ~/.kaggle/kaggle.json
```


## 1.4 Loading Kaggle Dataset

아래는 movielens 예제 입니다. 

```python
import pandas as pd
import kaggle.api as kaggle
from tempfile import gettempdir
from pathlib import Path

data_dir = Path(gettempdir()) / 'movielens'

kaggle.authenticate()
kaggle.dataset_download_files('grouplens/movielens-20m-dataset',
                              data_dir, 
                              unzip=True)
```