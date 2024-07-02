---
layout: post
title:  "LightGBM Java"
date:   2024-04-13 01:00:00
categories: "machine-learning"
asset_path: /assets/images/
tags: ['java']
---


# 1. Installation

```
# Mac
brew install libomp

# Debian Linux
sudo apt install libgomp1
```

# 2. LightGBM

## 2.1 Python 

Python code에서는 학습 그리고 evaluation을 작성하고, 이후에 txt 파일로 모델을 저장하는 것 까지 보여줍니다. <br>
즉 Java 에서는 prediction관련해서만 콬드를 보여줍니다. 


```python
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score


scale_pos_weight = sum(y_train == 0) / len(y_train)

model = LGBMClassifier(
    metrics="prauc",
    n_estimators=100,
    scale_pos_weight=scale_pos_weight,
    random_state=32
)
model.fit(x_train, y_train)

# 예측
y_prob = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]
```


txt 파일로 저장합니다. <br> 
텍스트 파일로 저장시 실제로 인간이 이해할수 있는 txt 정보로 저장이 됩니다.

```python
model.booster_.save_model("model.txt")
```



## 2.2 LightGBM Java

**build.gradle**

lightgbm4j 가 필요합니다. 

```bash
dependencies {
    implementation 'io.github.metarank:lightgbm4j:4.3.0-1'
    implementation group: 'com.google.guava', name: 'guava', version: '11.0.2'
}
```


**Example**

```java
package ai.incredible.lightgbm;

import com.google.common.base.Charsets;
import com.google.common.io.Files;
import com.microsoft.ml.lightgbm.PredictionType;
import io.github.metarank.lightgbm4j.LGBMBooster;

import java.io.File;
import java.util.Arrays;

public class Main {
	static private final String MODEL_PATH = "/home/anderson/Desktop/model.txt";
	static private final String DATA_PARQUET_DIR = "/tmp/lightgbm4j/";

	@lombok.SneakyThrows
	public static void main(String[] args) {
		File file = new File(MODEL_PATH);
		String modelContent;
		modelContent = Files.toString(file, Charsets.UTF_8);
		LGBMBooster model = LGBMBooster.loadModelFromString(modelContent);
		System.out.println(model);

		float[] input =
			new float[] { 0.700720f, 1.287160f, -2.085664f, -0.004941f, 0.249742f, -0.323739f,
				-1.946551f, 1.496363f };

		double[] pred = model.predictForMat(input, 1, 8, true,
			PredictionType.C_API_PREDICT_NORMAL);
		System.out.println(Arrays.toString(pred));
	}
}
```
