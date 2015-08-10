---
layout: post
title:  "How to choose Block and Thread dimensions in CUDA"
date:   2015-08-10 02:00:00
categories: "cuda"
asset_path: /assets/posts/How-To-Choose-Block-And-Thread/

---

Neural Network, Image Recognition, Voice Recognition 등등 <br>
게임 이외의 다양한 분야에서 GPU의 활용은 어느때보다도 핵인기? 인듯 합니다.

오늘 다룰 내용은 아주 기본적인 CUDA연산처리입니다. <br>
그리고 Block과 Thread의 dimension size를 어떻게 처리할것인가 입니다.

알다시피 그래픽 카드의 종류, 버젼에 따라서 Block 과 Thread의 크기는 각각 다릅니다.<br>
중요한것은 어떻게 하면 기기가 다름에도 불구하고 유연하게 처리를 하는 방법입니다.

우리가 다룰 예제에서는 두개의 array를 GPU에서 더하기 연산을 하는 간단한 예제 입니다.


### Code Download

[test.cu][code]


### Example...

초기화 ... N값은 array의 길이 입니다.
{% highlight c %}
#include <stdio.h>
#define N (1024*33)
{% endhighlight %}


먼저 main 함수안에서 다음의 변수들을 declarations 해줍니다.<br>
a, b, c variables들은 host에서 사용되며 dev_a, dev_b, dev_c는 device에서 사용되는 변수들입니다.<br>
이후에 a와 b를 초기화합니다.

> 실제로는 a, b가 host에서 필요없습니다. 하지만 예제에서 보여주는 것은
> Python이든  Java이든 하드디스크 또는 메모리 상에 있는 Data를 host에 불러와졌다고 가정하고 있습니다.

{% highlight c %}
int a[N];
int b[N];
int c[N];
int *dev_a;
int *dev_b;
int *dev_c;

for(int i=0; i<N; i++){
    a[i] = i;
    b[i] = i*2;
}
{% endhighlight %}


### Memory Allocations and Data Trasfers
다음으로 Device에다가 Memory Allocations을 해줍니다.<br>
그리고 device로 데이터를 이전시킵니다.
{% highlight c %}
cudaMalloc((void**)&dev_a, sizeof(int)*N);
cudaMalloc((void**)&dev_b, sizeof(int)*N);
cudaMalloc((void**)&dev_c, sizeof(int)*N);

cudaMemcpy(dev_a, a, sizeof(int)*N, cudaMemcpyHostToDevice);
cudaMemcpy(dev_b, b, sizeof(int)*N, cudaMemcpyHostToDevice);
{% endhighlight %}



### Run!
device에서 실행시킬 add함수를 실행시킵니다.<br>
포인트는 device갯수, thread갯수를 마음대로 정해도 결과는 동일하게 나오도록 add함수가 만들어졌습니다.

{% highlight c %}
add<<<128, 128>>>(dev_a, dev_b, dev_c);
{% endhighlight %}

add함수는 다음과 같습니다.

{% highlight c %}
__global__ void add(int *a, int *b, int *result){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while(tid<N){
		result[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}
{% endhighlight %}

여기서 몇가지 용어를 정의하자면 다음과 같습니다.

| Name | Description | Limitation |
|:-----|:-----|
| gridDim | The number of blocks in a grid | 65535 |
| blockDim | The number of threads in a block | 512, 1024 |
| blockIdx | Block Index | |
| threadIdx | Thread Index | |

중요 포인트는 tid += blockDim.x * gridDim.x 인데.. 즉.. <br>
블럭갯수 * 쓰레드갯수를 통해서 전체 쓰레드 갯수를 구하는 것입니다.


### Result
결론적으로 add<<<x, y>>>> 에서 x값, y값 어떻게 넣어도 (Hardware limitaions를 넘지 않는 선에서..)
동일한 결과값을 얻을수 있습니다.

{% highlight c %}
cudaMemcpy(c, dev_c, sizeof(int)*N, cudaMemcpyDeviceToHost);

double result =0;
for(int i=0; i<N; i++){
    result += c[i];
}
printf("result: %f", result);

cudaFree(dev_a);
cudaFree(dev_b);
cudaFree(dev_c);
{% endhighlight %}


[code]: {{page.asset_path}}test.cu