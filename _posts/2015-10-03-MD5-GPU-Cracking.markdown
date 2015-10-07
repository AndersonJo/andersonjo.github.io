---
layout: post
title:  "Cracking MD5 with GPU"
date:   2015-10-03 01:00:00
categories: "security"
asset_path: /assets/posts/MD5-GPU-Crack/
tags: ['nvcc', 'Nsight', 'makefile', 'Python Wrapper']
---
<div>
    <img src="{{ page.asset_path }}password.jpg" class="img-responsive img-rounded">
</div>

## MD5

MD5는 One-way hash function 으로서 Decrypt는 할수 없습니다.<br> 
즉 hash을 하면서 original data는 손실이 된다고 생각하면 됩니다.
MD5는 128 bits long 이고.. 즉.. 2^128개의 해쉬가 존재한다고 볼 수 있습니다. (340282366920938463463374607431768211456개)<br>
스트링으로는 16진수의 32개의 글자로 출력됩니다. (16^32 == 2^128) 


#### Cracking MD5 with Nvidia GPU

[Git Repository for MD5 GPU Cracker][git-gpu-cracker]

앞서 말했듯이 One-way hash 이기 때문에 어떤 알고리즘으로 다시 해독하는건 힘들고, Brutal Force 로 크래킹하는게 좋습니다. 
2가지 방법이 있는데 큰 Hash Table을 만들고 찾는 방법과, 또는 GPU로 돌려서 알아내는 방법이 있습니다. 
전문적으로 크래킹해주는 웹싸이트들이 있는데 그런곳은 미리 만들어놓은 테이블에서 찾는 방법이고, 
우리가 할 방법은 GPU로 뚫는 방법입니다.

물론 MD5를 실무에서 패스워드로 사용하는 경우는 매우 드뭅니다. 하지만 MD5나 SHA나 다른 해쉬 펑션들도 같은 방법으로 크래킹이 가능하기 때문에, 
MD5로 예를 들어보겠습니다.

위의 [Git Repository for MD5 GPU Cracker][git-gpu-cracker] 링크는 제가 집접 만든 Python + CUDA C 언어로 만든 
크래킹 툴입니다.

먼저 파이썬으로 MD5 해쉬키를 만들겠습니다. 

{% highlight python %}
m = md5()
m.update('ab14%P')
m.hexdigest()
# 'fe5f329d483283b7d03b03fc1e48e90c'
{% endhighlight %}





6자리 영어 대소문자 + 숫자 + 기호.. 그냥 아스키 코드에서 잘쓰는거 대부분다. 조합으로 찾아냅니다.

저위의 'ab14%P' 를 찾는데 대략 20분정도 걸리는듯 합니다. 

2틀동안 CUDA 짜느라.. 고생했네요. ;;

지금 자야되서.. ㅡ .ㅡ; 시간되면 Python라이브러리로 만들어서 올리겠습니다. 

{% highlight bash %}
Progress: aaan[x
Progress: aabqDp
...
Progress: abPIM3
Progress: abQK�*
ANSWER: ab14%P
{% endhighlight %}



## NVCC Compile

개발과정중에 얻은 지식을 좀 공유하도록 하겠습니다. 


#### Build Python Wrapper 

Shared Library (.so파일) 을 다음의 명령어로 얻을수 있습니다.

{% highlight bash %}
python setup.py build_ext --inplace
{% endhighlight %}

또는 nvcc를 이용한다면..

{% highlight bash %}
nvcc wrapper.cu -g -optf opt -o md5.so
{% endhighlight %}

#### Integrating Python with Nsight

Nsight에서 Python Wrapper를 nvcc 로 build하기 위해서는 다음과 같은 코드를 환경설정 파일을 저장하고.. 

**Project >> Properties >> Build >> Settings >> Tool Settings >> Miscellaneous <br>
Command Line Option File (-optf)의 위치를 적어주면 됩니다.

**Option File**

{% highlight text %}
-I/usr/local/lib/python2.7/dist-packages/numpy/core/include 
-I/usr/local/cuda-7.5/include 
-I/usr/include/python2.7 
-I/usr/include
-arch=sm_20 
-shared
--ptxas-options=-v 
--compiler-options '-fPIC'
{% endhighlight %}

|Option|Desc|
|:--|:--|
| -I | to include Python Header Files |
| -L | to include Python Binary Library Files |
| -g | Debug |
| -shared | Shared File |

**중요한점은 shared일 경우에는 (.so파일) -c 옵션을 빼야한다.** 



## Python Wrapper 2.7

#### Initial Function

{% highlight c %}
PyMODINIT_FUNC init<name>(void) {
	PyObject *m;

	m = Py_InitModule("<name>", md5_gpu_crack_methods);
	if (m == NULL)
		return;
}
{% endhighlight %}

함수 이름이 매우 중요한데, init<name> 이 부분이 -o &lt;name&gt;.so 하고 동일해야 합니다.<br>
또한 Py_InitModule("&lt;name&gt;", md5_gpu_crack_methods); 코드에서도 &lt;name&gt;이 동일해야 합니다.

#### Registration Table

{% highlight c %}
static PyMethodDef md5_gpu_crack_methods[] = {
	{ "crack", md5_gpu_crack_wrapper, METH_VARARGS, "MD5 GPU Cracking" },
	{NULL, NULL, 0, NULL}
};
{% endhighlight %}

순서대로.. <br> 
Name, &function, fmt, doc<br>
{NULL, NULL, 0, NULL} <- End of table marker


#### Function : METH_VARARGS

Arguments 를 받는 형태의 Function형태이고, PyArg_ParseTuple로 Convert를 합니다.

{% highlight c %}
static PyObject * md5_gpu_crack_wrapper(PyObject * self, PyObject * args) {
	char * input;

	if (!PyArg_ParseTuple(args, "s", &input)) {
		return NULL;
	}
	return Py_None;
}
{% endhighlight %}

#### Function : METH_KEYWORDS | METH_KEYWORDS

{% highlight c %}
static PyObject * md5_gpu_crack_wrapper(PyObject * self, PyObject * args, PyObject * keywds) {
	static char* kwlist[] = {"hash", "digit", "string" };
	char *hash;
	char* string = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\0";
	int digit = 4;
	printf("Hello md5_gpu_crack_wrapper\n");
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "s|is", kwlist, &hash, &digit, &string)) {
		return NULL;
	}
	Py_INCREF(Py_None);
	return Py_None;
}
{% endhighlight %}

| Variable | Desc | 
|:--|:--|
| kwlist[] = { "keyname" } | kwargs의 key names | 


[git-gpu-cracker]: https://github.com/AndersonJo/MD5-GPU-Cracker