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


## Message Digest

#### MD5 

MD5, SHA-256, SHA-512처럼 One-way Hash Function을 말합니다.<br>
즉.. arbitrary sized data를 받고 일정한 크기의 (fixed-length)를 갖은 Hash Value를 내놓는 알고리즘들을 말합니다.

{% highlight python %}
from hashlib import sha512
sha = sha512()
sha.update('password!')
sha.hexdigest()
# '35fddd95911c6e8ced55b83b4dddcb9a06a2af8471d412ad7427878369d3d18bb7ac00ec561a41d819596540d2edf600df56376c906f49ff5c93d04c0c42546d'
{% endhighlight %}

## Shared-Key Encryption (Symmetric)

간단하게.. Sender와 Receiver사이에 하나의 Key 를 서로 공유합니다. 해당 Key를 이용해서 Message를 Encrypt 또는 Decrypt 할수 있습니다.
Symmetric Key는 매우 간단하고 단순하지만 문제는 Shared Key를 서로 Private으로 전달할 방법이 있어야 한다는 것입니다.
이 문제를 극복한 것이 Public Key Encryption입니다. (Public Key Encryption은 Public Key를 non-secure way로 내보내지만 Private Key는 Trasmit 되지 않습니다.)

* 동일한 키가 Sender 와 Receiver에게서 사용이 됩니다.
* Public Key Encryption 에 비해서 더 빠릅니다.
* 키가 안전하게 보관이 되어야만 합니다.
* Twofish, Serpent, AES, Blowfish, CAST5, RC4, 3DES, SEED


먼저 설치..

{% highlight bash %}
sudo apt-get install libffi-dev
sudo pip install cryptography
{% endhighlight %}

#### Fernet (Symmetric Encryption)

cryptography 라이브러리에서 제공하는 Symmetric Encryption 툴.. 

{% highlight python %}
from cryptography.fernet import Fernet
key = Fernet.generate_key() # 'sxir3dIM7NtsyUPsLkau6pPr4whGPgpx1eHBmDA-hQw='
f = Fernet(key)
token = f.encrypt(b"My Secret Weapon")
f.decrypt(token) # 'My Secret Weapon'
{% endhighlight %}

#### AES (Symmetric Encryption)


{% highlight python%}
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
backend = default_backend()
key = os.urandom(32)
iv = os.urandom(16)
cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
encryptor = cipher.encryptor()
ct = encryptor.update(b"a secret message") + encryptor.finalize()
decryptor = cipher.decryptor()
decryptor.update(ct) + decryptor.finalize()
# 'a secret message'
{% endhighlight %}

## Public Key Encryption

* Public Key 와 Private Key를 두개를 사용합니다.
* 제3자가 Public Key를 갖었다 하더라도, Private키 없이는 Decrypt할 수 없습니다.
* Encryption은 Public Key만 있어도 할 수 있습니다.
* Symmetric Encryption에 비해서 느립니다.

{% highlight python %}
from Crypto.PublicKey import RSA
from Crypto import Random
random_generator = Random.new().read
key = RSA.generate(1024, random_generator)
# <_RSAobj @0x7f60cf1b57e8 n(1024),e,d,p,q,u,private>

public_key = key.publickey()
enc_data = public_key.encrypt('abcdefgh', 32)
# ('\x11\x86\x8b\xfa\x82\xdf\xe3sN ~@\xdbP\x85
# \x93\xe6\xb9\xe9\x95I\xa7\xadQ\x08\xe5\xc8$9\x81K\xa0\xb5\xee\x1e\xb5r
# \x9bH)\xd8\xeb\x03\xf3\x86\xb5\x03\xfd\x97\xe6%\x9e\xf7\x11=\xa1Y<\xdc
# \x94\xf0\x7f7@\x9c\x02suc\xcc\xc2j\x0c\xce\x92\x8d\xdc\x00uL\xd6.
# \x84~/\xed\xd7\xc5\xbe\xd2\x98\xec\xe4\xda\xd1L\rM`\x88\x13V\xe1M\n X
# \xce\x13 \xaf\x10|\x80\x0e\x14\xbc\x14\x1ec\xf6Rs\xbb\x93\x06\xbe',)

key.decrypt(enc_data)
# 'abcdefgh'


{% endhighlight %}


[git-gpu-cracker]: https://github.com/AndersonJo/MD5-GPU-Cracker