---
layout: post
title:  "OpenSSL Explained for Kubernetes"
date:   2020-10-02 01:00:00
categories: "kubernetes"
asset_path: /assets/images/
tags: ['kubernetes', 'openssl', 'ssl', 'tsa', 'security', '보안', 'ca']
---

# 1. Symmetric Cryptography 

## 1.1 Explained

Symmetric Encription은 동일한 키를 사용해서 encrypt 그리고 decrypt 를 하게 됩니다. 

<img src="{{ page.asset_path }}ssl_symmetric_encryption.png" class="img-responsive img-rounded img-fluid center">

 - 일반적으로 여러개의 crypto algorithms 을 사용해서 symmetric encryption 을 만듭니다. (ex. AES-256-CBC -PBKDF2)

## 1.2 Generate Key and Data

Symmetric encryption algorithm 중의 하나의 `aes-256-cbc` 사용해서 encrypt 를 했습니다. <br>

먼저 데이터를 생성 합니다.  

{% highlight bash %}
# 데이터 넣기
$ echo "Hello Anderson!" > data.txt
{% endhighlight %}

Symmetric Key 를 생성합니다.

{% highlight bash %}
# Symmetric Key 생성
$ openssl rand 128 > symmetric.key

# Key값 확인
$ cat symmetric.key
Z�vfh���*"��� S3�&�n=�`+�is@����ȡ˫<���fF��I|�S<생략>
{% endhighlight %}


## 1.3 Encryption

**Encryption**은 다음과 같이 합니다. 

 - `-k symmetric.key`: 해당 옵션을 빼면 암호를 넣으라고 나올겁니다. 

{% highlight bash %}
# Encrypt (데이터 + 패스워드 -> encrypted.txt)
$ openssl aes-256-cbc -pbkdf2 -base64 -in data.txt -out encrypted.txt -k symmetric.key

# 데이터 확인 (-base64 형식으로 저장됨)
$ cat encrypted.txt 
U2FsdGVkX1+K/AX4sYGnasTiOmW6gLxDw4APT/3B9OVFCHizUYLe1Bbo0gPcHrYL
{% endhighlight %}

## 1.4 Decryption

**Decryption**는 다음과 같이 합니다. 

 - `-k symmetric.key` 를 빼고 encrypt했다면, descrypt할때도 빼야 합니다.

{% highlight bash %}
# Decrypt (동일하게 -pbkdf2 -base64 사용)
$ openssl aes-256-cbc -d -pbkdf2 -base64 -in encrypted.txt  -out descrypted.txt -k symmetric.key

# 데이터 확인
$ cat descrypted.txt 
Hello Anderson!
{% endhighlight %}


# 2. Asymmetric Cryptography (Public Key)

## 2.1 Explained

<img src="{{ page.asset_path }}ssl_asymmetric_encryption.png" class="img-responsive img-rounded img-fluid center">

 - **Public Key**: Encrypt 하는 데 사용 
 - **Private Key**: Decrypt 하는 데 사용 

## 2.2 Generate Private Key 

먼저 RSA 알고리즘을 사용해서 Private Key 를 생성합니다. 

 -  `-aes256`, `-aria256`, `-des`, `-des3` 등의 옵션으로 private key를 다시한번 특정 cipher로 encrypt합니다.

{% highlight bash %}
$ openssl genrsa -out private.key 2048

$ cat private.key
-----BEGIN RSA PRIVATE KEY-----
MIIEpQIBAAKCAQEA5z95XeHLsbiaaDsM+7eHqL8wtNBOrDlge47Y/pSi1g6xWuC2
u+AawKhFluJHbN4yQMS3PMQGNjVZVRCUwwSacmLXEQD/ekWoZ2RBNdmhPlg+u0kd
6iEzLIFJy3jqW324lAcY9EfC/5Rdk0y3Yc184fQK4T/h0NZ5gOwJnL3XI4uv7rhR
<많은 생략>
84u58DUCgYEAk8It6g7hGgBqBeYY7pl8dfR1d9futVwqoP4mxmVUvua4jBQfFok3
Vxzj1fKLe+srTpI5srhhBDI1sYtVsaFuaXKaP1QIHlPqB2sF5AEnEbOX5w4pS7TR
zzxqPaelCfYudRjfw/Ma5cJWu50dW30JdLcapY/8wA+F1GtJaKlFD/o=
-----END RSA PRIVATE KEY-----
{% endhighlight %}

이후 private.key를 갖고서 public key를 생성합니다. 

## 2.3 Extract Public Key 

{% highlight bash %}
$ openssl rsa -in private.key -pubout -out public.key

$ cat public.key 
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA19I1QgwX1QLmilD9odP3
dwYIX66n4pEMXXsUM5zVYVhKCo4wMeXWIE8auu3IZ+P23/rTUxY3fogesvH5dfaO
<짧은 생략>
rK29doGZeDHdqyIBqi5rzxOu8VtRWAL91+VaaYtTKhzZE3fI1VPsESayKUtKc+fX
BwIDAQAB
-----END PUBLIC KEY-----

{% endhighlight %}


## 2.4 Small File

먼저 데이터 **Encrypt**는 다음과 같이 합니다.

{% highlight bash %}
# 데이터 생성 
$ echo "Hello Anderson!" > data.txt

# Encrypt data.txt 
$ openssl rsautl -encrypt -pubin -inkey public.key -in data.txt -out encrypted.txt

# encrypted.txt
$ cat encrypted.txt
�2�M-/L��pw]%M�U|*���EL�#�~'&Ƃ+X�,��%*�U��>�����}�Q�����ߙ��ct���"CA�����cئ0C�w�A�"y��Ғ�gȏ�Y��`�g�SIN�����I.\
{% endhighlight %}


**Decrypt**는 다음과 같이 합니다.

{% highlight bash %}
$ openssl rsautl -decrypt -inkey private.key -in encrypted.txt -out decrypted.txt
$ cat decrypted.txt 
Hello Anderson!
{% endhighlight %}

## 2.5  Large File 

Asymmetric cryptography로 RSA 를 사용하였고, RSA의 문제는 116bytes 까지의 데이터만 encrypt 할 수 있습니다. 

1. `openssl rand 128` 등의 명령어로 먼저 Symmetric Key를 생성 파일은 116 bytes를 넘을 수 없습니다.


# 3. Certificate Signing Request (CSR)

## 3.1 SSL/TLS Explained 

HTTPS 모르는 사람은 없을 것이고, HTTP인데 그냥 SSL/TLS 기술로 data encryption이 들어간 protocol이라고 생각하면 됩니다. <br>
SSL(Secure Socket Layer)은 90년대 넷스케이프에서 개발되었으며 현재는 deprecation 되었으며,<br> 
현재는 TLS(Transport Layer Security)를 주로 사용합니다.<br>

TLS Handshake 구조를 보면 다음과 같습니다. 

<img src="{{ page.asset_path }}openssl-tls-ssl-handshake.png" class="img-responsive img-rounded img-fluid center">

1. **Client Hello**: 클라쪽 SSL버젼 정보, Cipher Suite list (지원하는 암호화 방식), 무작위 바이트 문자열 -> 서버로 보냄 
2. **Server Hello**: 암호화 방법 선택 이후 SSL Certificate, 무작위 바이트 문자열 -> 클라로 보냄
    - 서버가 보낸 SSL Certificate에는 서버측 public key, 그리고 서비스 정보를 담고 있다
3. **CA에서 인증**: 클라는 서버에서 SSL Certificate을 받았지만, 신뢰할수 있는지 확인하기 위해서 CA에서 확인을 하게 됨
    - CA: certificate authority 로서 GeoTrust, IdenTrust를 의미 
    - 클라는 서버가 전달해준 certificate을 CA로 보냄
    - certificate에서 public key를 꺼내고 CA의 private key를 사용해서 encrypted data를 decrypt함
    - decrypt가 잘됐다면 CA에서 인증한 certificate이기 때문에 신뢰함
4. **Premaster Secret**: 클라는 서버측 certificate에 있는 public key를 갖고서 premaster secret을 만들고 -> 서보로 보냄
5. **Private Key Used**: 서버는 premaster secret을 서버측 private key 로 decrypt 한다
6. **Session Key Created**: 


**SSL Certificate**은 CA (Certificate Authority. 예 GeoTrust)로 부터 발급됩니다. 


## 3.2 Generate CSR 

**CSR**은 Certificate Signing Request의 약자로서 "인증서 서명 요청"이라는 뜻으로, <br> 
SSL Certificate을 CA로 부터 얻을때 사용합니다. 즉.. SSL Certificate 구매할때 반드시 만들어야 함. <br>


{% highlight bash %}
$ openssl req -new -newkey rsa:2048 -nodes -keyout private.key -out cert.csr 
Generating a RSA private key
.............................................+++++
Country Name (2 letter code) [AU]:KR
State or Province Name (full name) [Some-State]:Gyeonggi-do
Locality Name (eg, city) []:Goyang-si
Organization Name (eg, company) [Internet Widgits Pty Ltd]:Anderson
Organizational Unit Name (eg, section) []:R&D
Common Name (e.g. server FQDN or YOUR name) []:incredible.ai
Email Address []:a141890@gmail.com
{% endhighlight %}


**잘 생성 됐는지 확인**<br>
cert.csr 파일안에 어떤 내용이 들어있는지 확인하기 위해서 다음의 명령어를 사용합니다.<br>
국가, 주소, 이름, 회사, 도메인, 이메일 등등 정확하게 기입되었는지 확인합니다. 

{% highlight bash %}
$ openssl req -in cert.csr -noout -text
{% endhighlight %}

**기존 private key로 CSR생성**

{% highlight bash %}
$ openssl req -new -out cert.csr -key private.key
{% endhighlight %}

## 3.3 Generate Self-Signed Certificate 

Self-signed certificate은 보통 내부망 또는 개발환경에서 테스트시에 사용이 됩니다. <br>
다음의 명령어로 self-signed certificate을 생성할 수 있습니다. 

 - `-x509`: self-signed certificate 이라고 알림
 - `-nodes`: no des 라는 뜻으로 암호화 하지 않겠다는 것. 이거 빼면.. 생성할때 암호 쓰라고 나옴. 
 - `-new`: 국가, 이름, 회사, 도메인, 이메일 등등의 물어보는 prompt가 나오면서 새롭게 생성하게 됨
 - `cert.key`: Certificate
 - `private.key`: private key
 
{% highlight bash %}
$ openssl req -x509 -newkey rsa:2048 -nodes -keyout private.key -out cert.key
-----
Country Name (2 letter code) [AU]:KR
State or Province Name (full name) [Some-State]:Gyeonggi-do
Locality Name (eg, city) []:Goyang-si
Organization Name (eg, company) [Internet Widgits Pty Ltd]:Anderson
Organizational Unit Name (eg, section) []:R&D
Common Name (e.g. server FQDN or YOUR name) []:incredible.ai
Email Address []:a141890@gmail.com
{% endhighlight %}

**기존의 private key 그리고 CSR로 부터 Self-Signed Certificate 생성**

{% highlight bash %}
$ openssl x509 -signkey private.key -in cert.key -req  -out cert.crt
{% endhighlight %}

## 3.4 Verify

**CSR**

{% highlight bash %}
$ openssl req -text -noout -verify -in cert.csr
verify OK
{% endhighlight %}

**Private Key**

{% highlight bash %}
$ openssl rsa -in private.key -check
RSA key ok
{% endhighlight %}

**SSL Certificate**

{% highlight bash %}
$ openssl x509 -in cert.key -text -noout
{% endhighlight %}

**SSL Certificate 그리고 Private Key 가 서로 일치하는지 확인**

{% highlight bash %}
$ openssl x509 -noout -modulus -in cert.key | openssl md5
(stdin)= d31b4f5b1a438bb878b847062abc3e26

$ openssl rsa -noout -modulus -in private.key | openssl md5
(stdin)= d31b4f5b1a438bb878b847062abc3e26
{% endhighlight %}

# 4. Self-Signed Certificate in Kubernetes 

## 4.1 Generate CA 

CA 생성에는 두가지 방법이 있습니다. 

1. EKS에서 이미 만들어져 있는 것을 가져온다
2. 새로 생성

**EKS 에서 Private Key** 가져오는 방법은 다음과 같습니다.<br>
먼저 EKS -> Cluster -> Certificate Authority 를 복사해서 <Certificate-Authority> 부분을 교체합니다.

{% highlight bash %}
$ echo <Certificate-Authority> | base64 -d > ca.key
$ cat ca.key
-----BEGIN CERTIFICATE-----
MIICyDCCAbCgAwIBAgIBADANBgkqhkiG9w0BAQsFADAVMRMwEQYDVQQDEwprdWJl
cm5ldGVzMB4XDTIwMTAxNTE4MjczMFoXDTMwMTAxMzE4MjczMFowFTETMBEGA1UE
<생략>
3s8dl/pTqzbMwJtyRLnUHAcoYqYJiICkOWzIqSjuiNPMbSw4bWIKAc4ItQWukiGy
BcDZMt1MvWH2csacIJHMmgyOsIRqI4XyNJRr34DwMUMkMDfTrNQ8mNUPRcY=
-----END CERTIFICATE-----
{% endhighlight %}


**CA Key**를 새로 생성하는 방법은 다음과 같습니다.

- MASTER_IP: EKS Cluster에서 API Server Endpoint 를 찾아서 넣습니다. 이때 https는 뺍니다.

{% highlight bash %}
$ export MASTER_IP=EAA012E181CFAF7CDEB872D695DBF864.gr7.us-east-2.eks.amazonaws.com
$ openssl genrsa -out private.key 2048
$ openssl req -x509 -new -nodes -key ca.key -subj "/CN=${MASTER_IP}" -days 10000 -out cert.key
{% endhighlight %}

**Server Key**를 발급합니다.

{% highlight bash %}
$ openssl genrsa -out server.key 2048
{% endhighlight %}

**CSR (Certificate Signing Request)**를 생성하기 위해서 config 파일을 생성합니다. <br>

# Abbreviations 

 - CSR: Certificate Signing Request
 - DER: Distinguished Encoding Rules
 - PEM: Privacy Enhanced Mail
 - PKCS: Public-Key Cryptography Standards
 - SHA: Secure Hash Algorithm
 - SSL: Secure Socket Layer
 - TLS: Transport Layer Security


# References 

1. [Encryption: Symmetric and Asymmetric](https://cryptobook.nakov.com/encryption-symmetric-and-asymmetric)
2. [encrypt_openssl.md](https://gist.github.com/dreikanter/c7e85598664901afae03fedff308736b)
3. [파일 확장자 설명](https://serverfault.com/questions/9708/what-is-a-pem-file-and-how-does-it-differ-from-other-openssl-generated-key-file)
4. [What happens in a TLS Handshake?](https://www.cloudflare.com/learning/ssl/what-happens-in-a-tls-handshake/)