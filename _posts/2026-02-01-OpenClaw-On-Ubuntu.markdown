---
layout: post
title:  "OpenClaw on Ubuntu + Docker Isolation"
date:   2026-02-01 01:00:00
categories: "openclaw"
asset_path: /assets/images/
tags: []
---

# 1. Installation

```bash
$ git clone git@github.com:openclaw/openclaw.git 

# go to openclaw directory
$ ./docker-setup.sh
```

Change docker-compose.yml.<br>
I added `extra_hosts : "host.docker.internal:host-gateway"`. 

```yaml
services:
  openclaw-gateway:
    image: ${OPENCLAW_IMAGE:-openclaw:local}
    extra_hosts:
      - "host.docker.internal:host-gateway"
```



install brew manually.<br>
it seesm the docker container doesn't have brew and during installation it fails if brew is not installed. 

```bash
mkdir ~/.linuxbrew && curl -L https://github.com/Homebrew/brew/tarball/master | tar xz --strip 1 -C ~/.linuxbrew

# set up environment variables 
echo 'export PATH="$HOME/.linuxbrew/bin:$HOME/.linuxbrew/sbin:$PATH"' >> ~/.bashrc
echo 'export MANPATH="$HOME/.linuxbrew/share/man:$MANPATH"' >> ~/.bashrc
echo 'export INFOPATH="$HOME/.linuxbrew/share/info:$INFOPATH"' >> ~/.bashrc

# 3. 현재 세션에 적용
source ~/.bashrc

```



### Model Provider

**vLLM**

you can serve local LLM like this. 

```bash
vllm serve openai/gpt-oss-20b \
    --host 127.0.0.1 \
    --port 8045 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.3 \
    --trust-remote-code \
    --async-scheduling \
    --max-num-batched-tokens 8192 \
    --max-model-len 35096 \
    --api-key 1234
```

connect to Docker terminal and then change the connection information like this.<br>
no need to restart the openclaw container. 

```bash
# change address to 172.17.0.1
sed -i 's/127.0.0.1/host.docker.internal/' ~/.openclaw/openclaw.json

# change vLLM API key (1234)
sed -i 's/"VLLM_API_KEY"/"1234"/' ~/.openclaw/openclaw.json
```


### Setup OpenClaw Gateway

connect to Docker terminal and you can get a gateway token 

```bash
$ echo $OPENCLAW_GATEWAY_TOKEN
05c5df07ef01<eradicated>
```

1. http://127.0.0.1:18789/overview
2. in Gateway Access, put the token in Gateway Token


### Token Issue

If you happen to run `./docker-setup.sh` multiple times, it generates a new random gateway token. <br> 
this creates a mismatch between `.env` and `~/.openclaw/openclaw.json`

```bash
$ grep OPENCLAW_GATEWAY_TOKEN .env
OPENCLAW_GATEWAY_TOKEN=05c5df0<eradicated>

$ grep -o '"token": "[^"]*"' ~/.openclaw/openclaw.json
"token": "05c5df0<eradicated>"
```

if both configurations are different, you need to update ~/.openclaw/openclaw.json file to match the .env token<br>
and then **restart docker**

```bash
$ docker restart openclaw-openclaw-gateway-1
```

### Slack

```bash
$ ./openclaw.mjs pairing list --channel slack
```