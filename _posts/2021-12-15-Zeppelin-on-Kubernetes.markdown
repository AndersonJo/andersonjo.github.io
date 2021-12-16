---
layout: post 
title:  "Zeppelin on Kubernetes with Connections"
date:   2021-12-15 01:00:00 
categories: ""
asset_path: /assets/images/ 
tags: ['helm', 'k8s', 'secret', 'gcp', 'eks', 'kubernetes', 'bigquery', 'mariadb']
---

# 1. Installation

## 1.1 Installing Zeppelin on Kubernetes

먼저  zeppelin-server.yaml 을 다운로드 받습니다.

{% highlight bash %}
$ curl -s -O https://raw.githubusercontent.com/apache/zeppelin/master/k8s/zeppelin-server.yaml
{% endhighlight %}

다운 받은후 설정 버그를 수정합니다. (뭔가 업데이트가 잘 안되고 있네요)

- data.ZEPPELIN_K8S_CONTAINER_IMAGE (32줄)
  - `apache/zeppelin-interpreter:0.10.0` ->  `apache/zeppelin:0.10.0`
- spec.template.spec.containers.image (118줄)
  - `apache/zeppelin-server:0.10.0` ->  `apache/zeppelin:0.10.0`

Image Bug를 수정후 만약 EKS 를 사용중이고, LoadBalancer를 달려고 한다면 Service 부분을 찾아서 다음과 같이 수정합니다.

{% highlight yaml %}
kind: Service
apiVersion: v1
metadata:
  name: zeppelin-server
spec:
  type: LoadBalancer
  ports:
    - name: http
      port: 80
      targetPort: 80
    - name: rpc            # port name is referenced in the code. So it shouldn't be changed.
      port: 12320
  selector:
    app.kubernetes.io/name: zeppelin-server
{% endhighlight %}


아래와 같이 `ZEPPELIN_RUN_MODE` 를 추가합니다. <br>
Kubernetes cluster 모드는 현재 불안정한게 많습니다.<br>
아무래도 관리가 전혀 되고 있지 않은거 같습니다. 

Kubernetes 환경에서는 뭔가 실행시 새로운 pod을 열게 됩니다.<br>
문제는 bigquery 에 접속할때 필요한 authentication등이나 환경변수가 전혀 복사되지 않습니다. <br> 
이로 인해서 접속이 안되게 됩니다. <br>
따라서 local 환경으로 작동시킵니다. 

 - ZEPPELIN_RUN_MODE: Run mode. 'auto|local|k8s'. 'auto' autodetect environment. 'local' runs interpreter as a local process. k8s runs interpreter on Kubernetes cluster

{% highlight yaml %}
apiVersion: v1
kind: ConfigMap
metadata:
  name: zeppelin-server-conf-map
data:
  ZEPPELIN_HOME: /opt/zeppelin
  ZEPPELIN_RUN_MODE: local
{% endhighlight %}




{% highlight bash %}
$ kubectl apply -f zeppelin-server.yaml
{% endhighlight %}









## 1.2 Persistent Volume for Zeppelin



persistent volume 을 요청하는 부분을 작성합니다.

{% highlight yaml %}
[생략]
spec:
  serviceAccountName: zeppelin-server
  volumes:
  - name: zeppelin-server-conf-pv              # persistent volume 요청
    persistentVolumeClaim:
      claimName: zeppelin-server-conf-pvc
  - name: zeppelin-server-notebook-pv          # persistent volume 요청
    persistentVolumeClaim:
      claimName: zeppelin-server-notebook-pvc
{% endhighlight %}

`spec.template.spec.containers.volumeMounts` 에 persistent volume을 연결 시킵니다.ㄴ 

{% highlight yaml %}
spec:
  template:
    spec:
      serviceAccountName: zeppelin-server
      containers:
        volumeMounts:
        - name: zeppelin-server-notebook-pv         # configure this to persist notebook
          mountPath: /opt/zeppelin/notebook
        - name: zeppelin-server-conf-pv             # configure this to persist Zeppelin configuration
          mountPath: /opt/zeppelin/conf
{% endhighlight %}

맨 아래로 내려가서 다음을 추가합니다.

{% highlight yaml %}
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: zeppelin-server-conf-pvc
spec:
  resources:
    requests:
      storage: 1Gi
  accessModes:
    - ReadWriteOnce
  storageClassName: gp2
  volumeMode: Filesystem
---
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: zeppelin-server-notebook-pvc
spec:
  resources:
    requests:
      storage: 1Gi
  accessModes:
    - ReadWriteOnce
  storageClassName: gp2
  volumeMode: Filesystem
{% endhighlight %}


마지막으로 권한을 수정합니다.<br>
securityContext 를 추가 합니다.

{% highlight yaml %}
spec:
  template:
    spec:
      containers:
      - name: zeppelin-server
        image: apache/zeppelin:0.10.0
        securityContext:
          runAsUser: 0
{% endhighlight %}








잘되었는지 확인해 봅니다.

{% highlight bash %}
$ kubectl get pvc
NAME                           STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS
zeppelin-server-conf-pvc       Bound    pvc-12345678-bb06-4b4b-bbdb-aa1234567890   1Gi        RWO            gp2         
zeppelin-server-notebook-pvc   Bound    pvc-abcdefgh-a2de-4ea3-a433-bb0987654321   1Gi        RWO            gp2

$ kubectl get pv
NAME                                       CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM                                           STORAGECLASS
pvc-12345678-bb06-4b4b-bbdb-aa1234567890   1Gi        RWO            Delete           Bound    default/zeppelin-server-conf-pvc                gp2         
pvc-abcdefgh-a2de-4ea3-a433-bb0987654321   1Gi        RWO            Delete           Bound    default/zeppelin-server-notebook-pvc            gp2         
{% endhighlight %}


설치가 잘됐는지 확인해 봅니다.

{% highlight bash %}
$ kubectl exec -it <zeppelin-server-pod-name> -- sh

$ cat /proc/mounts | grep zeppelin
/dev/nvme2n1 /zeppelin/notebook ext4 rw,relatime 0 0
/dev/nvme3n1 /zeppelin/conf ext4 rw,relatime 0 0
{% endhighlight %}



## 1.3 BigQuery Setting

먼저 secret key 를 파일로 부터 생성해줍니다. <br>
이름은 service account라는 뜻으로 gcp-sa-secret-key 로 만들어 줍니다. <br>
.json 파일은 반드시 absolute path 이어야 합니다. <br>

{% highlight bash %}
$ kubectl create secret generic gcp-sa-secret-key --from-file=<absolute path of gcp-sa.json>
$ kubectl get secret gcp-sa-secret-key -o yaml
{% endhighlight %}


해당 secret key를 volume으로 올려주기 위해서 zeppelin-server.yaml 파일을 다시 수정합니다.<br>
`kind: Deployment` 를 찾고 `spec.template.spec.volumes` 에다가 다음을 추가 합니다. 

- gcp-sa-credentials-volume: volume 이름
- gcp-secret.json: 
  - `kubectl get secret gcp-sa-secret-key -o yaml` 실행한뒤.. 
  - data안에 key를 확인하면 됨
  - 보통 파일이름이 그대로 들어감

{% highlight yaml %}
[생략]
spec:
  serviceAccountName: zeppelin-server
  volumes:
  - name: gcp-sa-credentials-volume
    secret:
      secretName: gcp-sa-secret-key
      items:
      - key: gcp-secret.json
        path: gcp-sa-credentials.json
{% endhighlight %}


생성한 volume을 mount 시켜줍니다.<br>
`spec.template.spec.containers.volumeMounts` 아래에다가 넣습니다. 

- gcp-sa-credentials-volume: 우에서 생성한 volume 이름을 그대로 넣습니다.

{% highlight yaml %}
spec:
  template:
    spec:
      serviceAccountName: zeppelin-server
      containers:
        volumeMounts:
        - name: gcp-sa-credentials-volume
          mountPath: /etc/gcp
          readOnly: true
{% endhighlight %}


환경변수를 설정합니다. 

{% highlight yaml %}
spec:
  template:
    spec:
      serviceAccountName: zeppelin-server
      containers:
        env:
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: /etc/gcp/gcp-sa-credentials.json
{% endhighlight %}


실제 pod안에 들어가서 잘되는지 확인해 봅니다. 

{% highlight bash %}
$ kubectl exec -it <zeppelin-server-pod-name> -- sh

# 잘되는지 확인
$ env | grep -i GOOGLE_APPLICATION_CREDENTIALS
$ cat /etc/gcp/gcp-sa-credentials.json
{% endhighlight %}


## 1.4 MariaDB

Interpreter 에서 JDBC 를 설정하면 됩니다. 

 - default.url: `jdbc:mariadb://127.0.0.1:3306`
 - default.user: 유저 이름
 - default.password: 암호
 - default.driver: org.mariadb.jdbc.Driver
 - Dependencies 
   - `org.mariadb.jdbc:mariadb-java-client:3.0.2-rc`

이렇게 설정해주고.. 노트북에서는 `%jdbc` 로 사용합니다. 
