---
layout: post
title:  "HDFS Client Recipes"
date:   2016-07-23 01:00:00
categories: "hadoop"
asset_path: /assets/posts/HDFS-Client/
tags: []

---

<header>
    <img src="{{ page.asset_path }}elephant.jpg" class="img-responsive img-rounded" style="width:100%">
</header>

# Java Client 101

#### **Gradle, SBT**

**SBT**<br>
{% highlight bash %}
libraryDependencies += "org.apache.hadoop" % "hadoop-client" % "2.7.3"
{% endhighlight %}

**Gradle**<br>
{% highlight bash %}
compile group: 'org.apache.hadoop', name: 'hadoop-client', version: '2.7.3'
{% endhighlight %}





#### **클라이언트 설정**

{% highlight java %}
Configuration conf = new Configuration();
conf.set("fs.defaultFS", "hdfs://[PUBLIC-DNS]:8020");
{% endhighlight %}

#### **파일및 디렉토리 출력 + 싸이즈**

{% highlight java %}
FileSystem fs = FileSystem.get(conf);
FileStatus[] fileStatuses = fs.listStatus(new Path("/"));

for (FileStatus fileStatus : fileStatuses) {
    String path = fileStatus.getPath().toString();
    long size = fileStatus.getLen();
    System.out.format("%s [%d]\n", path, size);
}
{% endhighlight %}

{% highlight bash %}
# 결과
hdfs://PUBLIC_DNS:8020/app-logs [0]
hdfs://PUBLIC_DNS:8020/apps [0]
hdfs://PUBLIC_DNS:8020/sample.csv [2154853]
{% endhighlight %}


#### **디렉토리 생성및 유저 변경**

{% highlight java %}
UserGroupInformation ugi = UserGroupInformation.createRemoteUser("hdfs");
ugi.doAs(new PrivilegedExceptionAction<Void>() {
    @Override
    public Void run() throws Exception {
        FileSystem fs = FileSystem.get(conf);
        FsPermission permission = FsPermission.getDirDefault();
        fs.mkdirs(new Path("/sample"), permission);

        return null;
    }
});
{% endhighlight %}

{% highlight bash %}
# 결과
Permission  Owner Group Size  Last Modified               Replication  Block Size  Name
drwxr-xr-x  hdfs  hdfs	 0B   2014. 2. 23. 오후 8:21:57   0            0B          sample
{% endhighlight %}


#### **파일 업로드**

{% highlight java %}
ugi.doAs(new PrivilegedExceptionAction<Void>() {
    @Override
    public Void run() throws Exception {
        FileSystem hdfs = FileSystem.get(conf);

        ClassLoader cl = this.getClass().getClassLoader();
        File nyFile = new File(cl.getResource("ny/4300.txt").getFile());
        BufferedInputStream in = new BufferedInputStream(new FileInputStream(nyFile));

        FSDataOutputStream os = hdfs.create(new Path("/sample/ny.txt"), true);
        IOUtils.copyBytes(in, os, conf); // Upload local file to HDFS
        return null;
    }
});
{% endhighlight %}