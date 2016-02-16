from pyspark import SparkContext, SparkConf

spconf = SparkConf().setMaster('local').setAppName('Tutorial')
sc = SparkContext(conf=spconf)

textFile = sc.textFile('README.md')
print textFile.count()  # 36 ;the number of lines
print textFile.first()  # Examples for Learning Spark

pythonLines = textFile.filter(lambda line: 'python' in line.lower())
print pythonLines.first()