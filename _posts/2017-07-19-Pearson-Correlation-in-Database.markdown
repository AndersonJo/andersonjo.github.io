---
layout: post
title:  "Pearson Correlation in Database"
date:   2017-07-16 01:00:00
categories: "statistics"
asset_path: /assets/posts2/statistics/
tags: ['postgres', 'mysql', 'python']

---


<header>
    <img src="{{ page.asset_path }}ls_wallpaper.jpg" class="img-responsive img-rounded" style="width:100%">
    <div style="text-align:right;">
    <small><a href="https://unsplash.com/?photo=vmlJcey6HEU">Vladimir Kudinov의 사진</a>
    <br> 기본으로 돌아가서.. 천천히..
    </small>
    </div>
</header>


보통 RDBMS에 데이터가 저장되어 있는데.. 어떠한 이유로 Python이나, R등에서 다양한 컬럼들을 pearson correlation해야할때가 있습니다.<br>
이경우 데이터를 모두 가져와서 Python또는 R에서 돌리면 그 과정 자체가 매우 느립니다. <br>
특히 R은 많은 데이터를 처리시 Python에 비해서 더 잘 뻗어버리는 경향이 있어서 더욱 어렵습니다.<br>
그냥 Database에서 돌려버리고 결과만 받는다면 데이터를 다운받거나 그럴 필요가 없기 때문에 매우 효율적입니다.


# Pearson Correlation

Pearson Correlation의 공식은 다음과 같습니다.

$$ r = \frac{n \left( \sum xy \right) - \left( \sum x \right) \left( \sum y \right)}
{\sqrt{ \left[ n \sum x^2 - \left( \sum x \right)^2 \right]
        \left[ n \sum y^2 - \left( \sum y \right)^2 \right] }} $$

## Data

{% highlight python %}
data = pd.DataFrame({'city': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                     'age': [43, 21, 25, 42, 57, 59, 70, 60, 50, 40, 35, 24],
                     'glucose': [99, 65, 79, 75, 87, 81, 1, 3, 6, 10, 20, 22]})

city1 = data[data['city'] == 1]
city2 = data[data['city'] == 2]
city1.plot(y=['age', 'glucose'], title='city 1')
city2.plot(y=['age', 'glucose'], title='city 2')
{% endhighlight %}

## Numpy

{% highlight python %}
def pearson_corr(x, y):
    N = len(x)
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    x_squared = np.sum(x**2)
    y_squared = np.sum(y**2)

    a = N * (np.sum(x * y)) - x_sum * y_sum
    b = (N * x_squared - x_sum**2) * (N * y_squared - y_sum**2)
    r = a / np.sqrt(b)

    return r

print('city1:', pearson_corr(city1['age'], city1['glucose']))
print('city2:', pearson_corr(city2['age'], city2['glucose']))
{% endhighlight %}

city1: 0.52980890189 <br>
city2: -0.945671421045


## Pandas

{% highlight python %}
data.groupby('city').corr(method='pearson')
{% endhighlight %}

<img src="{{ page.asset_path }}pearson_pandas.png" class="img-responsive img-rounded">

## SQL

먼저 데이터를 올립니다.<br>
ORM은 [PonyORM](https://ponyorm.com/) 을 사용하였습니다.

{% highlight python %}
db = orm.Database()
db.bind(provider='postgres', user='postgres', password='1234', host='localhost', database='test')

# Create Table
class Pearson(db.Entity):
    id = orm.PrimaryKey(int, auto=True)
    city = orm.Required(int)
    age = orm.Required(int)
    glucose = orm.Required(int)

db.generate_mapping(create_tables=True)

# Insert Data
def insert_data(data):
    if Pearson.select().count():
        return None

    for i, d in data.iterrows():
        city = int(d.city)
        age = int(d.age)
        glucose = int(d.glucose)

        ent = Pearson(city=city, age=age, glucose=glucose)

    orm.commit()

insert_data(data)
{% endhighlight %}

{% highlight python %}
def create_pearson_corr_query(x:str, y:str, x_on='id', y_on='id', group_by:list=[]):
    """
    @param x<str>: [table].[column]
    @param y<str>: [table].[column]
    @param x_on<str>: JOIN Column Name
    @param y_on<str>: JOIN Column Name
    """
    x_table, x_column = x.split('.')
    y_table, y_column = y.split('.')

    # GROUP BY
    group_columns = []
    group_sub_select = []
    group_select = []

    for i, g in enumerate(group_by):
        g_table, g_column = g.split('.')
        g_table = 'd1' if g_table == x_table else 'd2'
        group_columns.append(f'{g_table}.{g_column}')
        group_sub_select.append(f'{g_table}.{g_column} as name{i}')
        group_select.append(f'name{i}')

    group_columns = ', '.join(group_columns)
    group_sub_select = ', '.join(group_sub_select)
    group_select = ', '.join(group_select)

    group_by_sql = ''
    if group_columns:
        group_by_sql = f'GROUP BY {group_columns}'
        group_sub_select += ', '
        group_select += ', '

    query = '''SELECT {group_select}
(N * xy_psum - sum1 * sum2)/
    SQRT((N * sqsum1 - POW(sum1, 2)) *
    (N * sqsum2 - POW(sum2, 2)))
FROM
    (SELECT {group_sub_select}
        COUNT(*) AS N,
        SUM(d1.{x} * d2.{y}) AS xy_psum,
        SUM(d1.{x}) AS sum1,
        SUM(d2.{y}) AS sum2,
        SUM(POW(d1.{x}, 2)) AS sqsum1,
        SUM(POW(d2.{y}, 2)) AS sqsum2
    FROM {x_table} AS d1
    LEFT JOIN {y_table} AS d2
    ON d1.{x_on} = d2.{y_on}
    {group_by_sql}) as pcorr
    '''.format(x_table=x_table, y_table=y_table,
               x=x_column, y=y_column,
               x_on=x_on, y_on=y_on,

               group_select = group_select,
               group_sub_select= group_sub_select,
               group_by_sql=group_by_sql)
    return query

query = create_pearson_corr_query(x='pearson.age', y='pearson.glucose', group_by=['pearson.city'])
db.select(query)
{% endhighlight %}

{% highlight text %}
SELECT name0,
(N * xy_psum - sum1 * sum2)/
    SQRT((N * sqsum1 - POW(sum1, 2)) *
    (N * sqsum2 - POW(sum2, 2)))
FROM
    (SELECT d1.city as name0,
        COUNT(*) AS N,
        SUM(d1.age * d2.glucose) AS xy_psum,
        SUM(d1.age) AS sum1,
        SUM(d2.glucose) AS sum2,
        SUM(POW(d1.age, 2)) AS sqsum1,
        SUM(POW(d2.glucose, 2)) AS sqsum2
    FROM pearson AS d1
    LEFT JOIN pearson AS d2
    ON d1.id = d2.id
    GROUP BY d1.city) as pcorr
{% endhighlight %}

[(1, 0.529808901890174), (2, -0.945671421044784)]