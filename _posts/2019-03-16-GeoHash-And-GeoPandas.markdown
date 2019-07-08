---
layout: post
title:  "GeoHash, GeoPandas and Folium"
date:   2019-03-16 01:00:00
categories: "code-snippet"
asset_path: /assets/images/
tags: ['folium', 'geohash', 'latitude', 'longitude']
---



<header>
    <img src="{{ page.asset_path }}geohash-wallpaper.jpeg" class="img-responsive img-rounded img-fluid">
    <div style="text-align:right;">
    <a style="background-color:black;color:white;text-decoration:none;padding:4px 6px;font-family:-apple-system, BlinkMacSystemFont, &quot;San Francisco&quot;, &quot;Helvetica Neue&quot;, Helvetica, Ubuntu, Roboto, Noto, &quot;Segoe UI&quot;, Arial, sans-serif;font-size:12px;font-weight:bold;line-height:1.2;display:inline-block;border-radius:3px" href="https://unsplash.com/@drewmark?utm_medium=referral&amp;utm_campaign=photographer-credit&amp;utm_content=creditBadge" target="_blank" rel="noopener noreferrer" title="Download free do whatever you want high-resolution photos from Andrew Stutesman"><span style="display:inline-block;padding:2px 3px"><svg xmlns="http://www.w3.org/2000/svg" style="height:12px;width:auto;position:relative;vertical-align:middle;top:-2px;fill:white" viewBox="0 0 32 32"><title>unsplash-logo</title><path d="M10 9V0h12v9H10zm12 5h10v18H0V14h10v9h12v-9z"></path></svg></span><span style="display:inline-block;padding:2px 3px">Andrew Stutesman</span></a> 
    </div>
</header>


# Introduction

## Latitude and Longitude

**Latitude는 y축으로 생각하면 되고, 가장 최극단에 북극점, 남극점이 존재합니다.**<br>
이론상 가장 남쪽은 -90이 나오고, 지구의 중간(equator 적도)은 0, 그리고 가장 북쪽은 90까지 나올수 있습니다. 

**Longitude는 x축으로 생각하면 되고, 영국 그리니치 천문대가 대략 0에서 시작해서, 지도상 오른쪽으로 갈수록 longitude의 값을 올라갑니다.**
그리니치 천문대를 중심으로 왼쪽으로 가면 음수가 잡히고, 오른쪽으로 가면 양수가 잡힙니다.

* Latitude range: -90 to 90
* Longitude range: -180 to 180


<img src="{{ page.asset_path }}geohash-lat-lng.jpg" class="img-responsive img-rounded img-fluid">



## GeoHash Legnth


| GeoHash Length | Area Width  | Area Height  | 
|:---------------|:------------|:-------------|
| 1              | 5,009.4km   |  4,992.6km   | 
| 2              | 1,252.3km   |  624.1km     | 
| 3              | 156.5km     | 156km        |
| 4              | 39.1km      | 19.5km       |
| 5              | 4.9km       | 4.9km        |
| 6              | 1.2km       | 609.4m       |
| 7              | 152.9m      | 152.4m       |
| 8              | 38.2m       | 19m          |
| 9              | 4.8m        | 4.8m         |
| 10             | 1.2m        | 59.5cm       |
| 11             | 14.9cm      | 14.9cm       |
| 12             | 3.7cm       | 1.9cm        |


## GeoJSON

예를 들어 아래와 같은 형태를 갖습니다.<br>
Python에서는 두가지 방법이 있는데, GeoJSON형태를 만들어서 사용하든지 또는 GeoPandas를 사용합니다.


{% highlight python %}
{
    "type": "FeatureCollection",
    "features": [
        {
            "properties": {"name": "Alabama"},
            "id": "AL",
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-87.359296, 35.00118], ...]]
                }
            },
        {
            "properties": {"name": "Alaska"},
            "id": "AK",
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [[[[-131.602021, 55.117982], ... ]]]
                }
            },
        ...
        ]
    }
{% endhighlight %}



# GeoHash Library Getting Started


## Converting 

Latitude 그리고 Longitude를 알고 있다면 geohash.encode(lat, lng, precision) 을 사용해서 geohash로 변환 가능 합니다.<br>
반대로 geohash.decode() 함수를 사용해서 해당 geohash의 latitude 그리고 longitude를 알아낼 수 있습니다.


{% highlight python %}
import geohash

lat, lng = (37.497868971527424, 127.0276489936216) # 강남 Lat, Lng
gangnam = geohash.encode(lat, lng, 5)
decoded_location = geohash.decode(gangnam)  # gangnam = 'wydm6'

print('Lat, Lng        :', lat, lng)
print('강남 geohash     :', gangnam)
print('Decoded Location:', *decoded_location)
{% endhighlight %}

{% highlight bash %}
Lat, Lng        : 37.497868971527424 127.0276489936216
강남 geohash     : wydm6
Decoded Location: 37.50732421875 127.02392578125
{% endhighlight %}

## decode_exactly function

더 자세한 정보가 필요시 decode_exactly 함수를 사용합니다.<br>
이때 return값은 **(latitude, longitude, latitude error margin, longitude error margin)** 입니다

{% highlight python %}
geohash.decode_exactly('wydm6')
{% endhighlight %}

{% highlight bash %}
(37.50732421875, 127.02392578125, 0.02197265625, 0.02197265625)
{% endhighlight %}

## Neighbors

GeoHash의 주변을 검색할 수 있습니다.

{% highlight python %}
geohash.neighbors(gangnam)
{% endhighlight %}

{% highlight bash %}
['wydm3', 'wydm7', 'wydm4', 'wydm1', 'wydm5', 'wydmd', 'wydm9', 'wydme']
{% endhighlight %}


# Visualization with Folium

## Choropleth

geo_data 와 data 두개를 연결시켜줘야 합니다. <br>
geo_data에는 **geometry** 정보가 있어서 여기서 polygon인지 point인지 위치 정보가 들어 있습니다.<br>
데이터에는 지도상에 색상으로 정보를 보여주기 위한 어떤 값이 존재하게 됩니다.<br>
두개의 데이터를 연결시키기 위해서 SQL의 join처럼 어떤 값을 기준으로 위치 정보와 값이 연결이 되게 되는데 **key_on** 값에서 설정하게 됩니다.

아래 GeoPandas DataFrame의 에서 geohash를 key_on값으로 사용했습니다. <br>
당연히 data로 사용되는 데이터에도 join으로 사용할 geohash column이 존재해야 합니다.

key_on='feature.properties.geohash' 이렇게 사용한 이유는 Pandas DataFrame을 json으로 바꾸면.. GeoJson으로 변환이 됩니다.<br>
보면 feature 안에 properties 안에 geohash가 존재하는것을 볼 수 있습니다. <br>
이 값을 사용한다는 뜻으로 feature.properties.geohash를 명시하였습니다.



{% highlight python %}
import folium
from polygon_geohasher.polygon_geohasher import geohash_to_polygon

locations = [(37.49786897152, 127.02764899362),
             (37.50732421875, 126.97998046875),
             (37.50732421875, 127.06787109375),
             (37.46337890625, 127.02392578125),
             (37.46337890625, 126.97998046875),
             (37.46337890625, 127.06787109375),
             (37.55126953125, 127.02392578125),
             (37.55126953125, 126.97998046875),
             (37.55126953125, 127.06787109375)]

# Create Geo Pandas DataFrame
df = gpd.GeoDataFrame({'location':locations, 'value': np.random.rand(9)})
df['geohash'] = df['location'].apply(lambda l: geohash.encode(l[0], l[1], 5))
df['geometry'] = df['geohash'].apply(geohash_to_polygon)
df.crs = {'init': 'epsg:4326'}


print('features.properties.geohash <- 요걸로 매핑함')
display(json.loads(df.to_json())['features'][0])
display(df.head())
{% endhighlight %}

{% highlight bash %}
features.properties.geohash <- 요걸로 매핑함

{'id': '0',
 'type': 'Feature',
 'properties': {'location': [37.49786897152, 127.02764899362],
  'value': 0.14382200834259584,
  'geohash': 'wydm6'},
 'geometry': {'type': 'Polygon',
  'coordinates': [[[127.001953125, 37.4853515625],
    [127.0458984375, 37.4853515625],
    [127.0458984375, 37.529296875],
    [127.001953125, 37.529296875],
    [127.001953125, 37.4853515625]]]}}
    
   location                           value     geohash                                      geometry
--------------------------------------------------------------------------------------------------------
0  (37.49786897152, 127.02764899362)  0.143822  wydm6  POLYGON ((127.001953125 37.4853515625, 127.045...
1  (37.50732421875, 126.97998046875)  0.183472  wydm3  POLYGON ((126.9580078125 37.4853515625, 127.00...
2  (37.50732421875, 127.06787109375)  0.654764  wydm7  POLYGON ((127.0458984375 37.4853515625, 127.08...
3  (37.46337890625, 127.02392578125)  0.076692  wydm4  POLYGON ((127.001953125 37.44140625, 127.04589...
4  (37.46337890625, 126.97998046875)  0.274943  wydm1  POLYGON ((126.9580078125 37.44140625, 127.0019...
{% endhighlight %}



{% highlight python %}
lat, lng = (37.497868971527424, 127.0276489936216) # 강남 Lat, Lng
m = folium.Map((lat, lng), zoom_start=12)
folium.Choropleth(geo_data=df, 
                  name='choropleth',
                  data=df,
                  columns=['geohash', 'value'],
                  key_on='feature.properties.geohash',
                  fill_color='YlGn',
                  fill_opacity=0.7,
                  line_opacity=0.2,
                  legend_name='asdf').add_to(m)
m
{% endhighlight %}

<img src="{{ page.asset_path }}geohash-choropleth.png" class="img-responsive img-rounded img-fluid">


## GeoJson

원래는 GeoJSON object(python dictionary)가 들어가야 하는데, 그냥 GeoPandas의 GeoDataFrame넣어도 돌아갑니다. <br>
Choropleth 에서는 안되는 style_function을 통해서 마음대로 스타일링이 가능합니다.

{% highlight python %}
def my_style_function(feature):
    label = feature['properties']['kmean_label']
    color = cm.step.Set1_09.scale(0, 9)(label)
    
    return dict(fillColor=color, 
                color='black', 
                weight=1.2, 
                fillOpacity=0.7)

df['kmean_label'] = np.arange(9)

lat, lng = (37.497868971527424, 127.0276489936216) # 강남 Lat, Lng
m = folium.Map((lat, lng), zoom_start=12)
folium.GeoJson(df, style_function=my_style_function).add_to(m)
m
{% endhighlight %}

<img src="{{ page.asset_path }}geohash-geojson.png" class="img-responsive img-rounded img-fluid">


## WebBrowser

지도를 HTML파일 형태로 저장한 다음에, 웹브라우져에서도 확인할 수 있습니다.

{% highlight python %}
import webbrowser

m.save('map.html')
webbrowser.open('map.html')
{% endhighlight %}

## Color Maps

{% highlight python %}
import branca.colormap as cm

step = cm.StepColormap(
    ['green', 'yellow', 'red'],
    vmin=3, vmax=10,
    index=[3, 4, 8, 10],
    caption='step'
)

linear = folium.LinearColormap(
    ['green', 'yellow', 'red'],
    vmin=0, vmax=1
)

display(step)
display(linear)
display(cm.linear.Accent_03.scale(0, 24))
display(cm.linear.Accent_08)
display(cm.linear.Blues_03)
{% endhighlight %}


<img src="{{ page.asset_path }}geohash-colormap.png" class="img-responsive img-rounded img-fluid">





## Tiles

Folium은 다음의 tiles들을 사용 가능합니다.

1. openstreetmap (기본값)
2. Stamen Toner  (블랙 & 화이트)
3. cartodbdark_matter (블랙)
4. stamenwatercolor (주황색)
5. Mapbox Bright (밝은 바닐라 크림색 - 거의 흰색)
6. cartodbpositron (밝은 회색빛.. 화얀색)

타일 변경시 두가지 방법으로 변경 가능합니다.

```
m = folium.Map((37.5387343, 127.07511967), tiles='Stamen Toner')
```

또는

```
folium.TileLayer('Stamen Toner').add_to(m)
```

{% highlight python %}
m = folium.Map((37.53873434027448, 127.07511967328423), tiles='cartodbpositron')
# folium.TileLayer('Stamen Toner').add_to(m)
m
{% endhighlight %}

<img src="{{ page.asset_path }}geohash-tiles.png" class="img-responsive img-rounded img-fluid">


## Adding m meters to latitude and longitude 

{% highlight python %}
def location_addition(lat, lng, meter):
    new_lat = lat + (meter/1000/6359.0899) * (180/np.pi)
    new_lng = lng + (meter/1000/6386) * (180/np.pi) / np.cos(lat * np.pi/180)
    return new_lat, new_lng
{% endhighlight %}   
 
 
 {% highlight python %}
 m = folium.Map(location=(lat, lng), zoom_start=12)
lat, lng = 37.499402, 127.054207

folium.Marker((lat, lng), popup='<b>A</b>').add_to(m)
new_lat, new_lng = location_addition(lat, lng, 500)
folium.Marker((new_lat, new_lng), popup='<b>500m</b>').add_to(m)

print('500m addition')
print('Latitude  Addition:', distance((lat, lng), (new_lat, lng)).m)
print('Longitude Addition:', distance((lat, lng), (lat, new_lng)).m)
print('Both      Addition:', distance((lat, lng), (new_lat, new_lng)).m)

new_lat, new_lng = location_addition(lat, lng, 1000)
folium.Marker((new_lat, new_lng), popup='<b>500m</b>').add_to(m)
print('1000m addition')
print('Latitude  Addition:', distance((lat, lng), (new_lat, lng)).m)
print('Longitude Addition:', distance((lat, lng), (lat, new_lng)).m)
print('Both      Addition:', distance((lat, lng), (new_lat, new_lng)).m)
print()

new_lat, new_lng = location_addition(lat, lng, 5000)
folium.Marker((new_lat, new_lng), popup='<b>500m</b>').add_to(m)
print('5000m addition')
print('Latitude  Addition:', distance((lat, lng), (new_lat, lng)).m)
print('Longitude Addition:', distance((lat, lng), (lat, new_lng)).m)
print('Both      Addition:', distance((lat, lng), (new_lat, new_lng)).m)
print()

new_lat, new_lng = location_addition(lat, lng, 10000)
folium.Marker((new_lat, new_lng), popup='<b>500m</b>').add_to(m)
print('10000m addition')
print('Latitude  Addition:', distance((lat, lng), (new_lat, lng)).m)
print('Longitude Addition:', distance((lat, lng), (lat, new_lng)).m)
print('Both      Addition:', distance((lat, lng), (new_lat, new_lng)).m)
print()


new_lat, new_lng = location_addition(lat, lng, 50000)
folium.Marker((new_lat, new_lng), popup='<b>500m</b>').add_to(m)
print('50000m addition')
print('Latitude  Addition:', distance((lat, lng), (new_lat, lng)).m)
print('Longitude Addition:', distance((lat, lng), (lat, new_lng)).m)
print('Both      Addition:', distance((lat, lng), (new_lat, new_lng)).m)
print()
{% endhighlight %}
 
 
{% highlight python %}
500m addition
Latitude  Addition: 500.00005283004964
Longitude Addition: 500.0049490970048
Both      Addition: 707.0996972853251
1000m addition
Latitude  Addition: 1000.0004879418325
Longitude Addition: 1000.0098977428217
Both      Addition: 1414.178421176091

5000m addition
Latitude  Addition: 5000.017733038395
Longitude Addition: 5000.049416527164
Both      Addition: 7070.052813071016

10000m addition
Latitude  Addition: 10000.073709301136
Longitude Addition: 10000.098381885806
Both      Addition: 14138.005608237756

50000m addition
Latitude  Addition: 50001.900230356274
Longitude Addition: 50000.41972205647
Both      Addition: 70605.6694353784
{% endhighlight %}

<img src="{{ page.asset_path }}geohash-adding-lat-lng.png" class="img-responsive img-rounded img-fluid">