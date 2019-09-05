---
layout: post
title:  "A/B Testing, Confidence Interval, P-Value"
date:   2019-05-25 01:00:00
categories: "statistics"
asset_path: /assets/images/
tags: ['confidence-interval', 'confidence-level', 'p-value', 'ab-testing', 'standard-error']
---



<header>
    <img src="{{ page.asset_path }}ab_beaker.jpg" class="img-responsive img-rounded img-fluid">
    <div style="text-align:right;">
    <a style="background-color:black;color:white;text-decoration:none;padding:4px 6px;font-family:-apple-system, BlinkMacSystemFont, &quot;San Francisco&quot;, &quot;Helvetica Neue&quot;, Helvetica, Ubuntu, Roboto, Noto, &quot;Segoe UI&quot;, Arial, sans-serif;font-size:12px;font-weight:bold;line-height:1.2;display:inline-block;border-radius:3px" href="https://unsplash.com/@_louisreed?utm_medium=referral&amp;utm_campaign=photographer-credit&amp;utm_content=creditBadge" target="_blank" rel="noopener noreferrer" title="Download free do whatever you want high-resolution photos from Louis Reed"><span style="display:inline-block;padding:2px 3px"><svg xmlns="http://www.w3.org/2000/svg" style="height:12px;width:auto;position:relative;vertical-align:middle;top:-2px;fill:white" viewBox="0 0 32 32"><title>unsplash-logo</title><path d="M10 9V0h12v9H10zm12 5h10v18H0V14h10v9h12v-9z"></path></svg></span><span style="display:inline-block;padding:2px 3px">Louis Reed</span></a> 
    </div>
</header>

# A/B Test Example

| Variation | Conversion | View | Conversion Rate | SE | Confidence Interval |  Change | Confidence | 
|:----------|:-----------|:-----|:----------------|:---|:--------------------|:--------|:-----------|
| Variation A (Control Group) | 330 | 1093 | 30.19% | 0.0138 | $$ \pm 2.72 $$ % | - | - |
| Variation B (Test Group)    | 491 | 1123 | 43.72% | 0.0148 | $$ \pm 2.25 $$ % | 44.82% | |

* SE: Standard Error
* 95% confidence level을 사용 
* 두 그룹간의 차의 z-score는 6.6663