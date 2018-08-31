---
layout: blog
title: Example content for posts  
categories: others
---


<p><small>This demo page has been used from <a href="http://jasonm23.github.io/markdown-css-themes/" target="_blank">http://jasonm23.github.io/markdown-css-themes/</a>.</small></p>

<h1>{{ page.title }}</h1>
<p>{{ page.date | date: '%B %d, %Y' }}</p>


### Code snippet

{% highlight python linenos %}
import numpy as np
import matplotlib.pyplot as plt

if __name__ =='__main__':
    img_thread = threading.Thread(target=downloadWallpaper)
    img_thread.start()
    st = '\rDownloading Image'
    current = 1
    while img_thread.is_alive():
        sys.stdout.write(st+'.'*((current)%5))
        current=current+1
        time.sleep(0.3)
    img_thread.join()
    print('\nImage of the day downloaded.')
    
    def fun(x):
		return x
    
{% endhighlight %}

{% highlight ruby linenos %}
require 'redcarpet'
markdown = Redcarpet.new("Hello World!")
puts markdown.to_html
{% endhighlight %}

<ul>
{% for post in site.posts %}
    <a href="{{ post.url }}/#about">
      <h5>{{ post.title }}</h5>
    </a>
{% endfor %}
</ul>
