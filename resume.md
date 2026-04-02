---
layout: resume
title: Resume
permalink: /resume/
cv_page: true
---

{% capture cv_body %}{% include cv.md %}{% endcapture %}
{{ cv_body | markdownify }}
