---
layout: default
title: Journals
---

<h1>Journal Entries</h1>

<ul>
  {% for post in site.posts %}
    {% if post.tags contains "journal" %}
      <li>
        <a href="{{ post.url }}">{{ post.title }}</a>
        <span style="color: gray; font-size: 0.9em;">
          — {{ post.date | date: "%B %d, %Y" }}
        </span>
      </li>
    {% endif %}
  {% endfor %}
</ul>
