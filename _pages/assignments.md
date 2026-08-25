---
layout: assignments
permalink: /assignments/
title: Programming Assignments
---

The course will be quite hands on.
We will have five programming assignments which are to be completed in groups of two. Please register your groups in this [google form](https://forms.gle/HTuDDKHuKyp1sytB6){:target="_blank"} by Saturday(8/29). Only one of you needs to do so. You will be assigned a group number which is to be used in the submission format. Those that filled the form please check your group number [here](https://docs.google.com/spreadsheets/d/1gqZPBOdgAd2ViVLmAQw8IVMt3Q2tH4bmN8zG5BI9s7k/edit?usp=sharing){:target="_blank"}.

Each assignment will be due 2-3 weeks after it is assigned.

Prior experience has shown that students who begin assignments shortly after they are assigned are more likely to succeed. Assignments submitted after the day and time they are due will be penalized 10% of the total points of the assignment per day, unless you have made prior arrangements with me due to extenuating circumstances.

All code and results you submit must be your original work. Cheating and plagiarism will not be tolerated and will be dealt with in accordance with the [University of Texas policies and procedures](https://deanofstudents.utexas.edu/conduct/index.php).

<!-- Assignment specifications, release dates, due dates, and links will be added when they are available. -->



{% for item in site.data.assignments %}
{% assign assignment = item %}


<tr>
    <th scope="row">{{ assignment.number }}</th>
    <th scope="row">
        {% if assignment.link %}
        {% if assignment.link contains '://' %}
        <a href="{{ assignment.link }}" target="_blank">{{ assignment.title }}</a>
        {% else %}
        <a href="{{ assignment.link | relative_url }}">{{ assignment.title }}</a>
        {% endif %}
        {% else %}
        {{ assignment.title }}
        {% endif %}
    </th>
    <th scope="row">{{ assignment.release_date }}</th>
    <th scope="row">{{ assignment.due_date }}</th>
</tr>
{% endfor %}
