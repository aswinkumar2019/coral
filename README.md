# coral

Implemented people counter on google coral.
Google coral will detect the objects in that frame,we will extract the objects with id person and,draw a line from left to right or in any direction
The person who crosses the line will be counted.

Issues faced:
This is one of my best in maths part :)

The person while he is crossing will be counted as a new person,we should not increase the counter unless the person is crossed.
As the thickness of the line is small,the person gone undetected so,we drew a imaginary line not visible in maths part for counting.
To find if the person is near the line,we found if the center of the person is collinear with the line drawn by us.
I also deployed this code on laptop
