Exercise 1
Spam detection with machine learning

The first task of this unit is about analyzing the content of emails and recognizing spam. The data consists of extracted email content that has already been prepared using various techniques and can thus be analyzed more easily. An example of a message is here:

  Subject: mid-year 2000 performance feedback
  note : you will receive this message each time you are selected
  as a reviewer. you have been selected to participate in the mid
  ...
The format for your system's predictions is as follows:

  data/spam1-test/dslkfhkajsdhfkj.x;0
  data/spam1-test/wueziqwewuewefs.x;1
  data/spam1-test/xmnbxcmnxuedasf.x;0
  ...
The first field is the filename of the email and the second field is your prediction. In the example, the email wueziqwewuewefs.x is classified as spam. Please pay attention to the correct path and file names.
The performance of your system is evaluated with the "Balanced Accuracy" (BACC). This is the accuracy of the system averaged individually for both classes.

**********************************************************************************************************

Exercise 2
Spam detection in practice

Surprise! There is a second task. It's all about distinguishing spam from ham again. However, this time the emails are not pre-processed and you are dealing with different headers and formats. Perhaps you can simply adapt the approach from the first task; maybe you have to parse a bit first and bring order to the emails. Try it!
The format and the calculation of the recognition performance are identical to the first task.