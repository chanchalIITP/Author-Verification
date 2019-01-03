# Author-Verification
Author verification is a problem of verifying whether two documents belong to the same author or not.

We have used siamese network for author verification. C50 data-set is used as the data-set for author verification.We have re-created the data-set as per siamese network requirement.

text1, text2, class
e.g : text11, text21, YES
      text12, text22, NO
      text13, text23, YES
In this way, we have used the documents of same author as class label YES, and for other authors the class label is assigned as NO.

This code works well for python 2.7.15.
For training the siamese model, just run python Data_classify.py

Run the following code to get the result, where data_1.txt, data_2.txt, data_3.txt are known documents and data_4.txt is the unknown document (place the unknown doc. in the last).

python testing.py data_1.txt data_2.txt data_3.txt data_4.txt
