# Experiment Log

Model V3 was making inaccurate predictions because the data of matches that ended in 0-0, 1-1, 2-0, 3-0 etc all looked similar. I've decided to see what changes I can make to the data to make the differences of these outcomes more clear

The performance of Model V3 is:

```
Benchmark Score Accuracy: 0.18
Benchmark Win Accuracy  : 0.7542
Training Score Accuracy : 0.17871986699916875
Training Win Accuracy   : 0.5103906899418121
CV Set Score Accuracy   : 0.07142857142857142
CV Set Win Accuracy     : 0.47959183673469385

-----------------------------------------------

    Precision and recall
 1-0   TP:0  FP:3  FN:11: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 3  Sample:11
 0-2   TP:4  FP:24  FN:6: Prec:0.1429  Rec:0.4000  F1:19.0000  #Pred: 28  Sample:10
 0-4   TP:0  FP:0  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:10
 3-0   TP:2  FP:15  FN:8: Prec:0.1176  Rec:0.2000  F1:27.0000  #Pred: 17  Sample:10
 0-1   TP:1  FP:21  FN:9: Prec:0.0455  Rec:0.1000  F1:64.0000  #Pred: 22  Sample:10
 0-3   TP:0  FP:14  FN:8: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 14  Sample:8
 4-0   TP:0  FP:0  FN:8: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:8
 2-0   TP:0  FP:6  FN:8: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 6  Sample:8
 0-5   TP:0  FP:0  FN:7: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:7
 0-0   TP:0  FP:8  FN:6: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 8  Sample:6
 5-0   TP:0  FP:0  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:3
 6-0   TP:0  FP:0  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:3
 0-6   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
10-10  TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 0-9   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 7-0   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
```

## Offense Defense Model Experiment

Training the model on this new offense defense dataset resulted in a model that was in total slightly worse performing than original. Though the are improvements in the performance of multiple classes. The classes it performed worse in turned out to be those with large number of samples, leading to lower overall performance. Interestingly the model seems to have high variance. Results are below:

```
Benchmark Score Accuracy: 0.18
Benchmark Win Accuracy  : 0.7542
Training Score Accuracy : 0.1745635910224439
Training Win Accuracy   : 0.5045719035743973
CV Set Score Accuracy   : 0.061224489795918366
CV Set Win Accuracy     : 0.46938775510204084

-------------------------------------------------------

    Precision and recall
 1-0   TP:2  FP:24  FN:9: Prec:0.0769  Rec:0.1818  F1:37.0000  #Pred: 26  Sample:11
 0-2   TP:0  FP:4  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 4  Sample:10
 3-0   TP:0  FP:15  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 15  Sample:10
 0-4   TP:0  FP:0  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:10
 0-1   TP:1  FP:16  FN:9: Prec:0.0588  Rec:0.1000  F1:54.0000  #Pred: 17  Sample:10
 2-0   TP:1  FP:5  FN:7: Prec:0.1667  Rec:0.1250  F1:28.0000  #Pred: 6  Sample:8
 4-0   TP:0  FP:0  FN:8: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:8
 0-3   TP:1  FP:16  FN:7: Prec:0.0588  Rec:0.1250  F1:50.0000  #Pred: 17  Sample:8
 0-5   TP:0  FP:0  FN:7: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:7
 0-0   TP:1  FP:12  FN:5: Prec:0.0769  Rec:0.1667  F1:38.0000  #Pred: 13  Sample:6
 6-0   TP:0  FP:0  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:3
 5-0   TP:0  FP:0  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:3
 7-0   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
10-10  TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 0-9   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 0-6   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
```

Typical signs of high variance where performance in training set is much better than that of CV set. I wonder how a 2 layer architecture would perform with this.

## Offense Defense 2 layer Model Experiment

The model had a lower training set performance but a much higher CV set performance. The CV set performance and training set performance are both below the benchmark which is indicative of a model with high bias. This then means that the previous model most definetly had high variance. The results are below:

```
Benchmark Score Accuracy: 0.18
Benchmark Win Accuracy  : 0.7542
Training Score Accuracy : 0.12219451371571072
Training Win Accuracy   : 0.4613466334164589
CV Set Score Accuracy   : 0.12244897959183673
CV Set Win Accuracy     : 0.5

--------------------------------
    Precision and recall
 1-0   TP:5  FP:19  FN:6: Prec:0.2083  Rec:0.4545  F1:14.0000  #Pred: 24  Sample:11
 3-0   TP:0  FP:10  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 10  Sample:10
 0-1   TP:0  FP:6  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 6  Sample:10
 0-2   TP:7  FP:29  FN:3: Prec:0.1944  Rec:0.7000  F1:13.1429  #Pred: 36  Sample:10
 0-4   TP:0  FP:0  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:10
 4-0   TP:0  FP:0  FN:8: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:8
 2-0   TP:0  FP:0  FN:8: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:8
 0-3   TP:0  FP:22  FN:8: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 22  Sample:8
 0-5   TP:0  FP:0  FN:7: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:7
 0-0   TP:0  FP:0  FN:6: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:6
 5-0   TP:0  FP:0  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:3
 6-0   TP:0  FP:0  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:3
10-10  TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 0-6   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 0-9   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 7-0   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
```

For the LOLs I want to try out the different architecture I thought of where I predict the home team goals seperately from away team goals. Meaning I'd have 22 units in the output layer instead of 121. If it works as well as first architecture I'll use it from now on. I'll call this **Reduced Output**