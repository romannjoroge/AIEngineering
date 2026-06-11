# Results storage

Storing results of different strategies so that I can compare them

## *Strategy*: Difference strategy with only output layer

Results:

```
Benchmark Score Accuracy: 0.18
Benchmark Win Accuracy  : 0.7542
Training Score Accuracy : 0.13632585203657524
Training Win Accuracy   : 0.5361596009975063
CV Set Score Accuracy   : 0.11224489795918367
CV Set Win Accuracy     : 0.4897959183673469

---------------------------------------------------------

    Precision and recall
 1-0   TP:1  FP:7  FN:10: Prec:0.1250  Rec:0.0909  F1:38.0000  #Pred: 8  Sample:11
 0-4   TP:0  FP:0  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:10
 0-2   TP:4  FP:30  FN:6: Prec:0.1176  Rec:0.4000  F1:22.0000  #Pred: 34  Sample:10
 3-0   TP:3  FP:19  FN:7: Prec:0.1364  Rec:0.3000  F1:21.3333  #Pred: 22  Sample:10
 0-1   TP:0  FP:0  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:10
 0-3   TP:2  FP:25  FN:6: Prec:0.0741  Rec:0.2500  F1:35.0000  #Pred: 27  Sample:8
 4-0   TP:0  FP:0  FN:8: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:8
 2-0   TP:1  FP:6  FN:7: Prec:0.1429  Rec:0.1250  F1:30.0000  #Pred: 7  Sample:8
 0-5   TP:0  FP:0  FN:7: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:7
 0-0   TP:0  FP:0  FN:6: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:6
 6-0   TP:0  FP:0  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:3
 5-0   TP:0  FP:0  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:3
10-10  TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 0-9   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 7-0   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 0-6   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
```

This is a low bias model which is not a suprise since it only has 1 layer. This was more a test to see if difference works well. It seems to does as it gets arguably better results than my previous model implementation

## *Strategy*: Average Difference strategy with only output layer

Results:

```
Benchmark Score Accuracy: 0.18
Benchmark Win Accuracy  : 0.7542
Training Score Accuracy : 0.13632585203657524
Training Win Accuracy   : 0.5361596009975063
CV Set Score Accuracy   : 0.11224489795918367
CV Set Win Accuracy     : 0.4897959183673469

-----------------------------------------------

    Precision and recall
 1-0   TP:1  FP:7  FN:10: Prec:0.1250  Rec:0.0909  F1:38.0000  #Pred: 8  Sample:11
 0-4   TP:0  FP:0  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:10
 0-2   TP:4  FP:30  FN:6: Prec:0.1176  Rec:0.4000  F1:22.0000  #Pred: 34  Sample:10
 3-0   TP:3  FP:19  FN:7: Prec:0.1364  Rec:0.3000  F1:21.3333  #Pred: 22  Sample:10
 0-1   TP:0  FP:0  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:10
 0-3   TP:2  FP:25  FN:6: Prec:0.0741  Rec:0.2500  F1:35.0000  #Pred: 27  Sample:8
 4-0   TP:0  FP:0  FN:8: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:8
 2-0   TP:1  FP:6  FN:7: Prec:0.1429  Rec:0.1250  F1:30.0000  #Pred: 7  Sample:8
 0-5   TP:0  FP:0  FN:7: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:7
 0-0   TP:0  FP:0  FN:6: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:6
 6-0   TP:0  FP:0  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:3
 5-0   TP:0  FP:0  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:3
10-10  TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 0-9   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 7-0   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 0-6   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
```
Interesting that it's the exact same results. Thought they'd be slightly different but maybe difference of sums and difference of averages is just the same (especially after normalization)

## *Strategy*: Self-weighting average strategy with only output layer

Results:

```
Benchmark Score Accuracy: 0.18
Benchmark Win Accuracy  : 0.7542
Training Score Accuracy : 0.1396508728179551
Training Win Accuracy   : 0.5361596009975063
CV Set Score Accuracy   : 0.08163265306122448
CV Set Win Accuracy     : 0.42857142857142855

    Precision and recall
 1-0   TP:1  FP:12  FN:10: Prec:0.0769  Rec:0.0909  F1:48.0000  #Pred: 13  Sample:11
 0-4   TP:0  FP:1  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 1  Sample:10
 0-2   TP:1  FP:25  FN:9: Prec:0.0385  Rec:0.1000  F1:72.0000  #Pred: 26  Sample:10
 3-0   TP:4  FP:23  FN:6: Prec:0.1481  Rec:0.4000  F1:18.5000  #Pred: 27  Sample:10
 0-1   TP:0  FP:0  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:10
 0-3   TP:1  FP:25  FN:7: Prec:0.0385  Rec:0.1250  F1:68.0000  #Pred: 26  Sample:8
 4-0   TP:0  FP:0  FN:8: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:8
 2-0   TP:1  FP:3  FN:7: Prec:0.2500  Rec:0.1250  F1:24.0000  #Pred: 4  Sample:8
 0-5   TP:0  FP:1  FN:7: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 1  Sample:7
 0-0   TP:0  FP:0  FN:6: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:6
 6-0   TP:0  FP:0  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:3
 5-0   TP:0  FP:0  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:3
10-10  TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 0-9   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 7-0   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 0-6   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
```

Performance is worse for this though prediction amounts are more spread out. Though the number of TPs looks roughly the same

## *Strategy*: Difference strategy with 2 layers

First layer is (27, 9) second layer is (121, 27)

Results:

```
Benchmark Score Accuracy: 0.18
Benchmark Win Accuracy  : 0.7542
Training Score Accuracy : 0.18703241895261846
Training Win Accuracy   : 0.5037406483790524
CV Set Score Accuracy   : 0.11224489795918367
CV Set Win Accuracy     : 0.45918367346938777

-------------------------------------------------------

    Precision and recall
 1-0   TP:0  FP:3  FN:11: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 3  Sample:11
 0-4   TP:0  FP:0  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:10
 0-2   TP:5  FP:11  FN:5: Prec:0.3125  Rec:0.5000  F1:10.4000  #Pred: 16  Sample:10
 3-0   TP:2  FP:26  FN:8: Prec:0.0714  Rec:0.2000  F1:38.0000  #Pred: 28  Sample:10
 0-1   TP:0  FP:6  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 6  Sample:10
 0-3   TP:2  FP:20  FN:6: Prec:0.0909  Rec:0.2500  F1:30.0000  #Pred: 22  Sample:8
 4-0   TP:0  FP:0  FN:8: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:8
 2-0   TP:1  FP:10  FN:7: Prec:0.0909  Rec:0.1250  F1:38.0000  #Pred: 11  Sample:8
 0-5   TP:0  FP:0  FN:7: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:7
 0-0   TP:1  FP:10  FN:5: Prec:0.0909  Rec:0.1667  F1:34.0000  #Pred: 11  Sample:6
 5-0   TP:0  FP:1  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 1  Sample:3
 6-0   TP:0  FP:0  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:3
10-10  TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 0-9   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 7-0   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 0-6   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
```

## *Strategy*: Self-weighting average strategy with 3 layers

First layer is (27, 9), second layer (81, 27) and third is (121, 81)

Results:

```
Benchmark Score Accuracy: 0.18
Benchmark Win Accuracy  : 0.7542
Training Score Accuracy : 0.2119700748129676
Training Win Accuracy   : 0.541978387364921
CV Set Score Accuracy   : 0.07142857142857142
CV Set Win Accuracy     : 0.42857142857142855

-------------------------------------------------


    Precision and recall
 1-0   TP:0  FP:6  FN:11: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 6  Sample:11
 0-4   TP:0  FP:0  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:10
 0-2   TP:3  FP:20  FN:7: Prec:0.1304  Rec:0.3000  F1:22.0000  #Pred: 23  Sample:10
 3-0   TP:2  FP:27  FN:8: Prec:0.0690  Rec:0.2000  F1:39.0000  #Pred: 29  Sample:10
 0-1   TP:0  FP:7  FN:10: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 7  Sample:10
 0-3   TP:1  FP:13  FN:7: Prec:0.0714  Rec:0.1250  F1:44.0000  #Pred: 14  Sample:8
 4-0   TP:0  FP:0  FN:8: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:8
 2-0   TP:1  FP:9  FN:7: Prec:0.1000  Rec:0.1250  F1:36.0000  #Pred: 10  Sample:8
 0-5   TP:0  FP:1  FN:7: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 1  Sample:7
 0-0   TP:0  FP:8  FN:6: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 8  Sample:6
 6-0   TP:0  FP:0  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:3
 5-0   TP:0  FP:0  FN:3: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:3
10-10  TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 0-9   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 7-0   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
 0-6   TP:0  FP:0  FN:1: Prec:0.0000  Rec:0.0000  F1:0.0000  #Pred: 0  Sample:1
```

Seems that difference does not relate to performance of team. Values with very similar difference vectors ended up having a large array of different scores!