# Tensorflow_DuplicateModel
I will try to create model that doubles the input given by using tensorflow, keras, pandas, numpy

# What this code contains and what it does 
My aim before starting this code was just creating a simple tensorflow model. Only to see how can you make a model work on embedded devices. But before i go with the embedded size. I just want to create a simple tf model. Then Convert it to the tflite model to use in mcu's.
and of course i wanted to build in on my pc before i go deeper into the embedded device. So this code creates a model and converts it to the tflite model for further use. Surely you can create better, complex and simpler models. But that is how i did it. C++ implementation of the code will be uploaded by me in the future. 

# GRAPHS
Loss and prediction graphs

![lossplot](https://github.com/Sedatekinci4/Tensorflow_DuplicateModel/assets/57107943/0c5c7eee-3439-4e6e-9c65-8b32b89e3fba)

![pred_graph](https://github.com/Sedatekinci4/Tensorflow_DuplicateModel/assets/57107943/a9c50bfd-2b83-4949-bbdc-c948d26df500)

-----------------------------
EXAMPLE OUTPUT (given data is 100.5)
------------------------------------------------------------------------
```diff
2.13.0

Input data has been created.
The input data is:

[[ 1.]
 [ 2.]
 [ 3.]
 [ 4.]
 [ 5.]
 [ 6.]
 [ 7.]
 [ 8.]
 [ 9.]
 [10.]
 [11.]
 [12.]
 [13.]
 [14.]
 [15.]
 [16.]
 [17.]
 [18.]
 [19.]
 [20.]
 [21.]
 [22.]
 [23.]
 [24.]
 [25.]
 [26.]
 [27.]
 [28.]
 [29.]
 [30.]
 [31.]
 [32.]
 [33.]
 [34.]
 [35.]
 [36.]
 [37.]
 [38.]
 [39.]
 [40.]
 [41.]
 [42.]
 [43.]
 [44.]
 [45.]
 [46.]
 [47.]
 [48.]
 [49.]
 [50.]]
[[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17. 18.
  19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35. 36.
  37. 38. 39. 40. 41. 42. 43. 44. 45. 46. 47. 48. 49. 50.]]

The output data is:

[[  2.]
 [  4.]
 [  6.]
 [  8.]
 [ 10.]
 [ 12.]
 [ 14.]
 [ 16.]
 [ 18.]
 [ 20.]
 [ 22.]
 [ 24.]
 [ 26.]
 [ 28.]
 [ 30.]
 [ 32.]
 [ 34.]
 [ 36.]
 [ 38.]
 [ 40.]
 [ 42.]
 [ 44.]
 [ 46.]
 [ 48.]
 [ 50.]
 [ 52.]
 [ 54.]
 [ 56.]
 [ 58.]
 [ 60.]
 [ 62.]
 [ 64.]
 [ 66.]
 [ 68.]
 [ 70.]
 [ 72.]
 [ 74.]
 [ 76.]
 [ 78.]
 [ 80.]
 [ 82.]
 [ 84.]
 [ 86.]
 [ 88.]
 [ 90.]
 [ 92.]
 [ 94.]
 [ 96.]
 [ 98.]
 [100.]]

Array concatenated..

[[  1.   2.]
 [  2.   4.]
 [  3.   6.]
 [  4.   8.]
 [  5.  10.]
 [  6.  12.]
 [  7.  14.]
 [  8.  16.]
 [  9.  18.]
 [ 10.  20.]
 [ 11.  22.]
 [ 12.  24.]
 [ 13.  26.]
 [ 14.  28.]
 [ 15.  30.]
 [ 16.  32.]
 [ 17.  34.]
 [ 18.  36.]
 [ 19.  38.]
 [ 20.  40.]
 [ 21.  42.]
 [ 22.  44.]
 [ 23.  46.]
 [ 24.  48.]
 [ 25.  50.]
 [ 26.  52.]
 [ 27.  54.]
 [ 28.  56.]
 [ 29.  58.]
 [ 30.  60.]
 [ 31.  62.]
 [ 32.  64.]
 [ 33.  66.]
 [ 34.  68.]
 [ 35.  70.]
 [ 36.  72.]
 [ 37.  74.]
 [ 38.  76.]
 [ 39.  78.]
 [ 40.  80.]
 [ 41.  82.]
 [ 42.  84.]
 [ 43.  86.]
 [ 44.  88.]
 [ 45.  90.]
 [ 46.  92.]
 [ 47.  94.]
 [ 48.  96.]
 [ 49.  98.]
 [ 50. 100.]]

Added the column names for better look...


Dataset has been created....

    Input  Output
0     1.0     2.0
1     2.0     4.0
2     3.0     6.0
3     4.0     8.0
4     5.0    10.0
5     6.0    12.0
6     7.0    14.0
7     8.0    16.0
8     9.0    18.0
9    10.0    20.0
10   11.0    22.0
11   12.0    24.0
12   13.0    26.0
13   14.0    28.0
14   15.0    30.0
15   16.0    32.0
16   17.0    34.0
17   18.0    36.0
18   19.0    38.0
19   20.0    40.0
20   21.0    42.0
21   22.0    44.0
22   23.0    46.0
23   24.0    48.0
24   25.0    50.0
25   26.0    52.0
26   27.0    54.0
27   28.0    56.0
28   29.0    58.0
29   30.0    60.0
30   31.0    62.0
31   32.0    64.0
32   33.0    66.0
33   34.0    68.0
34   35.0    70.0
35   36.0    72.0
36   37.0    74.0
37   38.0    76.0
38   39.0    78.0
39   40.0    80.0
40   41.0    82.0
41   42.0    84.0
42   43.0    86.0
43   44.0    88.0
44   45.0    90.0
45   46.0    92.0
46   47.0    94.0
47   48.0    96.0
48   49.0    98.0
49   50.0   100.0

Train datas are:

    Input  Output
28   29.0    58.0
11   12.0    24.0
10   11.0    22.0
41   42.0    84.0
2     3.0     6.0
27   28.0    56.0
38   39.0    78.0
31   32.0    64.0
22   23.0    46.0
4     5.0    10.0
33   34.0    68.0
35   36.0    72.0
26   27.0    54.0
34   35.0    70.0
18   19.0    38.0
7     8.0    16.0
14   15.0    30.0
45   46.0    92.0
48   49.0    98.0
29   30.0    60.0
15   16.0    32.0
30   31.0    62.0
32   33.0    66.0
16   17.0    34.0
42   43.0    86.0
20   21.0    42.0
43   44.0    88.0
8     9.0    18.0
13   14.0    28.0
25   26.0    52.0
5     6.0    12.0
17   18.0    36.0
40   41.0    82.0
49   50.0   100.0
1     2.0     4.0
12   13.0    26.0
37   38.0    76.0
24   25.0    50.0
6     7.0    14.0
23   24.0    48.0

Test features are:

    Input  Output
0     1.0     2.0
3     4.0     8.0
9    10.0    20.0
19   20.0    40.0
21   22.0    44.0
36   37.0    74.0
39   40.0    80.0
44   45.0    90.0
46   47.0    94.0
47   48.0    96.0
        count    mean        std  min    25%   50%    75%    max
Input    40.0  25.025  13.743973  2.0  13.75  25.5  35.25   50.0
Output   40.0  50.050  27.487946  4.0  27.50  51.0  70.50  100.0

Train features are:

    Input
28   29.0
11   12.0
10   11.0
41   42.0
2     3.0
27   28.0
38   39.0
31   32.0
22   23.0
4     5.0
33   34.0
35   36.0
26   27.0
34   35.0
18   19.0
7     8.0
14   15.0
45   46.0
48   49.0
29   30.0
15   16.0
30   31.0
32   33.0
16   17.0
42   43.0
20   21.0
43   44.0
8     9.0
13   14.0
25   26.0
5     6.0
17   18.0
40   41.0
49   50.0
1     2.0
12   13.0
37   38.0
24   25.0
6     7.0
23   24.0

Train labels are:

28     58.0
11     24.0
10     22.0
41     84.0
2       6.0
27     56.0
38     78.0
31     64.0
22     46.0
4      10.0
33     68.0
35     72.0
26     54.0
34     70.0
18     38.0
7      16.0
14     30.0
45     92.0
48     98.0
29     60.0
15     32.0
30     62.0
32     66.0
16     34.0
42     86.0
20     42.0
43     88.0
8      18.0
13     28.0
25     52.0
5      12.0
17     36.0
40     82.0
49    100.0
1       4.0
12     26.0
37     76.0
24     50.0
6      14.0
23     48.0
Name: Output, dtype: float64

Test features are:

    Input
0     1.0
3     4.0
9    10.0
19   20.0
21   22.0
36   37.0
39   40.0
44   45.0
46   47.0
47   48.0

Test labels are:

0      2.0
3      8.0
9     20.0
19    40.0
21    44.0
36    74.0
39    80.0
44    90.0
46    94.0
47    96.0
Name: Output, dtype: float64
          mean        std
Input   25.025  13.743973
Output  50.050  27.487946
2023-09-22 14:58:12.429151: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
[[25.025]]
First
    Input
28   29.0
11   12.0
10   11.0
41   42.0
2     3.0
27   28.0
38   39.0
31   32.0
22   23.0
4     5.0
33   34.0
35   36.0
26   27.0
34   35.0
18   19.0
7     8.0
14   15.0
45   46.0
48   49.0
29   30.0
15   16.0
30   31.0
32   33.0
16   17.0
42   43.0
20   21.0
43   44.0
8     9.0
13   14.0
25   26.0
5     6.0
17   18.0
40   41.0
49   50.0
1     2.0
12   13.0
37   38.0
24   25.0
6     7.0
23   24.0
First example: [[29.]]

Normalized: [[0.29]]
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 normalization_1 (Normaliza  (None, 1)                 3         
 tion)                                                           
                                                                 
 dense (Dense)               (None, 1)                 2         
                                                                 
=================================================================
Total params: 5 (24.00 Byte)
Trainable params: 2 (8.00 Byte)
Non-trainable params: 3 (16.00 Byte)
_________________________________________________________________
None
1/1 [==============================] - 0s 47ms/step
[[-0.455]
 [ 1.49 ]
 [ 1.604]
 [-1.942]
 [ 2.519]
 [-0.34 ]
 [-1.598]
 [-0.798]
 [ 0.232]
 [ 2.29 ]]
         loss  val_loss  epoch
295  0.053507  0.042991    295
296  0.041221  0.023393    296
297  0.033780  0.038718    297
298  0.042709  0.027752    298
299  0.027786  0.016842    299
8/8 [==============================] - 0s 0s/step
1/1 [==============================] - 0s 16ms/step

The guessed value for the given input is: 
[[200.883]]
              Output Error
linear_model      0.024977
2023-09-22 14:58:22.970872: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
2023-09-22 14:58:22.971009: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
2023-09-22 14:58:22.971803: I tensorflow/cc/saved_model/reader.cc:45] Reading SavedModel from: C:\Users\tades\PycharmProjects\ModelforDuplicate\duplicate_model.pb
2023-09-22 14:58:22.972964: I tensorflow/cc/saved_model/reader.cc:91] Reading meta graph with tags { serve }
2023-09-22 14:58:22.973084: I tensorflow/cc/saved_model/reader.cc:132] Reading SavedModel debug info (if present) from: C:\Users\tades\PycharmProjects\ModelforDuplicate\duplicate_model.pb
2023-09-22 14:58:22.974961: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:375] MLIR V1 optimization pass is not enabled
2023-09-22 14:58:22.975587: I tensorflow/cc/saved_model/loader.cc:231] Restoring SavedModel bundle.
2023-09-22 14:58:22.999051: I tensorflow/cc/saved_model/loader.cc:215] Running initialization op on SavedModel bundle at path: C:\Users\tades\PycharmProjects\ModelforDuplicate\duplicate_model.pb
2023-09-22 14:58:23.006288: I tensorflow/cc/saved_model/loader.cc:314] SavedModel load for tags { serve }; Status: success: OK. Took 34485 microseconds.
2023-09-22 14:58:23.014579: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:255] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.

Process finished with exit code 0
