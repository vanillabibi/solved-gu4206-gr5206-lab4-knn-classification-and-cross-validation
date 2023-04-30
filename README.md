Download Link: https://assignmentchef.com/product/solved-gu4206-gr5206-lab4-knn-classification-and-cross-validation
<br>
Make sure that you upload a knitted pdf or html file to the canvas page (this should have a .pdf or .html extension). Also upload the .Rmd file. Include output for each question in its own individual code chunk and don’t print out any vector that has more than 20 elements.

Objectives: KNN Classification and Cross-Validation

<h1>Background</h1>

Today we’ll be using the <em>Weekly </em>dataset from the <em>ISLR </em>package. This data is similar to the <em>Smarket </em>data from class. The dataset contains 1089 weekly returns from the beginning of 1990 to the end of 2010. Make sure that you have the <em>ISLR </em>package installed and loaded by running (without the code commented out) the following:

<em># install.packages(“ISLR”) </em><strong>library</strong>(ISLR)

## Warning: package ‘ISLR’ was built under R version 4.0.2

We’d like to see if we can accurately predict the direction of a week’s return based on the returns over the last five weeks. <em>Today </em>gives the percentage return for the week considered and <em>Year </em>provides the year that the observation was recorded. <em>Lag1 </em>– <em>Lag5 </em>give the percentage return for 1 – 5 weeks previous and <em>Direction </em>is a factor variable indicating the direction (‘UP’ or ‘DOWN’) of the return for the week considered.

<strong>Part 1: Visualizing the relationship between this week’s returns and the previous week’s returns.</strong>

<ol>

 <li>Explore the relationship between a week’s return and the previous week’s return. You should plot more graphs for yourself, but include in the lab write-up a scatterplot of the returns for the weeks considered (<em>Today</em>) vs the return from two weeks previous (<em>Lag2</em>), and side-by-side boxplots for the lag one week previous (<em>Lag1</em>) divided by the direction of this week’s Reuther (<em>Direction</em>).</li>

</ol>

<h2>Returns</h2>

Two Weeks Ago

<h2>Returns</h2>

Direction

<h1>Part 2: Building a classifier</h1>

Recall the KNN procedure. We classify a new point with the following steps: – Calculate the Euclidean distance between the new point and all other points.

<ul>

 <li>Create the set <em>N<sub>new </sub></em>containing the <em>K </em>closest points (or, nearest neighbors) to the new point.</li>

 <li>Determine the number of ‘UPs’ and ‘DOWNs’ in <em>N<sub>new </sub></em>and classify the new point according to the most frequent.

  <ol start="2">

   <li>We’d like to perform KNN on the <em>Weekly </em>data, as we did with the <em>Smarket </em>data in class. In class we wrote the following function which takes as input a new point (<em>Lag</em>1<em><sub>new</sub>,Lag</em>2<em><sub>new</sub></em>) and provides the KNN decision using as defaults <em>K </em>= 5, Lag1 data given in <em>Smarket$Lag1</em>, and Lag2 data given in <em>Smarket$Lag2</em>. Update the function to calculate the KNN decision for weekly market direction using the <em>Weekly </em>dataset with <em>Lag1 </em>– <em>Lag5 </em>as predictors. Your function should have only three input values: (1) a new point which should be a vector of length 5, (2) a value for K, and (3) the Lag data which should be a data frame with five columns (and n rows).</li>

  </ol></li>

</ul>

<table width="632">

 <tbody>

  <tr>

   <td width="632">KNN.decision &lt;- <strong>function</strong>(Lag1.new, Lag2.new, K = 5, Lag1 = Smarket<strong>$</strong>Lag1, Lag2 = Smarket<strong>$ </strong>n &lt;- <strong>length</strong>(Lag1)<strong>stopifnot</strong>(<strong>length</strong>(Lag2) <strong>== </strong>n, <strong>length</strong>(Lag1.new) <strong>== </strong>1, <strong>length</strong>(Lag2.new) <strong>== </strong>1, K <strong>&lt;= </strong>n)dists &lt;- <strong>sqrt</strong>((Lag1<strong>–</strong>Lag1.new)<strong>^</strong>2 <strong>+ </strong>(Lag2<strong>–</strong>Lag2.new)<strong>^</strong>2) neighbors &lt;- <strong>order</strong>(dists)[1<strong>:</strong>K] neighb.dir &lt;- Smarket<strong><sub>$</sub></strong>Direction[neighbors] choice         &lt;- <strong>names</strong>(<strong>which.max</strong>(<strong>table</strong>(neighb.dir))) <strong>return</strong>(choice)}</td>

  </tr>

 </tbody>

</table>

Lag2) {

<ol start="3">

 <li>Now train your model using data from 1990 – 2008 and use the data from 2009-2010 as test data. To do this, divide the data into two data frames, <em>test </em>and <em>train</em>. Then write a loop that iterates over the test points in the test dataset calculating a prediction for each based on the training data with <em>K </em>= 5. Save these predictions in a vector. Finally, calculate your test error, which you should store as a variable named <em>error</em>. The test error calculates the proportion of your predictions which are incorrect (don’t match the actual directions).</li>

 <li>Do the same thing as in question 3, but instead use <em>K </em>= 3. Which has a lower test error?</li>

</ol>

<h1>Part 3: Cross-validation</h1>

Ideally we’d like to use our model to predict future returns, but how do we know which value of <em>K </em>to choose? We could choose the best value of <em>K </em>by training with data from 1990 – 2008, testing with the 2009 – 2010 data, and selecting the model with the lowest test error as in the previous section. However, in order to build the best model, we’d like to use ALL the data we have to train the model. In this case, we could use all of the <em>Weekly </em>data and choose the best model by comparing the training error, but unfortunately this isn’t usually a good predictor of the test error.

In this section, we instead consider a class of methods that estimate the test error rate by holding out a

(random) subset of the data to use as a test set, which is called <em>k</em>-fold cross validation. (Note this lower case k is different than the upper case K in KNN. They have nothing to do with each other, it just happens that the standard is to use the same letter in both.) This approach involves randomly dividing the set of observations into <em>k </em>groups, or folds, of equal size. The first fold is treated as a test set, and the model is fit on the remaining <em>k − </em>1 folds. The error rate, ERR1, is then computed on the observations in the held-out fold. This procedure is repeated <em>k </em>times; each time, a different group of observations is treated as a test set. This process results in <em>k </em>estimates of the test error: ERR1, ERR2, …, ERRk. The <em>k</em>-fold CV estimate of the test error is computed by averaging these values,

<em>k</em>

<h2>1 X</h2>

<em>CV</em>(<em>k</em>) =     <em>         ERR</em><em>k.</em>

<em>k</em>

<em>i</em>=1

We’ll run a 9-fold cross-validation in the following. Note that we have 1089 rows in the dataset, so each fold will have exactly 121 members.

<ol start="5">

 <li>Create a vector <em>fold </em>which has <em>n </em>elements, where <em>n </em>is the number of rows in <em>Weekly</em>. We’d like for the <em>fold </em>vector to take values in 1-9 which assign each corresponding row of the <em>Weekly </em>dataset to a fold. Do this in two steps: (1) create a vector using <em>rep() </em>with the values 1-9 each repeated 121 times (note 1089 = 121 9), and (2) use <em>sample() </em>to randomly reorder the vector you created in (1).</li>

 <li>Iterate over the 9 folds, treating a different fold as the test set and all others the training set in each iteration. Using a KNN classifier with <em>K </em>= 5 calculate the test error for each fold. Then calculate the cross-validation approximation to the test error which is the average of ERR1, ERR2, …, ERR9.</li>

 <li>Repeat step (6) for <em>K </em>= 1, <em>K </em>= 3, and <em>K </em>= 7. For which set is the cross-validation approximation to the test error the lowest?</li>

</ol>