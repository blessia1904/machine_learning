import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf 
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
import math
from tqdm import tqdm



def question123():
# 1. Using hotel_cancellation.csv Download hotel_cancellation.csv, write code to estimate the treatment effects if a ‘different room is 
# assigned’ as the treatment indicator and interpret its effect on the room being ‘canceled’. Use all the other columns 
# as the covariates. Write your observations for the results.


    Hotel_Cancellationdata = pd.read_csv("hotel_cancellation.csv")
    Hotel_Cancellationdata["different_room_assigned"] = (Hotel_Cancellationdata['different_room_assigned'] == True).astype(int)
    Hotel_Cancellationdata["is_canceled"] = (Hotel_Cancellationdata['is_canceled'] == True).astype(int)

    formula1 = 'is_canceled ~ different_room_assigned + days_in_waiting_list + arrival_date_day_of_month + arrival_date_week_number + arrival_date_year + lead_time'
    model2_1 = smf.glm(formula1,family=sm.families.Binomial(),data = Hotel_Cancellationdata)
    result2_1 = model2_1.fit()
    print(result2_1.summary())
    print(math.exp(result2_1.params["different_room_assigned"]))
    print("The value of beta co-effecient for different room assigned is" + str(result2_1.params["different_room_assigned"]))

    # The co-efficeints of different room assigned is negative with the magnitude of 2.55, and it's statistically significant,
    # meaing that whether be assigned a different room or not affects the likelihood for a customer canceling the room.
    # And that if people were assigned different rooms then the chances of cancellation are less.  
    # People who are assigned different room have (.08) times less odds than people who arent assigned a different room
    # Additionally, we see the parameters of arrival date are significant, because cancellations are affected by seasonality factors. 
    # Lead time is also a significant factor with books with higher lead time are prone to cancellations


# 2. now use double logistic regression to measure the effect of ‘different room is assigned’ on the room being ‘canceled’.

    # used smf.glm(family=Binomial) here, basically the same as sm.logit

    # Logit 1: Use Different room assigned as Y and regress it on all covariates (except is cancelled). 
    Logit1_formula = 'different_room_assigned ~ days_in_waiting_list + arrival_date_day_of_month + arrival_date_week_number + arrival_date_year + lead_time'
    Q2Model_Logit1 = smf.glm(Logit1_formula,family=sm.families.Binomial(),data = Hotel_Cancellationdata)

    X = Hotel_Cancellationdata.drop(columns=['different_room_assigned', 'is_canceled'])

    Q2Model_Logit1_Fit = Q2Model_Logit1.fit()
    print(Q2Model_Logit1_Fit.summary())

    # Then get the Y_hat (in class we refer it as h_hat) from this logit 
    Y_Hat = (Q2Model_Logit1_Fit.predict(X))

    Hotel_Cancellationdata["Y_Hat"] = Y_Hat

    # Logit 2: Use Is Cancelled as Y and regress it on all covariates including Y_Hat
    Logit2_formula = 'is_canceled ~ Y_Hat + different_room_assigned + days_in_waiting_list + arrival_date_day_of_month + arrival_date_week_number + arrival_date_year + lead_time'
    Q2Model_Logit2 = smf.glm(Logit2_formula,family=sm.families.Binomial(),data = Hotel_Cancellationdata)
    Q2Model_Logit2_Fit = Q2Model_Logit2.fit()

    print(Q2Model_Logit2_Fit.summary())
    print("The value of beta co-effecient for different room assigned is" + str(Q2Model_Logit2_Fit.params["different_room_assigned"]))


# 3. Use bootstrap to estimate the standard error of the treatment effects measured in (2).
    Q3_Hotel_Cancellationdata = Hotel_Cancellationdata.drop(columns=['Y_Hat'])
    DataSize = Q3_Hotel_Cancellationdata.shape[0]

    beta_coeff = np.zeros((1000))
    print("Total no of rows in data are:"+str(DataSize))

    for i in tqdm(range(0,1000)):
        Bootstrapsample = Q3_Hotel_Cancellationdata.sample(DataSize,replace=True)     
        
        Logit1_formula = 'different_room_assigned ~ days_in_waiting_list + arrival_date_day_of_month + arrival_date_week_number + arrival_date_year + lead_time'
        Q2Model_Logit1 = smf.glm(Logit1_formula,family=sm.families.Binomial(),data = Bootstrapsample)

        X = Bootstrapsample.drop(columns=['different_room_assigned', 'is_canceled'])

        Q2Model_Logit1_Fit = Q2Model_Logit1.fit()

        Y_Hat = (Q2Model_Logit1_Fit.predict(X))

        Bootstrapsample["Y_Hat"] = Y_Hat

        Logit2_formula = 'is_canceled ~ Y_Hat + different_room_assigned + days_in_waiting_list + arrival_date_day_of_month + arrival_date_week_number + arrival_date_year + lead_time'
        Q2Model_Logit2 = smf.glm(Logit2_formula,family=sm.families.Binomial(),data = Bootstrapsample)
        Q2Model_Logit2_Fit = Q2Model_Logit2.fit()

        beta_coeff[i] = Q2Model_Logit2_Fit.params["different_room_assigned"]

    coeff_mean = np.mean(beta_coeff, axis=0)
    coeff_se = np.std(beta_coeff, axis=0)
    print("After bootstrapping, the mean estimate of beta co-efficient for different room assigned is"+str(coeff_mean))
    print("After bootstrapping, the SE of beta co-efficient estimator is"+str(coeff_se))
    # the results from bootstrapping are really close to what I got from double logistics.



def question4():
# Keeping 21 as the threshold for age, explore the data with an RDD by writing very simple code (no package needed, 
# just average to one side of the threshold minus average to the other side) to determine if alcohol increases 
# the chances of death by accident, suicide and/or others (the three given columns) 
# and comment on the question “Should the legal age for drinking be reduced from 21?” based on the results. 
# For this problem, choose the bandwidth to be 1 year (i.e., 21 +- 1). 

    drinking = pd.read_csv("drinking.csv")
    drinking = drinking.dropna()

    # untreated -- age 20 to 21
    below = drinking[(drinking['age'] >= 20) & (drinking['age'] < 21)]
    # print(below)
    # treated -- age 21 to 22
    above = drinking[(drinking['age'] > 21) & (drinking['age'] <= 22)]
    # print(above)

    below_others = below['others'].mean()
    below_accident = below['accident'].mean()
    below_suicide = below['suicide'].mean()
    
    above_others = above['others'].mean()
    above_accident = above['accident'].mean()
    above_suicide = above['suicide'].mean()

    others_TE = above_others - below_others
    accident_TE = above_accident - below_accident
    suicide_TE = above_suicide - below_suicide

    Q4table = pd.DataFrame({"Age from 20 to 21": [below_accident, below_suicide, below_others],
                   "Age from 21 to 22": [above_accident, above_suicide, above_others],
                   "Marginal treatment effect": [accident_TE, suicide_TE, others_TE],
                   "Result": ["Accident", "Suicide", "Others"]}).set_index("Result")
    
    print(Q4table)

# Should the legal age for drinking be reduced from 21?
    # No, since the TE are all positives, meaning alcohol might have a causal effect on increasing chances of 
    # death by accidenyt, suicide, and others. Thus, it's better not to reduce the legal drinking age.


# Plot graphs to show the discontinuity (if any) and to show results for the change in chances of death with all the three features. 
    x_new = drinking[(drinking['age'] >= 20) & (drinking['age'] <= 22)]
    age_x = x_new[['age']]

    # others vs age
    y_others = x_new[['others']]
    plt.scatter(age_x, y_others)
    plt.axvline(x = 21, color = 'b')
    plt.title("others vs age")
    plt.xlabel("Age from 20 to 22")
    plt.ylabel("Others")
    plt.show()

    # accident vs age
    y_accident = x_new[['accident']]
    plt.scatter(age_x, y_accident)
    plt.axvline(x = 21, color = 'b')
    plt.title("accident vs age")
    plt.xlabel("Age from 20 to 22")
    plt.ylabel("Accident")
    plt.show()

    # suicide vs age
    y_suicide = x_new[['suicide']]
    plt.scatter(age_x, y_suicide)
    plt.axvline(x = 21, color = 'b')
    plt.title("suicide vs age")
    plt.xlabel("Age from 20 to 22")
    plt.ylabel("Suicide")
    plt.show()


# What might be the effect of choosing a smaller bandwidth? 
    # In this case, I used age 21+- 0.5 as an example. 
    # Smaller bandwidth means we are choosing values that are even closer to the threshold value. 
    # And thoretically, this should reduce bias and increase precision of estimating the treatment effect since the values are more likely to be similar. 
    # Based on the resulted table, marginal treatment effect is magnified: accident, suicide, and others all increased.
    # However, smaller bandwidth also means smaller sample size, which can lead to lower statistical power and larger standard errors.

    # untreated -- age 20.5 to 21
    below_small = drinking[(drinking['age'] >= 20.5) & (drinking['age'] < 21)]

    # treated -- age 21 to 21.5
    above_small = drinking[(drinking['age'] > 21) & (drinking['age'] <= 21.5)]

    sbelow_others = below_small['others'].mean()
    sbelow_accident = below_small['accident'].mean()
    sbelow_suicide = below_small['suicide'].mean()
    
    sabove_others = above_small['others'].mean()
    sabove_accident = above_small['accident'].mean()
    sabove_suicide = above_small['suicide'].mean()

    sothers_TE = sabove_others - sbelow_others
    saccident_TE = sabove_accident - sbelow_accident
    ssuicide_TE = sabove_suicide - sbelow_suicide

    Q4table_small = pd.DataFrame({"Age from 20.5 to 21": [sbelow_accident, sbelow_suicide, sbelow_others],
                   "Age from 21 to 21.5": [sabove_accident, sabove_suicide, sabove_others],
                   "Marginal treatment effect": [saccident_TE, ssuicide_TE, sothers_TE],
                   "Result": ["Accident", "Suicide", "Others"]}).set_index("Result")
    
    print(Q4table_small)


# What if we chose the maximum bandwidth?
    # In this case, I used age 21+- 2 as an example. 
    # Means that we are choosing values aroung the threshold as many as possible. 
    # And this can increase bias and decrease precision of estimating the treatment effect since values are more likely to be un-similar,
    # allowing cofounders to confuse the effect, thus less likely to reflect the intervention(alcohol)'s effect.
    # Based on the resulted table, marginal treatment effect of accident is reversed, suicide and others both decreased, which seems not reasonable.

    # untreated -- age 19 to 21
    below_large = drinking[(drinking['age'] >= 19) & (drinking['age'] < 21)]
    # print(below)
    # treated -- age 21 to 23
    above_large = drinking[(drinking['age'] > 21) & (drinking['age'] <= 23)]
    # print(above)

    lbelow_others = below_large['others'].mean()
    lbelow_accident = below_large['accident'].mean()
    lbelow_suicide = below_large['suicide'].mean()
    
    labove_others = above_large['others'].mean()
    labove_accident = above_large['accident'].mean()
    labove_suicide = above_large['suicide'].mean()

    lothers_TE = labove_others - lbelow_others
    laccident_TE = labove_accident - lbelow_accident
    lsuicide_TE = labove_suicide - lbelow_suicide

    Q4table_large = pd.DataFrame({"Age from 19 to 21": [lbelow_accident, lbelow_suicide, lbelow_others],
                   "Age from 21 to 23": [labove_accident, labove_suicide, labove_others],
                   "Marginal treatment effect": [laccident_TE, lsuicide_TE, lothers_TE],
                   "Result": ["Accident", "Suicide", "Others"]}).set_index("Result")
    
    print(Q4table_large)


def question5():
# How does the performance of k-nearest neighbors change as k takes on the following values: 1, 3, 5, 7? 

    iris = pd.read_csv("iris.csv")
    x1 = iris.iloc[:, :-1]
    y1 = iris[['variety']]
    # split dataset into training and testing
    x1_train , x1_test , y1_train , y1_test = train_test_split(x1, y1, test_size= 0.25 , random_state=44)

    # standardize data (since we are not told whether the units/sacles is the same in the dataset)
    scaler = StandardScaler()
    x1_train = scaler.fit_transform(x1_train)
    x1_test = scaler.fit_transform(x1_test)

    # using metric = euclidean (the one I would choose for final use)
    k_values = [1, 3, 5, 7]
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(x1_train, np.ravel(y1_train))
        y_pred_i = knn.predict(x1_test)
        accuracy = accuracy_score(y_pred_i, y1_test)
        print("metric = euclidean, k =", k, ": accuracy =", accuracy)

    # using metric = manhattan
    k_values = [1, 3, 5, 7]
    for k in k_values:
        knn_m = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
        knn_m.fit(x1_train, np.ravel(y1_train))
        y_pred_m = knn_m.predict(x1_test)
        accuracy = accuracy_score(y_pred_m, y1_test)
        print("metric = manhattan, k =", k, ": accuracy =", accuracy)

    # using metric = cosine
    k_values = [1, 3, 5, 7]
    for k in k_values:
        knn_c = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn_c.fit(x1_train, np.ravel(y1_train))
        y_pred_c = knn_c.predict(x1_test)
        accuracy = accuracy_score(y_pred_c, y1_test)
        print("metric = cosine, k =", k, ": accuracy =", accuracy)

# Which of these is the optimal value of k? 
    # Based on above results, the optimal value of k in each metric is when :
    # k=7(for euclidean metric) where accuracy is the highest ~97%; 
    # k=1,5,7(for manhattan metric) where accuracy is the highest ~97%; 
    # k= 1,3 (cosine metric) where accuracy is the highest ~89%. 

# Which distance/similarity metric did you choose to use and why?
    # I would choose euclidean as the distance metric because:
    # it only has one optimal k in this case, and the accuracy rate is high (same as manhattan)
    # it is based on the straight line distance between two points, which is easy to understand and interpret;
    # it works well with continuous data and can be calculated efficiently;
    # it can deal with outliers, so it is less affected by extreme values in the data.


if __name__ == '__main__':
    question123()
    question4()
    question5()

