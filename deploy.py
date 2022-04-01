import streamlit as st
import pandas as pd
from PIL import Image

# Streamlit 
html_temp = """
<div style ="background-color:white;padding:3.5px">
<h1 style="color:black;text-align:center;">T IC3151 Artificial Intelligence</h1>
</div><br>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# Question 1 
st.title("Vacation Planner")
st.write("The vacation planning problem is solved by Genetic Algorithm (GA)." 
" Therefore, there are 3 types of GA method have been applied where Selection, Mutation and Crossover."
" The results of these methods have been compare as well. ")

if st.checkbox("Single Point Algorithm "):
    img = Image.open('single point algorithm.jpg')
    st.write("The Single Point Algorithm has included 0.05 %" " of selection and 0.01% " "of mutation."
    " For each sequence pair selected, a location is chosen from the"
    " genes in between, except for the first and last gene. This is the crossover point."
    " The genes that come after this point are mutually displaced in both sequences."
    " For this operation, the arrays must be of the same length.")
    st.image(img,width=500)
pass

if st.checkbox("Two Point Algorithm"):
    img = Image.open('two point algorithm.jpg')
    st.write("The Two Point Algorithm has included 0.1 %" " of selection and 0.1% " "of mutation."
    " In this method, two point are selected on the sequence, except for the"
    " first and last genes. The crossover process is the displacement of genes between these"
    " two selected points and genes are exchanged between these points.")
    st.image(img,width=500)
pass

if st.checkbox('Uniform Algorithm'):
    img = Image.open('uniform algorithm.jpg')
    st.write("The Uniform Algorithm has included 0.25 %" " of selection and 0.25% " "of mutation."
    " Changes are made between randomly selected pairs of chromosomes using a probability value.")
    st.image(img,width=500)
pass

st.write("")

if st.button('Overall Comparison'):
    img = Image.open('compare.jpg')
    st.write("Based on the table, it shows that the money saved for both Single Point Algorithm and Two Point Algorithm method"
    "is RM 2 while the Uniform Algorithm used up all the money.")
    st.image(img,width=500)
pass

# Question 2
st.title("Vaccine Distribution Modelling")
st.write("A vaccine distribution modelling has requested to support the national vaccination activity. The main task of this model"
" is to assign the right vaccine types and sufficient amount of vaccine to the vaccination center.")
st.write("There are three types of vaccine which Vaccine A is suitable for citizens that is 60 years old and above, Vaccine B is "
" mainly for citizens that age between 35 to 60 and vaccine C is suitable for citizens that is 35 years and below. Also, there are"
" 5 states which means that the vaccine have to distribute to every different state accordingly.")
st.write(" With this, the constraint satisfaction problem has formulate to solve the problem.")

if st.button("Overall Outputs"):
    img = Image.open("vaccine.jpg")
    st.write("The best solution that have the minimum rental and maximum vaccination per day has choosen by comparing each solution."
    "The amount of each vaccine type is calculating with the formula (Amount Vaccine Type Needed / Total Population of a State) * (Maximum Capacity of a State)."
    "Lastly, the day number of a state fully vacinated is calculate with the formula (Total Population of a State / Maximum Capacity of a State).")
    st.write("The final amount of vaccine needed, vaccination center, types of vaccine needed, rental and number of vaccination days needed in each state have"
    " distributed accordingly and displayed in a dataframe.")
    st.image(img,width=700)
pass 

# Dataset
st.set_option('deprecation.showPyplotGlobalUse', False)
@st.cache(allow_output_mutation=True)

# Read in Dataset
def load_data(n_rows):
    df = pd.read_csv("Bank_CreditScoring.csv")
    df.columns = df.columns.str.upper()
    return df

df = load_data(2350)

df_copy = df.copy()

X = df_copy.drop(['DECISION'],axis=1)
y = df_copy[['DECISION']]

X = pd.get_dummies(X, drop_first=True)

# Question 3
st.title("Loan Application Modelling")
st.write("A loan application modelling problem has been implemented with both supervised machine "
"learning method and unsupervised machine learning. The aim of this problem is to modify the parameters"
" to achieve the highest accuracy.")

st.markdown("### Description of the Dataset")
st.write("To create a loan application modelling, the dataset is first read in."
" The dataset consists of 2350 rows and 21 columns. The columns of the dataste has change from"
" lower case to upper case for a better view.")
st.write(df)

st.markdown("### Graph Visualizations")
st.write("There are types of visualization have plotted as correlation plot, bar charts and line graph." 
" These graphs have shown the characteristics and relationships among the attributes of the dataset."
" The graphs below mainly shows the characteristics over employment type in number of side income, monthly salary, score and loan amount."
" Also, the overall loan decision of employment type has shown as well.")

if st.checkbox('Correlation Plot'):
    img = Image.open('correlation.png')
    st.write("The overall correlation between each attributes has shown in the correlation plot. Based on the plot,"
    " the YEARS_TO_FINANCIAL_FREEDOM attibute is highly correlated to the CREDIT_CARD_MORE_THAN_MONTHS and NUMBER_OF_DEPENDETS."
    " with a correlation value of 0.8.")
    st.image(img,width=800)
pass

if st.checkbox("Number of Side Income"):
    img = Image.open('side income.png')
    img2 = Image.open('employee side income.png')
    st.write("The bar chart shows the total amount of side income among employment type. Based on the bar chart, "
    " employee has a total of 1211 side income which indicates the highest amount of side income among the other employment type.")
    st.image(img,width=800)
    st.write("From the bar chart above, it shows that employee has the highest total amount of side income. The number of side"
    " incomes of employee has determined and the results are compared as well. From the graph, it shows that 213 employees have"
    " 2 side incomes followed by 199 employees has 3 side income and lastly 188 of them have only 1 side income.")
    st.image(img2,width=800)
pass 

if st.checkbox("Monthly Salary"):
    img = Image.open('monthly salary.png')
    st.write("The line plot shows the maximum number of monthly salary among employment type. It shows that work as an employee"
    " has the maximum number of monthly salary where government employment type has the least number of monthly salary.")
    st.image(img,width=800)
pass 

if st.checkbox("Score"):
    img = Image.open('trust.png')
    st.write("The bar chart shows the overall score among employment type. There are 4 score rating given to each employment type as"
    " from the range 6 to 9. From the bar chart, employee has the high score in 9 compared to others which 9 indicates a highest score among"
    " the scoring system. Therefore, the overall score of employee is high and the financial status of employee is considered as trustable.")
    st.image(img,width=800)
pass

if st.checkbox("Loan Amount"):
    img = Image.open('loan amount.png')
    st.write("The line graph below shows the loan amount over employment type. Based on the line graph, it shows employee"
    " has the higher chance to borrow a higher number of loan compared to others. Also, employer has the lower chance to"
    " borrow a higher number of loan.")
    st.image(img,width=800)
pass 

if st.button("Overall Decision"):
    img = Image.open('decision.png')
    st.write("The bar chart shows the decision among the employment type. Employee has the highest accept decision compared to others. This"
    " may due to the employee has the most number of side incomes, highest amount of monthly salary and highest trust score. Therefore,"
    " it can be conclude that employee has the possibility to borrow a higher amount of loan and more number of loan to approve. Then, it indicates"
    " that the accept decision in employee type is higher than other employment type.")
    st.image(img, width=800)
pass

st.markdown("### Model Constructions")
st.write("For model construction, both supervised machine learning and unsupervised machine learning have been applied."
" There are two types of supervised machine learning method where Naive Bayes and Decision Tree Classifier. In addition,"
" K-Means clustering has applied as types of unsupervised machine learning method.")

if st.checkbox("Classification"):

    st.write("Classification is a two step process. It has been distributed into model construction and model usage."
    " Model construction describe a set of predetermined class and is determined by the class label attribute. Also, "
    "it represents classification rules, decision trees and mathematical formula.")
    st.write("Despite, model usage is used for classifying future or unknown objects. The known label of test sample is compared"
    " with the classified result from model construction. The accuracy shows how is the test sample are correctly classified by the model.")
    st.write("To begin with, the dataset has undergoes train test split and has split into 70""%" " of train dataset and 30""%" " for testing purpose."
    " The 'DECISION' attribute with the data of reject and accept has drop for testing purpose and defined as y variable."
    " Then, the remaining dataset has undergoes dummfying as it is suitable to label the categorical variable in binary forms.")
    st.write(X)
    
    st.write(" ")
    option = st.selectbox("Select a model", ['Naive Bayes','Decision Tree Classifier'])

    if option == 'Naive Bayes':
        img = Image.open('nb.png')
        st.write("Naive Bayes is a classification technique that based on the Bayesian theorem with an assumption of independence among predictors."
            " The bayes theorem provides a way of calculation of posterior probability like likelihood and prior probability. Also, Naive Bayes classifier"
            " handle missing values by ignoring the instance during probability estimate calculations.")
        st.write("The figure below shows both training and testing process in Naive Bayes classifier. The accuracy score of the classifier is 77% which is"
            " considered as a good model.")
        st.image(img,width=500)

    else:
        img = Image.open('dtree.jpg')
        img2 = Image.open('dtree2.jpg')
        st.write("Decision tree algorithm is able to solve for both classification and regression problems. In this case,"
        " the decision tree classifier is used to create a training model that can predict the value of the target variable by learning"
        " simple decision rules that inferred from train datasets.")
        st.write("Therefore, the algorithm starts from the root of the tree in terms to predict a class label. The tree is constructed"
        " in a top-down recursive divide-and-conquer manner. Also, it will only stop partitioning if there are no remaining attributes for further splitting."
        "With this, both gini and entropy criteria has include and compare to select the best classifier.")
        st.write("The figure shows the training and testing algorithm in Decision Tree Classifier. The accuracy with gini criteria is 76%.")
        st.image(img,width=600)
        st.write("As comparison, the accuracy score of Decision tree classifier with entropy criteria is 74% which is lower than classifier with gini criteria.")
        st.image(img2,width=600)
        st.write(" As result, gini criteria turns out to be better compared to entropy for selecting the best features. This may due to the gini index has values"
        " in the range of 0 to 0.5 whereas entropy in the range of 0 to 1. Furthermore, the maximum depth controls the overall complexity of the classifier."
        " It indicates how deep the tree can be. The more deeper the tree, the more splits it has and captures more information about the data.")
    pass

    if st.button('Overall Classifier Comparison'):
        img = Image.open('classification.jpg')
        st.write("As result, accuracy for both classification model has stored in a list. From the table, it shows that the Decision Tree Classifier"
        " has a higher accuracy score compared as Naive Bayes. Despite, the Decision Tree Classifier with gini criteria is more outperformed than classifier"
        " with entropy criteria.")
        st.image(img, width=400)
    pass
pass

if  st.checkbox("Cluster Analysis"):
    st.write("Cluster analysis is grouping a set of data objects into clusters. It is an unsupervised machine learning method where no predefined classes or"
    " dependent variable. A good clustering method will produce high quality clusters with high intra-class similarity and low inter-class similarity."
    " The clustering analysis that will be applied is the K-Means clustering.")

    if st.checkbox("K-Means Clustering"):
        st.write("K-Means clustering is types of partitioning algorithms by constructing various partitions and evaluate them by some criterion."
        " K-Means clustering is conceptually the most simple method that should be used first on new data.")
        st.write("The K-Means clustering has implement to find the best k clusters for the dataset. There are two types of method have been applied"
        " which included the elbow method and silhouette method.")
        st.write("First, the number of cluster 3 is implement. Then, the elbow method is plot as well to get the best number of clusters as for comparison.")

        option2 = st.selectbox("Select a method",['Elbow Method','Silhouette Score','Silhouette Visualizer'])

        if option2 == 'Elbow Method':
            img = Image.open('elbow.jpg')
            img2 = Image.open('clusters.png')
            st.write("Based on the plot, the point at which elbow shape is created at 4 where the k value and the optimal number of the cluster is 4."
            " Therefore, the K-Means model will now train for 4 clusters. ")
            st.image(img, width=500)
            st.write("Both original clusters and the new clusters are plot to show the difference. The plot shows the relationships between monthly salary"
            " and loan amount with the hue of decisions of loan status.")
            st.image(img2,width=700)
        
        elif option2 == 'Silhouette Score':
            img = Image.open('score.jpg')
            st.write("Silhouette score used to evaluate the quality of clusters that created with clusters algorithm. Its value is range from 1 to -1."
            " Tha value 1 represents the clusters are well apart from each other and clearly distinguished while value 0 represents clusters are indifferent,"
            " or it can be conclude that the distance between clusters is not significant. The value -1 defined clusters are assigned in the wrong way.")
            st.write("Therefore, for every k in k-means, there is a silhouette for it. Then, the silhouette score is calculated for clusters to show the performance"
            " of the K-Means algorithm. Based on the score, it shows that the performance of the cluster 3 is better than the cluster 4.")
            st.image(img,width=600)
        
        else:
            img = Image.open('visualize1.png')
            img2 = Image.open('visualize.png')
            st.write("The silhouette plot mainly shows the number of clusters and perform silhouette analysis to find the most appropriate clusters."
            " The yellow-brick library is used to draw the silhouette plots and perform comparative analysis. The thickness of the plot representing"
            " each cluster and is also a deciding point.")
            st.write("The silhouette visualizer for both cluster 3 and cluster 4 are shown as below: ")
            st.image(img,width=500)
            st.image(img2,width=500)
        pass 
    pass 

pass