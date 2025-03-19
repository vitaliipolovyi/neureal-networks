# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config
def dummy_npwarn_decorator_factory():
  def npwarn_decorator(x):
    return x
  return npwarn_decorator
np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)

# %%
df = pd.read_csv('car_data.csv')

# %%
df['Gender'] = df['Gender'].astype('category')
df['Gender_Code'] = df['Gender'].cat.codes

# %%
df

# %%
df.isnull().sum()

# %%
df.duplicated().sum()

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
fig,ax=plt.subplots(2,3,figsize=(25,15))
sns.distplot(df['Age'],ax=ax[0,0])
sns.boxplot(y=df['Age'],ax=ax[0,1])
sns.histplot(data=df,x='Age',ax=ax[0,2],hue='Purchased',kde=True)

sns.distplot(df['AnnualSalary'],ax=ax[1,0])
sns.boxplot(y=df['AnnualSalary'],ax=ax[1,1])
sns.histplot(data=df,x='AnnualSalary',ax=ax[1,2],hue='Purchased',kde=True)
    
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.suptitle("Visualizing Continuous Columns",fontsize=30)

# %%
df.drop(['User ID'],axis=1,inplace=True)

# %%
sns.countplot(data=df, x='Gender')

# %%
plt.pie(df['Purchased'].value_counts(),labels=df['Purchased'].value_counts().index,autopct='%.2f',explode=[0,0.1])
plt.title("Class Imbalance")
plt.show()

# %%
sns.pairplot(df,hue='Purchased')
plt.show()

# %%
sns.heatmap(df[['Age', 'AnnualSalary', 'Gender_Code']].corr(), annot=True)

# %%
x = df.drop(columns = ['Purchased'])
x

# %%
y = df[['Purchased']]
y.info()

# %%
x.info()

# %%
from sklearn.preprocessing import StandardScaler     
from sklearn.preprocessing import OneHotEncoder                    
from sklearn.compose import make_column_transformer
from sklearn import set_config
from sklearn.model_selection import train_test_split
from imblearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler
#Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn import tree
import graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

# %%
def scores(method, pipe, x_train, y_train, x_test, y_test):
    list = []
    
    list.append({
        'Method': method,
        'Score': 'Default',
        'Train': '{: .1%}'.format(pipe.score(x_train, y_train)),
        'Test': '{: .1%}'.format(pipe.score(x_test, y_test))
    })
        
    predict_train = pipe.predict(x_train)
    predict_test = pipe.predict(x_test)

    scores = [accuracy_score, precision_score, recall_score, f1_score]
    for score in scores:
        scores_train = score(y_train, predict_train)
        scores_test = score(y_test, predict_test)
        
        list.append({
            'Method': method,
            'Score': score.__name__,
            'Train': '{: .1%}'.format(scores_train),
            'Test': '{: .1%}'.format(scores_test)
        })
        
    list.append({
        'Method': method,
        'Score': roc_auc_score.__name__,
        'Train': '{: .3}'.format(roc_auc_score(y_train, predict_train)),
        'Test': '{: .3}'.format(roc_auc_score(y_test, predict_test))
    })
    
    confusion_matrix_result = confusion_matrix(y_test, predict_test)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result, display_labels=[0, 1])
    cm_display.plot()
    plt.show()

    train_probs = pipe.predict_proba(x_train)
    train_probs = train_probs[:, 1]
    test_probs = pipe.predict_proba(x_test)
    test_probs = test_probs[:, 1]
    
    fpr_train, tpr_train, _ = roc_curve(y_train, train_probs)
    plt.plot(fpr_train, tpr_train, marker='.', label='Train')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    fpr_test, tpr_test, _ = roc_curve(y_test, test_probs)
    plt.plot(fpr_test, tpr_test, marker='.', label='Test')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    #df = pd.DataFrame(list, columns=['Score', 'Train', 'Test'])    
    
    return list

# %%
trans = make_column_transformer(
    (OneHotEncoder(), [0]),
    (StandardScaler(),[1, 2]), 
    remainder = 'passthrough'
)
set_config(display = 'diagram')
trans

# %%
sns.countplot(y, x='Purchased')

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

# %%
lr = LogisticRegression()
pipe_lr = make_pipeline(trans, lr)

# %%
scores_list = []

# %%
pipe_lr.fit(x_train, y_train)

# %%
scrs = scores('Logistic regression', pipe_lr, x_train, y_train, x_test, y_test)
df_scrs = pd.DataFrame(scrs)
df_scrs.set_index('Score', inplace=True)
df_scrs.drop(columns=['Method'], axis=1, inplace=True)
df_scrs

# %%
ROS = RandomOverSampler()
o_x, o_y = ROS.fit_resample(x, y)
sns.countplot(o_y, x='Purchased')

# %%
o_x_train, o_x_test, o_y_train, o_y_test = train_test_split(o_x, o_y, test_size=0.3, shuffle=False)

# %%
pipe_lr.fit(o_x_train, o_y_train)

# %%
scrs = scores('Linear Regression', pipe_lr, o_x_train, o_y_train, o_x_test, o_y_test)
df_scrs = pd.DataFrame(scrs)
df_scrs.set_index('Score', inplace=True)
df_scrs.drop(columns=['Method'], axis=1, inplace=True)
df_scrs

# %%
pipe_s_lr = make_pipeline(trans, ROS, lr)
#pipe_s_lr

# %%
scores_list.extend(
  scores('Linear Regression', pipe_s_lr, x_train, y_train, x_test, y_test)
)

# %%
dt = DecisionTreeClassifier(max_depth=5)
pipe_s_dt = make_pipeline(trans, ROS, dt)
#pipe_s_dt

# %%
pipe_s_dt.fit(x_train, y_train)

# %%
scores_list.extend(
  scores('Decision Tree', pipe_s_dt, x_train, y_train, x_test, y_test)
)

# %%
x_test.columns

# %%
dot_data = tree.export_graphviz(
    pipe_s_dt['decisiontreeclassifier'],
    feature_names=df.columns, 
    class_names=y_train['Purchased'].unique().astype('str'),
    filled=True
)

graph = graphviz.Source(dot_data) 
graph

# %%
svc = SVC(kernel="linear", C=0.025, probability=True)
pipe_s_svc = make_pipeline(trans, ROS, svc)
#pipe_s_svc

# %%
pipe_s_svc.fit(x_train, y_train)

# %%
scores_list.extend(
  scores('SVC', pipe_s_svc, x_train, y_train, x_test, y_test)
)

# %%
neigh = KNeighborsClassifier(n_neighbors=5)
pipe_s_neigh = make_pipeline(trans, ROS, neigh)
#pipe_s_neigh

# %%
pipe_s_neigh.fit(x_train, y_train)

# %%
scores_list.extend(
  scores('KNN', pipe_s_neigh, x_train, y_train, x_test, y_test)
)

# %% 
naive_bayes = GaussianNB()
pipe_s_nb = make_pipeline(trans, ROS, naive_bayes)
#pipe_s_nb

# %%
pipe_s_nb.fit(x_train, y_train)

# %%
scores_list.extend(
  scores('Naive Bayes', pipe_s_nb, x_train, y_train, x_test, y_test)
)

# %%
ada_boost = AdaBoostClassifier(n_estimators=100, random_state=0)
pipe_s_ab = make_pipeline(trans, ROS, ada_boost)
#pipe_s_ab

# %%
pipe_s_ab.fit(x_train, y_train)

# %%
scores_list.extend(
  scores('Ada Boost', pipe_s_ab, x_train, y_train, x_test, y_test)
)

# %%
gradient_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
pipe_s_gb = make_pipeline(trans, ROS, gradient_boost)
#pipe_s_gb

# %%
pipe_s_gb.fit(x_train, y_train)

# %%
scores_list.extend(
  scores('Gradient Boost', pipe_s_gb, x_train, y_train, x_test, y_test)
)

# %%
bagging_random_forest = BaggingClassifier(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), n_estimators=10, random_state=0)
pipe_s_brf = make_pipeline(trans, ROS, bagging_random_forest)

# %%
pipe_s_brf.fit(x_train, y_train)

# %%
scores_list.extend(
  scores('Bagging Random Forest', pipe_s_brf, x_train, y_train, x_test, y_test)
)

# %%
estimators = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)),
    ('bag', BaggingClassifier(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), n_estimators=10, random_state=0)),
]
clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)
pipe_s_stacking = make_pipeline(trans, ROS, clf)

# %%
pipe_s_stacking.fit(x_train, y_train)

# %%
scores_list.extend(
  scores('Stacking', pipe_s_stacking, x_train, y_train, x_test, y_test)
)

# %%
df_scores = pd.DataFrame(
    scores_list,
    columns=['Method', 'Score', 'Train', 'Test'],
)
df_scores.set_index(['Method', 'Score'], inplace=True)

# %%
df_scores

# %%
