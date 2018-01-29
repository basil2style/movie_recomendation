
from surprise import Reader, Dataset, SVD, evaluate

# Define the format
reader = Reader(line_format='user item rating timestamp', sep='\t')

#Download dataset from http://files.grouplens.org/datasets/movielens/ml-100k.zip
# Load data from the dataset
data = Dataset.load_from_file('./ml-100k/u.data', reader=reader);

#Split data into 5 folds
data.split(n_folds=5)

#We are using SVD
algo = SVD()
evaluate(algo, data, measures=['RMSE','MAE'])

#retrive the trainset
trainset = data.build_full_trainset()
algo.fit(trainset)

#Actual predication
userid = str(196)
itemid = str(302)
actual_rating = 4

print(algo.predict(userid, 302, 4))