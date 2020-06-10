import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

movies = pd.read_csv("./ml-latest-small/movies.csv")
links = pd.read_csv("./ml-latest-small/links.csv")
ratings = pd.read_csv("./ml-latest-small/ratings.csv")
tags = pd.read_csv("./ml-latest-small/tags.csv")

reader = Reader()


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

trainset = data.build_full_trainset()
svd.fit(trainset)
print("Training complete")


# print(svd.predict(1, 302, 3))

#Testing
wish = input("Do you wish to predict?(y/n)")
while(wish == 'y'):
	uid = input("Enter User ID : ")
	mid = input("Enter Movie ID : ")
	prediction = svd.predict(uid, mid, 3)
	rating = "{:.2f}".format(prediction[3])
	print("Predicted rating is : " + str(rating))
	print("***************************")
	wish = input("Do you wish to predict?(y/n)")


