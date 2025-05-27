from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# 总结文章链接
# https://blog.csdn.net/h3076817064/article/details/148267403?sharetype=blogdetail&sharerId=148267403&sharerefer=PC&sharesource=h3076817064&spm=1011.2480.3001.8118

data = load_digits()
x, y = data.data, data.target
x_stand = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_stand, y, test_size=0.2)
pca = PCA(n_components=0.95)
svm = SVC(kernel="rbf")
module = Pipeline([('pca', pca), ('svm', svm)])
module.fit(x_train, y_train)
y_predict = module.predict(x_test)
print(f"accuracy:{accuracy_score(y_test, y_predict):.2f}")
