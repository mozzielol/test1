
from ex2_cnn import cnn_model
from utils import load_mnist_drift


if __name__=='__main__':
	X_train,y_train,X_test,y_test = load_mnist_drift(split=True)
	infer = cnn_model('info_gain')
	infer.fit(X_train,y_train)
	infer.evaluate(X_test,y_test)
	model = cnn_model('nor')
	model.fit(X_train,y_train)
	model.evaluate(X_test,y_test)
	para = cnn_model('para')
	para.fit(X_train,y_train)
	para.evaluate(X_test,y_test)
	