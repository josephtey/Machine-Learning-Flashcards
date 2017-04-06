from sklearn.externals import joblib
def predict(model, input, time_elapsed):
    h_power = model.predict(input)
    h = math.pow(2, h_power)
    p = math.pow(2, (-time_elapsed)/h)
    return p

model = joblib.load('models/model.pkl')
print(predict(model, [5,1,3.1,2,2], 200))