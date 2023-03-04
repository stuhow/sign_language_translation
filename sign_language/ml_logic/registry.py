from tensorflow.keras import models


def save_model_local(model):
    model.save('models.model.h5')

def load_model_local():
    model = models.load_model('models.model.h5')
    return model
