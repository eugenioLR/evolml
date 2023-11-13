def restart_model(trained_model):
    return type(trained_model)(**trained_model.get_params())