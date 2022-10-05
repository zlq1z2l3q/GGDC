def create_model(opt):
    from .GGDC_model import GGDC, InferenceModel
    if opt.isTrain:
        model = GGDC()
    else:
        model = InferenceModel()

    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    return model
