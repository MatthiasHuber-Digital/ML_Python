from utils.model_loaders import OptionsClassifModel, get_zoo_model

if __name__ == "__main__":
    mdl_type = OptionsClassifModel.EFFICIENT_NET

    myfunc, model_options = get_zoo_model(model_class=mdl_type)

    print(myfunc(pat))
