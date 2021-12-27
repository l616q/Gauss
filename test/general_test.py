from utils.bunch import Bunch


def replace_type(report_configure: dict):
    """
    This method will replace folder name in a json dict by recursion method.
    :param report_configure: Bunch object.
    :return: dict object.
    """
    if isinstance(report_configure, Bunch):
        report_configure = dict(report_configure)

    for key in report_configure.keys():
        if isinstance(report_configure[key], Bunch):
            report_configure[key] = dict(report_configure[key])
            replace_type(
                report_configure=report_configure[key]
            )
    return report_configure


def restore_type(report_configure: dict):
    """
    This method will replace folder name in a json dict by recursion method.
    :param report_configure: Bunch object.
    :return: dict object.
    """
    if not isinstance(report_configure, Bunch) and isinstance(report_configure, dict):
        report_configure = Bunch(**report_configure)

    for key in report_configure.keys():
        if not isinstance(report_configure[key], Bunch) and isinstance(report_configure[key], dict):
            report_configure[key] = Bunch(**report_configure[key])
            restore_type(
                report_configure=report_configure[key]
            )
    return report_configure


test = Bunch(
            entity_configure=Bunch(
                dataset=Bunch(success=Bunch(s="a", t=""), info="test"),
                feature_conf=Bunch(),
                loss=Bunch(),
                metric=Bunch(),
                model=Bunch()
            ),
            component_configure=Bunch(
                type_inference=Bunch(),
                data_clear=Bunch(),
                label_encode=Bunch(),
                feature_generation=Bunch(),
                unsupervised_feature_selector=Bunch(),
                supervised_feature_selector=Bunch()
            ),
            main_pipeline=Bunch(),
            success_flag=False
        )


report_configure = replace_type(report_configure=test)
print(report_configure)
print(report_configure.keys())
print(type(report_configure))
print(report_configure["entity_configure"])
print(type(report_configure["entity_configure"]))
print(type(report_configure["entity_configure"]["dataset"]))
print(type(report_configure["entity_configure"]["dataset"]["success"]))

report_configure = restore_type(report_configure=report_configure)
print(report_configure)
print(report_configure.keys())
print(type(report_configure))
print(report_configure["entity_configure"])
print(type(report_configure["entity_configure"]))
print(type(report_configure["entity_configure"]["dataset"]))
print(type(report_configure["entity_configure"]["dataset"]["success"]))

