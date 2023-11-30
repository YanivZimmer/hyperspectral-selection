from feature_selector.feature_selector_l1_conv import FeatureSelectorL1Conv

class FeatureSelectionL1Wrapper:
    def __init__(
        self,
    ):
        print("?@?@?@?@?@?@?@?@?@?@?@?@?@?@?@?@?@?@?@?@?@?@?@?@?@?")

    def __init__(self, n_input_channels, threshold=1e-4):
        self.feature_selector = FeatureSelectorL1Conv(n_input_channels,threshold)

    def forward(self, x):
        return self.feature_selector.forward(x)

    def regularization(self):
        return self.feature_selector.regularization()
