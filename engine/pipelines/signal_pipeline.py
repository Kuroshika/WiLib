from engine.pipelines import filters

class SignalPipeline:
    def __init__(self):
        self.pipeline = []

    def __call__(self, signal):
        for filter in self.pipeline:
            signal = filter(signal)
        return signal

    def add_transform(self, filter):
        assert isinstance(filter, filters.Filter)
        self.pipeline.append(filter)


