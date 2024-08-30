from .baseclass import Transform


class Pipeline(Transform):


    def __init__(self, transforms):
        self._transforms = transforms


    def __call__(self, x):
        
        for transform in self._transforms:

            x = transform(x)

        return x
