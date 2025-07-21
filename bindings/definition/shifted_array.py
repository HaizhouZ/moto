class shifted_array(list):
    """
    A class that provides a shifted view of a given array.
    It allows for accessing elements with an offset, which can be useful in various applications.
    """

    def __init__(self, dtype, shift: int, size: int):
        self.dtype = dtype
        self.shift = shift
        self.size = size
        super().__init__([self.dtype()] * self.size)

    def __getitem__(self, index):
        assert 0 <= index - self.shift < self.size, "Index out of bounds"
        return super().__getitem__(index - self.shift)

    def __setitem__(self, index, value):
        assert 0 <= index - self.shift < self.size, "Index out of bounds"
        assert isinstance(value, self.dtype), f"Value must be of type {self.dtype}"
        super().__setitem__(index - self.shift, value)

    def __len__(self):
        return super().__len__()

    def __repr__(self):
        return f"shifted_array(dtype={self.dtype}, shift={self.shift}, size={self.size})"
