from collections import OrderedDict, MutableMapping


class Cache(MutableMapping):
    # Cache with limited maximum capacit.
    # This is a simplified LRU caching scheme,
    #  when the cache is full and a new page is referenced which is not there in cache,
    # it will remove the least recently used frame to spare space for new page.
    # source: https://stackoverflow.com/questions/2437617/how-to-limit-the-size-of-a-dictionary
    def __init__(self, maxlen, items=None):
        self._maxlen = maxlen
        self.d = OrderedDict()
        if items:
            for k, v in items:
                self[k] = v

    @property
    def maxlen(self):
        return self._maxlen

    def __getitem__(self, key):
        self.d.move_to_end(key)
        return self.d[key]

    def __setitem__(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        elif len(self.d) == self.maxlen:
            self.d.popitem(last=False)
        self.d[key] = value

    def __delitem__(self, key):
        del self.d[key]

    def __iter__(self):
        return self.d.__iter__()

    def __len__(self):
        return len(self.d)


class MaxStack(list):
    '''
    A stack-like list class that can contain only a certain number of elements, 
    any additional elements pushed to this list that cause it to exceed the limit will result in the elements 
    at the tail of the stack being removed.
    '''

    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    # Simply here to adhere more to a stack
    def push(self, element):
        self.append(element)

    def append(self, element):
        super().append(element)
        # If the list has now exceeded the maximum size, remove the element at the tail of the list
        if super().__len__() > self.max_size:
            super().__delitem__(0)


class Dictate(object):
    """Object view of a dict, updating the passed in dict when values are set
    or deleted. "Dictate" the contents of a dict...: """

    def __init__(self, d):
        # since __setattr__ is overridden, self.__dict = d doesn't work
        object.__setattr__(self, '_Dictate__dict', d)

    # Dictionary-like access / updates
    def __getitem__(self, name):
        value = self.__dict[name]
        if isinstance(value, dict):  # recursively view sub-dicts as objects
            value = Dictate(value)
        return value

    def __setitem__(self, name, value):
        self.__dict[name] = value

    def __delitem__(self, name):
        del self.__dict[name]

    # Object-like access / updates
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.__dict)

    def __str__(self):
        return str(self.__dict)
