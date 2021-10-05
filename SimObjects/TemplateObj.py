class ForceObjTemplate:
    @property
    def force():
        NotImplementedError
    @property
    def moment():
        NotImplementedError

class DynamicObjTemplate:
    def derivative(state):
        NotImplementedError

